"""
Real-time fire classification (+ optional CAM heatmap) on Jetson Nano using
a TensorRT engine. Run this ON the Nano.

Requirements on the Nano (JetPack 4.6.x ships tensorrt + pycuda already):
    - tensorrt, pycuda (system python from JetPack)
    - opencv-python (JetPack-provided build recommended for CSI camera use)
    - numpy

Usage:
    python3 trt_infer.py --engine fireclf_fp16.trt \
        --cam_weights fireclf_cam_weights.npy \
        --source 0 --img_size 224

    --source: 0 (USB webcam) | csi (Jetson CSI camera) | video path | RTSP URL
    --no_cam: skip heatmap overlay, just print fire/not_fire (slightly faster)
"""
import argparse
import time

import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def gstreamer_pipeline(capture_width=1280, capture_height=720, framerate=30):
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), "
        f"width={capture_width}, height={capture_height}, "
        f"framerate={framerate}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


class TRTFireClassifier:
    def __init__(self, engine_path, cam_weights_path, img_size=224):
        self.img_size = img_size
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Binding order matches export order: input, logit, features
        self.idx_input = self.engine.get_binding_index("input")
        self.idx_logit = self.engine.get_binding_index("logit")
        self.idx_feat = self.engine.get_binding_index("features")

        feat_shape = tuple(self.engine.get_binding_shape(self.idx_feat))  # (1, 576, 7, 7)
        self.feat_shape = feat_shape

        self.h_input = cuda.pagelocked_empty(3 * img_size * img_size, dtype=np.float32)
        self.h_logit = cuda.pagelocked_empty(1, dtype=np.float32)
        self.h_feat = cuda.pagelocked_empty(int(np.prod(feat_shape)), dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_logit = cuda.mem_alloc(self.h_logit.nbytes)
        self.d_feat = cuda.mem_alloc(self.h_feat.nbytes)
        self.stream = cuda.Stream()

        self.cam_weights = np.load(cam_weights_path)  # shape (576,)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, frame_bgr):
        img = cv2.resize(frame_bgr, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        return np.ascontiguousarray(img, dtype=np.float32)

    def infer(self, frame_bgr, compute_cam=True):
        inp = self.preprocess(frame_bgr)
        np.copyto(self.h_input, inp.ravel())

        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        bindings = [None] * self.engine.num_bindings
        bindings[self.idx_input] = int(self.d_input)
        bindings[self.idx_logit] = int(self.d_logit)
        bindings[self.idx_feat] = int(self.d_feat)
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_logit, self.d_logit, self.stream)
        if compute_cam:
            cuda.memcpy_dtoh_async(self.h_feat, self.d_feat, self.stream)
        self.stream.synchronize()

        logit = float(self.h_logit[0])
        prob_fire = 1 / (1 + np.exp(-logit))

        heatmap = None
        if compute_cam:
            feat = self.h_feat.reshape(self.feat_shape)[0]  # (576, h, w)
            # CAM: weighted sum of feature maps using the classifier's own
            # learned weights -- this is what makes fire-ish regions light up.
            heatmap = np.tensordot(self.cam_weights, feat, axes=([0], [0]))  # (h, w)
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

        return prob_fire, heatmap


def overlay_cam(frame_bgr, heatmap, alpha=0.4):
    h, w = frame_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_color, alpha, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--cam_weights", required=True)
    parser.add_argument("--source", default="0")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--no_cam", action="store_true")
    parser.add_argument("--no_display", action="store_true")
    args = parser.parse_args()

    if args.source == "csi":
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    elif args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    clf = TRTFireClassifier(args.engine, args.cam_weights, img_size=args.img_size)

    fps_smooth = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        prob_fire, heatmap = clf.infer(frame, compute_cam=not args.no_cam)
        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = fps if fps_smooth is None else 0.9 * fps_smooth + 0.1 * fps

        is_fire = prob_fire > args.thresh
        if is_fire:
            print(f"[ALERT] fire detected: P(fire)={prob_fire:.2f}")

        display = frame
        if heatmap is not None and is_fire:
            display = overlay_cam(frame, heatmap)

        label = f"FIRE {prob_fire:.2f}" if is_fire else f"clear {prob_fire:.2f}"
        color = (0, 0, 255) if is_fire else (0, 200, 0)
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display, f"FPS: {fps_smooth:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not args.no_display:
            cv2.imshow("Fire Classifier", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()