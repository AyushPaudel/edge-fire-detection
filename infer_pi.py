"""
Real-time fire classification + CAM heatmap on a Raspberry Pi 4 using
ONNX Runtime (CPU). Works with the FP32 model or the INT8-quantized one.

Pi setup (64-bit Raspberry Pi OS is required for onnxruntime wheels):
    python3 -m venv --system-site-packages env   # system pkgs => picamera2
    source env/bin/activate
    pip install -r req_pi.txt

Usage:
    python infer_pi.py --model fireclf_int8.onnx --cam_weights fireclf_cam_weights.npy
    python infer_pi.py --model fireclf_int8.onnx --source picam      # CSI camera
    python infer_pi.py --model fireclf_int8.onnx --source video.mp4 --no_display
    python infer_pi.py --model fireclf_int8.onnx --source test_clip.mp4 --no_display --save out.mp4  # over SSH

    --source: 0 (USB webcam) | picam (Pi camera via picamera2) | video path | RTSP URL
"""
import argparse
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

from preprocess import preprocess


class FrameGrabber:
    """
    Reads frames on a background thread and keeps only the newest one, so
    slow inference never builds up a backlog of stale frames (= lag).
    """

    def __init__(self, source):
        self.picam = None
        self.is_file = False
        if source == "picam":
            from picamera2 import Picamera2  # only needed for the CSI camera
            self.picam = Picamera2()
            self.picam.configure(self.picam.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"}))  # RGB888 == BGR in numpy
            self.picam.start()
        else:
            self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source: {source}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_file = not source.isdigit() and "://" not in source

        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        if not self.is_file:
            threading.Thread(target=self._loop, daemon=True).start()

    def _read(self):
        if self.picam is not None:
            return True, self.picam.capture_array()
        return self.cap.read()

    def _loop(self):
        while not self.stopped:
            ok, frame = self._read()
            if not ok:
                self.stopped = True
                break
            with self.lock:
                self.frame = frame

    def read(self):
        # Video files are read synchronously so no frames get skipped.
        if self.is_file:
            ok, frame = self._read()
            return frame if ok else None
        while self.frame is None and not self.stopped:
            time.sleep(0.005)
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        if self.picam is not None:
            self.picam.stop()
        else:
            self.cap.release()


class ONNXFireClassifier:
    def __init__(self, model_path, cam_weights_path, img_size=224, threads=4,
                 interp="linear2area"):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads   # Pi 4 has 4 Cortex-A72 cores
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, so,
                                         providers=["CPUExecutionProvider"])
        self.img_size = img_size
        self.interp = interp
        self.cam_weights = np.load(cam_weights_path).astype(np.float32)  # (576,)

    def infer(self, frame_bgr, compute_cam=True):
        x = preprocess(frame_bgr, self.img_size, self.interp)[None]
        outputs = ["logit", "features"] if compute_cam else ["logit"]
        res = self.sess.run(outputs, {"input": x})
        prob_fire = float(1.0 / (1.0 + np.exp(-res[0].item())))

        heatmap = None
        if compute_cam:
            feat = res[1][0]                                    # (576, 7, 7)
            heatmap = np.tensordot(self.cam_weights, feat, axes=([0], [0]))
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
        return prob_fire, heatmap


def overlay_cam(frame_bgr, heatmap, alpha=0.4):
    # Convert + colorize the tiny 7x7 map first, then upscale the uint8 color
    # image: avoids float math on all 640x480 pixels (~2x faster on the Pi).
    h, w = frame_bgr.shape[:2]
    hm = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame_bgr, 1 - alpha, hm, alpha, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fireclf_int8.onnx")
    parser.add_argument("--cam_weights", default="fireclf_cam_weights.npy")
    parser.add_argument("--source", default="0")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--smooth", type=float, default=0.6,
                        help="EMA factor on P(fire) across frames (0 = off); "
                             "suppresses one-frame false alarms")
    parser.add_argument("--no_cam", action="store_true")
    parser.add_argument("--no_display", action="store_true",
                        help="no window (needed over SSH / without a monitor)")
    parser.add_argument("--save", default=None, metavar="OUT.mp4",
                        help="write the annotated video (works headless)")
    parser.add_argument("--save_fps", type=float, default=10)
    parser.add_argument("--interp", default="linear2area",
                        choices=["area", "linear2area", "linear"],
                        help="resize method (see preprocess.py and README: Pipeline optimization)")
    args = parser.parse_args()

    clf = ONNXFireClassifier(args.model, args.cam_weights, args.img_size, args.threads,
                              args.interp)
    grabber = FrameGrabber(args.source)
    print(f"Running {args.model} with {args.threads} threads. Press q to quit.")

    p_smooth, fps_smooth, last_alert = None, None, 0.0
    writer = None
    infer_ms, n_frames, n_fire_frames = [], 0, 0
    t_start = time.perf_counter()
    try:
        while True:
            frame = grabber.read()
            if frame is None:
                break

            # Timed section = preprocess + model + CAM (the full per-frame
            # pipeline cost, unlike benchmark.py which times the model only).
            t0 = time.perf_counter()
            prob, heatmap = clf.infer(frame, compute_cam=not args.no_cam)
            dt = time.perf_counter() - t0
            infer_ms.append(dt * 1000)
            n_frames += 1
            fps_smooth = 1 / dt if fps_smooth is None else 0.9 * fps_smooth + 0.1 / dt
            p_smooth = prob if p_smooth is None else \
                args.smooth * p_smooth + (1 - args.smooth) * prob

            is_fire = p_smooth > args.thresh
            n_fire_frames += int(is_fire)
            if is_fire and time.time() - last_alert > 1.0:   # rate-limit prints
                print(f"[ALERT] fire detected: P(fire)={p_smooth:.2f}")
                last_alert = time.time()

            if args.no_display and not args.save:
                continue   # headless & nothing to save: skip drawing entirely
            display = overlay_cam(frame, heatmap) if (is_fire and heatmap is not None) else frame
            label = f"FIRE {p_smooth:.2f}" if is_fire else f"clear {p_smooth:.2f}"
            color = (0, 0, 255) if is_fire else (0, 200, 0)
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display, f"{dt*1000:.0f} ms | {fps_smooth:.1f} FPS", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if args.save:
                if writer is None:
                    h, w = display.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.save_fps, (w, h))
                writer.write(display)
            if not args.no_display:
                cv2.imshow("Fire Classifier (Pi)", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass   # Ctrl+C in headless mode still prints the summary below
    finally:
        grabber.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if n_frames:
        wall = time.perf_counter() - t_start
        t = np.array(infer_ms)
        print(f"\nSummary: {n_frames} frames, fire on {n_fire_frames} "
              f"({n_fire_frames / n_frames:.0%})")
        print(f"  per-frame pipeline (preprocess+model{'' if args.no_cam else '+CAM'}): "
              f"mean={t.mean():.1f} ms  p95={np.percentile(t, 95):.1f} ms  "
              f"(~{1000 / t.mean():.1f} FPS)")
        print(f"  end-to-end incl. video decode/drawing: {n_frames / wall:.1f} FPS")
        if args.save:
            print(f"  saved annotated video -> {args.save}")


if __name__ == "__main__":
    main()
