"""
Shared numpy/OpenCV preprocessing for ONNX inference (quantize, infer_pi,
benchmark). Must match dataset.get_eval_transforms(): plain resize to a
square (no crop), RGB, scale to [0,1], ImageNet mean/std, CHW float32.

Optimized for the Pi 4 (see README: Pipeline optimization, profile_preprocess.py):
normalization is ONE fused multiply-add with precomputed per-channel
constants instead of three passes (/255, -mean, /std): identical output,
~3 ms faster per frame on the Pi.
"""
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# (x/255 - mean)/std  ==  x * SCALE + BIAS   (per RGB channel, shaped for CHW)
SCALE = (1.0 / (255.0 * STD)).astype(np.float32)[:, None, None]
BIAS = (-MEAN / STD).astype(np.float32)[:, None, None]

INTERP = {
    # Closest OpenCV match to torchvision's antialiased resize used in training.
    "area": cv2.INTER_AREA,
    # Fastest, but aliases when shrinking a lot (640 -> 224): verify accuracy.
    "linear": cv2.INTER_LINEAR,
}


def load_bgr(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def resize(frame_bgr, img_size, interp="area"):
    if interp == "linear2area":
        # Bilinear to 2x the target, then an exact 2:1 area average (fast path).
        frame_bgr = cv2.resize(frame_bgr, (2 * img_size, 2 * img_size),
                               interpolation=cv2.INTER_LINEAR)
        return cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(frame_bgr, (img_size, img_size), interpolation=INTERP[interp])


def preprocess(frame_bgr, img_size=224, interp="area"):
    img = resize(frame_bgr, img_size, interp)
    # HWC uint8 BGR -> CHW, reversing channels (BGR->RGB) as a free view;
    # astype() makes the single contiguous float32 copy.
    out = img.transpose(2, 0, 1)[::-1].astype(np.float32)
    out *= SCALE
    out += BIAS
    return out
