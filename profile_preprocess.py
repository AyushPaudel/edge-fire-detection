"""
Profile the preprocessing step on the target device and compare faster
variants. Preprocessing turned out to cost ~7.6 ms per frame on the Pi 4
(more than half the INT8 model's time), so it's worth optimizing.

For each variant it reports time per frame and the max/mean absolute
difference from the reference implementation (preprocess.py) — a variant is
only usable if it's faster AND numerically (nearly) identical, or if its
accuracy is re-verified with benchmark.py.

Usage (run on the Pi):
    python profile_preprocess.py --source test_clip.mp4
"""
import argparse
import time

import cv2
import numpy as np

from preprocess import MEAN, STD, preprocess


def reference(frame, size):
    """The ORIGINAL preprocess (before Entry 11's optimization), kept for comparison."""
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)

SIZE = 224


def sub_steps(frame):
    """Time each line of the reference preprocess separately."""
    steps = {}
    t = time.perf_counter
    t0 = t(); img = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA); steps["resize INTER_AREA"] = t() - t0
    t0 = t(); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); steps["BGR->RGB"] = t() - t0
    t0 = t(); img = img.astype(np.float32) / 255.0; steps["to float /255"] = t() - t0
    t0 = t(); img = (img - MEAN) / STD; steps["(x-mean)/std"] = t() - t0
    t0 = t(); np.ascontiguousarray(img.transpose(2, 0, 1)); steps["HWC->CHW copy"] = t() - t0
    return steps


def timeit(fn, frame, runs, warmup=20):
    for _ in range(warmup):
        fn(frame)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(frame)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(t)), float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="test_clip.mp4")
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {args.source}")
    print(f"Frame {frame.shape[1]}x{frame.shape[0]}, cv2 {cv2.__version__}, "
          f"numpy {np.__version__}, cv2 threads={cv2.getNumThreads()}")

    print("\nOriginal preprocess, per sub-step (mean over runs):")
    acc = {}
    for _ in range(args.runs):
        for k, v in sub_steps(frame).items():
            acc[k] = acc.get(k, 0.0) + v
    for k, v in acc.items():
        print(f"  {k:20s} {v / args.runs * 1000:6.2f} ms")

    ref = reference(frame, SIZE)
    print("\nVariants (time | difference from original):")
    for name, fn in [("original (3-pass normalize)", lambda f: reference(f, SIZE)),
                     ("preprocess.py interp=area", lambda f: preprocess(f, SIZE, "area")),
                     ("preprocess.py interp=linear2area", lambda f: preprocess(f, SIZE, "linear2area")),
                     ("preprocess.py interp=linear", lambda f: preprocess(f, SIZE, "linear"))]:
        mean, p95 = timeit(fn, frame, args.runs)
        diff = np.abs(fn(frame) - ref)
        print(f"  {name:28s} mean={mean:6.2f} ms  p95={p95:6.2f} ms  "
              f"max|diff|={diff.max():.2e}  mean|diff|={diff.mean():.2e}")


if __name__ == "__main__":
    main()
