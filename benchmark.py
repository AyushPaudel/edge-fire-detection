"""
Benchmark one or more ONNX models: latency, model size, and accuracy /
precision / recall / F1 for the fire class. Produces the numbers for the
README / CV (FP32 vs INT8 on the Pi).

Run latency ON THE PI (that's the number that matters). Accuracy is the same
on any machine, so you can run it on the Mac with the full test set, or on
the Pi with --max_per_class to keep it quick.

    python benchmark.py --models fireclf.onnx fireclf_int8.onnx \
        --test_dir dataset/test --threads 4 --runs 200

    python benchmark.py --models fireclf.onnx fireclf_int8.onnx --no_accuracy
"""
import argparse
import os
import platform
import time

import numpy as np
import onnxruntime as ort

from preprocess import preprocess, load_bgr

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def bench_latency(sess, img_size, runs, warmup=20):
    x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(warmup):
        sess.run(["logit"], {"input": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(["logit"], {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"mean": t.mean(), "p50": np.percentile(t, 50),
            "p95": np.percentile(t, 95), "fps": 1000 / t.mean()}


def bench_pipeline(path, frame_source, img_size, threads, runs, interp, warmup=20):
    """
    Time each stage of infer_pi.py's per-frame pipeline separately on a real
    frame: preprocess, model (logit+features), CAM heatmap, overlay drawing.
    Tells you where the time goes beyond the model itself.
    """
    import cv2
    from infer_pi import ONNXFireClassifier, overlay_cam

    cap = cv2.VideoCapture(frame_source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {frame_source}")

    clf = ONNXFireClassifier(path, os.path.join(os.path.dirname(path) or ".",
                                                "fireclf_cam_weights.npy"), img_size, threads, interp)
    x = preprocess(frame, img_size, interp)[None]
    feat = clf.sess.run(["logit", "features"], {"input": x})[1][0]
    heat = np.maximum(np.tensordot(clf.cam_weights, feat, axes=([0], [0])), 0)
    heat /= max(heat.max(), 1e-6)

    stages = {
        f"preprocess ({interp})": lambda: preprocess(frame, img_size, interp),
        "model (logit+features)": lambda: clf.sess.run(["logit", "features"], {"input": x}),
        "CAM heatmap": lambda: np.tensordot(clf.cam_weights, feat, axes=([0], [0])),
        "overlay (resize+colormap+blend)": lambda: overlay_cam(frame, heat),
    }
    print(f"  frame {frame.shape[1]}x{frame.shape[0]} from {frame_source}")
    total = 0.0
    for name, fn in stages.items():
        for _ in range(warmup):
            fn()
        t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            t.append((time.perf_counter() - t0) * 1000)
        m = float(np.mean(t))
        total += m
        print(f"    {name:32s} mean={m:6.2f} ms  p95={np.percentile(t, 95):6.2f} ms")
    print(f"    {'TOTAL':32s} mean={total:6.2f} ms  (~{1000 / total:.1f} FPS)")


def load_test_set(test_dir, max_per_class, img_size, interp="area"):
    """Returns (inputs, labels) with label 1 = fire, 0 = not fire."""
    xs, ys = [], []
    for cls in sorted(os.listdir(test_dir)):
        cdir = os.path.join(test_dir, cls)
        if not os.path.isdir(cdir):
            continue
        is_fire = 1 if cls == "fire" else 0
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith(IMG_EXTS))
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            img = load_bgr(os.path.join(cdir, f))
            if img is None:
                continue
            xs.append(preprocess(img, img_size, interp))
            ys.append(is_fire)
    return xs, np.array(ys)


def eval_accuracy(sess, xs, ys, thresh=0.5):
    probs = np.array([
        1 / (1 + np.exp(-sess.run(["logit"], {"input": x[None]})[0].item()))
        for x in xs
    ])
    pred = (probs > thresh).astype(int)
    tp = int(((pred == 1) & (ys == 1)).sum())
    fp = int(((pred == 1) & (ys == 0)).sum())
    fn = int(((pred == 0) & (ys == 1)).sum())
    acc = float((pred == ys).mean())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["fireclf.onnx", "fireclf_int8.onnx"])
    parser.add_argument("--test_dir", default="dataset/test")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--max_per_class", type=int, default=0,
                        help="limit test images per class (0 = all)")
    parser.add_argument("--no_accuracy", action="store_true")
    parser.add_argument("--pipeline", default=None, metavar="VIDEO_OR_IMAGE",
                        help="also time each stage of the infer_pi.py pipeline "
                             "(preprocess / model / CAM / overlay) on a frame from this file")
    parser.add_argument("--pipeline_interp", default="linear2area",
                        choices=["area", "linear2area", "linear"])
    parser.add_argument("--interp", default="area", choices=["area", "linear2area", "linear"],
                        help="resize method used for the accuracy test (see preprocess.py)")
    args = parser.parse_args()

    print(f"Host: {platform.machine()} | {platform.platform()} | "
          f"onnxruntime {ort.__version__} | threads={args.threads}")

    xs = ys = None
    if not args.no_accuracy:
        print(f"Loading test set from {args.test_dir} (resize: {args.interp}) ...")
        xs, ys = load_test_set(args.test_dir, args.max_per_class, args.img_size, args.interp)
        print(f"  {len(ys)} images ({int(ys.sum())} fire, {int((ys == 0).sum())} not fire)")

    rows = []
    for path in args.models:
        print(f"\n== {path}")
        sess = make_session(path, args.threads)
        row = {"model": os.path.basename(path), "size_mb": os.path.getsize(path) / 1e6}
        row.update(bench_latency(sess, args.img_size, args.runs))
        print(f"  latency mean={row['mean']:.1f} ms  p50={row['p50']:.1f}  "
              f"p95={row['p95']:.1f}  ({row['fps']:.1f} FPS)")
        if args.pipeline:
            print("  pipeline breakdown:")
            bench_pipeline(path, args.pipeline, args.img_size, args.threads, args.runs,
                           args.pipeline_interp)
        if xs is not None:
            row.update(eval_accuracy(sess, xs, ys))
            print(f"  acc={row['acc']:.4f}  precision={row['precision']:.4f}  "
                  f"recall={row['recall']:.4f}  f1={row['f1']:.4f}")
        rows.append(row)

    # Markdown table, ready to paste into a README.
    print("\n| Model | Size (MB) | Mean latency (ms) | p95 (ms) | FPS |"
          + (" Accuracy | Precision | Recall | F1 |" if xs is not None else ""))
    print("|---|---|---|---|---|" + ("---|---|---|---|" if xs is not None else ""))
    for r in rows:
        line = (f"| {r['model']} | {r['size_mb']:.2f} | {r['mean']:.1f} | "
                f"{r['p95']:.1f} | {r['fps']:.1f} |")
        if xs is not None:
            line += (f" {r['acc']:.4f} | {r['precision']:.4f} | "
                     f"{r['recall']:.4f} | {r['f1']:.4f} |")
        print(line)


if __name__ == "__main__":
    main()
