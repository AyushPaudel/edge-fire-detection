"""
Diagnose WHY an INT8-quantized model loses accuracy.

Answers three questions, in order:
  1. How confident is the FP32 model? (logit ranges -- huge logits usually
     mean wide activation ranges, which 8-bit integers represent poorly)
  2. Is the INT8 damage just a shifted decision threshold, or did ranking
     itself break? (AUC: if INT8 AUC is still ~FP32 AUC, re-tuning the
     threshold is enough; if AUC dropped, the features are damaged)
  3. Which operator types are quantization-sensitive? Quantizes ONE op type
     at a time (everything else stays FP32) and measures accuracy/recall.

Run on the Mac (it re-quantizes several times; a few minutes):
    python diagnose_quant.py --fp32 fireclf.onnx --int8 fireclf_int8.onnx \
        --calib_dir dataset_clean/train --test_dir dataset_clean/test
"""
import argparse
import os
import random
import tempfile

import numpy as np
import onnx
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from benchmark import load_test_set, make_session
from quantize import FireCalibReader, list_images


def logits_of(sess, xs):
    return np.array([sess.run(["logit"], {"input": x[None]})[0].item() for x in xs])


def auc(scores, y):
    """Probability that a random fire image scores higher than a random non-fire one."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def summarize(name, lg, y):
    pred = (lg > 0).astype(int)
    acc = (pred == y).mean()
    rec = ((pred == 1) & (y == 1)).sum() / max(1, y.sum())
    print(f"  {name:28s} acc={acc:.3f} recall={rec:.3f} AUC={auc(lg, y):.4f} "
          f"logit fire mean={lg[y == 1].mean():6.2f} non-fire mean={lg[y == 0].mean():6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default="fireclf.onnx")
    ap.add_argument("--int8", default="fireclf_int8.onnx")
    ap.add_argument("--calib_dir", default="dataset_clean/train")
    ap.add_argument("--test_dir", default="dataset_clean/test")
    ap.add_argument("--max_per_class", type=int, default=200)
    ap.add_argument("--num_calib", type=int, default=300)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--per_node", nargs="*", default=None, metavar="OP_TYPE",
                    help="step [4]: quantize ONE node at a time for these op types "
                         "(e.g. --per_node Conv Mul) and rank nodes by damage. Slow.")
    args = ap.parse_args()

    xs, ys = load_test_set(args.test_dir, args.max_per_class, args.img_size)
    print(f"Test subset: {len(ys)} images ({int(ys.sum())} fire)\n")

    print("[1+2] FP32 vs INT8 (threshold at logit 0 = P(fire) 0.5)")
    lg32 = logits_of(make_session(args.fp32, 4), xs)
    summarize("FP32", lg32, ys)
    if os.path.exists(args.int8):
        lg8 = logits_of(make_session(args.int8, 4), xs)
        summarize("INT8 (full)", lg8, ys)
        print(f"  correlation FP32 vs INT8 logits: {np.corrcoef(lg32, lg8)[0, 1]:.3f}  "
              f"(near 1.0 = INT8 tracks FP32)")

    print("\n[3] Sensitivity: quantize ONE op type at a time")
    random.seed(0)
    calib = []
    for d in sorted(os.listdir(args.calib_dir)):
        imgs = list_images(os.path.join(args.calib_dir, d))
        random.shuffle(imgs)
        calib += imgs[:args.num_calib // 2]

    with tempfile.TemporaryDirectory() as tmp:
        prep = os.path.join(tmp, "prep.onnx")
        quant_pre_process(args.fp32, prep, skip_symbolic_shape=True)
        op_types = sorted({n.op_type for n in onnx.load(prep).graph.node}
                          - {"Flatten", "Reshape", "Shape"})
        for op in op_types:
            out = os.path.join(tmp, f"{op}.onnx")
            quantize_static(prep, out, FireCalibReader(calib, args.img_size),
                            quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            per_channel=True, op_types_to_quantize=[op])
            summarize(f"only {op}", logits_of(make_session(out, 4), xs), ys)

        if args.per_node:
            # [4] Per-node sensitivity: which individual layers do the damage?
            # Metric: mean |INT8 logit - FP32 logit| when only that node is
            # quantized. A few nodes with huge error => keep just those FP32.
            nodes = [n.name for n in onnx.load(prep).graph.node if n.op_type in args.per_node]
            print(f"\n[4] Per-node sensitivity over {len(nodes)} nodes "
                  f"({', '.join(args.per_node)}) -- this takes a while")
            results = []
            for i, name in enumerate(nodes):
                out = os.path.join(tmp, "node.onnx")
                quantize_static(prep, out, FireCalibReader(calib, args.img_size),
                                quant_format=QuantFormat.QDQ,
                                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                                per_channel=True, nodes_to_quantize=[name])
                err = np.abs(logits_of(make_session(out, 4), xs) - lg32).mean()
                results.append((err, name))
                print(f"  {i + 1:3d}/{len(nodes)} err={err:.3f}  {name}", flush=True)
            results.sort(reverse=True)
            total = sum(e for e, _ in results)
            print("\n  Most sensitive nodes (share of total error):")
            cum = 0.0
            for err, name in results[:15]:
                cum += err
                print(f"   err={err:.3f}  cumulative={cum / total:5.1%}  {name}")

    print("\nRead it like this: op types whose row drops far below FP32 are the "
          "sensitive ones -> keep them FP32 (quantize.py --keep_fp32_ops) or use QAT.")


if __name__ == "__main__":
    main()
