"""
Static INT8 post-training quantization of the exported ONNX model, for fast
CPU inference on a Raspberry Pi 4 (ARM Cortex-A72) with ONNX Runtime.

Run this on your Mac/PC (not the Pi), after export.py:

    python export.py --ckpt checkpoints/best.pt --out fireclf.onnx
    python quantize.py --model fireclf.onnx --calib_dir dataset/train \
        --out fireclf_int8.onnx --num_calib 300

"Static" means activation ranges are measured once, here, by running a few
hundred real training images through the model (calibration). The Pi then
runs pure INT8 math with no runtime range estimation.
"""
import argparse
import os
import random

import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from preprocess import preprocess, load_bgr

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Quantization-sensitive nodes found by `diagnose_quant.py --per_node`.
STEM_NODES = [
    "/model/features/features.0/features.0.0/Conv",
    "/model/features/features.0/features.0.2/Mul",
    "/model/features/features.1/block/block.0/block.0.0/Conv",
]


def list_images(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, f))
    return paths


class FireCalibReader(CalibrationDataReader):
    """Feeds preprocessed calibration images one at a time."""

    def __init__(self, image_paths, img_size):
        self.image_paths = image_paths
        self.img_size = img_size
        self._iter = iter(self.image_paths)

    def get_next(self):
        for path in self._iter:
            img = load_bgr(path)
            if img is None:
                continue
            return {"input": preprocess(img, self.img_size)[None]}
        return None

    def rewind(self):
        self._iter = iter(self.image_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fireclf.onnx")
    parser.add_argument("--calib_dir", default="dataset/train",
                        help="folder of images (searched recursively); "
                             "use training images, never the test set")
    parser.add_argument("--out", default="fireclf_int8.onnx")
    parser.add_argument("--num_calib", type=int, default=300)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--method", choices=["minmax", "entropy", "percentile"],
                        default="minmax")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep_fp32_ops", nargs="*", default=[],
                        help="op types to leave in FP32, e.g. HardSigmoid Mul "
                             "(MobileNetV3's hard-swish / squeeze-excite parts)")
    parser.add_argument("--keep_head_fp32", action="store_true",
                        help="leave the final GAP + Linear (fc) in FP32")
    parser.add_argument("--keep_last_convs_fp32", type=int, default=0,
                        help="leave the last N Conv layers in FP32")
    parser.add_argument("--keep_stem_fp32", action="store_true",
                        help="RECOMMENDED for MobileNetV3: keep the 3 stem nodes that "
                             "diagnose_quant.py found cause ~92%% of the INT8 error in FP32")
    parser.add_argument("--keep_fp32_nodes", nargs="*", default=[],
                        help="node names (or name prefixes) to leave in FP32 -- "
                             "take them from diagnose_quant.py --per_node output")
    args = parser.parse_args()

    # Balanced random sample across both classes (fire / non fire).
    random.seed(args.seed)
    class_dirs = sorted(
        os.path.join(args.calib_dir, d) for d in os.listdir(args.calib_dir)
        if os.path.isdir(os.path.join(args.calib_dir, d))
    )
    per_class = max(1, args.num_calib // max(1, len(class_dirs)))
    calib_paths = []
    for d in class_dirs:
        imgs = list_images(d)
        random.shuffle(imgs)
        calib_paths += imgs[:per_class]
    random.shuffle(calib_paths)
    print(f"Calibrating on {len(calib_paths)} images from {class_dirs}")

    # Shape inference + graph optimization first; recommended by ORT before
    # quantizing, and avoids quantizing ops that would be fused away anyway.
    prep_path = args.model.replace(".onnx", "_prep.onnx")
    quant_pre_process(args.model, prep_path, skip_symbolic_shape=True)

    method = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[args.method]

    # Pick nodes to leave in FP32 (mixed precision) — quantization-sensitive
    # layers can be excluded at a small latency cost.
    graph = onnx.load(prep_path).graph
    exclude = [n.name for n in graph.node if n.op_type in args.keep_fp32_ops]
    if args.keep_head_fp32:
        exclude += [n.name for n in graph.node if n.name.startswith(("/model/fc", "/model/gap"))]
    if args.keep_last_convs_fp32:
        convs = [n.name for n in graph.node if n.op_type == "Conv"]
        exclude += convs[-args.keep_last_convs_fp32:]
    keep_nodes = list(args.keep_fp32_nodes)
    if args.keep_stem_fp32:
        # First conv on the raw image, its hard-swish, and the next conv that
        # reads that hard-swish output (see README: Fixing INT8 accuracy).
        keep_nodes += [n for n in STEM_NODES if n not in keep_nodes]
    if keep_nodes:
        missing = [k for k in keep_nodes if not any(n.name.startswith(k) for n in graph.node)]
        if missing:
            raise SystemExit(f"Node(s) not found in graph: {missing}")
        prefixes = tuple(keep_nodes)
        exclude += [n.name for n in graph.node if n.name.startswith(prefixes)]
    exclude = sorted(set(exclude))
    if exclude:
        print(f"Keeping {len(exclude)} nodes in FP32")

    quantize_static(
        model_input=prep_path,
        model_output=args.out,
        calibration_data_reader=FireCalibReader(calib_paths, args.img_size),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,   # per-channel weights: much smaller accuracy drop
        calibrate_method=method,
        nodes_to_exclude=exclude,
    )
    os.remove(prep_path)

    fp32_mb = os.path.getsize(args.model) / 1e6
    int8_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved {args.out}: {fp32_mb:.2f} MB (FP32) -> {int8_mb:.2f} MB (INT8)")
    print("\nCopy to the Pi: fireclf.onnx, fireclf_int8.onnx, "
          "fireclf_cam_weights.npy, preprocess.py, infer_pi.py, benchmark.py")


if __name__ == "__main__":
    main()
