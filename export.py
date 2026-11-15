"""
Export the trained FireClassifier to ONNX (for TensorRT), and separately
dump the final Linear layer's weights as a .npy file — the TensorRT-side
inference script needs these raw weights to compute the CAM heatmap, since
that computation happens in plain numpy after the TensorRT forward pass.

Usage:
    python export_onnx.py --ckpt checkpoints/best.pt --img_size 224 \
        --out fireclf.onnx
"""
import argparse
import numpy as np
import torch

from model import FireClassifier


class ONNXWrapper(torch.nn.Module):
    """Wraps the model so ONNX export gets two clean outputs: logit + features."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logit, feat = self.model(x, return_features=True)
        return logit, feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--out", default="fireclf.onnx")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    model = FireClassifier(pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    wrapper = ONNXWrapper(model).eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size)

    # dynamo=False forces the older, more stable ONNX exporter, which
    # respects opset_version exactly. This matters if your Jetson's
    # TensorRT (e.g. TensorRT 8.2 on JetPack 4.6) doesn't support the
    # newer opsets the default "dynamo" exporter may silently upgrade to.
    export_kwargs = dict(
        input_names=["input"],
        output_names=["logit", "features"],
        opset_version=args.opset,
        dynamic_axes=None,  # fixed batch=1 -- simplest and fastest on Nano
    )
    try:
        torch.onnx.export(wrapper, dummy, args.out, dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch versions don't have a `dynamo` argument at all --
        # their default exporter already behaves like dynamo=False.
        torch.onnx.export(wrapper, dummy, args.out, **export_kwargs)
    print(f"Exported ONNX model to {args.out}")

    cam_weights_path = args.out.replace(".onnx", "_cam_weights.npy")
    np.save(cam_weights_path, model.cam_weights())
    print(f"Saved CAM weight vector to {cam_weights_path} "
          f"(shape={model.cam_weights().shape})")

    print(f"\nNext (Raspberry Pi path): python quantize.py --model {args.out}")
    print("(Jetson only: trtexec --onnx=... --saveEngine=fireclf_fp16.trt --fp16)")


if __name__ == "__main__":
    main()