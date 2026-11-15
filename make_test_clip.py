"""
Build a short demo video from held-out test images, so infer_pi.py can be
tested without a camera (on the Mac or on the Pi).

The clip alternates blocks of non-fire and fire images, so you should see
the label flip between "clear" and "FIRE" (with the heatmap) as it plays.

Usage:
    python make_test_clip.py --test_dir dataset_clean/test --out test_clip.mp4
    python infer_pi.py --model fireclf_int8.onnx --source test_clip.mp4
"""
import argparse
import os
import random

import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="dataset_clean/test")
    ap.add_argument("--out", default="test_clip.mp4")
    ap.add_argument("--blocks", type=int, default=4, help="number of non-fire/fire block pairs")
    ap.add_argument("--per_block", type=int, default=20, help="images per block")
    ap.add_argument("--hold", type=int, default=3, help="frames each image is shown")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pools = {}
    for cls in ["non fire", "fire"]:
        d = os.path.join(args.test_dir, cls)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))
        rng.shuffle(files)
        pools[cls] = [os.path.join(d, f) for f in files]

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640, 480))
    n = 0
    for b in range(args.blocks):
        for cls in ["non fire", "fire"]:
            for path in pools[cls][b * args.per_block:(b + 1) * args.per_block]:
                img = cv2.imread(path)
                if img is None:
                    continue
                frame = cv2.resize(img, (640, 480))
                for _ in range(args.hold):
                    writer.write(frame)
                    n += 1
    writer.release()
    print(f"Wrote {args.out}: {n} frames ({n / args.fps:.0f} s at {args.fps} FPS)")


if __name__ == "__main__":
    main()
