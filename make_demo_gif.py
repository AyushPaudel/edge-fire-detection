"""
Turn an annotated video from infer_pi.py --save into a README-sized GIF and
a still grid of heatmap examples.

Usage:
    python infer_pi.py --model fireclf_int8.onnx --source test_clip.mp4 --no_display --save out.mp4
    python make_demo_gif.py --video out.mp4 --gif assets/demo.gif --grid assets/heatmaps.jpg
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image


def read_frames(path, step):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gif", default="assets/demo.gif")
    ap.add_argument("--grid", default="assets/heatmaps.jpg")
    ap.add_argument("--step", type=int, default=3, help="keep every Nth frame "
                    "(make_test_clip.py holds each image for 3 frames)")
    ap.add_argument("--per_block", type=int, default=20, help="images per block in the clip")
    ap.add_argument("--max_frames", type=int, default=40)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--ms_per_frame", type=int, default=700)
    args = ap.parse_args()

    frames = read_frames(args.video, args.step)
    if not frames:
        raise SystemExit(f"No frames read from {args.video}")
    # Spread the picks over the clip so both classes appear.
    idx = np.linspace(0, len(frames) - 1, min(args.max_frames, len(frames))).astype(int)
    h = int(frames[0].shape[0] * args.width / frames[0].shape[1])
    picked = [cv2.resize(frames[i], (args.width, h), interpolation=cv2.INTER_AREA) for i in idx]

    os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=128) for f in picked]
    imgs[0].save(args.gif, save_all=True, append_images=imgs[1:],
                 duration=args.ms_per_frame, loop=0, optimize=True)
    print(f"Wrote {args.gif}: {len(imgs)} frames, {os.path.getsize(args.gif) / 1e6:.1f} MB")

    # Grid: use the clip's known block structure (make_test_clip.py: blocks of
    # --per_block images, alternating non-fire / fire) and take images from the
    # MIDDLE of blocks, so the smoothed label has settled (no transition lag).
    per = args.per_block
    mid = [k for k in range(len(frames)) if per // 4 <= k % per < per - per // 4]
    fire_k = [k for k in mid if (k // per) % 2 == 1]
    clear_k = [k for k in mid if (k // per) % 2 == 0]
    pick = [fire_k[i] for i in np.linspace(0, len(fire_k) - 1, 4).astype(int)] + \
           [clear_k[i] for i in np.linspace(0, len(clear_k) - 1, 2).astype(int)]
    tw = 320
    th = int(tw * frames[0].shape[0] / frames[0].shape[1])
    tiles = [cv2.resize(frames[k], (tw, th), interpolation=cv2.INTER_AREA) for k in pick]
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    cv2.imwrite(args.grid, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"Wrote {args.grid}")


if __name__ == "__main__":
    main()
