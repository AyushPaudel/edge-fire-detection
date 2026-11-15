"""
Audit an image-classification dataset for the problems that silently
inflate accuracy and make models overconfident:

  1. Exact duplicate files (same bytes) inside each split/class.
  2. The same image appearing in more than one split (train AND test).
  3. The same image labelled with two different classes.
  4. Near-duplicates (e.g. adjacent video frames) shared between train and
     the evaluation splits, found with a 64-bit difference hash (dHash).
  5. Basic stats: counts, image sizes, unreadable files.

Works with any layout of the form  <root>/<split>/<class>/<images>
(e.g. fire_detect/dataset, fire_detect/dataset_clean, monkey_detect/dataset).
Read-only: it never modifies the dataset.

Usage:
    python audit_dataset.py dataset
    python audit_dataset.py dataset_clean --thresh 4
    python audit_dataset.py ../monkey_detect/dataset --train_split train

How dHash works (why it finds near-duplicates): shrink the image to 9x8
grayscale and record, for each row, whether each pixel is brighter than
its right neighbour -> 64 bits. Small edits (resize, recompress, next
video frame) flip only a few bits, so two images whose hashes differ in
<= ~4 bits (Hamming distance) are almost certainly the same scene.
"""
import argparse
import collections
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from make_clean_split import IMG_EXTS, hash_image


def image_size(path):
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset root containing <split>/<class>/ folders")
    ap.add_argument("--thresh", type=int, default=4, help="max dHash Hamming distance "
                    "to count as a near-duplicate")
    ap.add_argument("--train_split", default="train")
    args = ap.parse_args()

    items = []  # (path, split, cls)
    for split in sorted(os.listdir(args.root)):
        sdir = os.path.join(args.root, split)
        if not os.path.isdir(sdir):
            continue
        for cls in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir):
                continue
            items += [(os.path.join(cdir, f), split, cls) for f in sorted(os.listdir(cdir))
                      if f.lower().endswith(IMG_EXTS)]
    print(f"Scanning {len(items)} images in {args.root} ...")

    paths = [p for p, _, _ in items]
    with ProcessPoolExecutor() as ex:
        hashed = list(ex.map(hash_image, paths, chunksize=64))
        sizes = list(ex.map(image_size, paths, chunksize=64))

    rows = [(s, c, md5, dh, wh) for (_, s, c), (_, md5, dh), wh in zip(items, hashed, sizes)]
    unreadable = sum(md5 is None for _, _, md5, _, _ in rows)
    rows = [r for r in rows if r[2] is not None]

    # --- 1 + 5: per split/class stats ---------------------------------------
    print(f"\n[1] Per split/class (unreadable files: {unreadable})")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r[0], r[1])].append(r)
    for (s, c), g in sorted(groups.items()):
        n_unique = len({r[2] for r in g})
        w = np.array([r[4][0] for r in g])
        h = np.array([r[4][1] for r in g])
        print(f"  {s}/{c:12s} files={len(g):6d} unique={n_unique:6d} "
              f"duplicate copies={len(g) - n_unique:5d} ({(len(g) - n_unique) / len(g):.0%})  "
              f"median size={int(np.median(w))}x{int(np.median(h))}  "
              f"smallest side<100px: {int((np.minimum(w, h) < 100).sum())}")

    # --- 2 + 3: exact overlap across splits / classes -----------------------
    where = collections.defaultdict(set)
    for s, c, md5, _, _ in rows:
        where[md5].add((s, c))
    multi_split = sum(len({s for s, _ in v}) > 1 for v in where.values())
    multi_class = sum(len({c for _, c in v}) > 1 for v in where.values())
    print(f"\n[2] Unique images present in more than one split: {multi_split}"
          f"   {'<-- LEAKAGE' if multi_split else 'OK'}")
    print(f"[3] Unique images with conflicting class labels:   {multi_class}"
          f"   {'<-- LABEL CONFLICT' if multi_class else 'OK'}")

    # --- 4: near-duplicates of eval images in train -------------------------
    train_h = np.array([r[3] for r in rows if r[0] == args.train_split], dtype=np.uint64)
    if len(train_h) == 0:
        print(f"\n[4] No split named '{args.train_split}'; skipping near-duplicate check")
        return
    print(f"\n[4] Eval images with a near-duplicate (dHash distance <= {args.thresh}) in "
          f"'{args.train_split}':")
    for (s, c), g in sorted(groups.items()):
        if s == args.train_split:
            continue
        h = np.array([r[3] for r in g], dtype=np.uint64)
        near = 0
        for i in range(0, len(h), 256):
            dist = np.bitwise_count(h[i:i + 256, None] ^ train_h[None]).min(1)
            near += int((dist <= args.thresh).sum())
        print(f"  {s}/{c:12s} {near:5d}/{len(g)} ({near / len(g):.0%})"
              f"   {'<-- LEAKAGE' if near / len(g) > 0.02 else 'OK'}")
    print("\nA small % (<~2%) can be dHash false matches; large % means the "
          "eval split is not measuring generalization.")


if __name__ == "__main__":
    main()
