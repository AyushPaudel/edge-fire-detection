"""
Build a leakage-free train/val/test split from the original dataset.

Problems found in the original dataset/ (audit on 2026-11-09):
  - 21,733 files but only 14,917 unique images (exact byte duplicates; the
    7805/7805 train balance was reached by duplicating images).
  - 2,122 unique images appear in BOTH train and test, and ~54% of test
    images have a near-duplicate (e.g. adjacent video frame) in train.
  -> test accuracy was inflated and the model learned to be overconfident.

What this script does:
  1. Exact dedupe by MD5 (keeps one copy of each image).
  2. Near-duplicate grouping with a 64-bit difference hash (dHash): images
     within Hamming distance --hash_thresh are union-found into one group.
  3. Group-aware, class-stratified split into train/val/test, so all
     near-duplicates of an image land in the same split.
  4. Writes dataset_clean/{train,val,test}/{fire,non fire}/ as HARD LINKS
     (no extra disk space, and they copy to the Pi like normal files).
     The original dataset/ is never modified.

Usage:
    python make_clean_split.py --src dataset --dst dataset_clean
"""
import argparse
import collections
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")


def hash_image(path):
    """Returns (path, md5, dhash) or (path, None, None) if unreadable."""
    try:
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        im = Image.open(path)
        im.seek(0)
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        dhash = int("".join("1" if b else "0" for b in bits), 2)
        return path, md5, dhash
    except Exception:
        return path, None, None


def group_near_duplicates(hashes, thresh):
    """Union-find over all pairs with Hamming(dHash) <= thresh."""
    h = np.array(hashes, dtype=np.uint64)
    parent = list(range(len(h)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(0, len(h), 512):
        dist = np.bitwise_count(h[i:i + 512, None] ^ h[None])
        for a, b in zip(*np.nonzero(dist <= thresh)):
            a += i
            if a < b:
                parent[find(a)] = find(b)
    return [find(i) for i in range(len(h))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--dst", default="dataset_clean")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--hash_thresh", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if os.path.exists(args.dst):
        raise SystemExit(f"{args.dst} already exists; delete it first to rebuild.")

    # Gather every image from every split; label comes from the class folder.
    items = []  # (path, cls)
    for split in sorted(os.listdir(args.src)):
        sdir = os.path.join(args.src, split)
        if not os.path.isdir(sdir):
            continue
        for cls in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir):
                continue
            for f in sorted(os.listdir(cdir)):
                if f.lower().endswith(IMG_EXTS):
                    items.append((os.path.join(cdir, f), cls))
    print(f"Found {len(items)} files")

    with ProcessPoolExecutor() as ex:
        hashed = list(ex.map(hash_image, [p for p, _ in items], chunksize=64))

    # 1) exact dedupe (first occurrence wins; no cross-class exact dups exist)
    seen, uniq, unreadable = set(), [], 0
    for (path, cls), (_, md5, dh) in zip(items, hashed):
        if md5 is None:
            unreadable += 1
            continue
        if md5 in seen:
            continue
        seen.add(md5)
        uniq.append((path, cls, dh))
    print(f"Unique images: {len(uniq)} (removed {len(items) - len(uniq) - unreadable} "
          f"exact duplicates, {unreadable} unreadable)")

    # 2) near-duplicate groups
    roots = group_near_duplicates([dh for _, _, dh in uniq], args.hash_thresh)
    groups = collections.defaultdict(list)
    for (path, cls, _), r in zip(uniq, roots):
        groups[r].append((path, cls))
    print(f"Near-duplicate groups: {len(groups)} "
          f"(largest {max(len(g) for g in groups.values())})")

    # 3) group-aware stratified split: bucket groups by majority class, then
    #    fill test, then val, until each reaches its share of that class.
    rng = random.Random(args.seed)
    split_of = {}
    by_cls = collections.defaultdict(list)
    for r, g in groups.items():
        majority = collections.Counter(c for _, c in g).most_common(1)[0][0]
        by_cls[majority].append(r)
    for cls, rs in by_cls.items():
        rng.shuffle(rs)
        total = sum(len(groups[r]) for r in rs)
        quota = {"test": args.test * total, "val": args.val * total}
        filled = {"test": 0, "val": 0}
        for r in rs:
            size = len(groups[r])
            # never put a giant group in val/test; it would distort them
            if filled["test"] + size <= quota["test"] and size < 0.2 * quota["test"]:
                s = "test"
            elif filled["val"] + size <= quota["val"] and size < 0.2 * quota["val"]:
                s = "val"
            else:
                s = "train"
            if s != "train":
                filled[s] += size
            split_of[r] = s

    # 4) hard-link into dst
    counts = collections.Counter()
    manifest = []
    for r, g in groups.items():
        s = split_of[r]
        for path, cls in g:
            out_dir = os.path.join(args.dst, s, cls)
            os.makedirs(out_dir, exist_ok=True)
            # prefix with source split to avoid name clashes (e.g. 1.jpg)
            src_split = path.split(os.sep)[-3]
            out = os.path.join(out_dir, f"{src_split}__{os.path.basename(path)}")
            os.link(path, out)
            counts[(s, cls)] += 1
            manifest.append({"src": path, "dst": out, "split": s, "cls": cls, "group": int(r)})

    print("\nClean split:")
    for s in ["train", "val", "test"]:
        row = {c: counts[(s, c)] for c in sorted({c for _, c in counts})}
        print(f"  {s:5s} {row}  total={sum(row.values())}")
    with open(os.path.join(args.dst, "manifest.json"), "w") as f:
        json.dump({"args": vars(args), "counts": {f"{s}/{c}": n for (s, c), n in counts.items()},
                   "files": manifest}, f)
    print(f"Wrote {args.dst}/manifest.json")


if __name__ == "__main__":
    main()
