"""
Train the fire classifier on your dataset.

Usage (run from wherever fire_detect/ lives):
    python train.py --dataset_root fire_detect/dataset \
        --epochs 30 --batch_size 32 --img_size 224 --out checkpoints/

Run this on a desktop/laptop GPU or Colab, not on the Jetson Nano — the Nano
is for inference only.
"""
import argparse
import os
import json

import torch
import torch.nn as nn
from tqdm import tqdm

from model import FireClassifier
from dataset import build_dataloaders


def run_epoch(model, loader, criterion, optimizer, device, fire_idx, train=True,
              label_smoothing=0.0):
    model.train() if train else model.eval()
    total_loss, total_correct, n = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if train else "eval")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            # ImageFolder gives 0/1 by alphabetical class name; remap so
            # target=1 always means "fire", matching what the model predicts.
            targets = (labels == fire_idx).float().unsqueeze(1).to(device)

            if train:
                optimizer.zero_grad()

            logits = model(imgs)
            # Label smoothing (train only): targets 1/0 -> 1-e/2 and e/2, so
            # the model is never pushed toward infinite logits. Keeps logits
            # in a small range, which also makes INT8 quantization easier.
            loss_targets = targets
            if train and label_smoothing > 0:
                loss_targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
            loss = criterion(logits, loss_targets)

            if train:
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == targets).sum().item()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            pbar.set_postfix(loss=total_loss / n, acc=total_correct / n)

    return total_loss / n, total_correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
                         help="path to the folder containing train/ and test/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="0 disables; 0.1 keeps logits roughly within +-3")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU (M-series)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        args.dataset_root, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    fire_idx = class_to_idx["fire"]
    print(f"class_to_idx: {class_to_idx} (fire_idx={fire_idx})")

    # Save this alongside checkpoints -- export/inference scripts need it.
    with open(os.path.join(args.out, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f)

    model = FireClassifier(pretrained=not args.no_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Model selection uses val only; test is evaluated once at the end.
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, fire_idx,
            train=True, label_smoothing=args.label_smoothing,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, fire_idx, train=False
        )
        scheduler.step()

        print(f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            print(f"  -> new best (val_acc={best_acc:.4f}), saved best.pt")

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device))
    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, optimizer, device, fire_idx, train=False
    )
    print(f"\nTraining done. Best val acc: {best_acc:.4f} | "
          f"held-out TEST acc (best.pt): {test_acc:.4f}")


if __name__ == "__main__":
    main()