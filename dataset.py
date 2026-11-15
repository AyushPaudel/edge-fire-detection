"""
Builds train/test dataloaders directly from your existing folder layout:

    fire_detect/
      dataset/
        train/fire/*.jpg
        train/not_fire/*.jpg
        test/fire/*.jpg
        test/not_fire/*.jpg

Uses torchvision.datasets.ImageFolder, which reads exactly this
class-per-subfolder structure with no changes needed on your end.
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

IMG_SIZE = 224

# ImageNet normalization stats (matches the pretrained MobileNetV3 backbone)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_eval_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_dataloaders(dataset_root, img_size=IMG_SIZE, batch_size=32, num_workers=4):
    """
    dataset_root: folder containing `train/`, `val/` and `test/` subfolders
                  (each with fire/ and "non fire/"), e.g. dataset_clean/ from
                  make_clean_split.py. If `val/` is missing, test doubles as
                  val (old layout) -- that leaks test into model selection.

    Returns: train_loader, val_loader, test_loader, class_to_idx
    """
    import os

    train_ds = torchvision.datasets.ImageFolder(
        f"{dataset_root}/train", transform=get_train_transforms(img_size)
    )
    test_ds = torchvision.datasets.ImageFolder(
        f"{dataset_root}/test", transform=get_eval_transforms(img_size)
    )
    if os.path.isdir(f"{dataset_root}/val"):
        val_ds = torchvision.datasets.ImageFolder(
            f"{dataset_root}/val", transform=get_eval_transforms(img_size)
        )
    else:
        print("WARNING: no val/ split found; using test/ for model selection "
              "(test accuracy will be optimistic). Run make_clean_split.py.")
        val_ds = test_ds

    # ImageFolder assigns indices alphabetically: {'fire': 0, 'non fire': 1}.
    for name, ds in [("val", val_ds), ("test", test_ds)]:
        assert ds.class_to_idx == train_ds.class_to_idx, (
            f"Class index mismatch between train and {name}: "
            f"{train_ds.class_to_idx} vs {ds.class_to_idx}"
        )

    # pin_memory only helps CUDA transfers; leave it off for MPS/CPU.
    pin_memory = torch.cuda.is_available()

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)

    return (loader(train_ds, True), loader(val_ds, False),
            loader(test_ds, False), train_ds.class_to_idx)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset_clean"
    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(root, batch_size=8)
    print("class_to_idx:", class_to_idx)
    imgs, labels = next(iter(train_loader))
    print("batch images:", imgs.shape, "labels:", labels[:8])