# train_quality_model.py

import os
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import EMNIST
from torchvision import transforms
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

# -------------------------
# Config
# -------------------------

DATA_ROOT = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 28  # 28x28 images
BATCH_SIZE = 256
EPOCHS = 8
LEARNING_RATE = 1e-3

NUM_LETTERS = 26  # A..Z
SHAPE_LABELS: List[str] = ["CIRCLE", "SQUARE", "TRIANGLE", "STAR"]
NUM_SHAPES = len(SHAPE_LABELS)
NUM_CLASSES = NUM_LETTERS + NUM_SHAPES  # 26 letters + 4 shapes = 30

QUALITY_MODEL_PATH = MODELS_DIR / "quality_cnn.pth"


# -------------------------
# Label mapping helpers
# -------------------------

def build_label_mapping():
    """
    Map logical labels to class indices used by the CNN.

    LETTER:A..Z --> 0..25
    SHAPE:CIRCLE/SQUARE/TRIANGLE/STAR --> 26..29
    """
    label_to_index = {}
    for i in range(NUM_LETTERS):
        ch = chr(ord("A") + i)
        label_to_index[f"LETTER:{ch}"] = i

    for j, shape in enumerate(SHAPE_LABELS):
        label_to_index[f"SHAPE:{shape}"] = NUM_LETTERS + j

    return label_to_index


LABEL_TO_INDEX = build_label_mapping()


# -------------------------
# Model definition
# -------------------------

class QualityCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# Datasets
# -------------------------

class EmnistLettersDataset(Dataset):
    """
    Wrap torchvision EMNIST(split='letters') and remap labels:

    EMNIST letters labels are 1..26 for A..Z.
    We turn them into 0..25.

    IMPORTANT:
    We keep EMNIST's transform=None so that it returns a PIL Image.
    Then we apply our own transform (with augmentations for train) here.
    """

    def __init__(self, root: Path, train: bool, transform=None):
        # transform=None => EMNIST returns PIL images
        self.ds = EMNIST(
            root=str(root),
            split="letters",
            train=train,
            download=True,
            transform=None,
        )
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, label = self.ds[idx]  # img is PIL.Image.Image
        # EMNIST letters labels: 1..26 => map to 0..25
        label = int(label) - 1
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class SyntheticShapesDataset(Dataset):
    """
    On-the-fly generation of simple shapes (circle, square, triangle, star).
    Each sample: (1, 28, 28) tensor, label in [26..29].
    """

    def __init__(self, num_per_class: int = 2000, img_size: int = IMG_SIZE, transform=None):
        self.num_per_class = num_per_class
        self.img_size = img_size
        self.transform = transform
        self.shape_labels = SHAPE_LABELS
        self.total = len(self.shape_labels) * self.num_per_class

    def __len__(self):
        return self.total

    def _draw_circle(self, draw: ImageDraw.ImageDraw):
        w = self.img_size
        margin = int(w * 0.15)
        draw.ellipse(
            [margin, margin, w - margin, w - margin],
            outline=255,
            width=2,
        )

    def _draw_square(self, draw: ImageDraw.ImageDraw):
        w = self.img_size
        margin = int(w * 0.2)
        draw.rectangle(
            [margin, margin, w - margin, w - margin],
            outline=255,
            width=2,
        )

    def _draw_triangle(self, draw: ImageDraw.ImageDraw):
        w = self.img_size
        margin = int(w * 0.15)
        p1 = (w // 2, margin)
        p2 = (margin, w - margin)
        p3 = (w - margin, w - margin)
        draw.polygon([p1, p2, p3], outline=255, width=2)

    def _draw_star(self, draw: ImageDraw.ImageDraw):
        w = self.img_size
        cx, cy = w // 2, w // 2
        r_outer = int(w * 0.4)
        r_inner = int(w * 0.18)
        points = []
        import math as pymath
        for i in range(10):
            angle = i * pymath.pi / 5.0
            r = r_outer if i % 2 == 0 else r_inner
            x = cx + int(r * pymath.cos(angle))
            y = cy + int(r * pymath.sin(angle))
            points.append((x, y))
        draw.polygon(points, outline=255, width=2)

    def _draw_shape_image(self, shape_name: str) -> Image.Image:
        img = Image.new("L", (self.img_size, self.img_size), color=0)
        draw = ImageDraw.Draw(img)
        if shape_name == "CIRCLE":
            self._draw_circle(draw)
        elif shape_name == "SQUARE":
            self._draw_square(draw)
        elif shape_name == "TRIANGLE":
            self._draw_triangle(draw)
        elif shape_name == "STAR":
            self._draw_star(draw)
        return img

    def __getitem__(self, idx: int):
        class_idx = idx // self.num_per_class
        class_idx = min(class_idx, len(self.shape_labels) - 1)
        shape_name = self.shape_labels[class_idx]
        img = self._draw_shape_image(shape_name)  # PIL image

        if self.transform is not None:
            img = self.transform(img)

        # global label index
        label = NUM_LETTERS + class_idx  # 26..29
        return img, label


# -------------------------
# Training + evaluation
# -------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def main():
    print("== Training QualityCNN on EMNIST letters + synthetic shapes ==")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -----------------------------
    # Data transforms
    # -----------------------------
    # TRAIN: add light augmentations so model becomes robust
    # to flipped / slightly distorted trajectories from the canvas.
    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=8, fill=0),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.9, 1.1),
            fill=0,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    # EVAL: no augmentation, just a clean view of the data.
    transform_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # EMNIST letters (now PIL->transform done ONLY in our wrapper)
    train_letters = EmnistLettersDataset(DATA_ROOT, train=True, transform=transform_train)
    test_letters = EmnistLettersDataset(DATA_ROOT, train=False, transform=transform_eval)

    # Synthetic shapes
    train_shapes = SyntheticShapesDataset(
        num_per_class=2000,
        img_size=IMG_SIZE,
        transform=transform_train,
    )
    test_shapes = SyntheticShapesDataset(
        num_per_class=300,
        img_size=IMG_SIZE,
        transform=transform_eval,
    )

    train_dataset = ConcatDataset([train_letters, train_shapes])
    test_dataset = ConcatDataset([test_letters, test_shapes])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = QualityCNN(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, test_loader, device)

        print(
            f"Train loss: {train_loss:.4f}, acc: {train_acc*100:.2f}% | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), QUALITY_MODEL_PATH)
            print(f"[INFO] Saved best model to {QUALITY_MODEL_PATH}")

    # Save label mapping (optional, for debugging)
    mapping_path = MODELS_DIR / "quality_label_to_index.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_TO_INDEX, f, indent=2)
    print(f"[INFO] Saved label mapping to {mapping_path}")
    print(f"[DONE] Best validation accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
