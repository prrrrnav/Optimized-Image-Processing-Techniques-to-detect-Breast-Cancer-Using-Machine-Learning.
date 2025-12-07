# train_hybrid_pytorch.py
# Hybrid CNN + Transformer training (PyTorch)
import os
import json
import random
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -----------------------
# Config - edit if needed
# -----------------------
ROOT = Path(".")
DATA_DIR = ROOT / "dataset"   # expected: dataset/train/<class> and dataset/test/<class>
IMG_SIZE = 128                # keep small for CPU; increase if you have GPU
BATCH_SIZE = 32
NUM_EPOCHS = 8
LR = 1e-3
NUM_WORKERS = 4 if os.name != "nt" else 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUT = "model.pth"
METRICS_OUT = "metrics.json"
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# Dataset loaders
# -----------------------
train_trans = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_trans = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_trans)
test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=test_trans)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

classes = train_ds.classes
num_classes = len(classes)
print(f"Found classes: {classes} (num={num_classes})")
print(f"Train samples: {len(train_ds)}  Test samples: {len(test_ds)}")

# -----------------------
# Small ViT-like transformer (patch embedding)
# -----------------------
class PatchTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_ch=3, emb_dim=128, depth=4, heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_ch, emb_dim, kernel_size=patch_size, stride=patch_size)  # (B, emb_dim, H/P, W/P)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + num_patches, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                       # (B, emb_dim, H/P, W/P)
        B, E, Hc, Wc = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, num_patches, emb_dim)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B,1,emb_dim)
        x = torch.cat([cls_token, x], dim=1)   # (B, 1+num_patches, emb_dim)
        x = x + self.pos_emb
        x = x.transpose(0,1)                   # Transformer expects (seq_len, B, emb_dim)
        x = self.transformer(x)                # (seq_len, B, emb_dim)
        x = x.transpose(0,1)                   # (B, seq_len, emb_dim)
        cls_out = x[:, 0]                      # (B, emb_dim)
        cls_out = self.norm(cls_out)
        return cls_out                         # (B, emb_dim)

# -----------------------
# Hybrid Model
# -----------------------
class HybridModel(nn.Module):
    def __init__(self, num_classes, img_size=128, patch_size=16):
        super().__init__()
        # CNN backbone (MobileNetV2 without classifier)
        cnn = models.mobilenet_v2(pretrained=False)   # pretrained=False for reproducibility / smaller download
        cnn_features = cnn.features
        self.cnn_backbone = nn.Sequential(
            cnn_features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        cnn_out_dim = 1280  # mobilenet_v2 last_channel

        # Transformer branch
        self.transformer = PatchTransformer(img_size=img_size, patch_size=patch_size, emb_dim=256, depth=4, heads=4, mlp_dim=512, dropout=0.1)
        tr_out_dim = 256

        # classifier on concatenated features
        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim + tr_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (B,3,H,W)
        cnn_feat = self.cnn_backbone(x)     # (B, cnn_out_dim)
        tr_feat = self.transformer(x)       # (B, tr_out_dim)
        comb = torch.cat([cnn_feat, tr_feat], dim=1)
        out = self.head(comb)
        return out

# -----------------------
# Create model, loss, optimizer
# -----------------------
model = HybridModel(num_classes=num_classes, img_size=IMG_SIZE, patch_size=16)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# -----------------------
# Training loop
# -----------------------
best_test_acc = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    running = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - train")
    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running += loss.item() * labels.size(0)
        pbar.set_postfix(loss = running / total, acc = correct / total)
    scheduler.step()

    # validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Validate"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    print(f"Epoch {epoch+1} Test Acc: {test_acc:.4f}")

    # save best
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save({"model_state": model.state_dict(), "classes": classes}, MODEL_OUT)
        json.dump({"test_accuracy": best_test_acc, "classes": classes}, open(METRICS_OUT, "w"))
        print(f"Saved best model (acc={best_test_acc:.4f}) to {MODEL_OUT}")

print("Training complete. Best test acc:", best_test_acc)
if best_test_acc == 0.0:
    # still save something
    torch.save({"model_state": model.state_dict(), "classes": classes}, MODEL_OUT)
    json.dump({"test_accuracy": best_test_acc, "classes": classes}, open(METRICS_OUT, "w"))
    print("Saved final model (zero-acc fallback).")
