# train_synthetic.py
# This script needs NO dataset. It generates synthetic images and trains a tiny model.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import random
import json

IMG_SIZE = 64
NUM_CLASSES = 2
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# SYNTHETIC DATASET
# ----------------------------
class SyntheticDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = random.randint(0, 1)

        # Make synthetic 64x64 images (tensor filled with noise)
        img = torch.randn(3, IMG_SIZE, IMG_SIZE)

        return img, label

# Create train + test synthetic datasets
train_ds = SyntheticDataset(300)
test_ds = SyntheticDataset(60)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# ----------------------------
# MODEL
# ----------------------------
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ----------------------------
# TRAINING LOOP
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc = {correct/total:.2f}")

# ----------------------------
# TEST ACCURACY
# ----------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

test_accuracy = correct / total
print("Test Accuracy:", test_accuracy)

# ----------------------------
# SAVE MODEL + METRICS
# ----------------------------
torch.save(
    {
        "model_state": model.state_dict(),
        "classes": ["classA", "classB"]  # two fake classes
    },
    "model.pth"
)

with open("metrics.json", "w") as f:
    json.dump(
        {
            "test_accuracy": test_accuracy,
            "classes": ["classA", "classB"]
        },
        f
    )

print("\nSaved:")
print("- model.pth")
print("- metrics.json")
