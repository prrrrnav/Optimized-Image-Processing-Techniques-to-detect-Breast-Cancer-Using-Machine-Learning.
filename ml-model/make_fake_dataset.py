import os
from PIL import Image
import random

random.seed(0)

CLASSES = ["classA", "classB"]
SPLITS = ["train", "val", "test"]

def make_image(color, path):
    img = Image.new("RGB", (128, 128), color=color)
    img.save(path)

for split in SPLITS:
    for cls in CLASSES:
        folder = f"data/{split}/{cls}"
        os.makedirs(folder, exist_ok=True)
        for i in range(30 if split=="train" else 10):  # 30 train, 10 val, 10 test
            if cls == "classA":
                color = (255, random.randint(0,50), random.randint(0,50))  # reddish
            else:
                color = (random.randint(0,50), 255, random.randint(0,50))  # greenish
            make_image(color, f"{folder}/{cls}_{i}.jpg")

print("Fake dataset created successfully!")
