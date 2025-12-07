import os
import zipfile
import random
import shutil
from pathlib import Path

# CHANGE THIS: path to your downloaded Kaggle zip
ZIP_PATH = "IDC_regular_ps50_idx5.zip"
OUTPUT_DIR = Path("dataset")

def extract_subset(n_per_class=2000):
    print("Extracting subset...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "train").mkdir(exist_ok=True)
    (OUTPUT_DIR / "test").mkdir(exist_ok=True)

    temp = Path("temp_extract")
    temp.mkdir(exist_ok=True)

    # extract all only once
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(temp)

    benign = list(temp.rglob("*class0*.png"))
    malignant = list(temp.rglob("*class1*.png"))

    random.shuffle(benign)
    random.shuffle(malignant)

    benign = benign[:n_per_class]
    malignant = malignant[:n_per_class]

    def move_subset(files, label):
        cls_train = OUTPUT_DIR / "train" / label
        cls_test = OUTPUT_DIR / "test" / label
        cls_train.mkdir(parents=True, exist_ok=True)
        cls_test.mkdir(parents=True, exist_ok=True)

        split = int(0.8 * len(files))
        for f in files[:split]:
            shutil.copy(f, cls_train / f.name)
        for f in files[split:]:
            shutil.copy(f, cls_test / f.name)

    move_subset(benign, "benign")
    move_subset(malignant, "malignant")

    shutil.rmtree(temp)
    print("Dataset prepared at ./dataset")

extract_subset()
