import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------
# Patch Transformer (MUST MATCH TRAINING)
# -----------------------------------------
class PatchTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_ch=3, emb_dim=256, depth=4, heads=4, mlp_dim=512, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_ch, emb_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + num_patches, emb_dim))

        encoder = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.proj(x)       # [B, emb_dim, H/patch, W/patch]
        B, C, Hc, Wc = x.shape
        x = x.flatten(2).transpose(1, 2)

        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)

        x = x + self.pos_emb
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        return self.norm(x[:, 0])  # CLS token output


# -----------------------------------------
# Hybrid CNN + Transformer (MUST MATCH TRAINING)
# -----------------------------------------
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        cnn = models.mobilenet_v2(weights=None)
        self.cnn_backbone = nn.Sequential(
            cnn.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        cnn_out_dim = 1280

        self.transformer = PatchTransformer(
            img_size=128,
            patch_size=16,
            emb_dim=256,
            depth=4,
            heads=4,
            mlp_dim=512
        )
        tr_out_dim = 256

        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim + tr_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn_backbone(x)
        tr_feat = self.transformer(x)
        combined = torch.cat([cnn_feat, tr_feat], dim=1)
        return self.head(combined)


# -----------------------------------------------------
# Load checkpoint — this matches your actual checkpoint
# -----------------------------------------------------
print("Loading model.pth...")
ckpt = torch.load("model.pth", map_location=device)

class_list = ckpt["classes"]
num_classes = len(class_list)

model = HybridModel(num_classes=num_classes).to(device)
model_state = ckpt["model_state"]

missing, unexpected = model.load_state_dict(model_state, strict=False)
print("Loaded model with strict=False")
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

model.eval()
print("MODEL LOADED SUCCESSFULLY.")

# ----------------------
# Preprocessing
# ----------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ----------------------
# Flask App
# ----------------------
app = Flask(__name__)

@app.route("/infer", methods=["POST"])
def infer():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]

    try:
        img = Image.open(img_file.stream).convert("RGB")
    except:
        return jsonify({"error": "Invalid image"}), 400

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = probs.argmax()
    pred_label = class_list[pred_idx]
    confidence = float(probs[pred_idx])

    # Try loading metrics.json
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        acc = metrics.get("test_accuracy", None)
    except:
        acc = None

    return jsonify({
        "prediction": pred_label,
        "confidence": confidence,
        "test_accuracy": acc
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
