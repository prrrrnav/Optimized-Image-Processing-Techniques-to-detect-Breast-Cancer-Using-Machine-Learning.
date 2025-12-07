import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import ViTFeatureExtractor, TFViTModel
import json
import os

IMG_SIZE = 224

# -----------------------
# CNN feature extractor
# -----------------------
def build_cnn():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights="imagenet"
    )
    base.trainable = False
    return models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu")
    ])

# -----------------------
# Transformer (ViT)
# -----------------------
vit = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

def build_transformer():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.image.resize(inputs, (224, 224))
    x = layers.Lambda(lambda x: x / 255.0)(x)

    outputs = vit(x, training=False).pooler_output
    model = models.Model(inputs, outputs)
    return model

# -----------------------
# Hybrid Model
# -----------------------
def build_hybrid():
    cnn = build_cnn()
    transformer = build_transformer()

    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    cnn_out = cnn(inp)
    tr_out = transformer(inp)

    combined = layers.Concatenate()([cnn_out, tr_out])
    combined = layers.Dense(256, activation="relu")(combined)
    combined = layers.Dense(1, activation="sigmoid")(combined)

    model = models.Model(inp, combined)
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model

model = build_hybrid()
model.summary()

# -----------------------
# Dataset
# -----------------------
train = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16
)

test = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16
)

# -----------------------
# Training
# -----------------------
history = model.fit(train, validation_data=test, epochs=5)

test_acc = history.history["val_accuracy"][-1]
print("Test Accuracy:", test_acc)

# -----------------------
# Save model + metrics
# -----------------------
model.save("model.pth")

json.dump(
    {"test_accuracy": float(test_acc), "classes": ["benign", "malignant"]},
    open("metrics.json", "w")
)

print("Saved: model.pth + metrics.json")
