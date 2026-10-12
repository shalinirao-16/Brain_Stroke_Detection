import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ======================
# 1. Dataset Directory
# ======================
DATASET_DIR = "dataset"   # should contain Normal/ and Stroke/
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================
# 2. Data Preprocessing
# ======================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# Save class indices for Flask app
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
print("✅ Saved class indices:", train_generator.class_indices)

# ======================
# 3. Build MobileNetV2 Model
# ======================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

# Freeze most layers, fine-tune last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
preds = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# ======================
# 4. Handle Class Imbalance
# ======================
from collections import Counter
counter = Counter(train_generator.classes)
total = sum(counter.values())
class_weight = {cls: total/(len(counter)*count) for cls, count in counter.items()}
print("✅ Using class weights:", class_weight)

# ======================
# 5. Callbacks
# ======================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint("best_stroke_model.h5", save_best_only=True)
]

# ======================
# 6. Train Model
# ======================
EPOCHS = 15
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)

# ======================
# 7. Save Final Model
# ======================
model.save("stroke_model.h5")
print("✅ Model saved as stroke_model.h5")

# ======================
# 8. Evaluation
# ======================
val_generator.reset()
y_pred = model.predict(val_generator, verbose=1)
y_pred_classes = (y_pred > 0.5).astype("int32")
y_true = val_generator.classes

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=["Normal", "Stroke"]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Stroke"],
            yticklabels=["Normal", "Stroke"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ======================
# 9. Training Curves
# ======================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy Curve")

# Loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.show()
