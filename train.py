import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# =============================
# CONFIG
# =============================
DATA_DIR = "data"
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 50   # Can be high; EarlyStopping will stop automatically

MODEL_DIR = "model2"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "keras_model.h5")
LABELS_SAVE_PATH = os.path.join(MODEL_DIR, "labels.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# DATA GENERATORS
# =============================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = train_gen.num_classes
print("Classes:", train_gen.class_indices)

# =============================
# SAVE LABELS (cvzone compatible)
# =============================
labels = list(train_gen.class_indices.keys())
with open(LABELS_SAVE_PATH, "w") as f:
    for label in labels:
        f.write(label + "\n")

# =============================
# MODEL (CNN)
# =============================
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")
])

# =============================
# COMPILE
# =============================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
# CALLBACKS
# =============================

# Save ONLY the best model (lowest val_loss)
checkpoint_cb = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Stop training if no improvement
earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# =============================
# TRAIN
# =============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)

print("✅ Training complete")
print("✅ Best model saved to:", MODEL_SAVE_PATH)
print("✅ Labels saved to:", LABELS_SAVE_PATH)
