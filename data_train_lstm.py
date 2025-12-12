import os
import cv2
import copy
import itertools
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp

# ------------------------------ Settings ------------------------------
DATASET_DIR = "dataset/images"
PRED_WINDOW = 10  # sequence length
alphabet = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

mp_hands = mp.solutions.hands

# ------------------------------ Helpers ------------------------------
def calc_landmark_list(image, hand_landmarks):
    h, w = image.shape[:2]
    pts = []
    for lm in hand_landmarks.landmark:
        x = min(int(lm.x * w), w-1)
        y = min(int(lm.y * h), h-1)
        pts.append([x, y])
    return pts

def pre_process_landmark(landmarks):
    temp = copy.deepcopy(landmarks)
    base_x, base_y = temp[0]
    for i, p in enumerate(temp):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    temp = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, temp))
    if max_val == 0:
        return temp
    temp = [v/max_val for v in temp]
    return temp

# ------------------------------ Load Dataset ------------------------------
print("📦 Preparing dataset sequences...")
X_sequences = []
y_labels = []

label_map = {ch: idx for idx, ch in enumerate(alphabet)}
print("Label Map:", label_map)

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

for label in alphabet:
    folder = os.path.join(DATASET_DIR, label)
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}, skipping")
        continue

    images = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))])
    print(f"Processing: {label} ({len(images)} images)")

    pts_list = []
    for img_file in images:
        img_path = os.path.join(folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read {img_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand_lms = res.multi_hand_landmarks[0]
            pts = calc_landmark_list(image, hand_lms)
            processed = pre_process_landmark(pts)
            pts_list.append(processed)
            print(f"✅ Hand detected: {img_file}")
        else:
            print(f"❌ No hand detected: {img_file}")
            # fallback: flatten resized image
            small_img = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32,32))
            pts_list.append(small_img.flatten().tolist())

    # create sequences
    for i in range(0, len(pts_list) - PRED_WINDOW + 1):
        seq = pts_list[i:i+PRED_WINDOW]
        X_sequences.append(seq)
        y_labels.append(label_map[label])

    # handle partial sequences
    if len(pts_list) < PRED_WINDOW and len(pts_list) > 0:
        X_sequences.append(pts_list)
        y_labels.append(label_map[label])

hands.close()
# ------------------------------ Convert to numpy ------------------------------
if len(X_sequences) == 0:
    raise ValueError("No sequences found. Check your images and hand visibility.")

X = np.array(X_sequences, dtype=np.float32)
y = np.array(y_labels, dtype=np.int32)
print(f"\nDataset Sequences Loaded:")
print(f"X: {X.shape}, y: {y.shape}")

# ------------------------------ Train/Test Split ------------------------------
if len(X) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )
else:
    X_train, X_test, y_train, y_test = X, X, y, y  # fallback if too few samples

# Pad sequences to uniform length
max_seq_len = max([len(seq) for seq in X_train])
feature_dim = len(X_train[0][0])

def pad_sequence(seq, max_len):
    padded = seq.copy()
    while len(padded) < max_len:
        padded.append(padded[-1])
    return padded

X_train_pad = np.array([pad_sequence(seq, max_seq_len) for seq in X_train])
X_test_pad = np.array([pad_sequence(seq, max_seq_len) for seq in X_test])

# ------------------------------ Build LSTM Model ------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(max_seq_len, feature_dim)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(len(alphabet), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ------------------------------ Train Model ------------------------------
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),
    epochs=25,
    batch_size=8
)

# ------------------------------ Save Model ------------------------------
model.save("model_lstm.h5")
print("✅ LSTM Model saved as model_lstm.h5")

# ------------------------------ Plot Accuracy/Loss ------------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
