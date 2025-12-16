import os
import cv2
import time
import json
import csv
import numpy as np
import mediapipe as mp
import warnings
from collections import deque
from mpi4py import MPI
from tensorflow.keras.models import load_model

from src.common import mediapipe_detection, extract_keypoints, draw_landmarks

# =========================================================
# ---------------- MPI ----------------
# =========================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 4:
    raise RuntimeError("Run with at least 4 MPI ranks")

# =========================================================
# ---------------- CONFIG ----------------
# =========================================================
MODEL_PATH = "model/sign_lstm.h5"
LABEL_PATH = "model/labels.json"
CSV_PATH = "data/sentences.csv"

SEQ_LEN = 30
FPS_TARGET = 15
FRAME_SKIP = 2

SMOOTH_WIN = 5
COOLDOWN = 6
WARNING_HOLD_FRAMES = 15

warnings.filterwarnings("once")

# =========================================================
# ---------------- LOAD LABELS ----------------
# =========================================================
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

actions = {v: k for k, v in label_map.items()}

# =========================================================
# ---------------- UTILS ----------------
# =========================================================
def clean_label(lbl):
    return lbl.split("_", 1)[1] if "_" in lbl else lbl


def grammar_correct(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out[-10:]


def get_bbox(landmarks, shape):
    h, w, _ = shape
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def center(b):
    return (b[0] + b[2]) // 2, (b[1] + b[3]) // 2


def inside(b, area):
    cx, cy = center(b)
    x1, y1, x2, y2 = area
    return x1 <= cx <= x2 and y1 <= cy <= y2


def zero_hand(kp, side):
    if side == "left":
        kp[33:33+63] = 0
    else:
        kp[33+63:33+126] = 0
    return kp

# =========================================================
# ================= RANK 0: CAMERA =================
# =========================================================
if rank == 0:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        comm.send(frame, dest=1)

    cap.release()

# =========================================================
# ================= RANK 1: MEDIAPIPE =================
# =========================================================
elif rank == 1:
    mp_holistic = mp.solutions.holistic
    holistic = None

    min_det, min_track = 0.5, 0.5
    last_det, last_track = -1, -1

    while True:
        frame = comm.recv(source=0)

        while comm.Iprobe(source=3):
            min_det, min_track = comm.recv(source=3)

        if holistic is None or min_det != last_det or min_track != last_track:
            if holistic:
                holistic.close()
            holistic = mp_holistic.Holistic(
                min_detection_confidence=min_det,
                min_tracking_confidence=min_track
            )
            last_det, last_track = min_det, min_track

        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)

        bbox = {}
        hand_outside = False
        valid_area = None

        if results.pose_landmarks:
            torso_ids = [11, 12, 23, 24]
            torso = [results.pose_landmarks.landmark[i] for i in torso_ids]
            tb = get_bbox(torso, image.shape)
            bbox["valid"] = tb

            cx, cy = center(tb)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            vw, vh = int(tw * 2), int(th * 2)

            valid_area = (
                max(0, cx - vw // 2),
                max(0, cy - vh // 2),
                min(image.shape[1], cx + vw // 2),
                min(image.shape[0], cy + vh // 2),
            )
            bbox["valid"] = valid_area

        if results.left_hand_landmarks:
            lb = get_bbox(results.left_hand_landmarks.landmark, image.shape)
            bbox["lh"] = lb
            if valid_area and not inside(lb, valid_area):
                keypoints = zero_hand(keypoints, "left")
                hand_outside = True

        if results.right_hand_landmarks:
            rb = get_bbox(results.right_hand_landmarks.landmark, image.shape)
            bbox["rh"] = rb
            if valid_area and not inside(rb, valid_area):
                keypoints = zero_hand(keypoints, "right")
                hand_outside = True

        comm.send((image, keypoints, bbox, hand_outside), dest=2)

# =========================================================
# ================= RANK 2: MODEL =================
# =========================================================
elif rank == 2:
    model = load_model(MODEL_PATH)
    seq = []
    conf_buf = deque(maxlen=SMOOTH_WIN)
    frame_i = 0
    base_thresh = 0.18

    while True:
        image, kp, bbox, hand_outside = comm.recv(source=1)

        while comm.Iprobe(source=3):
            base_thresh = comm.recv(source=3)

        frame_i += 1
        seq.append(kp)
        seq = seq[-SEQ_LEN:]

        pred, conf = None, 0.0

        if len(seq) == SEQ_LEN and frame_i % FRAME_SKIP == 0 and not hand_outside:
            res = model.predict(np.expand_dims(seq, 0), verbose=0)[0]
            idx = int(np.argmax(res))
            conf = float(res[idx])
            conf_buf.append(conf)

            adaptive = max(base_thresh, np.mean(conf_buf) * 0.6)
            if conf > adaptive:
                pred = clean_label(actions[idx])

        comm.send((image, bbox, pred, conf, hand_outside), dest=3)

# =========================================================
# ================= RANK 3: UI + WARNINGS =================
# =========================================================
elif rank == 3:
    os.makedirs("data", exist_ok=True)
    f = open(CSV_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 200)

    cv2.createTrackbar("Base Threshold", "Controls", 18, 80, lambda x: None)
    cv2.createTrackbar("Min Detection", "Controls", 50, 99, lambda x: None)
    cv2.createTrackbar("Min Tracking", "Controls", 50, 99, lambda x: None)

    tokens = []
    cooldown = 0
    last_time = time.time()

    warning_active = False
    warning_timer = 0

    while True:
        base_thresh = cv2.getTrackbarPos("Base Threshold", "Controls") / 100
        min_det = max(0.1, cv2.getTrackbarPos("Min Detection", "Controls") / 100)
        min_track = max(0.1, cv2.getTrackbarPos("Min Tracking", "Controls") / 100)

        comm.send((min_det, min_track), dest=1)
        comm.send(base_thresh, dest=2)

        image, bbox, pred, conf, hand_outside = comm.recv(source=2)

        # ---------- WARNING STATE ----------
        if hand_outside:
            warning_timer = WARNING_HOLD_FRAMES
            if not warning_active:
                warnings.warn("Hand outside active area. Prediction paused.")
                warning_active = True
        else:
            warning_timer = max(0, warning_timer - 1)
            if warning_timer == 0:
                warning_active = False

        if pred and cooldown == 0:
            tokens.append(pred)
            cooldown = COOLDOWN
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), pred, conf])
            f.flush()

        cooldown = max(0, cooldown - 1)
        sentence = grammar_correct(tokens)

        for b in bbox.values():
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)

        fps = int(1 / max(time.time() - last_time, 1e-6))
        last_time = time.time()

        cv2.rectangle(image, (0, 0), (image.shape[1], 160), (0, 0, 0), -1)
        cv2.putText(image, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if warning_active:
            cv2.putText(image, "WARNING: Hands outside active area",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)

        if pred:
            cv2.putText(image, f"WORD: {pred}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(image, (10, 125),
                          (10 + int(conf * 300), 145),
                          (0, 255, 0), -1)

        cv2.putText(image, " ".join(sentence), (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("MPI Sign Language (Stable)", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    f.close()
    cv2.destroyAllWindows()
