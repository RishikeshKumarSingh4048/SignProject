import cv2
import mediapipe as mp
import numpy as np
import itertools, copy, string, time, json
from collections import deque
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output
from ipywidgets import Button
from ipywebrtc import CameraStream, ImageRecorder
#from google.colab.patches import cv2_imshow
from datetime import datetime
DATASET = "images"  # Folder containing subfolders per label
SEQUENCE_LENGTH = 20
PRED_WINDOW = 8
LETTER_CONF_THRESH = 0.85
COOLDOWN_FRAMES = 8

alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection

def calc_landmark_list(img, hand_lm):
    h,w = img.shape[:2]
    pts = [[int(lm.x*w), int(lm.y*h)] for lm in hand_lm.landmark]
    return pts

def pre_process_landmarks(pts):
    pts2 = copy.deepcopy(pts)
    base_x, base_y = pts2[0]
    for p in pts2:
        p[0] -= base_x
        p[1] -= base_y
    pts2 = list(itertools.chain.from_iterable(pts2))
    max_val = max(map(abs, pts2))
    return [p/max_val for p in pts2]

def get_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def hand_in_face(hand, face_det, w, h):
    if not face_det: return False
    wx, wy = hand.landmark[0].x*w, hand.landmark[0].y*h
    for f in face_det:
        b = f.location_data.relative_bounding_box
        x1,y1 = b.xmin*w, b.ymin*h
        x2,y2 = x1+b.width*w, y1+b.height*h
        if x1 <= wx <= x2 and y1 <= wy <= y2:
            return True
    return False

def hand_in_torso(hand, pose_lm, h):
    if pose_lm is None: return False
    lm = pose_lm.landmark
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_y = ((ls.y + rs.y)/2)*h
    top = shoulder_y - (0.15*h)
    bottom = shoulder_y + (0.25*h)
    wy = hand.landmark[0].y*h
    return top <= wy <= bottom

import os

labels = sorted([d for d in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET,d))])
label_map = {lab:i for i,lab in enumerate(labels)}
print("Label Map:", label_map)

X = []
y = []

hands_mp = mp_hands.Hands(max_num_hands=1)
pose_mp = mp_pose.Pose(model_complexity=0)
face_mp = mp_face.FaceDetection(model_selection=0)

for label in labels:
    folder = os.path.join(DATASET,label)
    if not os.path.isdir(folder): continue

    for file in os.listdir(folder):
        if not file.lower().endswith((".jpg",".png",".jpeg",".bmp")): continue
        path = os.path.join(folder,file)
        frame = cv2.imread(path)
        if frame is None: continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w = frame.shape[:2]

        hand_res = hands_mp.process(rgb)
        pose_res = pose_mp.process(rgb)
        face_res = face_mp.process(rgb)

        if hand_res.multi_hand_landmarks:
            hand = hand_res.multi_hand_landmarks[0]
            if hand_in_face(hand, face_res.detections if face_res else None, w,h): continue
            if not hand_in_torso(hand, pose_res.pose_landmarks, h): continue
            landmarks = pre_process_landmarks(calc_landmark_list(frame, hand))
            if len(landmarks)==42:
                X.append([landmarks]*SEQUENCE_LENGTH)
                y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("Dataset:", X.shape, y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=42)

# Build LSTM
model_lstm = models.Sequential([
    layers.Input(shape=(SEQUENCE_LENGTH,42)),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.25),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(64,activation="relu"),
    layers.Dense(len(labels),activation="softmax")
])

model_lstm.compile(optimizer=optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

history = model_lstm.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=40, batch_size=16)

model_lstm.save("model_lstm.h5")
with open("label_map.json","w") as f: json.dump(label_map,f,indent=4)

# Load saved model
model = keras.models.load_model("model_lstm.h5")

sequence = deque(maxlen=SEQUENCE_LENGTH)
pred_queue = deque(maxlen=PRED_WINDOW)
current_word = ""
sentence = ""
last_letter_time = time.time()
fps_queue = deque(maxlen=30)
prev_time = time.time()

stop_button = Button(description="Stop Detection", button_style="danger")
stop_flag = False
def stop_clicked(b):
    global stop_flag
    stop_flag = True
stop_button.on_click(stop_clicked)
display(stop_button)

camera = CameraStream(constraints={"video": True})
recorder = ImageRecorder(stream=camera)
display(camera)

print("Starting Detection...")

while not stop_flag:
    img_bytes = recorder.image.value
    if not img_bytes: continue
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None: continue

    h,w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_res = hands_mp.process(rgb)
    pose_res = pose_mp.process(rgb)
    face_res = face_mp.process(rgb)

    now = time.time()
    fps_queue.append(1/(now-prev_time))
    prev_time = now
    fps = int(np.mean(fps_queue))

    hand_lms = None
    if hand_res.multi_hand_landmarks:
        hand_lms = hand_res.multi_hand_landmarks[0]

    if hand_lms:
        pts = calc_landmark_list(frame, hand_lms)
        x1,y1,x2,y2 = get_bbox(pts)

        occluded = hand_in_face(hand_lms, face_res.detections if face_res else None, w,h)
        in_torso = hand_in_torso(hand_lms, pose_res.pose_landmarks, h)

        if occluded:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,"FACE BLOCK",(x1,y2+20),0,0.7,(0,0,255),2)
        elif not in_torso:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,"OUTSIDE TORSO",(x1,y2+20),0,0.7,(0,0,255),2)
        else:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            sequence.append(pre_process_landmarks(pts))

            if len(sequence) == SEQUENCE_LENGTH:
                seq_input = np.array(sequence).reshape(1,SEQUENCE_LENGTH,42)
                preds = model.predict(seq_input,verbose=0)
                conf = float(np.max(preds))
                cls = int(np.argmax(preds))

                if conf >= LETTER_CONF_THRESH:
                    pred_queue.append(cls)
                    if len(pred_queue) == PRED_WINDOW:
                        final = max(set(pred_queue), key=pred_queue.count)
                        letter = alphabet[final]
                        current_word += letter
                        last_letter_time = time.time()
                        cv2.putText(frame,f"Pred: {letter} ({conf:.2f})",(x1,y1-10),1,1,(0,255,0),2)
                else:
                    cv2.putText(frame,f"Low Conf ({conf:.2f})",(x1,y1-10),0,0.7,(0,255,255),2)

    if time.time() - last_letter_time > 1.2 and len(current_word)>0:
        sentence += current_word + " "
        current_word = ""

    cv2.putText(frame, "Sentence: "+sentence,(10,h-40),0,0.7,(255,255,255),2)
    cv2.putText(frame, "Word: "+current_word,(10,h-10),0,0.7,(0,255,0),2)
    cv2.putText(frame,f"FPS: {fps}",(10,30),0,0.7,(255,255,255),2)

    clear_output(wait=True)
    display(stop_button)
    cv2_imshow(frame)
