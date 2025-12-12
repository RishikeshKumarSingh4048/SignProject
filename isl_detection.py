import cv2
import mediapipe as mp
import numpy as np
import itertools
import copy
import string
from collections import deque
from datetime import datetime
import time
from tensorflow import keras

# -------------------------------------------------------------
# Load model + Mediapipe
# -------------------------------------------------------------
model = keras.models.load_model("model.h5")

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection

alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# -------------------------------------------------------------
# Settings
# -------------------------------------------------------------
PRED_WINDOW = 8
COOLDOWN_FRAMES = 8
LETTER_CONF_THRESH = 0.85

sentence_log = []
CSV_PATH = "sentence.csv"

fps_queue = deque(maxlen=30)

# Hand-tracking persistent IDs
TRACKED_HANDS = {
    "LEFT": None,
    "RIGHT": None
}

# Per-hand queues and cooldowns
pred_queues = {
    "LEFT": deque(maxlen=PRED_WINDOW),
    "RIGHT": deque(maxlen=PRED_WINDOW)
}
cooldowns = {"LEFT": 0, "RIGHT": 0}

# Word / sentence assembly
current_word = ""
sentence = ""
last_letter_time = time.time()
last_hand_time = time.time()


# -------------------------------------------------------------
#  Kalman smoothing for shoulders
# -------------------------------------------------------------
class Kalman1D:
    def __init__(self, q=1e-3, r=1e-2):
        self.x = 0
        self.P = 1
        self.Q = q
        self.R = r

    def update(self, m):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (m - self.x)
        self.P *= (1 - K)
        return self.x

kal_ls_x, kal_ls_y = Kalman1D(), Kalman1D()
kal_rs_x, kal_rs_y = Kalman1D(), Kalman1D()


# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def calc_landmark_list(image, hand_lms):
    h, w = image.shape[:2]
    pts = []
    for lm in hand_lms.landmark:
        pts.append([int(lm.x*w), int(lm.y*h)])
    return pts

def pre_process_landmark(pts):
    pts2 = copy.deepcopy(pts)
    base_x, base_y = pts2[0]

    for p in pts2:
        p[0] -= base_x
        p[1] -= base_y

    pts2 = list(itertools.chain.from_iterable(pts2))
    max_val = max(map(abs, pts2))
    pts2 = [v/max_val for v in pts2]
    return pts2

def get_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


# -------------------------------------------------------------
# HAND ID STABILITY  — MATCH BY WRIST X POSITION
# -------------------------------------------------------------
def match_hands_to_ids(multi_hand_lm, width):
    global TRACKED_HANDS

    if multi_hand_lm is None:
        TRACKED_HANDS = {"LEFT": None, "RIGHT": None}
        return

    wrists = []
    for h in multi_hand_lm:
        wx = h.landmark[0].x * width
        wrists.append(wx)

    # Sort by x position: smaller x ⇒ LEFT hand
    sorted_hands = sorted(zip(wrists, multi_hand_lm), key=lambda x: x[0])

    # Assign
    if len(sorted_hands) == 1:
        TRACKED_HANDS["LEFT"] = sorted_hands[0][1]
        TRACKED_HANDS["RIGHT"] = None

    elif len(sorted_hands) >= 2:
        TRACKED_HANDS["LEFT"] = sorted_hands[0][1]
        TRACKED_HANDS["RIGHT"] = sorted_hands[1][1]


# -------------------------------------------------------------
# Torso region + drawing
# -------------------------------------------------------------
def torso_info(pose_lm, face_det, w, h):
    if pose_lm is None:
        return None

    lm = pose_lm.landmark
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    ls_x = kal_ls_x.update(ls.x*w)
    ls_y = kal_ls_y.update(ls.y*h)
    rs_x = kal_rs_x.update(rs.x*w)
    rs_y = kal_rs_y.update(rs.y*h)

    shoulder_width = np.linalg.norm([ls_x - rs_x, ls_y - rs_y])
    center_y = (ls_y + rs_y)/2

    face_top = 0
    if face_det:
        fb = face_det[0].location_data.relative_bounding_box
        face_top = fb.ymin*h

    torso_top = face_top
    torso_bottom = center_y + 1.6*shoulder_width

    return {
        "ls": (ls_x, ls_y),
        "rs": (rs_x, rs_y),
        "top": torso_top,
        "bottom": torso_bottom
    }


def draw_torso(frame, torso):
    if torso is None:
        return
    h,w=frame.shape[:2]

    top = int(torso["top"])
    bottom = int(torso["bottom"])

    ls_x, ls_y = torso["ls"]
    rs_x, rs_y = torso["rs"]

    cv2.rectangle(frame,(0,top),(w,bottom),(60,80,255),2)
    cv2.line(frame,(int(ls_x),int(ls_y)),(int(rs_x),int(rs_y)),(0,255,255),2)
    mid_x = int((ls_x + rs_x)/2)
    cv2.line(frame,(mid_x,top),(mid_x,bottom),(200,200,0),1)


# -------------------------------------------------------------
# Face occlusion
# -------------------------------------------------------------
def hand_in_front_of_face(hand, face_det, w, h):
    if not face_det:
        return False
    wrist = hand.landmark[0]
    wx, wy = wrist.x*w, wrist.y*h

    for f in face_det:
        box = f.location_data.relative_bounding_box
        x1 = box.xmin*w
        y1 = box.ymin*h
        x2 = x1 + box.width*w
        y2 = y1 + box.height*h

        if x1 <= wx <= x2 and y1 <= wy <= y2:
            return True
    return False


def hand_in_torso(hand, torso, w, h):
    if torso is None:
        return True
    wrist = hand.landmark[0]
    wy = wrist.y*h
    return torso["top"] <= wy <= torso["bottom"]


# -------------------------------------------------------------
# UI Bars
# -------------------------------------------------------------
def draw_status(frame, x1, y1, conf, text, color):
    bar_w = 120
    bar_h = 10

    cv2.rectangle(frame, (x1,y1), (x1+bar_w,y1+bar_h),(140,140,140),1)
    cv2.rectangle(frame, (x1,y1),
                  (x1+int(bar_w*conf),y1+bar_h), color, -1)

    cv2.putText(frame, text, (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# -------------------------------------------------------------
# Camera Loop
# -------------------------------------------------------------
cap = cv2.VideoCapture(0)
prev_time = time.time()

with mp_hands.Hands(max_num_hands=2) as hands, \
     mp_pose.Pose(model_complexity=0) as pose, \
     mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame=cv2.flip(frame,1)
        h,w = frame.shape[:2]

        now = time.time()
        fps_queue.append(1/(now - prev_time))
        prev_time = now
        fps = int(np.mean(fps_queue))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = hands.process(rgb)
        pose_res = pose.process(rgb)
        face_res = face.process(rgb)

        # Update persistent hand IDs
        match_hands_to_ids(hand_res.multi_hand_landmarks, w)

        torso = torso_info(pose_res.pose_landmarks,
                           face_res.detections if face_res else None, w,h)

        draw_torso(frame, torso)

        detected = False

        for hand_id in ["LEFT", "RIGHT"]:
            hand_lms = TRACKED_HANDS[hand_id]
            if hand_lms is None:
                continue

            last_hand_time = time.time()

            pts = calc_landmark_list(frame, hand_lms)
            x1,y1,x2,y2 = get_bbox(pts)

            occluded = hand_in_front_of_face(hand_lms,
                        face_res.detections if face_res else None, w,h)
            in_torso = hand_in_torso(hand_lms, torso, w,h)

            if occluded:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                draw_status(frame,x1,y2+20,0,"OCCLUDED",(0,0,255))
                continue

            if not in_torso:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                draw_status(frame,x1,y2+20,0,"FAR",(0,0,255))
                continue

            # Predict
            processed = pre_process_landmark(pts)
            processed = np.array(processed).reshape(1,-1)
            preds = model.predict(processed,verbose=0)
            conf = float(np.max(preds))
            cls = int(np.argmax(preds))

            if conf < LETTER_CONF_THRESH:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                draw_status(frame,x1,y2+20,conf,"LOW CONF",(0,255,255))
                continue

            # Cooldown
            if cooldowns[hand_id] > 0:
                cooldowns[hand_id] -= 1
                continue

            pred_queues[hand_id].append(cls)

            if len(pred_queues[hand_id]) == PRED_WINDOW:
                final_cls = max(set(pred_queues[hand_id]),
                                key=pred_queues[hand_id].count)
                letter = alphabet[final_cls]

                cv2.putText(frame, letter, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,0),3)

                # word building
                #global last_letter_time, current_word, sentence

                current_word += letter
                last_letter_time = time.time()

                # log to CSV
                sentence_log.append([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    letter, current_word, sentence
                ])

                np.savetxt(CSV_PATH, sentence_log,
                           fmt="%s", delimiter=",")
                cooldowns[hand_id] = COOLDOWN_FRAMES

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            draw_status(frame,x1,y2+20,conf,"OK",(0,255,0))
            detected = True

        # WORD FINALIZATION LOGIC
        if time.time() - last_letter_time > 1.2 and len(current_word)>0:
            sentence += current_word + " "
            current_word = ""

        # DISPLAY sentence + current word
        cv2.putText(frame, "Sentence: " + sentence,
                    (10,h-45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)
        cv2.putText(frame, "Word: " + current_word,
                    (10,h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)

        # FPS
        cv2.putText(frame,f"FPS: {fps}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.imshow("ISL Detector",frame)
        if cv2.waitKey(1)&0xFF==27:
            break

cap.release()
cv2.destroyAllWindows()
