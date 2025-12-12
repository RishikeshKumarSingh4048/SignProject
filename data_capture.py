import cv2
import mediapipe as mp
import numpy as np
import os
import itertools
import copy
import time
from collections import deque

# -------------------------------------------------------------------
# EXACT PREPROCESSING == detection file (same functions)
# -------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection

alphabet = ['1','2','3','4','5','6','7','8','9'] + \
           [chr(ord('A')+i) for i in range(26)]

# -------------------------------------------------------------
# Torso + landmark helpers
# -------------------------------------------------------------
class Kalman1D:
    def __init__(self, q=1e-3, r=1e-2):
        self.x = 0
        self.P = 1
        self.Q = q
        self.R = r
    def update(self,m):
        self.P += self.Q
        K = self.P / (self.P+self.R)
        self.x += K*(m-self.x)
        self.P *= (1-K)
        return self.x

kal_ls_x, kal_ls_y = Kalman1D(), Kalman1D()
kal_rs_x, kal_rs_y = Kalman1D(), Kalman1D()

def calc_landmark_list(img, hand_lm):
    h,w = img.shape[:2]
    pts=[]
    for lm in hand_lm.landmark:
        pts.append([int(lm.x*w),int(lm.y*h)])
    return pts

def pre_process_landmarks(pts):
    pts2 = copy.deepcopy(pts)
    base_x,base_y = pts2[0]
    for p in pts2:
        p[0] -= base_x
        p[1] -= base_y

    pts2 = list(itertools.chain.from_iterable(pts2))
    max_val = max(map(abs,pts2))
    pts2 = [v/max_val for v in pts2]
    return pts2

def get_bbox(pts):
    xs=[p[0] for p in pts]
    ys=[p[1] for p in pts]
    return min(xs),min(ys),max(xs),max(ys)

def match_hands(multi_hand_lm, w):
    if multi_hand_lm is None: 
        return {"LEFT":None,"RIGHT":None}
    wrists=[]
    for h in multi_hand_lm:
        wrists.append(h.landmark[0].x*w)
    sorted_hands = sorted(zip(wrists,multi_hand_lm), key=lambda x: x[0])
    out={"LEFT":None,"RIGHT":None}
    if len(sorted_hands)==1:
        out["LEFT"]=sorted_hands[0][1]
    elif len(sorted_hands)>=2:
        out["LEFT"]=sorted_hands[0][1]
        out["RIGHT"]=sorted_hands[1][1]
    return out

def torso_info(pose_lm, face_det, w, h):
    if pose_lm is None: return None
    lm = pose_lm.landmark

    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    ls_x = kal_ls_x.update(ls.x*w)
    ls_y = kal_ls_y.update(ls.y*h)
    rs_x = kal_rs_x.update(rs.x*w)
    rs_y = kal_rs_y.update(rs.y*h)

    shoulder_w = np.linalg.norm([ls_x-rs_x, ls_y-rs_y])
    face_top = 0
    if face_det:
        box = face_det[0].location_data.relative_bounding_box
        face_top = box.ymin * h

    top = face_top
    bottom = (ls_y+rs_y)/2 + 1.6*shoulder_w

    return {
        "ls": (ls_x,ls_y),
        "rs": (rs_x,rs_y),
        "top": top,
        "bottom": bottom
    }

def hand_in_torso(hand, torso, w,h):
    if torso is None: return True
    wrist = hand.landmark[0]
    wy = wrist.y*h
    return torso["top"] <= wy <= torso["bottom"]

def hand_in_face(hand, face_det, w,h):
    if not face_det: return False
    wrist = hand.landmark[0]
    wx,wy = wrist.x*w, wrist.y*h

    for f in face_det:
        b = f.location_data.relative_bounding_box
        x1=b.xmin*w
        y1=b.ymin*h
        x2=x1+b.width*w
        y2=y1+b.height*h
        if x1<=wx<=x2 and y1<=wy<=y2: return True
    return False

def draw_torso(frame, torso):
    if torso is None: return
    h,w = frame.shape[:2]
    top=int(torso["top"])
    bottom=int(torso["bottom"])
    ls_x,ls_y = torso["ls"]
    rs_x,rs_y = torso["rs"]

    cv2.rectangle(frame,(0,top),(w,bottom),(50,120,220),2)
    cv2.line(frame,(int(ls_x),int(ls_y)),
                  (int(rs_x),int(rs_y)),(0,255,255),2)
    mid_x = int((ls_x+rs_x)/2)
    cv2.line(frame,(mid_x,top),(mid_x,bottom),(200,200,0),1)


# =============================================================
# DATA COLLECTION CONFIG
# =============================================================
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# -------------------------------------------------------------
# NAME OF CLASS (char/word/sentence)
# -------------------------------------------------------------
print("\n👉 Enter LABEL name for this recording session.")
print("   Examples: A, B, C, HELLO, GOOD MORNING, etc.")
label_name = input("Label: ").strip().upper().replace(" ", "_")

save_dir = os.path.join(DATASET_DIR, label_name)
os.makedirs(save_dir, exist_ok=True)

print(f"\n📁 Images will be saved to: {save_dir}\n")

capture_mode = None
print("Choose capture mode:")
print("1 = Capture single IMAGES")
print("2 = Capture VIDEO clips")
print("3 = Capture FRAME SEQUENCES")
mode = input("Select (1/2/3): ").strip()


# =============================================================
# START CAMERA
# =============================================================
cap=cv2.VideoCapture(0)
img_counter = 0
sequence_frames = []
sequence_id = 1
recording = False
video_writer = None

with mp_hands.Hands(max_num_hands=2) as hands, \
     mp_pose.Pose(model_complexity=0) as pose, \
     mp_face.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face:

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame,1)
        h,w = frame.shape[:2]

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        hand_res = hands.process(rgb)
        pose_res = pose.process(rgb)
        face_res = face.process(rgb)

        tracked = match_hands(hand_res.multi_hand_landmarks, w)
        torso = torso_info(pose_res.pose_landmarks,
                           face_res.detections if face_res else None,w,h)
        draw_torso(frame, torso)

        # ---------------------------------------------------------
        # Detect valid hand
        # ---------------------------------------------------------
        any_valid_hand = None

        for hid in ["LEFT","RIGHT"]:
            hlm = tracked[hid]
            if hlm is None: continue

            pts = calc_landmark_list(frame, hlm)
            x1,y1,x2,y2 = get_bbox(pts)

            if hand_in_face(hlm, face_res.detections if face_res else None,w,h):
                continue
            if not hand_in_torso(hlm, torso, w,h):
                continue

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            any_valid_hand = frame[y1:y2, x1:x2]

        # ---------------------------------------------------------
        # CAPTURE LOGIC
        # ---------------------------------------------------------
        if any_valid_hand is not None:

            if mode == "1":  # SINGLE IMAGES
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    img_counter += 1
                    filename = f"{label_name}_{img_counter:04d}.jpg"
                    cv2.imwrite(os.path.join(save_dir, filename), any_valid_hand)
                    print("Saved:", filename)

            elif mode == "2":  # VIDEO
                if recording:
                    video_writer.write(any_valid_hand)

            elif mode == "3":  # FRAME SEQUENCES
                if recording:
                    sequence_frames.append(any_valid_hand)

        # -------------------- video toggle ----------------------
        if mode == "2":
            if cv2.waitKey(1) & 0xFF == ord('r'):
                if not recording:
                    filename = f"{label_name}_video_{int(time.time())}.avi"
                    path = os.path.join(save_dir, filename)
                    print("▶ Recording video:", filename)

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(path, fourcc, 20, (200,200))
                    recording = True
                else:
                    print("⏹ Stopped recording.")
                    recording = False
                    video_writer.release()

        # -------------------- sequence toggle -------------------
        if mode == "3":
            if cv2.waitKey(1) & 0xFF == ord('r'):
                if not recording:
                    print("▶ Recording sequence...")
                    sequence_frames=[]
                    recording=True
                else:
                    print("⏹ Stopped. Saving sequence frames...")
                    for i,frm in enumerate(sequence_frames):
                        filename = f"{label_name}_seq{sequence_id:03d}_{i:03d}.jpg"
                        cv2.imwrite(os.path.join(save_dir, filename), frm)
                    sequence_id += 1
                    recording=False

        # ---------------------------------------------------------
        # UI text
        # ---------------------------------------------------------
        cv2.putText(frame, f"Label: {label_name}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.putText(frame, "Press C = Capture image, R = Start/Stop recording",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1)

        cv2.imshow("Data Capture", frame)
        if cv2.waitKey(1)&0xFF == 27: # ESC
            break

cap.release()
cv2.destroyAllWindows()
