import os
import cv2
import shutil
import mediapipe as mp

SOURCE = "dataset"
TARGET = "clean_dataset"

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection

hands = mp_hands.Hands(max_num_hands=1)
pose = mp_pose.Pose()
face = mp_face.FaceDetection()

def is_blurry(img):
    return cv2.Laplacian(img, cv2.CV_64F).var() < 50

def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Clean target
if os.path.exists(TARGET):
    shutil.rmtree(TARGET)
ensure(TARGET)

print("\n🧹 Cleaning Dataset...\n")

for label in os.listdir(SOURCE):
    src_label = os.path.join(SOURCE, label)
    dst_label = os.path.join(TARGET, label)
    ensure(dst_label)

    for file in os.listdir(src_label):
        path = os.path.join(src_label, file)
        if os.path.getsize(path) == 0:
            continue

        if not file.lower().endswith((".jpg",".png",".jpeg",".bmp",".mp4",".avi",".mov",".mkv")):
            continue

        # Copy valid image or video directly
        ok = True

        if file.lower().endswith((".jpg",".png",".jpeg",".bmp")):
            img = cv2.imread(path)
            if img is None: continue
            if is_blurry(img): continue
            cv2.imwrite(os.path.join(dst_label, file), img)

        else:
            # check video validity
            cap = cv2.VideoCapture(path)
            if not cap.isOpened(): continue
            cap.release()
            shutil.copy2(path, dst_label)

print("\n🎉 Dataset cleaning complete! Output saved to clean_dataset/")
