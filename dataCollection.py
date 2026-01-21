import cv2
import mediapipe as mp
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

counter =0

folder = "data"
directories = ["A", "B", "C", "D",  "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "_","_B"]
folderS = "data/_B"

for name in directories:
    path = os.path.join(folder, name)
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Successfully created directory:{path}")
    except OSError as e:
        print(f"Error creating directory {path}: e")


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hands = hands[0]
        x,y,w,h = hands['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y - offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        

        aspectRatio = h/w

        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap]= imgResize

        else:
            k = imgSize/w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap, :]= imgResize

        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folderS}/image_{time.time()}.jpg', imgWhite)
        print(counter)