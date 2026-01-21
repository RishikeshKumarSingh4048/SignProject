import sys
import cv2
import numpy as np
import math
import time
import csv
from datetime import datetime
import multiprocessing as mp

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QGridLayout
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

import psutil
import GPUtil

# =============================
# CONFIG
# =============================
IMG_SIZE = 300
OFFSET = 20
CONFIDENCE_THRESHOLD = 0.7
SIGN_DURATION = 2.0
BACKSPACE_REPEAT_DELAY = 2.0
BACKSPACE_REPEAT_RATE = 0.4

LABELS = ["A","B","C","D","E","F","G","H","I","J","K","L","M",
          "N","O","P","Q","R","S","T","U","V","W","X","Y","Z","_","_B"]

CSV_FILE = "sentences.csv"

# =============================
# CAMERA + MODEL THREAD
# =============================
class CameraWorker(QThread):
    frame_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.model = load_model("model2/keras_model.h5")
        self.running = True

    def run(self):
        prev_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            output = frame.copy()
            hands, _ = self.detector.findHands(frame)

            detected_letter = ""
            confidence = 0
            imgCrop = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgCrop = frame[y-OFFSET:y+h+OFFSET, x-OFFSET:x+w+OFFSET]

                if imgCrop.size != 0:
                    aspect = h / w
                    if aspect > 1:
                        k = IMG_SIZE / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                        wGap = (IMG_SIZE - wCal) // 2
                        imgWhite[:, wGap:wGap+wCal] = imgResize
                    else:
                        k = IMG_SIZE / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                        hGap = (IMG_SIZE - hCal) // 2
                        imgWhite[hGap:hGap+hCal, :] = imgResize

                    inp = imgWhite.astype(np.float32) / 255.0
                    inp = np.expand_dims(inp, axis=0)
                    preds = self.model.predict(inp, verbose=0)
                    idx = np.argmax(preds)
                    confidence = preds[0][idx]
                    detected_letter = LABELS[idx]

                    cv2.rectangle(output, (x-OFFSET, y-OFFSET),
                                  (x+w+OFFSET, y+h+OFFSET), (255,0,255), 3)
                    cv2.putText(output,
                                f"{detected_letter} {confidence*100:.1f}%",
                                (x, y-20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255,255,255), 2)

            fps = 1 / max(time.time() - prev_time, 1e-6)
            prev_time = time.time()

            self.frame_ready.emit({
                "output": output,
                "crop": imgCrop,
                "white": imgWhite,
                "letter": detected_letter,
                "confidence": confidence,
                "fps": fps
            })

    def stop(self):
        self.running = False
        self.cap.release()
        self.quit()
        self.wait()

# =============================
# SYSTEM MONITOR PROCESS
# =============================
def system_monitor(queue):
    while True:
        cpu = psutil.cpu_percent(percpu=True)
        ram = psutil.virtual_memory().percent
        gpus = GPUtil.getGPUs()
        gpu = f"{gpus[0].load*100:.1f}%" if gpus else "N/A"
        queue.put((cpu, ram, gpu))
        time.sleep(1)

# =============================
# MAIN WINDOW
# =============================
class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition System")
        self.setGeometry(0, 0, 1920, 1080)

        # ---- STATE ----
        self.current_letter = ""
        self.letter_start_time = 0
        self.last_backspace_time = 0
        self.word = ""
        self.sentence = []

        # ---- CSV ----
        try:
            with open(CSV_FILE, "x", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "Word", "Sentence"])
        except FileExistsError:
            pass

        # ---- UI ----
        self.init_ui()

        # ---- Camera thread ----
        self.cam_thread = CameraWorker()
        self.cam_thread.frame_ready.connect(self.process_frame)
        self.cam_thread.start()

        # ---- System monitor ----
        self.sys_queue = mp.Queue()
        self.sys_proc = mp.Process(target=system_monitor,
                                   args=(self.sys_queue,),
                                   daemon=True)
        self.sys_proc.start()

        self.sys_timer = QTimer()
        self.sys_timer.timeout.connect(self.update_system_panel)
        self.sys_timer.start(1000)

    # =============================
    # UI
    # =============================
    def init_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        grid = QGridLayout(central)

        self.camera_label = QLabel()
        self.crop_label = QLabel()
        self.white_label = QLabel()
        self.text_panel = QLabel()
        self.debug_panel = QLabel()

        for lbl in [self.camera_label, self.crop_label,
                    self.white_label, self.text_panel, self.debug_panel]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:black;color:white")

        grid.addWidget(self.camera_label, 0, 0, 2, 2)
        grid.addWidget(self.crop_label, 2, 0)
        grid.addWidget(self.white_label, 2, 1)
        grid.addWidget(self.text_panel, 0, 2)
        grid.addWidget(self.debug_panel, 1, 2, 2, 1)

    # =============================
    # FRAME PROCESSING (LOGIC PRESERVED)
    # =============================
    def process_frame(self, data):
        now = time.time()

        detected_letter = data["letter"]
        confidence = data["confidence"]

        if detected_letter and confidence >= CONFIDENCE_THRESHOLD:
            if detected_letter != self.current_letter:
                self.current_letter = detected_letter
                self.letter_start_time = now
                self.last_backspace_time = 0
            else:
                elapsed = now - self.letter_start_time
                if self.current_letter == "_B":
                    if elapsed >= BACKSPACE_REPEAT_DELAY and \
                       now - self.last_backspace_time >= BACKSPACE_REPEAT_RATE:
                        self.word = self.word[:-1]
                        self.last_backspace_time = now
                elif elapsed >= SIGN_DURATION:
                    if self.current_letter == "_":
                        if self.word:
                            self.sentence.append(self.word)
                            self.word = ""
                    else:
                        self.word += self.current_letter

                    with open(CSV_FILE, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [datetime.now(), self.word, " ".join(self.sentence)]
                        )
                    self.current_letter = ""
        else:
            self.current_letter = ""
            self.letter_start_time = 0

        # ---- TEXT PANEL ----
        text_panel = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(text_panel, f"Char: {detected_letter}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(text_panel, f"Word: {self.word}", (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        cv2.putText(text_panel,
                    f"Sentence: {' '.join(self.sentence)}",
                    (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(text_panel, f"FPS: {data['fps']:.2f}", (10,160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # ---- DISPLAY ----
        self.camera_label.setPixmap(self.to_pixmap(data["output"], 960, 540))
        self.crop_label.setPixmap(self.to_pixmap(data["crop"], 480, 270))
        self.white_label.setPixmap(self.to_pixmap(data["white"], 480, 270))
        self.text_panel.setPixmap(self.to_pixmap(text_panel, 400, 300))

    # =============================
    # SYSTEM PANEL
    # =============================
    def update_system_panel(self):
        if self.sys_queue.empty():
            return
        cpu, ram, gpu = self.sys_queue.get()

        debug = np.zeros((500, 400, 3), dtype=np.uint8)
        y = 30
        for i, c in enumerate(cpu):
            cv2.putText(debug, f"CPU {i}: {c:.1f}%", (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            y += 18
        cv2.putText(debug, f"RAM: {ram:.1f}%", (10,y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(debug, f"GPU: {gpu}", (10,y+60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        self.debug_panel.setPixmap(self.to_pixmap(debug, 400, 500))

    def to_pixmap(self, img, w, h):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        h_, w_, ch = img.shape
        return QPixmap.fromImage(QImage(img.data, w_, h_, ch*w_, QImage.Format_RGB888))

    def closeEvent(self, event):
        self.cam_thread.stop()
        self.sys_proc.terminate()
        event.accept()

# =============================
# RUN
# =============================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())

