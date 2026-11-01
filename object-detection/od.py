
import cv2
import mediapipe as mp
import numpy as np
import customtkinter as ctk
from PIL import Image
from PIL import ImageTk
from ultralytics import YOLO
import threading
import queue
import time
import csv
import os
import urllib.request
from datetime import datetime
import pyttsx3
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing


# =====================================================
# CONFIGURATION
# =====================================================

APP_TITLE = "AI Hand Gesture Recognition System"
CAMERA_INDEX = 0
YOLO_MODEL = "yolov8n.pt"
YOLO_ENABLED = False

SAVE_DIR = "gesture_results"
os.makedirs(SAVE_DIR, exist_ok=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# =====================================================
# SPEECH ENGINE
# =====================================================

class SpeechEngine:

    def __init__(self):
        self.messages = queue.Queue()
        self.enabled = True

        self.worker = threading.Thread(
            target=self.speech_worker,
            daemon=True
        )
        self.worker.start()

    def speech_worker(self):
        engine = None

        try:
            engine = pyttsx3.init()
        except Exception as e:
            print("Speech initialization error:", e)

        while True:
            message = self.messages.get()

            if message is None:
                break

            if engine and self.enabled:
                try:
                    engine.say(message)
                    engine.runAndWait()
                except Exception as e:
                    print("Speech error:", e)

    def speak(self, message):
        if self.enabled:
            self.messages.put(message)


# =====================================================
# HAND GESTURE RECOGNITION
# =====================================================

class HandGestureRecognizer:

    def __init__(self):

        model_path = os.path.join(SAVE_DIR, "hand_landmarker.task")
        if not os.path.exists(model_path):
            download_path = f"{model_path}.download"
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/"
                "hand_landmarker.task"
            )
            try:
                urllib.request.urlretrieve(model_url, download_path)
                os.replace(download_path, model_path)
            except Exception:
                if os.path.exists(download_path):
                    os.remove(download_path)
                raise

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.65,
            min_tracking_confidence=0.60
        )
        self.mp_hands = mp_vision.HandLandmarksConnections
        self.mp_draw = mp_drawing
        self.hands = mp_vision.HandLandmarker.create_from_options(options)

        self.previous_positions = {}
        self.last_direction = {}
        self.last_gesture = {}

    def distance(self, p1, p2):

        return np.linalg.norm(
            np.array(p1) - np.array(p2)
        )

    def get_finger_states(self, landmarks):

        lm = landmarks

        # Thumb extension based on its distance from index MCP.
        thumb_tip = [lm[4].x, lm[4].y]
        thumb_ip = [lm[3].x, lm[3].y]
        index_mcp = [lm[5].x, lm[5].y]

        thumb_open = (
            self.distance(thumb_tip, index_mcp)
            >
            self.distance(thumb_ip, index_mcp) * 1.15
        )

        # Other fingers: tip must be above PIP.
        index_open = lm[8].y < lm[6].y
        middle_open = lm[12].y < lm[10].y
        ring_open = lm[16].y < lm[14].y
        pinky_open = lm[20].y < lm[18].y

        return [
            thumb_open,
            index_open,
            middle_open,
            ring_open,
            pinky_open
        ]

    def recognize_gesture(self, lm, fingers):

        thumb, index, middle, ring, pinky = fingers

        count = sum(fingers)

        # OK sign: thumb and index tips close together.
        thumb_index_distance = self.distance(
            [lm[4].x, lm[4].y],
            [lm[8].x, lm[8].y]
        )

        hand_scale = max(
            self.distance(
                [lm[0].x, lm[0].y],
                [lm[9].x, lm[9].y]
            ),
            0.001
        )

        ok_sign = (
            thumb_index_distance < hand_scale * 0.35
            and middle and ring and pinky
        )

        if ok_sign:
            return "OK Sign"

        # Thumbs up/down.
        if thumb and not any(
            [index, middle, ring, pinky]
        ):
            if lm[4].y < lm[2].y:
                return "Thumbs Up"
            elif lm[4].y > lm[2].y:
                return "Thumbs Down"
            return "Thumb"

        # Common static gestures.
        if count == 0:
            return "Closed Fist"

        if count == 5:
            return "Open Palm"

        if fingers == [
            False, True, True, False, False
        ]:
            return "Victory / Peace"

        if fingers == [
            False, True, False, False, False
        ]:
            return "Pointing"

        if fingers == [
            False, True, True, True, False
        ]:
            return "Three Fingers"

        if fingers == [
            False, True, True, True, True
        ]:
            return "Four Fingers"

        if fingers == [
            True, True, False, False, True
        ]:
            return "Rock Sign"

        if fingers == [
            True, False, False, False, True
        ]:
            return "Call Me"

        if count == 2:
            return "Two Fingers"

        if count == 3:
            return "Three Fingers"

        if count == 4:
            return "Four Fingers"

        if count == 1:
            return "One Finger"

        return "Unknown Gesture"

    def recognize_direction(self, hand_id, lm):

        wrist = np.array([
            lm[0].x,
            lm[0].y
        ])

        previous = self.previous_positions.get(hand_id)

        direction = "Stationary"

        if previous is not None:

            dx = wrist[0] - previous[0]
            dy = wrist[1] - previous[1]

            threshold = 0.035

            if abs(dx) > threshold or abs(dy) > threshold:

                if abs(dx) > abs(dy):
                    direction = (
                        "Moving Right" if dx > 0
                        else "Moving Left"
                    )
                else:
                    direction = (
                        "Moving Down" if dy > 0
                        else "Moving Up"
                    )

        self.previous_positions[hand_id] = wrist

        return direction

    def process(self, frame):

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        results = self.hands.detect_for_video(
            image,
            time.monotonic_ns() // 1_000_000
        )

        detected = []

        if results.hand_landmarks:

            for i, hand_landmarks in enumerate(
                results.hand_landmarks
            ):

                lm = hand_landmarks

                handedness = "Hand"

                if results.handedness:
                    handedness = (
                        results.handedness[i][0].category_name
                    )

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                fingers = self.get_finger_states(lm)

                count = sum(fingers)

                gesture = self.recognize_gesture(
                    lm, fingers
                )

                direction = self.recognize_direction(
                    i, lm
                )

                # Approximate heuristic score, not a
                # calibrated model probability.
                confidence = 95 if gesture != "Unknown Gesture" else 40

                h, w, _ = frame.shape

                x_coordinates = [
                    int(point.x * w) for point in lm
                ]

                y_coordinates = [
                    int(point.y * h) for point in lm
                ]

                x1 = max(0, min(x_coordinates) - 15)
                y1 = max(0, min(y_coordinates) - 15)

                x2 = min(w, max(x_coordinates) + 15)
                y2 = min(h, max(y_coordinates) + 15)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{handedness}: {gesture}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Fingers: {count}/5",
                    (x1, min(h - 10, y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                detected.append({
                    "hand": handedness,
                    "gesture": gesture,
                    "fingers": count,
                    "finger_states": fingers,
                    "direction": direction,
                    "confidence": confidence
                })

        return frame, detected


# =====================================================
# YOLO OBJECT DETECTION
# =====================================================

class YOLODetector:

    def __init__(self):

        self.model = None
        self.enabled = False

        try:
            self.model = YOLO(YOLO_MODEL)
            print("YOLO model loaded successfully.")

        except Exception as e:
            print("YOLO model unavailable:", e)

    def detect(self, frame):

        if not self.enabled or self.model is None:
            return frame, []

        try:

            results = self.model(
                frame,
                verbose=False,
                conf=0.40
            )

            detections = []

            for result in results:

                for box in result.boxes:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    class_id = int(
                        box.cls[0]
                    )

                    label = self.model.names[class_id]

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 150, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{label} {confidence:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 150, 0),
                        2
                    )

                    detections.append(label)

            return frame, detections

        except Exception as e:
            print("YOLO error:", e)
            return frame, []


# =====================================================
# MAIN APPLICATION
# =====================================================

class GestureApplication(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1250x780")
        self.minsize(950, 650)

        self.cap = None
        self.running = False
        self.camera_thread = None

        self.recognizer = HandGestureRecognizer()
        self.yolo = YOLODetector()
        self.speech = SpeechEngine()

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_info = []

        self.last_added_gesture = {}
        self.last_added_time = {}

        self.gesture_history = []
        self.sentence = []

        self.previous_time = time.time()
        self.fps = 0

        self.create_interface()

        self.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

        self.after(30, self.update_interface)

    # -------------------------------------------------
    # GUI
    # -------------------------------------------------

    def create_interface(self):

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(
            self,
            text="AI HAND GESTURE RECOGNITION",
            font=("Arial", 25, "bold")
        )

        title.grid(
            row=0,
            column=0,
            columnspan=2,
            pady=15
        )

        # Webcam panel.

        self.video_panel = ctk.CTkLabel(
            self,
            text="Camera is stopped",
            fg_color="#171717",
            corner_radius=12
        )

        self.video_panel.grid(
            row=1,
            column=0,
            padx=15,
            pady=10,
            sticky="nsew"
        )

        # Right dashboard.

        self.sidebar = ctk.CTkFrame(
            self,
            corner_radius=12
        )

        self.sidebar.grid(
            row=1,
            column=1,
            padx=10,
            pady=10,
            sticky="nsew"
        )

        self.sidebar.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.sidebar,
            text="RECOGNITION PANEL",
            font=("Arial", 18, "bold")
        ).pack(pady=15)

        self.gesture_label = ctk.CTkLabel(
            self.sidebar,
            text="Gesture: None",
            font=("Arial", 17, "bold"),
            wraplength=230
        )

        self.gesture_label.pack(pady=10)

        self.finger_label = ctk.CTkLabel(
            self.sidebar,
            text="Raised Fingers: 0",
            font=("Arial", 15)
        )

        self.finger_label.pack(pady=8)

        self.hand_label = ctk.CTkLabel(
            self.sidebar,
            text="Hand: None"
        )

        self.hand_label.pack(pady=8)

        self.direction_label = ctk.CTkLabel(
            self.sidebar,
            text="Movement: None"
        )

        self.direction_label.pack(pady=8)

        self.confidence_label = ctk.CTkLabel(
            self.sidebar,
            text="Recognition score: --"
        )

        self.confidence_label.pack(pady=8)

        self.fps_label = ctk.CTkLabel(
            self.sidebar,
            text="FPS: 0"
        )

        self.fps_label.pack(pady=8)

        # Text output.

        ctk.CTkLabel(
            self.sidebar,
            text="GESTURE TEXT",
            font=("Arial", 15, "bold")
        ).pack(pady=(20, 5))

        self.text_output = ctk.CTkTextbox(
            self.sidebar,
            height=100
        )

        self.text_output.pack(
            fill="x",
            padx=10,
            pady=5
        )

        # Controls.

        self.start_button = ctk.CTkButton(
            self.sidebar,
            text="Start Camera",
            command=self.start_camera
        )

        self.start_button.pack(
            fill="x",
            padx=15,
            pady=6
        )

        self.stop_button = ctk.CTkButton(
            self.sidebar,
            text="Stop Camera",
            command=self.stop_camera
        )

        self.stop_button.pack(
            fill="x",
            padx=15,
            pady=6
        )

        self.yolo_switch = ctk.CTkSwitch(
            self.sidebar,
            text="Enable YOLO",
            command=self.toggle_yolo
        )

        self.yolo_switch.pack(pady=8)

        self.speech_switch = ctk.CTkSwitch(
            self.sidebar,
            text="Enable Speech",
            command=self.toggle_speech
        )

        self.speech_switch.select()
        self.speech_switch.pack(pady=8)

        self.screenshot_button = ctk.CTkButton(
            self.sidebar,
            text="Take Screenshot",
            command=self.save_screenshot
        )

        self.screenshot_button.pack(
            fill="x",
            padx=15,
            pady=6
        )

        self.save_button = ctk.CTkButton(
            self.sidebar,
            text="Export CSV History",
            command=self.export_csv
        )

        self.save_button.pack(
            fill="x",
            padx=15,
            pady=6
        )

        self.clear_button = ctk.CTkButton(
            self.sidebar,
            text="Clear Text",
            command=self.clear_text
        )

        self.clear_button.pack(
            fill="x",
            padx=15,
            pady=6
        )

        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="Status: Ready",
            text_color="lightgreen"
        )

        self.status_label.pack(pady=15)

    # -------------------------------------------------
    # CAMERA
    # -------------------------------------------------

    def start_camera(self):

        if self.running:
            return

        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():

            self.cap.release()
            self.cap = None

            self.status_label.configure(
                text="Camera unavailable",
                text_color="red"
            )

            return

        self.running = True

        self.camera_thread = threading.Thread(
            target=self.camera_loop,
            daemon=True
        )

        self.camera_thread.start()

        self.status_label.configure(
            text="Camera running",
            text_color="lightgreen"
        )

    def stop_camera(self):

        self.running = False

        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_info = []

        self.video_panel.configure(
            image=None,
            text="Camera is stopped"
        )

        self.video_panel.image = None

        self.status_label.configure(
            text="Camera stopped"
        )

    def camera_loop(self):

        while self.running:

            if self.cap is None:
                break

            success, frame = self.cap.read()

            if not success:
                time.sleep(0.05)
                continue

            frame, information = self.recognizer.process(
                frame
            )

            if self.yolo.enabled:
                frame, _ = self.yolo.detect(frame)

            current_time = time.time()

            elapsed = current_time - self.previous_time

            if elapsed > 0:
                self.fps = 1 / elapsed

            self.previous_time = current_time

            # Add gesture history with a cooldown.
            for item in information:

                hand = item["hand"]
                gesture = item["gesture"]

                previous = self.last_added_gesture.get(hand)
                last_time = self.last_added_time.get(hand, 0)

                if (
                    gesture != "Unknown Gesture"
                    and gesture != previous
                    and current_time - last_time > 1.0
                ):

                    self.add_gesture(item)

                    self.last_added_gesture[hand] = gesture
                    self.last_added_time[hand] = current_time

            with self.frame_lock:

                self.latest_frame = frame.copy()
                self.latest_info = information.copy()

    # -------------------------------------------------
    # GESTURE HISTORY
    # -------------------------------------------------

    def add_gesture(self, item):

        gesture = item["gesture"]

        record = {
            "time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "hand": item["hand"],
            "gesture": gesture,
            "fingers": item["fingers"],
            "direction": item["direction"]
        }

        self.gesture_history.append(record)
        self.sentence.append(gesture)

        self.speech.speak(gesture)

    # -------------------------------------------------
    # GUI UPDATE
    # -------------------------------------------------

    def update_interface(self):

        with self.frame_lock:

            frame = (
                self.latest_frame.copy()
                if self.latest_frame is not None
                else None
            )

            information = self.latest_info.copy()

        if frame is not None:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(rgb)

            width = max(
                400,
                self.video_panel.winfo_width()
            )

            height = max(
                300,
                self.video_panel.winfo_height()
            )

            image.thumbnail((width, height))

            photo = ImageTk.PhotoImage(image)

            self.video_panel.configure(
                image=photo,
                text=""
            )

            self.video_panel.image = photo

        if information:

            item = information[0]

            self.gesture_label.configure(
                text=f"Gesture: {item['gesture']}"
            )

            self.finger_label.configure(
                text=f"Raised Fingers: {item['fingers']}/5"
            )

            self.hand_label.configure(
                text=f"Hand: {item['hand']}"
            )

            self.direction_label.configure(
                text=f"Movement: {item['direction']}"
            )

            self.confidence_label.configure(
                text=f"Recognition score: {item['confidence']}%"
            )

        else:

            self.gesture_label.configure(
                text="Gesture: None"
            )

            self.finger_label.configure(
                text="Raised Fingers: 0"
            )

            self.hand_label.configure(
                text="Hand: None"
            )

            self.direction_label.configure(
                text="Movement: None"
            )

            self.confidence_label.configure(
                text="Recognition score: --"
            )

        self.fps_label.configure(
            text=f"FPS: {self.fps:.1f}"
        )

        self.text_output.delete("1.0", "end")

        self.text_output.insert(
            "end",
            " ".join(self.sentence[-30:])
        )

        self.after(30, self.update_interface)

    # -------------------------------------------------
    # SETTINGS
    # -------------------------------------------------

    def toggle_yolo(self):

        self.yolo.enabled = (
            self.yolo_switch.get() == 1
        )

    def toggle_speech(self):

        self.speech.enabled = (
            self.speech_switch.get() == 1
        )

    def clear_text(self):

        self.sentence.clear()

        self.text_output.delete(
            "1.0",
            "end"
        )

    # -------------------------------------------------
    # SCREENSHOT
    # -------------------------------------------------

    def save_screenshot(self):

        with self.frame_lock:

            if self.latest_frame is None:
                self.status_label.configure(
                    text="No camera frame available"
                )
                return

            frame = self.latest_frame.copy()

        filename = os.path.join(
            SAVE_DIR,
            "screenshot_" +
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".jpg"
        )

        if cv2.imwrite(filename, frame):

            self.status_label.configure(
                text="Screenshot saved"
            )

    # -------------------------------------------------
    # CSV EXPORT
    # -------------------------------------------------

    def export_csv(self):

        if not self.gesture_history:

            self.status_label.configure(
                text="No gesture history"
            )

            return

        filename = os.path.join(
            SAVE_DIR,
            "gesture_history_" +
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".csv"
        )

        try:

            with open(
                filename,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "time",
                        "hand",
                        "gesture",
                        "fingers",
                        "direction"
                    ]
                )

                writer.writeheader()
                writer.writerows(self.gesture_history)

            self.status_label.configure(
                text="CSV exported successfully"
            )

        except Exception as e:

            self.status_label.configure(
                text=f"CSV error: {e}"
            )

    # -------------------------------------------------
    # APPLICATION SHUTDOWN
    # -------------------------------------------------

    def close_application(self):

        self.running = False

        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.speech.messages.put(None)

        self.destroy()


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app = GestureApplication()
    app.mainloop()
