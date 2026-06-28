"""
breathing_worker.py
-------------------

Module chính của hệ thống đo nhịp thở.

Pipeline

Camera
    ↓
YOLO Pose
    ↓
Chest ROI
    ↓
Optical Flow
    ↓
Signal Buffer
    ↓
Signal Processing
    ↓
Peak Detection
    ↓
Respiration Rate
"""

import cv2
import time
import numpy as np
from collections import deque
from mqtt_client import MQTTClient
from camera import Camera
from pose_detector import PoseDetector
from roi_tracker import ROITracker
from optical_flow import OpticalFlowTracker
from signal_processing import SignalProcessor
from peak_detector import PeakDetector

import json
from config import *


class BreathingWorker:

    def __init__(self):

        print("=" * 50)
        print("Initializing Breathing Worker")
        print("=" * 50)

        # Camera
        self.camera = Camera()
        self.mqtt = MQTTClient()
        self.fps = self.camera.get_fps()

        if self.fps <= 1:
            self.fps = TARGET_FPS

        print("FPS :", self.fps)

        # Pose
        self.pose = PoseDetector()

        # ROI
        self.roi_tracker = ROITracker()

        # Optical Flow
        self.flow = OpticalFlowTracker()

        # Signal Processing
        self.processor = SignalProcessor(self.fps)

        # Peak Detector
        self.detector = PeakDetector(self.fps)

        # Buffer tín hiệu
        self.signal_buffer = deque(
            maxlen=int(self.fps * WINDOW_SECONDS)
        )

        # Buffer thời gian
        self.time_buffer = deque(
            maxlen=int(self.fps * WINDOW_SECONDS)
        )

        # BPM hiện tại
        self.current_bpm = 0

        self.current_confidence = 0

        # FPS
        self.prev_time = time.time()

        self.display_fps = 0

        # Trạng thái
        self.person_detected = False

        self.frame_count = 0

        print("Initialization Complete.")

    # --------------------------------------------------

    def update_fps(self):

        now = time.time()

        dt = now - self.prev_time

        self.prev_time = now

        if dt > 0:
            self.display_fps = 1.0 / dt

    # --------------------------------------------------

    def append_signal(self, value):

        self.signal_buffer.append(value)

        self.time_buffer.append(time.time())

    # --------------------------------------------------

    def enough_signal(self):

        return len(self.signal_buffer) >= self.fps * 15

    # --------------------------------------------------

    def estimate_breathing(self):

        signal = np.array(self.signal_buffer)

        signal = self.processor.process(signal)

        result = self.detector.estimate(

            signal,

            self.processor

        )

        if result is None:
            return

        self.current_bpm = result["bpm"]

        self.current_confidence = result["confidence"]

        # ===== Publish MQTT =====
        self.mqtt.publish(
            self.current_bpm,
            self.current_confidence,
            self.display_fps
        )

        with open("output/latest.json", "w") as f:
            json.dump({

                "bpm": self.current_bpm,

                "confidence": self.current_confidence,

                "fps": self.display_fps

            }, f)


    # --------------------------------------------------

    def draw_information(self, frame):

        cv2.putText(
            frame,
            f"BPM : {self.current_bpm:.1f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence : {self.current_confidence:.1f}%",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2
        )

        cv2.putText(
            frame,
            f"FPS : {self.display_fps:.1f}",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Samples : {len(self.signal_buffer)}",
            (20,160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,255),
            2
        )

        return frame

    # --------------------------------------------------

    def process_frame(self, frame):

        detection = self.pose.detect(frame)

        if detection is None:

            self.person_detected = False

            self.flow.reset()

            return frame

        self.person_detected = True

        chest_box = self.pose.get_chest_box(detection)

        roi, roi_box = self.roi_tracker.get_roi(
            frame,
            chest_box
        )

        frame = self.pose.draw(frame, detection)
        frame = self.roi_tracker.draw(frame, roi_box)

        if roi is None:
            return frame

        displacement = self.flow.update(roi)

        if displacement is not None:
            self.append_signal(displacement)

        roi = self.flow.draw_points(roi)

        cv2.imshow("Chest ROI", roi)

        return frame


    # --------------------------------------------------

    def update_estimation(self):

        """
        Chỉ bắt đầu tính BPM khi đã có đủ tín hiệu.
        """

        if not self.enough_signal():
            return

        try:

            self.estimate_breathing()

        except Exception as e:

            print("Estimate Error :", e)

    # --------------------------------------------------

    def save_csv(self):

        """
        Lưu tín hiệu để phân tích sau.
        """

        if len(self.signal_buffer) == 0:
            return

        try:

            signal = np.array(self.signal_buffer)

            np.savetxt(
                CSV_FILE,
                signal,
                delimiter=","
            )

        except Exception as e:

            print(e)

    # --------------------------------------------------

    def reset_tracking(self):

        """
        Reset khi mất người.
        """

        self.flow.reset()

        self.signal_buffer.clear()

        self.time_buffer.clear()

        self.current_bpm = 0

        self.current_confidence = 0

    # --------------------------------------------------

    def draw_status(self, frame):

        status = "Detected"

        color = (0,255,0)

        if not self.person_detected:

            status = "No Person"

            color = (0,0,255)

        cv2.putText(

            frame,

            status,

            (20,200),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            color,

            2

        )

        return frame

    # --------------------------------------------------

    def run(self):

        print("Start Monitoring...")

        last_estimate = time.time()

        estimate_interval = 5

        while True:

            frame = self.camera.read()

            if frame is None:

                break

            self.frame_count += 1

            self.update_fps()

            frame = self.process_frame(frame)

            now = time.time()

            if now - last_estimate >= estimate_interval:

                self.update_estimation()

                self.save_csv()

                last_estimate = now

            frame = self.draw_information(frame)

            frame = self.draw_status(frame)

            cv2.imshow(

                "Breathing AI",

                frame

            )

            key = cv2.waitKey(1)

            if key == 27:

                break

            elif key == ord("r"):

                print("Reset")

                self.reset_tracking()

        self.camera.release()

        cv2.destroyAllWindows()


# ======================================================

if __name__ == "__main__":

    worker = BreathingWorker()

    worker.run()