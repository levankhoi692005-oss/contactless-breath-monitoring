
"""
breathing_worker.py (skeleton)

Phiên bản khung cho chế độ đo 60 giây.
Hoàn thiện bằng cách ghép với các module:
camera.py, pose_detector.py, roi_tracker.py,
optical_flow.py, signal_processing.py, peak_detector.py
"""

import cv2
import time
import json
import numpy as np
from collections import deque

from camera import Camera
from pose_detector import PoseDetector
from roi_tracker import ROITracker
from optical_flow import OpticalFlowTracker
from signal_processing import SignalProcessor
from peak_detector import PeakDetector
from mqtt_client import MQTTClient
from config import *

from graph import GraphDrawer

class BreathingWorker:

    def __init__(self):

        self.camera = Camera()
        self.pose = PoseDetector()
        self.roi_tracker = ROITracker()
        self.flow = OpticalFlowTracker()

        self.graph = GraphDrawer()
        self.filtered_signal = None
        self.signal_peaks = []



        self.fps = self.camera.get_fps()
        if self.fps <= 1:
            self.fps = TARGET_FPS

        self.processor = SignalProcessor(self.fps)
        self.detector = PeakDetector(self.fps)
        self.mqtt = MQTTClient()

        self.graph = GraphDrawer()

        self.filtered_signal = None

        self.signal_peaks = []


        self.signal_buffer = deque(maxlen=int(self.fps * WINDOW_SECONDS))

        self.current_bpm = 0
        self.current_confidence = 0

        self.is_measuring = False
        self.measure_duration = 60
        self.measure_start = 0
        self.remaining_time = 60



    def start_measurement(self):
        self.signal_buffer.clear()
        self.flow.reset()
        self.is_measuring = True
        self.measure_start = time.time()
        self.remaining_time = self.measure_duration
        print("Measurement started.")

    def finish_measurement(self):
        self.is_measuring = False

        signal = np.array(self.signal_buffer)
        signal = self.processor.process(signal)

        self.filtered_signal = signal.copy()

        result = self.detector.estimate(signal, self.processor)
        if result is not None:
            self.signal_peaks = result["peaks"]

        if result:
            self.current_bpm = result["bpm"]
            self.current_confidence = result["confidence"]

            with open("output/latest.json", "w") as f:
                json.dump({
                    "bpm": self.current_bpm,
                    "confidence": self.current_confidence,
                    "fps": self.fps
                }, f)

            self.mqtt.publish(
                self.current_bpm,
                self.current_confidence,
                self.fps
            )

            print("BPM:", self.current_bpm)

    def update_measurement(self):

        if not self.is_measuring:
            return

        elapsed = time.time() - self.measure_start
        self.remaining_time = max(
            0,
            int(self.measure_duration - elapsed)
        )

        if elapsed >= self.measure_duration:
            self.finish_measurement()

    def run(self):

        while True:

            frame = self.camera.read()

            detection = self.pose.detect(frame)

            if detection is not None:

                chest_box = self.pose.get_chest_box(detection)

                roi, roi_box = self.roi_tracker.get_roi(
                    frame,
                    chest_box
                )

                frame = self.pose.draw(frame, detection)

                frame = self.roi_tracker.draw(frame, roi_box)

                if roi is not None:

                    displacement = self.flow.update(roi)

                    if self.is_measuring and displacement is not None:
                        self.signal_buffer.append(displacement)

                    cv2.imshow("Chest ROI", roi)

            if frame is None:
                break

            # TODO:
            # detection = self.pose.detect(frame)
            # roi = ...
            # displacement = ...
            # if self.is_measuring:
            #     self.signal_buffer.append(displacement)

            self.update_measurement()

            cv2.putText(
                frame,
                f"BPM: {self.current_bpm}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            if self.is_measuring:
                txt = f"Measuring: {self.remaining_time}s"
            else:
                txt = "Press S to Start"

            cv2.putText(
                frame,
                txt,
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )

            frame = self.graph.draw(

                frame,

                self.filtered_signal,

                self.signal_peaks

            )
            cv2.imshow("Breathing AI", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                self.start_measurement()

            elif key == 27:
                break

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    BreathingWorker().run()
