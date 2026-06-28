"""
camera.py
----------

Quản lý camera.
Hỗ trợ:

- Webcam
- Camera USB
- Video
- DroidCam
- Camo Studio

"""

import cv2

from config import *


class Camera:

    def __init__(self,
                 source=CAMERA_INDEX):

        self.source = source

        self.cap = cv2.VideoCapture(source)

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            TARGET_FPS
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                "Không mở được camera."
            )

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):

        self.cap.release()

    def get_width(self):

        return int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

    def get_height(self):

        return int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

    def get_fps(self):

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 1:
            fps = TARGET_FPS

        return fps