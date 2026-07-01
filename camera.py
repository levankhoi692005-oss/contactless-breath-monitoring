"""
camera.py
--------------------------------
Camera Module
"""

import cv2

from config import *


class Camera:

    def __init__(self):

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

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

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                "Cannot open camera!"
            )

    # =========================================

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    # =========================================

    def get_fps(self):

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 1:
            fps = TARGET_FPS

        return fps

    # =========================================

    def get_width(self):

        return int(

            self.cap.get(

                cv2.CAP_PROP_FRAME_WIDTH

            )

        )

    # =========================================

    def get_height(self):

        return int(

            self.cap.get(

                cv2.CAP_PROP_FRAME_HEIGHT

            )

        )

    # =========================================

    def release(self):

        if self.cap:

            self.cap.release()