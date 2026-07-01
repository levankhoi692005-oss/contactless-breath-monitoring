"""
optical_flow.py
--------------------------------
Lucas-Kanade Optical Flow
"""

import cv2
import numpy as np

from config import *


class OpticalFlowTracker:

    def __init__(self):

        self.prev_gray = None
        self.prev_points = None

        self.points = None

    # =======================================

    def reset(self):

        self.prev_gray = None
        self.prev_points = None
        self.points = None

    # =======================================

    def update(self, roi):

        if roi is None:
            return None

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        # ---------- First Frame ----------
        if self.prev_gray is None:

            self.prev_gray = gray

            self.prev_points = cv2.goodFeaturesToTrack(

                gray,

                maxCorners=MAX_CORNERS,

                qualityLevel=QUALITY_LEVEL,

                minDistance=MIN_DISTANCE,

                blockSize=BLOCK_SIZE

            )

            return None

        # ---------- Nếu mất hết điểm ----------
        if self.prev_points is None or len(self.prev_points) < 20:

            self.prev_points = cv2.goodFeaturesToTrack(

                gray,

                maxCorners=MAX_CORNERS,

                qualityLevel=QUALITY_LEVEL,

                minDistance=MIN_DISTANCE,

                blockSize=BLOCK_SIZE

            )

            self.prev_gray = gray

            return None

        # ---------- Optical Flow ----------

        next_points, status, err = cv2.calcOpticalFlowPyrLK(

            self.prev_gray,

            gray,

            self.prev_points,

            None,

            **LK_PARAMS

        )

        if next_points is None:

            self.reset()

            return None

        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]

        if len(good_new) < 5:

            self.reset()

            return None

        # ---------- Độ dịch chuyển ----------

        dy = np.mean(

            good_new[:, 1] -

            good_old[:, 1]

        )

        # lưu điểm để vẽ

        self.points = good_new

        self.prev_gray = gray.copy()

        self.prev_points = good_new.reshape(-1, 1, 2)

        return float(dy)

    # =======================================

    def draw_points(self, roi):

        if roi is None:
            return roi

        if self.points is None:
            return roi

        img = roi.copy()

        for p in self.points:

            x, y = p

            cv2.circle(

                img,

                (int(x), int(y)),

                2,

                (0, 255, 0),

                -1

            )

        return img