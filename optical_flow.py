"""
optical_flow.py
---------------

Theo dõi chuyển động vùng ngực bằng
Lucas Kanade Optical Flow

Output:
    displacement
"""

import cv2
import numpy as np

from config import *


class OpticalFlowTracker:

    def __init__(self):

        self.prev_gray = None

        self.prev_points = None

        self.displacements = []

    def reset(self):

        self.prev_gray = None

        self.prev_points = None

        self.displacements.clear()

    def initialize(self, roi):

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )
        # Nếu ROI thay đổi kích thước thì reset Optical Flow
        if self.prev_gray is not None:

            if gray.shape != self.prev_gray.shape:
                self.initialize(roi)

                return None

        points = cv2.goodFeaturesToTrack(
            gray,
            mask=None,
            maxCorners=MAX_CORNERS,
            qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_DISTANCE,
            blockSize=BLOCK_SIZE
        )

        self.prev_gray = gray
        self.prev_points = points

    def update(self, roi):

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        if self.prev_gray is None:

            self.initialize(roi)

            return None

        if self.prev_points is None:

            self.initialize(roi)

            return None

        next_points, status, err = cv2.calcOpticalFlowPyrLK(

            self.prev_gray,
            gray,
            self.prev_points,
            None,
            **LK_PARAMS
        )

        if next_points is None:

            self.initialize(roi)

            return None

        good_old = self.prev_points[
            status == 1
        ]

        good_new = next_points[
            status == 1
        ]

        if len(good_new) < 8:

            self.initialize(roi)

            return None

        motion = good_new - good_old

        dx = np.mean(
            motion[:,0]
        )

        dy = np.mean(
            motion[:,1]
        )

        displacement = float(dy)

        self.displacements.append(
            displacement
        )

        self.prev_gray = gray.copy()

        self.prev_points = good_new.reshape(
            -1,
            1,
            2
        )

        return displacement

    def draw_points(self, roi):

        if self.prev_points is None:

            return roi

        img = roi.copy()

        for p in self.prev_points:

            x,y = p.ravel()

            cv2.circle(

                img,

                (int(x),int(y)),

                3,

                (0,255,0),

                -1

            )

        return img

    def get_signal(self):

        return np.array(

            self.displacements,

            dtype=np.float32

        )

    def clear_signal(self):

        self.displacements.clear()

    def point_count(self):

        if self.prev_points is None:

            return 0

        return len(self.prev_points)