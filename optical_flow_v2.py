"""
optical_flow.py
=========================================
Lucas-Kanade Optical Flow cho đo nhịp thở

Output:
    displacement (pixel/frame)
"""

import cv2
import numpy as np

from config import *


class OpticalFlowTracker:

    def __init__(self):

        self.prev_gray = None
        self.prev_points = None

        self.signal = []

        self.min_points = 20

    # ==================================================

    def reset(self):

        self.prev_gray = None
        self.prev_points = None
        self.signal.clear()

    # ==================================================

    def initialize(self, roi):

        if roi is None:
            return False

        if roi.size == 0:
            return False

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        points = cv2.goodFeaturesToTrack(

            gray,

            maxCorners=MAX_CORNERS,

            qualityLevel=QUALITY_LEVEL,

            minDistance=MIN_DISTANCE,

            blockSize=BLOCK_SIZE

        )

        if points is None:

            return False

        self.prev_gray = gray

        self.prev_points = points

        return True

    # ==================================================

    def update(self, roi):

        if roi is None:
            return None

        if roi.size == 0:
            return None

        gray = cv2.cvtColor(

            roi,

            cv2.COLOR_BGR2GRAY

        )

        if self.prev_gray is None:

            ok = self.initialize(roi)

            if not ok:
                return None

            return None

        if self.prev_points is None:

            ok = self.initialize(roi)

            if not ok:
                return None

            return None

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(

            self.prev_gray,

            gray,

            self.prev_points,

            None,

            **LK_PARAMS

        )

        if next_pts is None:

            self.initialize(roi)

            return None

        status = status.reshape(-1)

        old = self.prev_points[status == 1]

        new = next_pts[status == 1]

        if len(new) < self.min_points:

            self.initialize(roi)

            return None

        motion = new - old

        dy = motion[:, 1]

        median = np.median(dy)

        mad = np.median(

            np.abs(

                dy - median

            )

        )

        if mad > 0:

            mask = (

                np.abs(

                    dy - median

                )

                <

                2.5 * mad

            )

            dy = dy[mask]

        if len(dy) == 0:

            displacement = median

        else:

            displacement = np.median(dy)

        self.signal.append(

            displacement

        )

        self.prev_gray = gray

        self.prev_points = new.reshape(

            -1,

            1,

            2

        )

        return float(displacement)

    # ==================================================

    def draw_points(self, roi):

        if roi is None:
            return roi

        img = roi.copy()

        if self.prev_points is None:

            return img

        for p in self.prev_points:

            x, y = p.ravel()

            cv2.circle(

                img,

                (int(x), int(y)),

                2,

                (0,255,0),

                -1

            )

        cv2.putText(

            img,

            f"Points: {len(self.prev_points)}",

            (10,20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255,255,0),

            2

        )

        return img

    # ==================================================

    def point_count(self):

        if self.prev_points is None:

            return 0

        return len(self.prev_points)

    # ==================================================

    def get_signal(self):

        return np.asarray(

            self.signal,

            dtype=np.float32

        )

    # ==================================================

    def clear_signal(self):

        self.signal.clear()