"""
roi_tracker.py
=========================
Theo dõi và làm mượt ROI vùng ngực.
"""

import cv2
import numpy as np

from config import ROI_PADDING
ROI_WIDTH = 220
ROI_HEIGHT = 160

class ROITracker:

    def __init__(self):

        self.alpha = 0.7

        self.last_box = None

    def _smooth(self, box):

        if self.last_box is None:

            self.last_box = np.array(box, dtype=np.float32)

            return box

        current = np.array(box, dtype=np.float32)

        self.last_box = (

            self.alpha * self.last_box +

            (1 - self.alpha) * current

        )

        return self.last_box.astype(int)

    def get_roi(self, frame, chest_box):

        if chest_box is None:

            return None, None

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = chest_box

        x1 -= ROI_PADDING
        y1 -= ROI_PADDING

        x2 += ROI_PADDING
        y2 += ROI_PADDING

        x1, y1, x2, y2 = self._smooth(
            (x1, y1, x2, y2)
        )

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        if x2 <= x1:
            return None, None

        if y2 <= y1:
            return None, None

        roi = frame[
            y1:y2,
            x1:x2
        ]

        if roi.size == 0:
            return None, None
        roi = cv2.resize(

            roi,

            (

                ROI_WIDTH,

                ROI_HEIGHT

            )

        )

        return roi, (x1, y1, x2, y2)

    def draw(self, frame, roi_box):

        if roi_box is None:
            return frame

        x1, y1, x2, y2 = roi_box

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            (0, 255, 255),

            2

        )

        cv2.putText(

            frame,

            "ROI",

            (x1, y1 - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 255),

            2

        )

        return frame

    def reset(self):

        self.last_box = None