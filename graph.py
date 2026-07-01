"""
graph.py
-------------------------
Realtime Respiration Signal Graph
"""

import cv2
import numpy as np


class GraphDrawer:

    def __init__(self,
                 width=600,
                 height=180):

        self.width = width
        self.height = height

    # ===================================

    def draw(self,
             frame,
             signal,
             peaks=None,
             x=20,
             y=520):

        if signal is None:
            return frame

        if len(signal) < 2:
            return frame

        graph = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )

        # Background
        graph[:] = (25, 25, 25)

        # Border
        cv2.rectangle(
            graph,
            (0, 0),
            (self.width - 1, self.height - 1),
            (180, 180, 180),
            1
        )

        # Center line
        cv2.line(
            graph,
            (0, self.height // 2),
            (self.width, self.height // 2),
            (80, 80, 80),
            1
        )

        sig = np.array(signal, dtype=np.float32)

        sig -= sig.min()

        if sig.max() > 0:
            sig /= sig.max()

        sig = self.height - 20 - sig * (self.height - 40)

        xs = np.linspace(
            10,
            self.width - 10,
            len(sig)
        ).astype(int)

        # Signal
        for i in range(len(sig) - 1):

            cv2.line(
                graph,
                (xs[i], int(sig[i])),
                (xs[i + 1], int(sig[i + 1])),
                (0, 255, 0),
                2
            )

        # Peaks
        if peaks is not None:

            for p in peaks:

                if p >= len(sig):
                    continue

                cv2.circle(
                    graph,
                    (xs[p], int(sig[p])),
                    5,
                    (0, 0, 255),
                    -1
                )

        cv2.putText(
            graph,
            "Respiration Signal (60 s)",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        h, w = graph.shape[:2]

        if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
            frame[y:y + h, x:x + w] = graph

        return frame