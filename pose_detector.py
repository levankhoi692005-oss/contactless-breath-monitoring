"""
pose_detector.py
=========================
YOLO11 Pose Detector
"""

import cv2
import numpy as np
from ultralytics import YOLO

from config import *

# COCO Keypoints
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_ELBOW = 7
RIGHT_ELBOW = 8

LEFT_WRIST = 9
RIGHT_WRIST = 10

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_KNEE = 13
RIGHT_KNEE = 14

LEFT_ANKLE = 15
RIGHT_ANKLE = 16


SKELETON = [

    (5,6),

    (5,7),

    (7,9),

    (6,8),

    (8,10),

    (5,11),

    (6,12),

    (11,12),

    (11,13),

    (13,15),

    (12,14),

    (14,16)

]


class PoseDetector:

    def __init__(self):

        print("Loading YOLO Pose...")

        self.model = YOLO(MODEL_PATH)

        print("Done.")

    def detect(self, frame):

        results = self.model.predict(

            frame,

            conf=YOLO_CONF,

            iou=YOLO_IOU,

            verbose=False

        )

        if len(results) == 0:
            return None

        r = results[0]

        if r.boxes is None:
            return None

        if len(r.boxes) == 0:
            return None

        boxes = r.boxes.xyxy.cpu().numpy()

        areas = []

        for b in boxes:

            x1,y1,x2,y2 = b

            areas.append(

                (x2-x1)*(y2-y1)

            )

        idx = np.argmax(areas)

        return {

            "box": boxes[idx],

            "score": float(

                r.boxes.conf[idx]

            ),

            "keypoints":

                r.keypoints.xy.cpu().numpy()[idx]

        }

    def get_chest_box(self,detection):

        if detection is None:
            return None

        pts = detection["keypoints"]

        ls = pts[LEFT_SHOULDER]

        rs = pts[RIGHT_SHOULDER]

        lh = pts[LEFT_HIP]

        rh = pts[RIGHT_HIP]

        if (

            np.any(ls==0)

            or

            np.any(rs==0)

            or

            np.any(lh==0)

            or

            np.any(rh==0)

        ):

            return None

        shoulder_y = (ls[1]+rs[1])/2

        hip_y = (lh[1]+rh[1])/2

        chest_h = hip_y-shoulder_y

        center_x = (

            ls[0]+

            rs[0]

        )/2

        shoulder_w = abs(

            rs[0]-ls[0]

        )

        width = shoulder_w*CHEST_WIDTH_SCALE

        height = chest_h*CHEST_HEIGHT_SCALE

        x1 = int(

            center_x-width

        )

        x2 = int(

            center_x+width

        )

        y1 = int(

            shoulder_y+10

        )

        y2 = int(

            y1+height

        )

        return (

            x1,

            y1,

            x2,

            y2

        )

    def draw(self,frame,detection):

        if detection is None:

            return frame

        box = detection["box"]

        pts = detection["keypoints"]

        x1,y1,x2,y2 = map(int,box)

        cv2.rectangle(

            frame,

            (x1,y1),

            (x2,y2),

            (0,255,0),

            2

        )

        for p in pts:

            x,y = p

            cv2.circle(

                frame,

                (int(x),int(y)),

                4,

                (0,0,255),

                -1

            )

        for a,b in SKELETON:

            p1 = pts[a]

            p2 = pts[b]

            cv2.line(

                frame,

                tuple(p1.astype(int)),

                tuple(p2.astype(int)),

                (255,0,0),

                2

            )

        chest = self.get_chest_box(detection)

        if chest is not None:

            x1,y1,x2,y2 = chest

            cv2.rectangle(

                frame,

                (x1,y1),

                (x2,y2),

                (0,255,255),

                2

            )

            cv2.putText(

                frame,

                "Chest ROI",

                (x1,y1-8),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0,255,255),

                2

            )

        return frame