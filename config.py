"""
config.py
=========================
Cấu hình hệ thống
"""

import os

# =====================================================
# CAMERA
# =====================================================

# 0: Webcam
# 1/2/3: Camo Studio nếu cần
CAMERA_INDEX = 0

ROI_WIDTH = 220
ROI_HEIGHT = 160

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

TARGET_FPS = 30

# =====================================================
# YOLO
# =====================================================

MODEL_PATH = "models/yolo11n-pose.pt"

YOLO_CONF = 0.45

YOLO_IOU = 0.45

# =====================================================
# ROI
# =====================================================

# Chỉ lấy vùng giữa ngực
CHEST_WIDTH_SCALE = 0.55
CHEST_HEIGHT_SCALE = 0.35

ROI_PADDING = 15

# =====================================================
# Optical Flow
# =====================================================

MAX_CORNERS = 300

QUALITY_LEVEL = 0.01

MIN_DISTANCE = 3

BLOCK_SIZE = 7

LK_PARAMS = dict(

    winSize=(21, 21),

    maxLevel=3,

    criteria=(
        3,
        20,
        0.03
    )

)

# =====================================================
# Signal Processing
# =====================================================

WINDOW_SECONDS = 30

LOWCUT = 0.10

HIGHCUT = 0.60

FILTER_ORDER = 4

# =====================================================
# Breathing
# =====================================================

MIN_BPM = 8

MAX_BPM = 40

UPDATE_INTERVAL = 5

MIN_SIGNAL_SECONDS = 15

# =====================================================
# MQTT
# =====================================================

MQTT_ENABLE = False

MQTT_BROKER = "127.0.0.1"

MQTT_PORT = 1883

MQTT_TOPIC = "respiration/bpm"

# =====================================================
# Dashboard
# =====================================================

ENABLE_DASHBOARD = True

# =====================================================
# Logging
# =====================================================

OUTPUT_DIR = "output"

LOG_DIR = "logs"

CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "breathing_signal.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(LOG_DIR, exist_ok=True)

# =====================================================
# Display
# =====================================================

SHOW_SKELETON = True

SHOW_ROI = True

SHOW_POINTS = True

SHOW_SIGNAL = True

SHOW_FPS = True

SHOW_BPM = True