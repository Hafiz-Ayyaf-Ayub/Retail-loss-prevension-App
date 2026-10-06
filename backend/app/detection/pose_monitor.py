import os
import threading
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from app import profiler

# Naye MediaPipe mein purana "mp.solutions.pose" hata diya gaya hai.
# Ab "Tasks API" (PoseLandmarker) use hoti hai, jisay ek model file (.task)
# chahiye. Yeh file pehli dafa khud download ho jayegi (internet chahiye),
# aur is file ke bilkul saath (detection/ folder mein) save hogi.
MODEL_FILENAME = "pose_landmarker_lite.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

_landmarkers = {}
_lock = threading.Lock()

# MediaPipe ke 33 points mein se humein sirf yeh chahiye (index numbers)
POINTS = {
    "left_wrist": 15,
    "right_wrist": 16,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}

MIN_VISIBILITY = 0.5   # is se kam "nazar aane ka yaqeen" ho to point ignore
CROP_PADDING = 0.10    # box ke charon taraf 10% extra jagah

# MediaPipe apne andar image ko chhota (lagbhag 256px) kar leta hai, is liye
# bari crop dene ka koi faida nahi - ulta convert/copy mein waqt jata hai.
# Isliye crop ki lambi side ko is se zyada nahi hone dete.
MAX_CROP_SIDE = 144

# HAATH KA CENTRE: kalai (wrist) haath ki sirf "jad" hai, jabke bottle
# hatheli/ungliyon mein hoti hai. Isliye kalai + pinky + index + thumb ke
# points ka average lete hain - yeh asal haath ke kahin zyada qareeb hai.
# (MediaPipe Pose mein yeh points pehle se maujood hain.)
HAND_LANDMARKS = {
    "left_hand": (15, 17, 19, 21),    # wrist, pinky, index, thumb
    "right_hand": (16, 18, 20, 22),
}
HAND_MIN_VISIBILITY = 0.3   # ungliyan aksar kam saaf hoti hain, isliye threshold kam


def _ensure_model():
    """Model file na ho to ek dafa download kar leta hai."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"[pose_monitor] Model download ho raha hai: {MODEL_FILENAME} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[pose_monitor] Model download ho gaya.")
    except Exception as e:
        # Adhoori file na reh jaye, warna agli baar corrupt file load hogi
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise RuntimeError(
            f"Pose model download nahi hua ({e}). Isay khud download karke "
            f"'{MODEL_PATH}' par rakh dein. Link: {MODEL_URL}"
        )


def _get_landmarker(cam_id):
    # Har camera ka apna alag PoseLandmarker (detector.py ki tarah)
    with _lock:
        if cam_id not in _landmarkers:
            _ensure_model()
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.IMAGE,  # har call alag insaan ka crop hai
                num_poses=1,
                min_pose_detection_confidence=0.5,
            )
            _landmarkers[cam_id] = vision.PoseLandmarker.create_from_options(options)
        return _landmarkers[cam_id]


def detect_pose(frame, person, cam_id=1):
    """
    Person ke box ko crop karke pose nikalta hai. Return: dict jisme
    left_wrist, right_wrist, left_shoulder, ... ke (x, y) pixel points
    hain (poori frame ke coordinates mein), ya None agar pose na mile.
    Jo point saaf nazar nahi aaya uski value None hoti hai.
    """
    frame_h, frame_w = frame.shape[:2]
    pad_x = int((person["x2"] - person["x1"]) * CROP_PADDING)
    pad_y = int((person["y2"] - person["y1"]) * CROP_PADDING)

    cx1 = max(0, person["x1"] - pad_x)
    cy1 = max(0, person["y1"] - pad_y)
    cx2 = min(frame_w, person["x2"] + pad_x)
    cy2 = min(frame_h, person["y2"] + pad_y)

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    # Bari crop ko chhota karo (points normalized hain, isliye neeche
    # asli crop_w/crop_h se wapis frame ke pixel mein badalna sahi rahega)
    crop_h_px, crop_w_px = crop.shape[:2]
    longest_side = max(crop_h_px, crop_w_px)
    if longest_side > MAX_CROP_SIDE:
        scale = MAX_CROP_SIDE / longest_side
        crop = cv2.resize(
            crop,
            (max(1, int(crop_w_px * scale)), max(1, int(crop_h_px * scale))),
            interpolation=cv2.INTER_AREA,
        )

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with profiler.stage(cam_id, "mp_detect"):
        result = _get_landmarker(cam_id).detect(mp_image)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]  # pehla (aur ab tak sirf ek) insaan
    crop_w, crop_h = cx2 - cx1, cy2 - cy1

    points = {}
    for name, index in POINTS.items():
        lm = landmarks[index]
        visibility = lm.visibility if lm.visibility is not None else 1.0
        if visibility < MIN_VISIBILITY:
            points[name] = None
        else:
            # crop ke andar ka (0-1) point -> poori frame ka pixel
            points[name] = (int(cx1 + lm.x * crop_w), int(cy1 + lm.y * crop_h))

    # Haath ka centre: jo points nazar aa rahe hain unka average
    for hand_name, indexes in HAND_LANDMARKS.items():
        xs, ys = [], []
        for index in indexes:
            lm = landmarks[index]
            visibility = lm.visibility if lm.visibility is not None else 1.0
            if visibility >= HAND_MIN_VISIBILITY:
                xs.append(cx1 + lm.x * crop_w)
                ys.append(cy1 + lm.y * crop_h)
        points[hand_name] = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys))) if xs else None

    return points