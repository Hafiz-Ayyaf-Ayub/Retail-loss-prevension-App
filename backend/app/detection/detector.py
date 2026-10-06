import threading
from ultralytics import YOLO

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.35        # insaan ke liye (pehle jaisa)
# Items (bottle/cup/phone) ke liye alag, kam threshold: haath mein pakdi
# cheez ungliyon se aadhi chhup jati hai, is liye YOLO ki confidence kam
# aati hai (aksar 0.3-0.5) aur 0.5 par woh box hi gayab ho jata tha.
ITEM_CONFIDENCE_THRESHOLD = 0.04
INFERENCE_SIZE = 1080  # 480 se kam kiya - ONNX export bhi isi size ka hai, CPU pe kaafi tez

# YOLO/COCO dataset ke standard class IDs — inhe hum "item" ke tor par track karenge
ITEM_CLASS_IDS = {
    39: "Bottle",
    41: "Cup",
    67: "Mobile Phone",
    73 : "book",
    
}

TRACKED_CLASSES = [PERSON_CLASS_ID] + list(ITEM_CLASS_IDS.keys())

# Har camera ka apna alag YOLO model instance — taake unke trackers
# (jo track_id assign karte hain) ek doosre ke frames se confuse na hon
_models = {}
_models_lock = threading.Lock()


def _get_model(cam_id):
    with _models_lock:
        if cam_id not in _models:
            _models[cam_id] = YOLO("yolo26n.onnx")  # ONNX version - CPU pe kaafi tez, .pt se
    return _models[cam_id]


def track_all(frame, cam_id=1):
    model = _get_model(cam_id)

    results = model.track(
        frame,
        persist=True,
        classes=TRACKED_CLASSES,
        conf=min(CONFIDENCE_THRESHOLD, ITEM_CONFIDENCE_THRESHOLD),  # neeche class ke hisab se filter hota hai
        tracker="bytetrack.yaml",  # botsort se halka/tez - CPU ke liye behtar
        # NOTE: single number (jaise 320) dene se Ultralytics frame ke
        # aspect-ratio (640x480) ke hisab se khud size "auto-calculate"
        # karta hai - jo humare case mein galti se 640x640 ban raha tha
        # (4 guna zyada compute, isi liye slow ho gaya tha). Exact square
        # tuple dene se yeh hamesha bilkul 320x320 par hi chalega.
        imgsz=(INFERENCE_SIZE, INFERENCE_SIZE),
        verbose=False
    )[0]

    persons = []
    items = []

    if results.boxes.id is not None:
        for box in results.boxes:
            track_id = int(box.id[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            data = {
                "id": track_id,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "confidence": confidence
            }

            if class_id == PERSON_CLASS_ID:
                if confidence < CONFIDENCE_THRESHOLD:
                    continue  # insaan ke liye wahi purani 0.5 wali sakht shart
                persons.append(data)
            else:
                if confidence < ITEM_CONFIDENCE_THRESHOLD:
                    continue
                data["class_name"] = ITEM_CLASS_IDS.get(class_id, "Item")
                items.append(data)

    return persons, items