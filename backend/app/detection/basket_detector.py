"""
basket_detector.py

Kaam: frame mein BASKET (tokri) dhoondna.

MASLA: aapka YOLO model COCO par trained hai, aur COCO ki 80 classes mein
"basket" hai hi nahi. Is liye detector.py basket dekh hi nahi sakta.

HAL (yahan): YOLO-World - ek "open-vocabulary" model. Isay training nahi
chahiye, hum bas text mein bata dete hain ke kya dhoondna hai
(model.set_classes(["shopping basket"])) aur yeh dhoondh leta hai.
Nuqsan: apne khud ke train kiye hue model se kam accurate hota hai, aur
COCO wale model se bhaari bhi. Is liye 3 tarkeebein lagayi hain taake FPS
na girey:

  1. BACKGROUND THREAD: detection alag thread mein chalti hai, camera ka
     main loop kabhi is ka intezaar nahi karta.
  2. THROTTLE: tokri tez nahi hilti, is liye har frame par nahi, sirf har
     DETECT_INTERVAL_SECONDS par chalate hain (aur result yaad rakhte hain).
  3. GATING: frame mein koi insaan hi na ho to chalate hi nahi.

Agar model load na ho sake (internet nahi / ultralytics purana), to system
band nahi hota - baskets khali list milti hai aur concealment pehle jaisa
chalta hai.

Camera stream mein istemal (persons = detector.track_all() wali list):

    from app.detection import basket_detector
    basket_detector.submit_frame(frame, cam_id, persons)   # foran wapis aa jata hai
    baskets = basket_detector.get_baskets(cam_id)          # pichla taaza result

Akele test karne ke liye (camera_stream ko chhue bagair):

    python -m app.detection.basket_detector tasveer.jpg      # image par
    python -m app.detection.basket_detector 0                # webcam par
"""

import hashlib
import os
import threading
import time

import cv2

MODEL_NAME = "yolov8s-worldv2.pt"    # pehli dafa khud download hoga

# Text mein bataya gaya "kya dhoondna hai". Yeh badlo to accuracy badal jati hai
# (jaise "red plastic basket", "wire shopping basket"). Sab ko "Basket" hi maana jata hai.
BASKET_PROMPTS = ["shopping basket", "basket"]

CONFIDENCE = 0.20        # zero-shot ka confidence aksar kam aata hai, is liye 0.5 nahi
IMGSZ = 320              # detector.py jitna hi, taake CPU par halka rahe
DETECT_INTERVAL_SECONDS = 7.0   # har camera par itni dair mein ek dafa. Kam = zyada sahi, zyada CPU
BASKET_TTL_SECONDS = 10.0        # aakhri detection ke baad itni dair basket "yaad" rahe
MIN_BOX_AREA_RATIO = 0.01       # frame ke 1% se chhota box = shor (noise), ignore
MAX_BOX_AREA_RATIO = 0.60       # frame ke 60% se bara box = galat, ignore

# Prompts se file ka naam banta hai, taake prompts badalne par purani saved
# file istemal na ho jaye.
_prompt_hash = hashlib.md5(",".join(BASKET_PROMPTS).encode()).hexdigest()[:8]
SAVED_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"basket_world_{_prompt_hash}.pt"
)

_model = None
_model_failed = False
_load_lock = threading.Lock()      # model sirf ek dafa load ho
_predict_lock = threading.Lock()   # ultralytics predict thread-safe nahi, ek waqt mein ek

_workers = {}
_workers_lock = threading.Lock()


def _load_model():
    from ultralytics import YOLOWorld

    # Pehle se "basket wali" saved file ho to seedha wahi (CLIP text model ki zaroorat nahi)
    if os.path.exists(SAVED_MODEL_PATH):
        try:
            return YOLOWorld(SAVED_MODEL_PATH)
        except Exception as e:
            print(f"[basket_detector] Saved model load nahi hua ({e}), dobara bana rahe hain.")

    print(f"[basket_detector] Model tayyar ho raha hai: {MODEL_NAME} (pehli dafa download ho sakta hai) ...")
    model = YOLOWorld(MODEL_NAME)
    model.set_classes(BASKET_PROMPTS)   # NOTE: yeh predict() se PEHLE ek hi dafa hona chahiye
    try:
        model.save(SAVED_MODEL_PATH)    # agli baar tez load ho
    except Exception as e:
        print(f"[basket_detector] Model save nahi hua (koi masla nahi): {e}")
    return model


def _get_model():
    global _model, _model_failed
    if _model is not None:
        return _model
    if _model_failed:
        return None
    with _load_lock:
        if _model is None and not _model_failed:
            try:
                _model = _load_model()
                print("[basket_detector] Model tayyar.")
            except Exception as e:
                _model_failed = True
                print(f"[basket_detector] WARNING: basket model load nahi hua ({e}). "
                      f"Basket detection band rahegi, baaki system normal chalega.")
    return _model


def _detect(frame):
    """Ek frame mein baskets dhoondta hai (SYNCHRONOUS - main loop se seedha mat bulao)."""
    model = _get_model()
    if model is None:
        return []

    frame_h, frame_w = frame.shape[:2]
    with _predict_lock:
        result = model.predict(
            frame,
            imgsz=IMGSZ,
            conf=CONFIDENCE,
            agnostic_nms=True,   # "basket" aur "shopping basket" ek hi cheez ke 2 box na banayen
            verbose=False,
        )[0]

    baskets = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(frame_w * frame_h)
        if area_ratio < MIN_BOX_AREA_RATIO or area_ratio > MAX_BOX_AREA_RATIO:
            continue
        baskets.append({
            "id": None,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": float(box.conf[0]),
            "class_name": "Basket",
        })
    return baskets


class _Worker(threading.Thread):
    """Ek camera ka background detector."""

    def __init__(self, cam_id):
        super().__init__(daemon=True, name=f"basket-detector-{cam_id}")
        self.baskets = []
        self.updated_at = 0.0
        self.last_submit = 0.0
        self.busy = False
        self._frame = None
        self._wake = threading.Event()

    def submit(self, frame):
        now = time.time()
        if self.busy or (now - self.last_submit) < DETECT_INTERVAL_SECONDS:
            return
        self.last_submit = now
        self.busy = True
        # copy zaroori hai: main loop isi frame par boxes draw karta hai.
        # (Yeh copy har DETECT_INTERVAL_SECONDS par ek dafa hoti hai, har frame nahi.)
        self._frame = frame.copy()
        self._wake.set()

    def run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            frame, self._frame = self._frame, None
            try:
                if frame is not None:
                    found = _detect(frame)
                    # Khali result par purana yaad rakho (TTL khud purana hata dega):
                    # ek miss se tokri "gayab" nahi honi chahiye.
                    if found:
                        self.baskets = found
                        self.updated_at = time.time()
            except Exception as e:
                print(f"[basket_detector] detection mein masla: {e}")
            finally:
                self.busy = False


def submit_frame(frame, cam_id=1, persons=None):
    """
    Frame detection ke liye bhejta hai. NON-BLOCKING: foran wapis aa jata hai
    (frame skip bhi kar deta hai agar abhi throttle/busy ho - yeh theek hai).
    persons: agar diya gaya aur khali hai (koi insaan nahi) to detection chalti hi nahi.
    """
    if _model_failed:
        return
    if persons is not None and len(persons) == 0:
        return

    with _workers_lock:
        worker = _workers.get(cam_id)
        if worker is None:
            worker = _Worker(cam_id)
            worker.start()
            _workers[cam_id] = worker
    worker.submit(frame)


def get_baskets(cam_id=1):
    """Is camera ke taaza baskets (har ek dict: x1,y1,x2,y2,confidence). Purane ho gaye to khali list."""
    worker = _workers.get(cam_id)
    if worker is None or (time.time() - worker.updated_at) > BASKET_TTL_SECONDS:
        return []
    return list(worker.baskets)


if __name__ == "__main__":
    # Akela test: python -m app.detection.basket_detector <image.jpg | camera_number>
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        frame = None
        for _ in range(10):          # pehle kuch frames camera "garam" hone dete hain
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
    else:
        frame = cv2.imread(source)

    if frame is None:
        raise SystemExit("Frame nahi mila (image path ya camera number check karo).")

    t0 = time.perf_counter()
    _detect(frame)   # pehla call: model load + warmup (time ginte nahi)
    print(f"Model load + pehli detection: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    found = _detect(frame)
    print(f"Ek detection ka asal time: {(time.perf_counter() - t0) * 1000:.0f} ms")
    print(f"Baskets mile: {len(found)}")

    for b in found:
        cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 165, 255), 3)
        cv2.putText(frame, f"basket {b['confidence']:.2f}", (b["x1"], max(20, b["y1"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        print(f"  box=({b['x1']},{b['y1']})-({b['x2']},{b['y2']}) confidence={b['confidence']:.2f}")

    cv2.imwrite("basket_test_result.jpg", frame)
    print("Result tasveer: basket_test_result.jpg (isay kholo, orange box dekho)")