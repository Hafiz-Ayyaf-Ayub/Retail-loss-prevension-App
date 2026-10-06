import os

# ---------------------------------------------------------------------------
# CPU THREAD-OVERSUBSCRIPTION FIX
# MASLA: is app mein 3 bhaari models EK SATH (alag threads mein) chalte hain -
# YOLO (yolo stage), MediaPipe Pose (mp_detect stage), aur basket ka YOLO-World
# model (basket_detector.py ka apna thread). Har library APNE TAUR PAR "sab
# CPU cores mere liye" soch kar apna internal thread-pool banati hai. Jab
# teeno ek sath chalte hain, to woh EK DOOSRE se cores ke liye takrate hain
# (jitne CPU cores hain us se kai guna zyada threads), aur har ek ka apna kaam
# ULTA slow ho jata hai (profiler mein yolo 199ms se 657ms tak chala gaya tha).
#
# HAL: har library ko batao ke sirf ITNE threads istemal karo (poore cores
# nahi), taake teeno milkar bhi CPU ko "oversubscribe" na karein.
#
# ZAROORI: yeh lines cv2/torch/onnxruntime import hone se PEHLE honi chahiye -
# yeh libraries apna thread-pool IMPORT hote hi bana leti hain, baad mein
# environment variable badalne ka koi asar nahi hota.
#
# Number (2) apne CPU ke physical cores se kam rakhna. Jaise 4-core CPU par 2,
# 8-core par 3-4. Bohat kam rakhoge to ek model khud hi slow ho jayega.
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import cv2
cv2.setNumThreads(8)   # OpenCV (resize/cvtColor wagera) ka apna thread-pool bhi
import time
import threading
from collections import deque
from app.detection.detector import track_all
from app.detection.zone_monitor import update_zone_tracking
from app.detection.concealment_monitor import update_concealment, get_concealment_flagged_ids
from app.detection import person_reid
from app.detection import item_tracker
from app.detection import incident_logger
from app.detection import pose_monitor
from app.detection import pickup_monitor
from app.detection import detection_worker
from app import profiler
from app.config import Settings

LINE_Y_RATIO = 0.5
CROSSING_COOLDOWN = 1.5

# NOTE: ab istemal nahi hoti - detection ab detection_worker.py mein background
# thread mein chalti hai, jo khud apni raftaar se chalta hai (busy-check khud
# throttle karta hai). Constant sirf compatibility ke liye rakhi hai, delete nahi ki.
DETECT_EVERY_N_FRAMES = 2

# Basket ka box screen par dikhana hai ya nahi (testing ke liye True rakho,
# taake pata chale basket sahi jagah detect ho raha hai; live/production mein
# False kar dena - orange box customer-facing screen par ajeeb lagega).
SHOW_BASKETS = True

# Incident video clip ke liye: item ghaib hone se PEHLE aur BAAD ka kitna
# footage shamil karna hai. Total clip lagbhag PRE + POST second ki hogi,
# jiske beech mein wo lamha aayega jab item asal mein ghaib hua tha.
PRE_INCIDENT_SECONDS = 5
POST_INCIDENT_SECONDS = 3

# Buffer itna bara rakhna zaroori hai ke jab tak POST wala hissa export
# hone ke liye ready ho (disappearance ke POST_INCIDENT_SECONDS baad),
# tab tak PRE wala purana hissa (disappearance se PRE_INCIDENT_SECONDS
# pehle) bhi buffer mein maujood ho. Isliye PRE + POST + thora extra
# margin (timing jitter ke liye) rakha hai.
VIDEO_BUFFER_SECONDS = PRE_INCIDENT_SECONDS + POST_INCIDENT_SECONDS + 4

# Reference resolution jis par font/line sizes originally design kiye gaye the
# (0.55 font scale, 2px thickness). Har camera ki apni resolution ho sakti hai
# (Device 0 vs Device 1 alag), aur agar text ek fixed pixel size mein draw
# karein to bari-resolution wali camera ka text frontend mein zyada chhota
# dikhta hai (kyunke usay barabar size ke box mein fit karne ke liye zyada
# scale-down hota hai). Isliye frame ki height ke hisab se scale nikalte hain.
REFERENCE_FRAME_HEIGHT = 480


# ---------------------------------------------------------------------------
# LIVE CAMERA KA "LAG" (mobile/IP camera mein video peeche reh jana) - FIX
#
# MASLA: cv2.VideoCapture andar se frames ki ek QUEUE rakhta hai. Laptop
# webcam mein frame tab banta hai jab hum maangte hain, isliye koi lag nahi.
# Lekin mobile camera (IP Webcam / DroidCam / RTSP) network par 30 frame/sec
# bhejta rehta hai. Agar hamara loop (decode + encode + browser ko bhejna)
# zara bhi slow ho, to queue mein purane frames jama hote jate hain, aur hum
# hamesha queue ka SABSE PURANA frame dekhte hain -> movement pehle ho jati
# hai, video baad mein aati hai, aur delay waqt ke sath barhta jata hai.
#
# HAL: har live camera ke liye ALAG thread jo lagatar padhta rahe aur sirf
# SABSE TAAZA frame rakhe (purane phenk de). generate_frames() jab bhi read()
# kare, use hamesha taaza frame mile. Queue ban hi nahi sakti.
#
# Yeh cv2.VideoCapture jaisa hi dikhta hai (isOpened/read/release/set/get),
# isliye generate_frames() mein kuch badalna nahi para.
# NOTE: sirf LIVE camera par lagta hai. Video FILE par nahi (file ko free-run
# karte to woh bohat tez chal jati).
# ---------------------------------------------------------------------------
class _ThreadedCapture:
    def __init__(self, cap, name="camera"):
        self._cap = cap
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._frame = None
        self._seq = 0            # har naye frame par barhta hai
        self._last_returned = 0  # read() ne aakhri baar kaunsa seq diya
        self._alive = True       # False = stream khatam/toot gayi
        self._stop = False
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name=f"capture-{name}")
        self._thread.start()

    def _reader_loop(self):
        failures = 0
        try:
            while not self._stop:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    failures += 1
                    if failures >= 30:   # lagatar ~30 dafa fail = stream sach mein band
                        break
                    time.sleep(0.02)
                    continue
                failures = 0
                with self._new_frame:
                    self._frame = frame          # purana frame yahin phenk diya
                    self._seq += 1
                    self._new_frame.notify_all()
        finally:
            with self._new_frame:
                self._alive = False
                self._new_frame.notify_all()
            try:
                self._cap.release()
            except Exception:
                pass

    def isOpened(self):
        return self._alive and self._cap.isOpened()

    def read(self, timeout=2.0):
        """Hamesha SABSE TAAZA frame deta hai. Naya frame na aaye to us ke aane
        tak intezaar (busy-loop nahi). Wahi frame do dafa nahi milta."""
        with self._new_frame:
            end = time.time() + timeout
            while self._seq == self._last_returned:
                if not self._alive:
                    return False, None
                remaining = end - time.time()
                if remaining <= 0:
                    return False, None
                self._new_frame.wait(remaining)
            self._last_returned = self._seq
            return True, self._frame

    def set(self, prop, value):
        return self._cap.set(prop, value)

    def get(self, prop):
        return self._cap.get(prop)

    def release(self):
        self._stop = True
        self._thread.join(timeout=3.0)   # thread khud cap.release() karta hai
        with self._new_frame:
            self._alive = False


def _open_capture(source):
    """
    VideoCapture kholta hai. Network stream (http/rtsp/rtmp) ho to FFmpeg ko
    "low latency" mode mein kholta hai (apni taraf ki buffering band) - yeh
    option sirf isi open ke liye lagta hai, baad mein pehle jaisa wapis.
    """
    is_network = isinstance(source, str) and source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://"))
    if not is_network:
        return cv2.VideoCapture(source)

    key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    old = os.environ.get(key)
    os.environ[key] = "fflags;nobuffer|flags;low_delay"
    try:
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _new_camera_state():
    """Har naye camera ke liye khali/default state banata hai."""
    return {
        "capture": None,
        "device_index": 0,
        # "camera" = live webcam, "file" = recorded mp4 chal raha hai
        "source_type": "camera",
        "loop_video": True,   # file khatam hote hi shuru se dobara chale
        "is_paused": False,
        "last_frame_bytes": None,
        "current_person_count": 0,
        "unique_visitor_ids": set(),
        "entry_count": 0,
        "exit_count": 0,
        "track_history": {},
        "last_crossing_time": {},
        "active_alerts": [],
        "active_concealment_alerts": [],
        # Har camera ka apna zone — shuru mein Settings se copy hota hai,
        # baad mein frontend se adjust ho sakta hai (set_zone() dekhein)
        "zone_ratios": {
            "x1": Settings.ZONE_X1_RATIO,
            "y1": Settings.ZONE_Y1_RATIO,
            "x2": Settings.ZONE_X2_RATIO,
            "y2": Settings.ZONE_Y2_RATIO,
        },
        "zone_enabled": True,
        # Speed optimization: har frame detect nahi karte, purana result reuse karte hain
        "frame_counter": 0,
        "last_persons": [],
        "last_items": [],
        "last_baskets": [],
        # Speed optimization: har track ID ka global_id ek dafa nikal kar yaad rakhte hain
        "global_id_cache": {},
        # Speed optimization: har detection par pose nahi chalate, pichla pose
        # kuch dair yaad rakhte hain (pickup_monitor.refresh_poses dekhein)
        "pose_cache": {},
        # Rolling video buffer: (timestamp, jpg_bytes) tuples, purane->naye.
        # Har asal frame yahan store hota hai (chahe detection chali ho ya
        # na ho), taake sustained-concealment confirm hote hi pichle
        # VIDEO_BUFFER_SECONDS ka clip nikala ja sake. JPEG bytes store
        # karte hain (raw frame nahi) taake RAM usage kam rahe.
        "frame_buffer": deque(),
        # Confirmed incidents jinke video ka "POST" hissa abhi ban raha hai -
        # yeh yahan wait karte hain jab tak enough future frames record na
        # ho jayen, phir export ho kar yahan se hat jate hain.
        # Format: {"incident_id", "alert", "export_at"}
        "pending_video_exports": [],
        # detection_worker.py (background thread) is list mein NAYE incidents
        # daalta hai, aur neeche _flush_ready_video_exports() (MAIN thread) isay
        # padh/khaali karta hai - do threads ek hi list chhoote hain, is liye
        # lock zaroori hai (warna beech mein daali gayi entry gum ho sakti hai).
        "pending_lock": threading.Lock(),
    }


# Sab cameras yahan store honge — abhi sirf 1 use ho raha hai (purana behaviour)
cameras = {
    1: _new_camera_state()
}


def _get_state(cam_id):
    if cam_id not in cameras:
        cameras[cam_id] = _new_camera_state()
    return cameras[cam_id]


def _zone_box_px(state, frame_width, frame_height):
    """Camera ke apne zone_ratios se actual pixel coordinates nikalta hai."""
    r = state["zone_ratios"]
    x1 = int(frame_width * r["x1"])
    y1 = int(frame_height * r["y1"])
    x2 = int(frame_width * r["x2"])
    y2 = int(frame_height * r["y2"])
    return x1, y1, x2, y2


def set_zone(cam_id, x1, y1, x2, y2):
    """Frontend se naye zone ratios (0.0 - 1.0) save karta hai is camera ke liye."""
    state = _get_state(cam_id)
    state["zone_ratios"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def set_zone_enabled(cam_id, enabled):
    """Zone ko on/off karta hai (remove button ke liye)."""
    state = _get_state(cam_id)
    state["zone_enabled"] = enabled


def start_camera(cam_id=1, device_index=None):
    state = _get_state(cam_id)

    if device_index is not None:
        state["device_index"] = device_index

    if state["capture"] is None or not state["capture"].isOpened():
        raw_capture = _open_capture(state["device_index"])
        state["capture"] = raw_capture

        # Speed optimization: webcam ko explicitly chhoti resolution + MJPG
        # format pe set karo. Agar yeh na kiya jaye to bohat se webcams apni
        # default (kabhi kabhi 720p/1080p) resolution pe frames dete hain,
        # jo decode/resize/encode sab ko slow kar deta hai - chahe YOLO khud
        # chhote imgsz pe chale. MJPG bhi USB webcams par capture ko tez karta hai.
        state["capture"].set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        state["capture"].set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        state["capture"].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        state["capture"].set(cv2.CAP_PROP_FPS, 30)
        state["capture"].set(cv2.CAP_PROP_BUFFERSIZE, 3)   # jahan backend support kare wahan andar ki queue 1 frame ki

        # Live camera: alag thread se lagatar taaza frame padho (lag/queue fix)
        if raw_capture.isOpened():
            state["capture"] = _ThreadedCapture(raw_capture, name=f"cam{cam_id}")

    state["is_paused"] = False
    return state["capture"].isOpened()


def start_camera_from_file(cam_id, file_path, loop=True):
    """
    Live webcam ki jagah ek recorded .mp4 file ko "camera" ki tarah chalata
    hai - baaki poora system (YOLO, pose, concealment, dashboard) isay
    normal camera hi samjhega, kyunke generate_frames() same tarah frames
    padhta hai chahe source webcam ho ya file.

    loop=True (default): video khatam hote hi khud shuru se dobara chalti
    hai - taake ek hi "chori" wali clip baar baar test ki ja sake bina
    dobara button dabaye.
    """
    state = _get_state(cam_id)

    if state["capture"] is not None:
        state["capture"].release()

    state["capture"] = cv2.VideoCapture(file_path)
    state["source_type"] = "file"
    state["loop_video"] = loop
    state["is_paused"] = False

    opened = state["capture"].isOpened()
    if not opened:
        state["capture"] = None
    return opened


def pause_camera(cam_id=1):
    state = _get_state(cam_id)
    state["is_paused"] = True


def resume_camera(cam_id=1):
    state = _get_state(cam_id)
    state["is_paused"] = False


def stop_camera(cam_id=1):
    state = _get_state(cam_id)
    if state["capture"] is not None:
        state["capture"].release()
    state["capture"] = None
    state["source_type"] = "camera"
    state["is_paused"] = False
    state["last_frame_bytes"] = None
    state["current_person_count"] = 0


def draw_tracks(frame, persons, items, baskets=None, zone_box=None, zone_enabled=True):
    # Zone ab video ke andar draw nahi hoti — frontend ka adjustable overlay hi
    # zone dikhata hai. Yahan zone_box sirf detection ke liye use hota hai (generate_frames mein).

    # Frame ki apni resolution ke hisab se font/line ko scale karo (upar
    # REFERENCE_FRAME_HEIGHT ka comment dekhein) - taake Camera 1 aur
    # Camera 2 ka label size, screen par dikhte waqt, barabar lage chahe
    # unki native camera resolution alag ho.
    frame_height = frame.shape[0]
    scale = frame_height / REFERENCE_FRAME_HEIGHT
    font_scale = max(0.35, 0.55 * scale)
    item_font_scale = max(0.32, 0.5 * scale)
    box_thickness = max(1, round(2 * scale))
    text_thickness = max(1, round(2 * scale))

    for track in persons:
        x1, y1, x2, y2 = track["x1"], track["y1"], track["x2"], track["y2"]
        track_id = track["id"]
        is_suspicious = track.get("suspicious", False)
        is_concealment = track.get("concealment_flag", False)
        global_id = track.get("global_id")

        color = (0, 0, 255) if (is_suspicious or is_concealment) else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        if is_concealment:
            label = f"ID {track_id} (G{global_id}) - CONCEALMENT ALERT"
        elif is_suspicious:
            label = f"ID {track_id} (G{global_id}) - WRONG ACTIVITY"
        else:
            dwell = track.get("dwell_time", 0)
            label = f"ID {track_id} (G{global_id})" + (f" ({dwell}s)" if dwell > 0 else "")

        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)

        # Step 1 (MediaPipe): kalai/kandhe/kolhe ke points peele dots mein dikhao
        pose = track.get("pose")
        if pose:
            for point in pose.values():
                if point is not None:
                    cv2.circle(frame, point, max(3, box_thickness * 2), (0, 255, 255), -1)

    if SHOW_BASKETS and baskets:
        for basket in baskets:
            bx1, by1, bx2, by2 = basket["x1"], basket["y1"], basket["x2"], basket["y2"]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), box_thickness)
            cv2.putText(frame, "Basket", (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, item_font_scale, (0, 165, 255), text_thickness)

    for item in items:
        x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]

        # Step 2: agar kisi ke haath mein hai to magenta, warna purana rang
        held_by = item.get("held_by")
        item_color = (255, 0, 255) if held_by is not None else (255, 200, 0)
        item_label = item["class_name"]
        if held_by is not None:
            item_label += f" HELD by G{held_by}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), item_color, box_thickness)
        cv2.putText(frame, item_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, item_font_scale, item_color, text_thickness)

    return frame


def check_line_crossing(state, tracks, line_y):
    now = time.time()

    for track in tracks:
        track_id = track["id"]
        cy = (track["y1"] + track["y2"]) // 2

        if track_id in state["track_history"]:
            prev_cy = state["track_history"][track_id]
            last_time = state["last_crossing_time"].get(track_id, 0)

            if now - last_time > CROSSING_COOLDOWN:
                if prev_cy < line_y <= cy:
                    state["entry_count"] += 1
                    state["last_crossing_time"][track_id] = now
                elif prev_cy > line_y >= cy:
                    state["exit_count"] += 1
                    state["last_crossing_time"][track_id] = now

        state["track_history"][track_id] = cy


def update_alerts(state, tracks):
    state["active_alerts"] = []
    for track in tracks:
        if track.get("suspicious", False):
            state["active_alerts"].append({
                "id": track["id"],
                "global_id": track.get("global_id"),
                "dwell_time": track.get("dwell_time", 0),
                "message": f"Person ID {track['id']} (Global {track.get('global_id')}) — {track.get('dwell_time', 0)}s in monitored zone"
            })


def apply_person_reid(frame, persons, state):
    """Har person ko ek Global ID deta hai, aur agar doosre camera se pehle se flagged hai to yahan bhi suspicious mark karta hai.
    Speed ke liye: agar is track_id ka global_id pehle nikal chuke hain, to dobara signature nahi nikalte."""
    cache = state["global_id_cache"]

    for track in persons:
        track_id = track["id"]

        if track_id in cache:
            global_id = cache[track_id]
        else:
            x1, y1, x2, y2 = track["x1"], track["y1"], track["x2"], track["y2"]
            signature, person_crop = person_reid.extract_signature(frame, x1, y1, x2, y2)
            face_encoding = person_reid.extract_face_encoding(frame, x1, y1, x2, y2)
            global_id = person_reid.match_or_register(signature, person_crop, face_encoding)
            cache[track_id] = global_id

        track["global_id"] = global_id

        # Agar ye insaan pehle kisi camera mein flag ho chuka hai, to yahan bhi turant dikhao
        was_suspicious, was_concealment = person_reid.is_flagged(global_id)
        if was_suspicious:
            track["suspicious"] = True

    return persons


def _handle_new_concealment_incidents(new_alerts, state, cam_id):
    """
    Naye CONFIRMED concealment incidents (sustained threshold cross kar
    chuke) ko yahan turant export NAHI karte - kyunke video mein
    "disappearance ke POST_INCIDENT_SECONDS baad" ka footage bhi chahiye,
    jo abhi record hi nahi hua. Isliye inhe "pending_video_exports" mein
    daal dete hain; asal export _flush_ready_video_exports() karega jab
    itna waqt guzar jaye.
    """
    for alert in new_alerts:
        incident_id = f"cam{cam_id}_g{alert['global_id']}_{int(alert['disappeared_at'] * 1000)}"
        state["pending_video_exports"].append({
            "incident_id": incident_id,
            "alert": alert,
            "export_at": alert["disappeared_at"] + POST_INCIDENT_SECONDS,
        })


def _flush_ready_video_exports(state, cam_id):
    """
    Jin pending incidents ka POST_INCIDENT_SECONDS wala intezaar poora ho
    chuka hai, unke liye ab poora clip (PRE + disappearance + POST) frame_buffer
    se nikal kar .mp4 save karta hai, aur metadata bhi (ek sath, video_path
    ke sath) log karta hai. Har frame par call hota hai (halka operation hai).
    """
    with state["pending_lock"]:
        pending = list(state["pending_video_exports"])

    if not pending:
        return

    now = time.time()
    still_pending = []

    for entry in pending:
        if now < entry["export_at"]:
            still_pending.append(entry)
            continue

        alert = entry["alert"]
        window_start = alert["disappeared_at"] - PRE_INCIDENT_SECONDS
        window_end = alert["disappeared_at"] + POST_INCIDENT_SECONDS

        clip_frames = [
            (ts, jpg) for ts, jpg in state["frame_buffer"]
            if window_start <= ts <= window_end
        ]

        video_path = incident_logger.save_incident_clip(clip_frames, entry["incident_id"])

        incident_logger.log_incident({
            "incident_id": entry["incident_id"],
            "camera_id": cam_id,
            "global_id": alert["global_id"],
            "person_id": alert["person_id"],
            "item": alert["item"],
            "message": alert["message"],
            "disappeared_at_time": time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime(alert["disappeared_at"])),
            "confirmed_at_time": time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime(alert["created_at"])),
            "video_path": video_path,
        })

    # Sirf woh entries hatao jo ABHI is call mein fully process ho chuki hain.
    # Blind reassignment (state["pending_video_exports"] = still_pending) khatarnak
    # tha: agar background worker (detection_worker.py) isi lamhe ek NAYI entry
    # add kar raha ho, to woh yahan reassign se gum ho jati. Ab sirf "done" wale
    # incident_ids nikalte hain, baaki (still-waiting + koi naya jo abhi aaya) rehte hain.
    done_ids = {e["incident_id"] for e in pending} - {e["incident_id"] for e in still_pending}
    with state["pending_lock"]:
        state["pending_video_exports"] = [
            e for e in state["pending_video_exports"] if e["incident_id"] not in done_ids
        ]


def _pace_file_playback(state):
    """
    Video FILE ko uski apni FPS par chalata hai (real-time). Iske bagair file
    jitni tez ho sake utni tez chalti hai (25 frame 0.05s mein!), aur time.time()
    par based sab timer (2s sustained concealment, 1.2s hold grace, 0.7s basket
    memory) video ke hisab se bohat chhote ho jate hain - file test ka natija
    jhoota aata hai. Live camera par yeh lagta hi nahi (woh khud real-time hai).
    """
    fps = state["capture"].get(cv2.CAP_PROP_FPS) or 30.0
    fps = min(max(fps, 5.0), 60.0)
    interval = 1.0 / fps

    now = time.time()
    next_t = state.get("_file_next_t")
    if next_t is None or now - next_t > 1.0:   # pehli dafa, ya pause/loop restart ke baad dobara sync
        next_t = now
    wait = next_t - now
    if wait > 0:
        time.sleep(wait)
    state["_file_next_t"] = next_t + interval


def generate_frames(cam_id=1):
    state = _get_state(cam_id)

    while True:
        if state["capture"] is None or not state["capture"].isOpened():
            break

        if state["is_paused"]:
            if state["last_frame_bytes"] is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + state["last_frame_bytes"] + b'\r\n')
            time.sleep(0.1)
            continue

        with profiler.stage(cam_id, "read"):
            ret, frame = state["capture"].read()

        if not ret:
            if state["source_type"] == "file" and state["loop_video"]:
                # Video file khatam ho gayi - shuru se dobara chalao (live
                # camera ke liye yeh kabhi True nahi hota, wahan waisa hi
                # rukega jaisa pehle tha)
                state["capture"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        if state["source_type"] == "file":
            _pace_file_playback(state)

        height, width = frame.shape[:2]
        line_y = int(height * LINE_Y_RATIO)
        zone_box = _zone_box_px(state, width, height)

        state["frame_counter"] += 1

        # Bhaari kaam (YOLO + Re-ID + Pose + Basket + Concealment) ab background
        # thread mein hota hai (detection_worker.py) - yeh call turant (non-blocking)
        # wapis aati hai, chahe andar YOLO 200ms le raha ho. Agar worker abhi pichle
        # frame par kaam kar raha ho to yeh frame chupke se chhor diya jata hai
        # (koi crash/queue-jama nahi hota) - jo bhi PICHLA result maujood hai woh
        # neeche seedha state se utha lete hain.
        with profiler.stage(cam_id, "submit"):
            detection_worker.submit_frame(frame, cam_id, state, zone_box, line_y, POST_INCIDENT_SECONDS)

        persons = state["last_persons"]
        items = state["last_items"]
        baskets = state["last_baskets"]

        with profiler.stage(cam_id, "draw"):
            frame = draw_tracks(frame, persons, items, baskets, zone_box, state["zone_enabled"])

        with profiler.stage(cam_id, "encode"):
            success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

        if not success:
            continue

        state["last_frame_bytes"] = buffer.tobytes()

        # Rolling video buffer: har asal frame yahan jaata hai (chahe
        # detection chali ho ya na ho), taake sustained-concealment confirm
        # hote hi disappearance ke aas-paas (PRE + POST) ka clip nikala ja sake.
        now = time.time()
        state["frame_buffer"].append((now, state["last_frame_bytes"]))
        while state["frame_buffer"] and (now - state["frame_buffer"][0][0]) > VIDEO_BUFFER_SECONDS:
            state["frame_buffer"].popleft()

        # Jo incidents ka POST-wait poora ho chuka hai, unka video ab export karo
        _flush_ready_video_exports(state, cam_id)

        profiler.frame_done(cam_id)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + state["last_frame_bytes"] + b'\r\n')