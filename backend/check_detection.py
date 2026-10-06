"""
detection_worker.py

MASLA: YOLO + Re-ID + Pose + Concealment ka poora pipeline (~200ms) pehle
camera_stream.generate_frames() ke andar, MAIN thread mein, SEEDHA chalta
tha. Is wajah se jab tak ye 200ms khatam na hon, agli camera frame na
parhi ja sakti thi, na screen par bheji ja sakti thi - isi liye video
"atak atak" kar chalta tha (profiler: yolo=199ms, FPS=8.2).

HAL: ye poora bhaari kaam ab ALAG BACKGROUND THREAD (har camera ka apna)
mein hota hai - bilkul basket_detector.py jaisa "submit + jo bhi pichla
result hai wo utha lo" tareeqa. generate_frames() ab sirf itna karta hai:
  1. Camera se frame parho             (~12ms)
  2. Worker ko frame de do              (non-blocking, turant wapis)
  3. Jo bhi PICHLA result maujood hai wahi draw karo, JPEG banao, bhejo

NATIJA: screen ab camera ki asal speed (~read+draw+encode) par chalegi,
chahe YOLO abhi bhi utna hi time le raha ho. Boxes ek detection-cycle
(~200ms) purane ho sakte hain - insaan ki aankh ko farq mehsoos nahi hota,
magar video buttery-smooth lagta hai.

*** ZAROORI: ye sirf VIDEO ko smooth karta hai, DETECTION ko tez nahi
karta. *** Item abhi bhi ~200ms + concealment_monitor ka
SUSTAINED_CONCEALMENT_SECONDS (2s) mein hi "confirm" hoga. Asal detection
tez karna alag kaam hai (chhota imgsz / OpenVINO export / GPU) - is file
ke saath jo guidance di gayi hai wahan dekho.

THREAD-SAFETY NOTE (dhyan se parhna): item_tracker.py, concealment_
monitor.py aur person_reid.py apne GLOBAL dictionaries istemal karte hain
(saari cameras ke liye SHARED, per-camera alag nahi). Agar ek se zyada
cameras chal rahi hain, to un sab ke workers ab yeh dictionaries EK SATH
chhoo sakte hain. Yeh khatra bilkul NAYA nahin hai (Flask pehle bhi har
camera ka generate_frames() apne alag request-thread mein chalata tha),
magar is change ke baad thora zyada "exercise" hoga. Agar aage 2+ cameras
ek sath chalte waqt koi ajeeb/dohri alert dikhe, mujhe batana - un
dictionaries ke gird locks lagane honge (abhi jaan-boojh kar nahi lagaye,
taake yeh change chhota aur test karne layak rahe).
"""

import threading
import time

from app.detection.detector import track_all
from app.detection.zone_monitor import update_zone_tracking
from app.detection.concealment_monitor import update_concealment, get_concealment_flagged_ids
from app.detection import person_reid
from app.detection import item_tracker
from app.detection import pose_monitor
from app.detection import pickup_monitor
from app.detection import basket_detector
from app import profiler
from app.config import Settings

CROSSING_COOLDOWN = 1.5

_workers = {}
_workers_lock = threading.Lock()


# ---- Yeh 3 functions camera_stream.py ke check_line_crossing / update_alerts /
# apply_person_reid ki HUBAHU nakal hain. Alag se yahan rakhi hain (copy) taake
# camera_stream.py <-> detection_worker.py ke beech circular import na bane, aur
# camera_stream.py ki original public functions (agar kahin aur use hoti hon)
# chhue bagair, jaisi thi waisi hi reh jayein.

def _check_line_crossing(state, tracks, line_y):
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


def _update_alerts(state, tracks):
    state["active_alerts"] = [
        {
            "id": track["id"],
            "global_id": track.get("global_id"),
            "dwell_time": track.get("dwell_time", 0),
            "message": f"Person ID {track['id']} (Global {track.get('global_id')}) — {track.get('dwell_time', 0)}s in monitored zone",
        }
        for track in tracks
        if track.get("suspicious", False)
    ]


def _apply_person_reid(frame, persons, state):
    cache = state["global_id_cache"]

    # Is frame ke jin logon ko Global ID pehle se mil chuki hai, unki IDs "istemal
    # mein" hain. Naya track in mein se kisi se match nahi ho sakta: ek hi frame mein
    # do alag track = do alag log (chahe kapre bilkul same hon). Yahan tak cache
    # sirf MAUJOODA frame ke tracks ka hota hai (gaye hue tracks _process() mein
    # pehle hi hata diye jate hain), isliye jo banda frame se chala gaya aur wapis
    # aaya (ya jiska track ID badal gaya) wo apni purani ID wapis pa sakta hai.
    in_use = {cache[t["id"]] for t in persons if t["id"] in cache}

    for track in persons:
        track_id = track["id"]
        if track_id in cache:
            global_id = cache[track_id]
        else:
            x1, y1, x2, y2 = track["x1"], track["y1"], track["x2"], track["y2"]
            signature, person_crop = person_reid.extract_signature(frame, x1, y1, x2, y2)
            face_encoding = person_reid.extract_face_encoding(frame, x1, y1, x2, y2)
            global_id = person_reid.match_or_register(
                signature, person_crop, face_encoding, exclude_ids=in_use
            )
            cache[track_id] = global_id
            in_use.add(global_id)   # isi frame ka agla naya banda ye ID nahi le sakta
        track["global_id"] = global_id
        was_suspicious, _ = person_reid.is_flagged(global_id)
        if was_suspicious:
            track["suspicious"] = True
    return persons


def _handle_new_concealment_incidents(new_alerts, state, cam_id, post_incident_seconds):
    """
    pending_video_exports list ko camera_stream._flush_ready_video_exports()
    (MAIN thread) bhi padhta/badalta hai - is liye state["pending_lock"] ke
    andar hi append karte hain, taake ek entry "gum" na ho jaye agar dono
    thread theek usi lamhe mein is list ko chhuen.
    """
    if not new_alerts:
        return
    with state["pending_lock"]:
        for alert in new_alerts:
            incident_id = f"cam{cam_id}_g{alert['global_id']}_{int(alert['disappeared_at'] * 1000)}"
            state["pending_video_exports"].append({
                "incident_id": incident_id,
                "alert": alert,
                "export_at": alert["disappeared_at"] + post_incident_seconds,
            })


class _CamWorker(threading.Thread):
    """Ek camera ka background detector. basket_detector._Worker jaisa hi pattern."""

    def __init__(self, cam_id):
        super().__init__(daemon=True, name=f"detect-worker-{cam_id}")
        self.cam_id = cam_id
        self.busy = False
        self._frame = None
        self._state = None
        self._zone_box = None
        self._line_y = None
        self._post_incident_seconds = 3
        self._wake = threading.Event()

    def submit(self, frame, state, zone_box, line_y, post_incident_seconds):
        if self.busy:
            return   # pichla frame abhi process ho raha hai - yeh frame chhor do (basket_detector jaisa)
        self.busy = True
        self._frame = frame.copy()
        self._state = state
        self._zone_box = zone_box
        self._line_y = line_y
        self._post_incident_seconds = post_incident_seconds
        self._wake.set()

    def run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            frame, state = self._frame, self._state
            zone_box, line_y = self._zone_box, self._line_y
            try:
                self._process(frame, state, zone_box, line_y)
            except Exception as e:
                print(f"[detection_worker] cam{self.cam_id} mein masla: {e}")
            finally:
                self.busy = False

    def _process(self, frame, state, zone_box, line_y):
        cam_id = self.cam_id

        with profiler.stage(cam_id, "yolo"):
            persons, items = track_all(frame, cam_id)

        current_ids = {p["id"] for p in persons}
        stale_ids = [tid for tid in state["global_id_cache"] if tid not in current_ids]
        for tid in stale_ids:
            del state["global_id_cache"][tid]
        current_item_ids = {it["id"] for it in items}

        with profiler.stage(cam_id, "basket"):
            basket_detector.submit_frame(frame, cam_id, persons)
            baskets = basket_detector.get_baskets(cam_id)

        state["current_person_count"] = len(persons)
        for p in persons:
            state["unique_visitor_ids"].add(p["id"])

        _check_line_crossing(state, persons, line_y)

        if state["zone_enabled"]:
            persons = update_zone_tracking(persons, zone_box, Settings.SUSPICIOUS_DWELL_SECONDS)

        with profiler.stage(cam_id, "reid"):
            persons = _apply_person_reid(frame, persons, state)

        with profiler.stage(cam_id, "pose+pickup"):
            pickup_monitor.refresh_poses(
                frame, persons, items, state["pose_cache"],
                lambda f, p: pose_monitor.detect_pose(f, p, cam_id),
            )
            pickup_monitor.update_pickups(persons, items, cam_id)

        for item in items:
            item_tracker.register_item_if_new(item, persons)

        _update_alerts(state, persons)

        for p in persons:
            if p.get("suspicious", False):
                person_reid.mark_suspicious(p["global_id"], True)

        with profiler.stage(cam_id, "conceal"):
            active_concealment_alerts, new_concealment_alerts = update_concealment(
                persons, items, baskets=baskets
            )
        state["active_concealment_alerts"] = active_concealment_alerts

        _handle_new_concealment_incidents(
            new_concealment_alerts, state, cam_id, self._post_incident_seconds
        )

        stale_item_ids = [iid for iid in item_tracker.item_origins if iid not in current_item_ids]
        for iid in stale_item_ids:
            item_tracker.cleanup_item(iid)

        concealment_flagged_ids = get_concealment_flagged_ids()
        for p in persons:
            p["concealment_flag"] = p["global_id"] in concealment_flagged_ids

        # Teeno list ko SAB SE AAKHIR mein, ek sath likhna zaroori hai -
        # taake main thread ko kabhi "adhoora" mix (jaise naye persons +
        # purane items) na mile.
        state["last_persons"] = persons
        state["last_items"] = items
        state["last_baskets"] = baskets

        profiler.frame_done(cam_id)


def submit_frame(frame, cam_id, state, zone_box, line_y, post_incident_seconds=3):
    """
    generate_frames() se HAR frame par bulao. Turant (non-blocking) wapis
    aata hai. Agar worker abhi pichla frame process kar raha hai to yeh
    naya frame chhupke se chhor diya jata hai (jaisa basket_detector mein
    hota hai) - koi crash/queue-buildup nahi hoga.
    """
    with _workers_lock:
        worker = _workers.get(cam_id)
        if worker is None:
            worker = _CamWorker(cam_id)
            worker.start()
            _workers[cam_id] = worker
    worker.submit(frame, state, zone_box, line_y, post_incident_seconds)