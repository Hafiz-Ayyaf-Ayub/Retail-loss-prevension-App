"""
pickup_monitor.py  (Step 2 - MediaPipe, v2: zyada stable)

Kaam: dekhna ke kisi insaan ka HAATH kisi item ke qareeb kuch dair tak
raha ya nahi. Agar haan, to item["held_by"] = us insaan ki Global ID.

v1 ki dikkatein (jo aapne test mein dekhi: "1-2 second ke liye aata hai"):
  1. Item ka box ek frame ke liye bhi gayab hua, ya haath ka point ek
     frame ke liye door gaya, to "held" turant khatam ho jata tha.
  2. Item ki track ID badalte hi (jo haath mein aksar hota hai) sab kuch
     dobara zero se shuru hota tha.
  3. Kalai (wrist) ka point item ke CENTRE se naapa jata tha, jabke
     bottle hatheli mein hoti hai.

v2 ke 3 fixes:
  1. GRACE TIME: "held" ek dam khatam nahi hota - haath thodi dair
     (RELEASE_GRACE_SECONDS) door ya item gayab rahe tab bhi held rehta hai.
  2. Hold ki yaad (insaan + item ki KISM) ke hisaab se hai, track ID se nahi,
     isliye ID badalne par bhi held foran wapis mil jata hai.
  3. Haath ke CENTRE (kalai+ungliyon ka average) se item ke BOX tak ka
     faasla naapte hain, sirf kalai se item ke centre tak nahi.

Abhi bhi sirf DIKHATA hai (item ka box magenta). Alert Step 3 mein.
"""

import math
import time

# Haath ka point, item ke box se kitna qareeb ho - insaan ke box ki HEIGHT
# ka %. (Fixed pixels nahi, taake camera door ho ya qareeb, dono mein chale.)
HAND_NEAR_RATIO = 0.12
HAND_NEAR_MIN_PX = 12

# Itni dair lagatar qareeb rahe tab "uthaya" maana jaye.
PICKUP_HOLD_SECONDS = 0.4

# "Held" khatam karne se pehle itni dair intezaar (flicker se bachne ke liye).
# Zyada rakhoge to item rakhne ke baad "held" der tak dikhega; kam rakhoge to
# flicker wapis aa sakta hai. 0.8 - 1.5 ke beech tune karo.
RELEASE_GRACE_SECONDS = 1.2

# Pose (MediaPipe) sirf unhi logon par chalate hain jinke aas-paas koi item
# ho - CPU bachane ke liye. Margin insaan ke box ke size ka %.
POSE_GATE_RATIO = 0.6
POSE_GATE_MIN_PX = 40

# ---- Temporary ON/OFF switch ----
# MediaPipe abhi sirf "HELD by G1" wala visual hint deta hai - concealment
# ka asal faisla (Step 3) isse judaa nahi hai. Is liye False karne se koi
# zaroori feature nahi tootega, sirf yeh magenta highlight gayab ho jayegi.
# CPU/thermal load kam karne ke liye abhi ke liye False kar dein - jab
# wapis chahiye ho, sirf yeh line True kar dena, baaki kuch chhedne ki
# zaroorat nahi.
POSE_ENABLED = True

# SPEED: MediaPipe pose bhaari hai (profile mein 100-230ms). Isliye har
# detection-frame par nahi, har POSE_EVERY_N_DETECTIONS-wein detection-frame
# par chalate hain, aur beech mein pichla pose dobara istemal karte hain.
# 1 rakho to har detection par chalega (sab se sahi, sab se slow).
POSE_EVERY_N_DETECTIONS = 2
# Purana pose is se zyada purana ho to zabardasti dobara chalao (haath aage nikal chuka hoga).
POSE_MAX_AGE_SECONDS = 0.8

# (cam_id, global_id, item_class) -> {"first_near": t, "last_near": t}
# Ek insaan ke haath mein ek hi kism ki 2 cheezein ho to woh ek hi maani
# jayengi - abhi ke liye theek hai.
_holds = {}


def _person_margin(person):
    px1, py1, px2, py2 = person["x1"], person["y1"], person["x2"], person["y2"]
    return max(POSE_GATE_MIN_PX, max(px2 - px1, py2 - py1) * POSE_GATE_RATIO)


def _item_near_person_box(person, item):
    margin = _person_margin(person)
    cx = (item["x1"] + item["x2"]) / 2
    cy = (item["y1"] + item["y2"]) / 2
    return ((person["x1"] - margin) <= cx <= (person["x2"] + margin)
            and (person["y1"] - margin) <= cy <= (person["y2"] + margin))


def person_has_item_nearby(person, items):
    """Kya koi item is insaan ke box ke (size-relative) margin ke andar hai?"""
    return any(_item_near_person_box(person, item) for item in items)


def refresh_poses(frame, persons, items, cache, detect_fn):
    """
    Har insaan ka p["pose"] set karta hai, lekin MediaPipe ko sirf zaroorat
    par chalata hai:
      - jis ke qareeb koi item nahi -> pose None (chalta hi nahi)
      - jis ka pose abhi taaza hai -> purana wala dobara istemal
      - warna detect_fn(frame, person) se naya nikalta hai

    cache: har camera ka apna dict (camera_stream ki state se aata hai).
    detect_fn: function jo (frame, person) leke pose deta hai.
    """
    if not POSE_ENABLED:
        # MediaPipe bilkul band - kisi ko bhi pose nahi milega, aur
        # detect_fn kabhi call hi nahi hoga (CPU load zero)
        for p in persons:
            p["pose"] = None
        cache.clear()
        return

    now = time.time()
    seen_ids = set()

    for person in persons:
        track_id = person["id"]
        seen_ids.add(track_id)

        if not person_has_item_nearby(person, items):
            person["pose"] = None
            cache.pop(track_id, None)
            continue

        entry = cache.get(track_id)
        if entry is not None:
            entry["skipped"] += 1
            still_fresh = (entry["skipped"] < POSE_EVERY_N_DETECTIONS
                           and (now - entry["time"]) <= POSE_MAX_AGE_SECONDS)
            if still_fresh:
                person["pose"] = entry["pose"]
                continue

        pose = detect_fn(frame, person)
        cache[track_id] = {"pose": pose, "time": now, "skipped": 0}
        person["pose"] = pose

    # Jo log ab frame mein nahi, unka cache saaf karo
    for track_id in [t for t in cache if t not in seen_ids]:
        del cache[track_id]


def _hand_points(person):
    """Insaan ke haath ke points: haath ka centre, warna kalai."""
    pose = person.get("pose")
    if not pose:
        return []

    points = []
    for hand_name, wrist_name in (("left_hand", "left_wrist"), ("right_hand", "right_wrist")):
        point = pose.get(hand_name) or pose.get(wrist_name)
        if point is not None:
            points.append(point)
    return points


def _distance_to_box(px, py, item):
    """Point (px, py) se item ke box tak ka faasla (point box ke andar ho to 0)."""
    dx = max(item["x1"] - px, 0, px - item["x2"])
    dy = max(item["y1"] - py, 0, py - item["y2"])
    return math.hypot(dx, dy)


def _find_holder(persons, item):
    """Is frame mein kis insaan ka haath is item ke sab se qareeb hai? (Global ID ya None)"""
    best_gid = None
    best_dist = None

    for person in persons:
        gid = person.get("global_id")
        if gid is None:
            continue

        person_height = person["y2"] - person["y1"]
        radius = max(HAND_NEAR_MIN_PX, person_height * HAND_NEAR_RATIO)

        for (hx, hy) in _hand_points(person):
            d = _distance_to_box(hx, hy, item)
            if d < radius and (best_dist is None or d < best_dist):
                best_dist = d
                best_gid = gid

    return best_gid


def _held_by_grace(persons_by_gid, items_class, item, cam_id, now):
    """
    Is frame mein haath item ke qareeb nahi mila (point flicker, ya haath
    ka point hi na mila) - lekin agar abhi-abhi (grace time ke andar) kisi ne
    is kism ki cheez confirm-held ki thi aur item abhi bhi us insaan ke
    qareeb hai, to held barqarar rakho.
    """
    for (c_id, gid, cls), rec in _holds.items():
        if c_id != cam_id or cls != items_class:
            continue
        if now - rec["last_near"] > RELEASE_GRACE_SECONDS:
            continue
        if rec["last_near"] - rec["first_near"] < PICKUP_HOLD_SECONDS:
            continue  # yeh kabhi confirm hi nahi hua tha
        person = persons_by_gid.get(gid)
        if person is not None and _item_near_person_box(person, item):
            return gid
    return None


def update_pickups(persons, items, cam_id=1):
    """
    Har item ke liye item["held_by"] set karta hai (Global ID ya None).
    Detection wale frame par har baar chalao (persons mein "pose" aur
    "global_id" pehle se maujood hone chahiye).
    """
    now = time.time()
    persons_by_gid = {p.get("global_id"): p for p in persons if p.get("global_id") is not None}

    for item in items:
        gid = _find_holder(persons, item)

        if gid is not None:
            key = (cam_id, gid, item["class_name"])
            rec = _holds.get(key)

            if rec is None or now - rec["last_near"] > RELEASE_GRACE_SECONDS:
                rec = {"first_near": now, "last_near": now}   # naya contact
                _holds[key] = rec
            else:
                rec["last_near"] = now                         # contact jaari

            item["held_by"] = gid if (now - rec["first_near"]) >= PICKUP_HOLD_SECONDS else None
        else:
            item["held_by"] = _held_by_grace(persons_by_gid, item["class_name"], item, cam_id, now)

    # Bohot purane records saaf karo (memory na barhe)
    expired = [k for k, rec in _holds.items() if now - rec["last_near"] > RELEASE_GRACE_SECONDS * 3]
    for k in expired:
        del _holds[k]