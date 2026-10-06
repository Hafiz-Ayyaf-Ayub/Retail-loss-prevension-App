import cv2
import numpy as np
import time
import os
import threading

# ---- Face recognition (naya, zyada reliable signal) ----
# Agar library install nahi hai to system crash nahi hoga - bas face
# feature off ho jayega aur system sirf clothes wale tareeqe pe chalega
# (jaisa pehlay se ho raha tha). "pip install face_recognition" chala kar
# isay enable karein.
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[person_reid] WARNING: 'face_recognition' library nahi mili — "
          "sirf clothes-based matching use hogi. 'pip install face_recognition' "
          "chala kar face matching enable karein.")

# Kitni "qareeb" honi chahiye rang taake same insaan maana jaye (0 se 1, 1 = perfect match)
# Ab score do metrics ka average hai (pehle sirf correlation tha), isliye threshold
# thora kam rakha hai. Debug images dekhne ke baad isko tune karna.
MATCH_THRESHOLD = 0.40

# face_recognition ka "distance" score - KAM (lower) matlab zyada match.
# 0.6 library ki common/recommended default hai (0.0 = identical chehra).
FACE_MATCH_THRESHOLD = 0.6

# Kitni dair tak purana record yaad rakhna hai (seconds) - warna list hamesha badhti rahegi
FORGET_AFTER_SECONDS = 300  # 5 minute

# Har insaan ke kitne purane samples yaad rakhein (sirf akhri wala nahi, taake match reliable ho)
MAX_SAMPLES_PER_PERSON = 3
MAX_FACE_SAMPLES_PER_PERSON = 3

# True kar do taake har comparison disk pe (terminal ke bagair) record ho:
# 1) app/detection/reid_cache/reid_debug.log mein ek line (score + threshold + candidate)
# 2) app/detection/reid_cache/debug/ folder mein ek side-by-side comparison image
#    (left = purana candidate jisse compare hua, right = abhi wala naya insaan)
# Filename mein score likha hota hai, jaise: score_0.32_vs_G1_1732000000.jpg
DEBUG_SCORES = True

# Naye insaan ki reference tasveer yahan save hogi (project ke andar hi)
REID_CACHE_DIR = os.path.join(os.path.dirname(__file__), "reid_cache")
os.makedirs(REID_CACHE_DIR, exist_ok=True)

DEBUG_DIR = os.path.join(REID_CACHE_DIR, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

LOG_PATH = os.path.join(REID_CACHE_DIR, "reid_debug.log")

# Global registry: { global_id: { "signatures": [...], "face_encodings": [...],
#                                  "last_seen": time, "suspicious": False, "concealment": False } }
# Ye dict saari cameras (threads) ke beech SHARED hai - isliye cross-camera match
# architecture ke lihaz se pehle se hi possible tha. Lock ke sath thread-safe bhi hai.
registry = {}
next_global_id = 1
_registry_lock = threading.Lock()


def normalize_illumination(bgr_image):
    """Gray-world white-balance normalization.

    Har camera ka apna alag exposure/white-balance hota hai, isliye ek hi
    insaan ki shirt ka color Camera 1 aur Camera 2 mein thora different
    dikhta hai. Ye function har channel (B,G,R) ko uske apne average ke
    hisab se scale karta hai taake color cast kam ho.
    """
    result = bgr_image.astype('float32')

    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    result[:, :, 0] *= (avg_gray / (avg_b + 1e-6))
    result[:, :, 1] *= (avg_gray / (avg_g + 1e-6))
    result[:, :, 2] *= (avg_gray / (avg_r + 1e-6))

    return np.clip(result, 0, 255).astype('uint8')


def extract_signature(frame, x1, y1, x2, y2):
    """Insaan ke sirf TORSO (beech ka, kapron wala) hisse se HSV color histogram nikalta hai -
    sar aur background ka asar kam karne ke liye. Crop image bhi return karta hai
    taake naye insaan ki reference tasveer disk pe save ki ja sake."""
    height = y2 - y1
    width = x2 - x1
    if height <= 0 or width <= 0:
        return None, None

    aspect_ratio = height / float(width)

    if aspect_ratio >= 1.3:
        # Box lamba hai (khara insaan, upper/full-body camera ka normal shot) -
        # beech ka torso hissa hi sahi rehta hai.
        ty1 = y1 + int(height * 0.30)
        ty2 = y1 + int(height * 0.85)
    else:
        # Box chora/squarish hai - matlab camera bohat qareeb hai aur box
        # zyada tar SIRF CHEHRA hai (reid_cache ki saved images se confirm
        # hua). Is surat mein beech ka hissa bhi chehre pe hi girta, isliye
        # box ke bilkul neeche wale hisse (collar/kandha) ko use karte hain
        # taake chehra shamil na ho.
        ty1 = y1 + int(height * 0.70)
        ty2 = y1 + int(height * 0.98)

    tx1 = x1 + int(width * 0.15)
    tx2 = x2 - int(width * 0.15)

    person_crop = frame[max(0, ty1):ty2, max(0, tx1):tx2]
    if person_crop.size == 0:
        return None, None

    normalized_crop = normalize_illumination(person_crop)

    hsv = cv2.cvtColor(normalized_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist, person_crop


def extract_face_encoding(frame, x1, y1, x2, y2):
    """
    Person ke bounding box ke andar chehra dhoondta hai aur uska 128-number
    ka "face encoding" (fingerprint) nikalta hai.

    Agar library install nahi hai, box ke andar koi chehra nahi mila, ya
    chehra bohot chhota/dhundla hai - None return karta hai (is surat mein
    match_or_register khud-ba-khud sirf clothes pe fall back kar jayega).
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return None

    person_crop = frame[max(0, y1):y2, max(0, x1):x2]
    if person_crop.size == 0:
        return None

    # face_recognition RGB images expect karta hai, OpenCV frames BGR hote hain
    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

    # "hog" model CPU pe fast chalta hai (webcam ke liye theek hai).
    # Zyada accuracy chahiye aur GPU available ho to "cnn" try kar saktay hain.
    face_locations = face_recognition.face_locations(rgb_crop, model="hog")
    if not face_locations:
        return None

    face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)
    if not face_encodings:
        return None

    return face_encodings[0]  # is box mein sabse pehla/bara chehra


def compare_signatures(sig1, sig2):
    """0 se 1 ke beech return karta hai - 1 matlab bilkul match.

    Do metrics ka average liya hai taake sirf ek lighting-sensitive metric
    pe pura inhisar na ho: correlation (overall shape match) aur intersection
    (kitna overlap hai, chhoti shifts ke against zyada tolerant)."""
    if sig1 is None or sig2 is None:
        return 0

    a = sig1.astype('float32')
    b = sig2.astype('float32')

    correl = cv2.compareHist(a, b, cv2.HISTCMP_CORREL)
    correl = max(0.0, correl)  # negative correlation ko 0 treat karo

    intersect = cv2.compareHist(a, b, cv2.HISTCMP_INTERSECT)
    intersect = max(0.0, min(intersect, 1.0))

    return (correl + intersect) / 2.0


def _best_score_against_samples(signature, samples):
    """Ek insaan ke sab saved clothes-samples mein se sabse acha (highest) match score dhoondta hai."""
    best = 0
    for s in samples:
        score = compare_signatures(signature, s)
        if score > best:
            best = score
    return best


def _best_face_distance_against_samples(face_encoding, face_samples):
    """Ek insaan ke sab saved face-samples mein se sabse chhota (best) distance dhoondta hai.
    Face_recognition mein KAM distance = zyada match, isliye yahan 'min' use hota hai
    (clothes wale _best_score function ke bilkul ulta - wahan 'max' hota hai)."""
    if not face_samples:
        return None
    distances = face_recognition.face_distance(face_samples, face_encoding)
    return float(min(distances))


def _log_debug(clothes_score, clothes_id, face_distance, face_id, matched_id, method):
    line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"clothes_score={clothes_score:.3f}(vs G{clothes_id}) "
            f"face_distance={('%.3f' % face_distance) if face_distance is not None else 'NA'}(vs G{face_id}) "
            f"matched=G{matched_id} method={method}\n")
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _save_debug_comparison(new_crop, candidate_id, score):
    """Candidate ki purani saved reference photo aur abhi wale naye crop ko
    side-by-side jor kar debug/ folder mein save karta hai, taake terminal
    dekhe bagair bhi pata chal sake ke comparison mein kya ho raha tha."""
    try:
        candidate_path = os.path.join(REID_CACHE_DIR, f"person_{candidate_id}.jpg")
        candidate_img = cv2.imread(candidate_path)
        if candidate_img is None or new_crop is None or new_crop.size == 0:
            return

        target_h = 200

        def resize_to_h(img, h):
            scale = h / img.shape[0]
            w = max(1, int(img.shape[1] * scale))
            return cv2.resize(img, (w, h))

        left = resize_to_h(candidate_img, target_h)
        right = resize_to_h(new_crop, target_h)
        gap = np.zeros((target_h, 10, 3), dtype='uint8')
        gap[:] = (0, 0, 255)
        combined = np.hstack([left, gap, right])

        ts = int(time.time() * 1000)
        fname = f"score_{score:.2f}_vs_G{candidate_id}_{ts}.jpg"
        cv2.imwrite(os.path.join(DEBUG_DIR, fname), combined)
    except Exception:
        pass  # debug save kabhi bhi asal detection ko na roke


def match_or_register(signature, person_crop=None, face_encoding=None, exclude_ids=None):
    """
    exclude_ids: un Global IDs ka set jo is match se BAHAR rakhne hain. Caller
    yahan woh IDs deta hai jo isi camera frame ke DOOSRE logon ke paas pehle se
    hain - jo do log ek hi frame mein ek saath khare hain wo lazmi alag alag
    log hain, chahe kapre bilkul same hon, is liye naya banda unki ID kabhi
    nahi le sakta (yehi "teeno ko G1 mil gayi" wala masla hal karta hai).
    Default None = purane jaisa behaviour (koi ID exclude nahi).

    Do signals se purane logon ke saath match karta hai (chahe wo kisi bhi
    camera se aaye hon, kyunke registry saari cameras mein shared hai):

      1) FACE (agar face_encoding mila) - zyada reliable, pehlay try hota hai
      2) CLOTHES (color histogram) - fallback, jab chehra na mila ho

    Match mile to wahi global_id (aur dono naye samples yaad rakh lo, jo bhi
    available hon), warna naya insaan register hota hai.
    """
    global next_global_id

    now = time.time()

    with _registry_lock:
        # Purane, bhoole hue logon ko list se nikal dein
        expired = [gid for gid, data in registry.items() if now - data["last_seen"] > FORGET_AFTER_SECONDS]
        for gid in expired:
            del registry[gid]

        # ---- Signal 1: FACE ----
        best_face_id = None
        best_face_distance = None
        if face_encoding is not None:
            for gid, data in registry.items():
                if exclude_ids and gid in exclude_ids:
                    continue
                dist = _best_face_distance_against_samples(face_encoding, data.get("face_encodings", []))
                if dist is not None and (best_face_distance is None or dist < best_face_distance):
                    best_face_distance = dist
                    best_face_id = gid

        # ---- Signal 2: CLOTHES ----
        best_clothes_id = None
        best_clothes_score = 0
        if signature is not None:
            for gid, data in registry.items():
                if exclude_ids and gid in exclude_ids:
                    continue
                score = _best_score_against_samples(signature, data["signatures"])
                if score > best_clothes_score:
                    best_clothes_score = score
                    best_clothes_id = gid

        # ---- Decision: face ko priority do (zyada reliable), clothes fallback ----
        best_match_id = None
        method = "none"

        if best_face_id is not None and best_face_distance <= FACE_MATCH_THRESHOLD:
            best_match_id = best_face_id
            method = "face"
        elif best_clothes_id is not None and best_clothes_score >= MATCH_THRESHOLD:
            best_match_id = best_clothes_id
            method = "clothes"

        if best_match_id is not None:
            data = registry[best_match_id]

            if signature is not None:
                data["signatures"].append(signature)
                if len(data["signatures"]) > MAX_SAMPLES_PER_PERSON:
                    data["signatures"].pop(0)

            if face_encoding is not None:
                data.setdefault("face_encodings", [])
                data["face_encodings"].append(face_encoding)
                if len(data["face_encodings"]) > MAX_FACE_SAMPLES_PER_PERSON:
                    data["face_encodings"].pop(0)

            data["last_seen"] = now
            result_id = best_match_id
            new_id = None
        else:
            new_id = next_global_id
            next_global_id += 1
            registry[new_id] = {
                "signatures": [signature] if signature is not None else [],
                "face_encodings": [face_encoding] if face_encoding is not None else [],
                "last_seen": now,
                "suspicious": False,
                "concealment": False,
            }
            result_id = new_id

    if DEBUG_SCORES and (best_face_id is not None or best_clothes_id is not None):
        _log_debug(best_clothes_score, best_clothes_id, best_face_distance, best_face_id, result_id, method)
        if best_clothes_id is not None:
            _save_debug_comparison(person_crop, best_clothes_id, best_clothes_score)

    # Reference tasveer disk pe save karo (sirf naye insaan ke liye)
    if new_id is not None and person_crop is not None:
        try:
            path = os.path.join(REID_CACHE_DIR, f"person_{new_id}.jpg")
            cv2.imwrite(path, person_crop)
        except Exception:
            pass

    return result_id


def mark_suspicious(global_id, value=True):
    if global_id in registry:
        registry[global_id]["suspicious"] = value


def mark_concealment(global_id, value=True):
    if global_id in registry:
        registry[global_id]["concealment"] = value


def is_flagged(global_id):
    data = registry.get(global_id)
    if not data:
        return False, False
    return data.get("suspicious", False), data.get("concealment", False)