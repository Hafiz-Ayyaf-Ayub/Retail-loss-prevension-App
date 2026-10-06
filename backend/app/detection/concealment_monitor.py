import time
from app.detection import item_tracker

item_last_seen = {}
concealment_history = []
recent_alerts = []
last_alert_time = {}

# Jab item pehli dafa gayab hota hai to turant alert nahi banti - pehle
# yahan "pending" mein daal dete hain. Agar item itni dair (SUSTAINED_
# CONCEALMENT_SECONDS) tak continuously gayab raha (turant wapis nahi
# aaya), tabhi woh "confirmed" alert banti hai. Isse false alarms kam
# hote hain (jaise YOLO ka koi ek frame miss ho jana).
#
# NOTE (fix): Key ab sirf GLOBAL ID hai - "{global_id}_{item_class}" nahi.
# Pehle agar item wapis dikhne par YOLO usay thori si alag class bol deta
# (jaise Bottle ko Cup), ya Re-ID naya Global ID de deta, to system
# "reappear" pehchanta hi nahi tha aur alert 60 second tak red atki rehti
# thi. Ab hum PERSON ko track karte hain, item ki class ko nahi - jo bhi
# item us insaan ke paas dobara nazar aaye (chahe uska naam kuch bhi ho),
# alert clear ho jati hai.
# Key format: global_id -> {"since": timestamp, ...}
pending_concealments = {}

# Yaad rakhta hai kaunse insaan (Global ID) ki concealment alert is waqt
# "active" (on-screen) hai. Global ID se hone ki wajah se yeh cross-camera
# kaam karta hai - jo banda Camera 1 mein flag hua wahi Camera 2 mein bhi
# red dikhta hai, chahe wo doosri camera mein chala jaye.
# Key format: global_id -> True/False
active_concealments = {}

ALERT_DISPLAY_SECONDS = 300
ALERT_COOLDOWN_SECONDS = 10
SUSTAINED_CONCEALMENT_SECONDS = 0.3  # itni dair continuously gayab rahe tab confirm ho

# NOTE: pehle yeh ek FIXED "NEARBY_DISTANCE = 20" (pixels) tha - jo bohat
# tight tha. Agar camera door thi ya bottle thori si bhi table pe alag
# (person ke box se 20px se zyada) thi, to "qareeb" hi count nahi hoti thi,
# isliye pending concealment kabhi banti hi nahi thi (asal chori bhi miss
# ho jati thi). Ab yeh item_tracker.py jaisa hi, PERSON KE SIZE ke hisab
# se relative hai - taake camera chahe door ho ya qareeb, sahi kaam kare.
NEARBY_DISTANCE_RATIO = 0.2
NEARBY_DISTANCE_MIN = 10  # bohat chhoti/door detection ke liye minimum margin

# ---- BASKET (tokri) LOGIC ----
# Agar item basket ke box ke SAATH gayab hua (yani gayab hone se pehle
# aakhri baar basket ke paas/andar dikha) to yeh CONCEALMENT nahi hai -
# customer ne maal tokri mein daala hai. Baaki har surat mein (jaise jacket/
# jeb mein gaya, basket ke paas nahi tha) purana concealment logic chalta hai.
#
# NOTE: hum item ki LAST POSITION dekhte hain, yeh nahi ke "banda tokri utha
# kar chal raha hai" - warna tokri wala banda jacket mein maal chhupa kar bhi
# bach jata.
# NOTE (fix): pehle yeh sirf "boxes kahin bhi chhoo rahe hain?" check karta tha -
# aur 20% margin itna bara tha ke seene ke paas pakri bottle ka box aur basket ka
# (bara kiya hua) box sirf CORNER se chhoo jate the (camera angle ki wajah se, jahan
# 2D mein overlap dikhta hai chahe asal mein bottle basket se door ho), aur system
# galti se "basket mein gaya" keh deta tha jabke banda jacket mein chupa raha tha.
#
# AB: sirf "chhoo rahe hain" kaafi nahi - ITEM KA ZYADA TAR HISSA basket ke andar
# hona chahiye (containment), taake seene ke paas ka item, jo basket se sirf 2D
# mein tange hua dikhta hai, "in_basket" na ban jaye.
BASKET_MARGIN_RATIO = 0.5       # basket box ke charon taraf sirf itna extra (detection jitter ke liye)
BASKET_CONTAINMENT_MIN = 0.85    # item ke box ka kam az kam itna % basket ke andar hona chahiye
BASKET_MEMORY_SECONDS = 0.8      # gayab hone se itni dair pehle tak basket ke andar dikha ho to "tokri mein gaya"


def _item_in_basket_zone(item, baskets):
    """
    Kya item ka box kisi basket ke (thora bara kiye hue) box ke ANDAR hai -
    sirf chhoona/corner-overlap kaafi nahi, ZYADA TAR hissa andar hona chahiye.
    """
    ix1, iy1, ix2, iy2 = item["x1"], item["y1"], item["x2"], item["y2"]
    item_area = max(1, (ix2 - ix1) * (iy2 - iy1))

    for b in baskets:
        mx = (b["x2"] - b["x1"]) * BASKET_MARGIN_RATIO
        my = (b["y2"] - b["y1"]) * BASKET_MARGIN_RATIO
        bx1, by1, bx2, by2 = b["x1"] - mx, b["y1"] - my, b["x2"] + mx, b["y2"] + my

        # dono boxes ka overlap wala hissa (rectangle intersection)
        ox1, oy1 = max(ix1, bx1), max(iy1, by1)
        ox2, oy2 = min(ix2, bx2), min(iy2, by2)
        if ox2 <= ox1 or oy2 <= oy1:
            continue   # koi overlap hi nahi

        overlap_area = (ox2 - ox1) * (oy2 - oy1)
        if (overlap_area / item_area) >= BASKET_CONTAINMENT_MIN:
            return True
    return False


def _is_near(cx, cy, person):
    """Kya (cx, cy) point is person ke box ke (size-relative) margin ke andar hai?"""
    px1, py1, px2, py2 = person["x1"], person["y1"], person["x2"], person["y2"]
    person_w = px2 - px1
    person_h = py2 - py1
    margin = max(NEARBY_DISTANCE_MIN, max(person_w, person_h) * NEARBY_DISTANCE_RATIO)
    return (px1 - margin) <= cx <= (px2 + margin) and (py1 - margin) <= cy <= (py2 + margin)


def update_concealment(persons, items, baskets=None):
    """
    baskets: detector.get_baskets(cam_id) ki list (har basket ka x1,y1,x2,y2).
    Optional hai - None/khali ho to bilkul pehle jaisa kaam karta hai.
    """
    global item_last_seen, concealment_history, recent_alerts, last_alert_time, active_concealments, pending_concealments

    now = time.time()
    current_item_ids = set()
    baskets = baskets or []

    for item in items:
        current_item_ids.add(item["id"])

        # Basket ke paas kab dikha tha - pichli value yaad rakho (sirf tab
        # update karo jab abhi bhi basket ke paas ho)
        prev = item_last_seen.get(item["id"])
        near_basket_t = prev["near_basket_t"] if prev else 0
        if baskets and _item_in_basket_zone(item, baskets):
            near_basket_t = now

        item_last_seen[item["id"]] = {
            "class_name": item["class_name"],
            "cx": (item["x1"] + item["x2"]) // 2,
            "cy": (item["y1"] + item["y2"]) // 2,
            "t": now,
            "near_basket_t": near_basket_t,
        }

    # Naye alerts jo ISI call mein "confirm" hue - camera_stream.py isay use
    # karega taake incident metadata log kare aur video clip save kare
    # (sirf EK dafa, jab pehli baar confirm ho, dobara nahi).
    new_alerts = []

    # ---- STEP 1: Reappearance check (PERSON-level, item ki class se independent) ----
    # Agar kisi (Global ID) person ke paas pehle "concealment" alert active
    # thi, aur ab KOI BHI item unke qareeb dobara nazar aa gaya hai (chahe
    # yeh kisi bhi camera mein ho, aur chahe YOLO usay thodi alag class bhi
    # bole) - to iska matlab shayad chori nahi thi. Active alert hata do.
    for item in items:
        item_class = item["class_name"]
        icx = (item["x1"] + item["x2"]) // 2
        icy = (item["y1"] + item["y2"]) // 2

        for person in persons:
            if _is_near(icx, icy, person):
                global_id = person["global_id"]

                # NOTE: Pehle yahan "pending" bhi turant clear ho jati thi
                # jaise hi item ek dafa bhi dobara nazar aata - lekin item
                # chupate waqt (haath mein le kar jeb tak) YOLO ka detection
                # khud hi flicker karta hai (ek-do frame ke liye halka sa
                # dikh jana), jis se asal concealment ka 2-second countdown
                # baar baar reset ho kar kabhi complete hi nahi hota tha.
                # Ab PENDING ko yahan clear NAHI karte - woh apna 2-second
                # countdown continuously chalata rahega chahe beech mein
                # flicker ho. Sirf CONFIRMED (active_concealments) wapis
                # green hoti hai jab koi bhi item genuinely dobara dikhe.
                if active_concealments.get(global_id):
                    recent_alerts = [
                        a for a in recent_alerts if a["global_id"] != global_id
                    ]
                    active_concealments[global_id] = False

                    concealment_history.append({
                        "time": time.strftime("%I:%M:%S %p"),
                        "message": f"✅ {item_class} reappeared near Person ID {person['id']} (Global {global_id}) — concealment alert cleared"
                    })
                break

    # ---- STEP 2: Disappearance check -> PENDING mein daalo (turant alert nahi) ----
    disappeared_ids = [tid for tid in item_last_seen if tid not in current_item_ids]

    for tid in disappeared_ids:
        last = item_last_seen[tid]

        # Item basket ke saath gayab hua (tokri mein gaya) -> concealment nahi.
        if last["near_basket_t"] > 0 and (last["t"] - last["near_basket_t"]) <= BASKET_MEMORY_SECONDS:
            concealment_history.append({
                "time": time.strftime("%I:%M:%S %p"),
                "message": f"🧺 {last['class_name']} basket mein gaya - concealment nahi (ignored)"
            })
            del item_last_seen[tid]
            continue

        for person in persons:
            if _is_near(last["cx"], last["cy"], person):
                # Agar yeh item person ka apna tha (jaise apna mobile),
                # to yeh chori nahi — alert mat banao, bas aage badh jao.
                origin = item_tracker.get_item_origin(tid)
                if origin == "personal":
                    break

                global_id = person["global_id"]

                # Sirf PEHLI dafa gayab hone par pending shuru karo - agar
                # pehle se pending hai to uska "since" time overwrite mat
                # karo (warna item kabhi bhi threshold cross nahi karega).
                if global_id not in pending_concealments:
                    pending_concealments[global_id] = {
                        "since": now,
                        "person_id": person["id"],
                        "global_id": global_id,
                        "item_class": last["class_name"],
                    }

                break

        del item_last_seen[tid]

    # ---- STEP 3: Pending concealments jo threshold cross kar chuke - CONFIRM karo ----
    for global_id, pending in list(pending_concealments.items()):
        if now - pending["since"] < SUSTAINED_CONCEALMENT_SECONDS:
            continue  # abhi itni dair nahi hui - abhi bhi sirf "pending"

        # Threshold cross ho gayi - is pending ko final decide kar do
        # (chahe cooldown ki wajah se skip ho jaye, pending se hata do -
        # dobara har frame check karne ki zaroorat nahi).
        del pending_concealments[global_id]

        last_time = last_alert_time.get(global_id, 0)
        if now - last_time <= ALERT_COOLDOWN_SECONDS:
            continue  # thodi dair pehle hi isi insaan ki alert ban chuki thi

        item_class = pending["item_class"]
        person_id = pending["person_id"]

        message = f"⚠ {item_class} disappeared near Person ID {person_id} (Global {global_id}) — Possible Concealment"

        alert = {
            "person_id": person_id,
            "global_id": global_id,
            "item": item_class,
            "message": message,
            "created_at": now,               # jab CONFIRM hui (disappearance + sustained threshold)
            "disappeared_at": pending["since"],  # jab item ASAL mein ghaib hua tha
        }

        recent_alerts.append(alert)
        new_alerts.append(alert)

        concealment_history.append({
            "time": time.strftime("%I:%M:%S %p"),
            "message": message
        })

        last_alert_time[global_id] = now
        active_concealments[global_id] = True

    recent_alerts = [a for a in recent_alerts if now - a["created_at"] < ALERT_DISPLAY_SECONDS]

    # Jin alerts ka on-screen display time khatam ho gaya, unhe bhi
    # "not active" mark kar do (taake active_concealments state sahi rahe)
    still_active_gids = {a["global_id"] for a in recent_alerts}
    for global_id in list(active_concealments.keys()):
        if active_concealments[global_id] and global_id not in still_active_gids:
            active_concealments[global_id] = False

    return recent_alerts, new_alerts


def get_concealment_flagged_ids():
    """
    Un GLOBAL IDs ka set deta hai jinki koi na koi concealment alert
    is waqt active hai — chahe wo kisi bhi camera mein trigger hui ho.
    camera_stream.py isay har camera mein use karega taake us person ki
    ID box (uska Global ID match kar ke) red/green dikhai ja sake — is
    liye wahi banda kisi bhi camera mein jaye, us camera ka feed usay
    red dikhayega jab tak alert clear na ho.
    """
    return {a["global_id"] for a in recent_alerts}