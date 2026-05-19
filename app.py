# ============================================================
# ORBITAL WATCH SYSTEM — Backend (app.py)
# Phase 1: Foundation + Physics + AI Pipeline
# ============================================================

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import sqlite3
import requests
import threading
import time
import math
import random
from datetime import datetime, timedelta, timezone
import pygad
from sgp4.api import Satrec, jday as sgp4_jday

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # ── Space-Track.org credentials ──
    # Register FREE at https://www.space-track.org/auth/createAccount
    # Fill these in once you have an account — leave blank to use Celestrak only
    "SPACETRACK_USER": "",   # ← paste your email here
    "SPACETRACK_PASS": "",   # ← paste your password here

    # ── Space-Track settings ──
    "ST_CDM_DAYS_BACK":  7,      # fetch CDMs from last N days
    "ST_CDM_MAX":        200,    # max CDMs to fetch per request
    "ST_TLE_MAX":        500,    # max TLEs from Space-Track

    "MAX_OBJECTS": 300,            # Satellites to track (more = slower scan)
    "PROPAGATION_HOURS": 24,       # How far ahead to predict (hours)
    "TIME_STEP_MINUTES": 5,        # Propagation resolution
    "CONJUNCTION_THRESHOLD": 50.0, # km — flag pairs closer than this
    "DATA_REFRESH_SECONDS": 3600,  # Re-fetch TLEs every hour

    # --- GA Planner ---
    "GA_GENERATIONS": 80,
    "GA_POPULATION": 30,
    "GA_DV_LIMIT": 0.05,           # Max delta-v per axis (km/s)
}

DB_PATH = "satellite_watch.db"

# ============================================================
# DATABASE
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS satellites (
        norad_id   INTEGER PRIMARY KEY,
        name       TEXT,
        tle1       TEXT,
        tle2       TEXT,
        object_type TEXT DEFAULT "PAYLOAD",
        last_updated TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS conjunction_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        sat1_name       TEXT,
        sat2_name       TEXT,
        miss_distance   REAL,
        pc              REAL,
        risk_level      TEXT,
        tca_minutes     INTEGER,
        detected_at     TEXT
    )''')
    conn.commit()
    conn.close()

def store_tles(tles):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    for t in tles:
        c.execute(
            "INSERT OR REPLACE INTO satellites VALUES (?,?,?,?,?,?)",
            (t["norad_id"], t["name"], t["tle1"], t["tle2"], t.get("object_type","PAYLOAD"), now)
        )
    conn.commit()
    conn.close()

def load_tles_from_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT norad_id, name, tle1, tle2, object_type FROM satellites")
    rows = c.fetchall()
    conn.close()
    return [{"norad_id": r[0], "name": r[1], "tle1": r[2], "tle2": r[3], "object_type": r[4]} for r in rows]

def log_conjunction(conj):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conjunction_log (sat1_name,sat2_name,miss_distance,pc,risk_level,tca_minutes,detected_at) VALUES (?,?,?,?,?,?,?)",
        (conj["sat1_name"], conj["sat2_name"], conj["miss_distance"],
         conj["probability_of_collision"], conj["risk_level"],
         conj["tca_minutes_from_now"], datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()

def get_conjunction_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM conjunction_log ORDER BY detected_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "sat1_name": r[1], "sat2_name": r[2], "miss_distance": r[3],
             "pc": r[4], "risk_level": r[5], "tca_minutes": r[6], "detected_at": r[7]} for r in rows]

# ============================================================
# TLE FETCHING
# ============================================================

CELESTRAK_GROUPS = [
    ("stations",       "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=TLE"),
    ("active",         "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=TLE"),
    ("active-csv",     "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=TLE&CATNR="),
    ("starlink",       "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=TLE"),
    ("oneweb",         "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=TLE"),
    ("iridium-debris", "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=TLE"),
    ("cosmos-debris",  "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=TLE"),
    ("last-30-days",   "https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=TLE"),
    # GEO satellites
    ("geo",            "https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=TLE"),
    ("geosync",        "https://celestrak.org/NORAD/elements/gp.php?GROUP=geosync&FORMAT=TLE"),
]

# GEO altitude band: 35,000 – 36,500 km
GEO_ALT_MIN = 35000
GEO_ALT_MAX = 36500
GEO_CONJ_THRESHOLD_KM = 5.0   # much tighter for GEO (sparse population)

FALLBACK_TLES = [
    {"norad_id": 25544, "name": "ISS (ZARYA)", "object_type": "STATION",
     "tle1": "1 25544U 98067A   24001.50000000  .00016717  00000-0  30306-3 0  9999",
     "tle2": "2 25544  51.6416 247.4627 0006703 130.5360 345.7693 15.49509837472918"},
    {"norad_id": 33777, "name": "IRIDIUM 33 DEB", "object_type": "DEBRIS",
     "tle1": "1 33777U 97051C   24001.50000000  .00000100  00000-0  11440-4 0  9997",
     "tle2": "2 33777  86.3980 220.2443 0001607 273.5744  86.3980 14.34149209381894"},
    {"norad_id": 48274, "name": "STARLINK-2139", "object_type": "PAYLOAD",
     "tle1": "1 48274U 21024BH  24001.50000000  .00001000  00000-0  10000-4 0  9999",
     "tle2": "2 48274  53.0539 120.0000 0001000   0.0000   0.0000 15.06000000000000"},
    {"norad_id": 40075, "name": "COSMOS 2251 DEB", "object_type": "DEBRIS",
     "tle1": "1 40075U 09005BG  24001.50000000  .00000050  00000-0  50000-5 0  9999",
     "tle2": "2 40075  74.0372 300.0000 0068000  90.0000 270.0000 14.28000000000000"},
]

# ============================================================
# SPACE-TRACK.ORG  — Official US Space Surveillance Network
# ============================================================

ST_BASE    = "https://www.space-track.org"
ST_LOGIN   = ST_BASE + "/ajaxauth/login"
# cdm_public query — TCA filter only, Python-side distance filter
# Space-Track date filter format: now-N (no "day" suffix)
ST_CDM_URL = (ST_BASE + "/basicspacedata/query/class/cdm_public"
              "/TCA/%3Enow-{days}"
              "/orderby/MISS_DISTANCE%20asc"
              "/limit/{limit}/format/json")
ST_TLE_URL = (ST_BASE + "/basicspacedata/query/class/gp"
              "/EPOCH/%3Enow-3"
              "/MEAN_MOTION/%3E11.25/ECCENTRICITY/%3C0.25"
              "/orderby/NORAD_CAT_ID/limit/{limit}/format/tle")

# Fetch TLE for a specific NORAD ID (for CDM path enrichment)
ST_TLE_BY_ID = (ST_BASE + "/basicspacedata/query/class/gp"
                "/NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20desc"
                "/limit/1/format/tle")


def spacetrack_login(session):
    """Login to Space-Track and store session cookie. Returns True on success."""
    user = CONFIG.get("SPACETRACK_USER","").strip()
    pw   = CONFIG.get("SPACETRACK_PASS","").strip()
    if not user or not pw:
        return False
    try:
        resp = session.post(ST_LOGIN,
                            data={"identity": user, "password": pw},
                            timeout=15)
        # Space-Track returns HTTP 200 for BOTH success and failure
        # Check the response body — failed login contains "Login" or "Failed"
        body = resp.text.strip()
        if resp.status_code == 200 and '"Login"' not in body and "Failed" not in body and len(body) < 100:
            print("  Space-Track login: OK")
            return True
        # Also check cookies — successful login sets a chocolatechip cookie
        if "chocolatechip" in session.cookies or "session" in [c.lower() for c in session.cookies]:
            print("  Space-Track login: OK (cookie found)")
            return True
        print(f"  Space-Track login failed — check credentials. Response: {body[:120]}")
        return False
    except Exception as e:
        print(f"  Space-Track login error: {e}")
        return False


def fetch_cdms_from_spacetrack():
    """
    Fetch official Conjunction Data Messages from Space-Track.
    Returns list of conjunction dicts in our internal format, or [] on failure.
    """
    user = CONFIG.get("SPACETRACK_USER","").strip()
    pw   = CONFIG.get("SPACETRACK_PASS","").strip()
    if not user or not pw:
        return []

    print("  Fetching CDMs from Space-Track.org...")
    session = requests.Session()

    if not spacetrack_login(session):
        return []

    try:
        url = ST_CDM_URL.format(
            days  = CONFIG["ST_CDM_DAYS_BACK"],
            limit = CONFIG["ST_CDM_MAX"],
        )
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Handle error response
        if isinstance(data, dict) and "error" in data:
            print(f"  Space-Track CDM error: {data}")
            return []
        cdms = data if isinstance(data, list) else []
        print(f"  Space-Track returned {len(cdms)} CDMs")
        if cdms:
            print(f"  CDM fields: {list(cdms[0].keys())[:20]}")
            # Filter by miss distance in Python (threshold in km)
            thresh_km = CONFIG["CONJUNCTION_THRESHOLD"]
            cdms = [c for c in cdms if float(c.get("MISS_DISTANCE",999999)) / 1000.0 < thresh_km]
            print(f"  After {thresh_km}km filter: {len(cdms)} CDMs")
        return [parse_cdm(cdm) for cdm in cdms if cdm]
    except Exception as e:
        print(f"  Space-Track CDM fetch error: {e}")
        return []
    finally:
        try:
            session.get(ST_BASE + "/ajaxauth/logout", timeout=5)
        except:
            pass


def fetch_tles_from_spacetrack():
    """
    Fetch fresh TLEs from Space-Track for all active LEO objects.
    Returns list of TLE dicts or [] on failure.
    """
    user = CONFIG.get("SPACETRACK_USER","").strip()
    pw   = CONFIG.get("SPACETRACK_PASS","").strip()
    if not user or not pw:
        return []

    print("  Fetching TLEs from Space-Track.org...")
    session = requests.Session()
    if not spacetrack_login(session):
        return []

    try:
        url  = ST_TLE_URL.format(limit=CONFIG["ST_TLE_MAX"])
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        tles = parse_tle_text(resp.text)
        print(f"  Space-Track TLEs: {len(tles)} objects")
        return tles
    except Exception as e:
        print(f"  Space-Track TLE fetch error: {e}")
        return []
    finally:
        try:
            session.get(ST_BASE + "/ajaxauth/logout", timeout=5)
        except:
            pass


def parse_cdm(cdm):
    """
    Convert Space-Track CDM JSON to internal format.
    IMPORTANT: Space-Track MISS_DISTANCE is in METRES, RELATIVE_SPEED in m/s.
    """
    try:
        # Miss distance: metres -> km
        miss_m = float(cdm.get("MISS_DISTANCE") or cdm.get("TCA_RANGE") or 999000)
        miss   = round(miss_m / 1000.0, 4)

        # Pc
        pc_raw = cdm.get("COLLISION_PROBABILITY") or cdm.get("PROBABILITY")
        try:
            pc = float(pc_raw) if pc_raw not in (None,"","N/A","0") else compute_pc(miss)
        except:
            pc = compute_pc(miss)

        # TCA -> minutes from now
        tca_str = cdm.get("TCA","")
        tca_min = 60
        if tca_str:
            try:
                from datetime import timezone as tz
                tca_dt  = datetime.strptime(tca_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=tz.utc)
                tca_min = max(0, int((tca_dt - datetime.now(tz.utc)).total_seconds() / 60))
            except Exception as te:
                print(f"  TCA parse: {te}")

        # NORAD IDs
        def get_norad(p):
            for k in [f"{p}_OBJECT_DESIGNATOR", f"{p}_CATALOG_NUMBER"]:
                v = cdm.get(k)
                if v not in (None,"","0",0):
                    try: return int(str(v).strip())
                    except: pass
            return 0

        # Names
        def get_name(p):
            for k in [f"{p}_CATALOG_NAME", f"{p}_OBJECT_NAME", f"{p}_NAME"]:
                v = (cdm.get(k) or "").strip()
                if v and v not in ("","0","TBA"): return v
            n = get_norad(p)
            return f"NORAD-{n}" if n else f"{p}-UNKNOWN"

        sat1_id   = get_norad("SAT1")
        sat2_id   = get_norad("SAT2")
        sat1_name = get_name("SAT1")
        sat2_name = get_name("SAT2")

        # Relative speed: m/s -> km/s
        rel_v = round(float(cdm.get("RELATIVE_SPEED") or cdm.get("TCA_RELATIVE_SPEED") or 0) / 1000.0, 3)

        risk    = classify_risk(miss, pc)
        tca_idx = max(2, tca_min // CONFIG["TIME_STEP_MINUTES"])

        # Basic orbital elements from CDM
        def cdm_orb(p):
            try:
                n_rd = float(cdm.get(f"{p}_MEAN_MOTION") or 15.5)
                mu   = 398600.4418
                n    = n_rd * 2 * math.pi / 86400
                a    = (mu / n**2) ** (1/3)
                inc  = float(cdm.get(f"{p}_INCLINATION") or cdm.get(f"{p}_INC") or 0)
                ecc  = float(cdm.get(f"{p}_ECCENTRICITY") or 0)
                return {"inclination": round(inc,2), "altitude_km": round(a-6371,1),
                        "velocity_kms": round(math.sqrt(mu/a),3),
                        "period_min":   round(2*math.pi*math.sqrt(a**3/mu)/60,1),
                        "eccentricity": round(ecc,6)}
            except:
                return {"inclination":0,"altitude_km":400,"velocity_kms":7.7,"period_min":92,"eccentricity":0}

        return {
            "sat1_id": sat1_id, "sat1_name": sat1_name,
            "sat1_type": cdm.get("SAT1_OBJECT_TYPE","PAYLOAD"),
            "sat2_id": sat2_id, "sat2_name": sat2_name,
            "sat2_type": cdm.get("SAT2_OBJECT_TYPE","DEBRIS"),
            "miss_distance": miss, "tca_minutes_from_now": tca_min,
            "probability_of_collision": pc, "risk_level": risk,
            "relative_velocity": rel_v, "tca_idx": tca_idx,
            "sat1_path": [], "sat2_path": [],
            "sat1_pos_at_tca": [0,0,0], "sat2_pos_at_tca": [0,0,0],
            "sat1_orbital": cdm_orb("SAT1"), "sat2_orbital": cdm_orb("SAT2"),
            "is_synthetic": False, "source": "space-track-cdm",
            "cdm_id": cdm.get("CDM_ID",""),
        }
    except Exception as e:
        print(f"  CDM parse error: {e} | keys: {list(cdm.keys())[:8]}")
        return None

def parse_tle_text(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    tles = []
    i = 0
    while i <= len(lines) - 3:
        if lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            try:
                norad_id = int(lines[i+1][2:7])
                name = lines[i]
                obj_type = "DEBRIS" if any(w in name.upper() for w in ["DEB","DEBRIS","FRAG"]) else "PAYLOAD"
                tles.append({"norad_id": norad_id, "name": name,
                             "tle1": lines[i+1], "tle2": lines[i+2],
                             "object_type": obj_type,
                             "is_geo": False})  # tagged later by altitude
                i += 3
            except:
                i += 1
        else:
            i += 1
    return tles

def fetch_tles(max_objects=300):
    all_tles = []
    for group, url in CELESTRAK_GROUPS:
        try:
            print(f"  Fetching {group} from Celestrak...")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            tles = parse_tle_text(r.text)
            all_tles.extend(tles)
            print(f"    → {len(tles)} objects")
            time.sleep(0.3)
        except Exception as e:
            print(f"    WARNING: {group} failed: {e}")

    if not all_tles:
        print("  Using hardcoded fallback TLEs")
        return FALLBACK_TLES

    seen, unique = set(), []
    for t in all_tles:
        if t["norad_id"] not in seen:
            seen.add(t["norad_id"])
            unique.append(t)

    random.shuffle(unique)
    return unique[:max_objects]

# ============================================================
# SGP4 PROPAGATION
# ============================================================

def dt_to_jd(dt):
    """Convert datetime to Julian date + fraction for sgp4."""
    return sgp4_jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                     dt.second + dt.microsecond / 1e6)

def get_time_grid(hours=24, step_min=5):
    """Build list of (jd, fr) tuples for propagation."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    steps = int(hours * 60 / step_min)
    grid = []
    for i in range(steps):
        t = now + timedelta(minutes=i * step_min)
        jd, fr = dt_to_jd(t)
        grid.append((jd, fr))
    return grid

def propagate(tle1, tle2, time_grid):
    """
    Propagate a satellite with SGP4.
    Returns np.array of shape (N, 3) in km (ECI frame), or None on failure.
    """
    try:
        sat = Satrec.twoline2rv(tle1, tle2)
        positions = []
        for jd, fr in time_grid:
            e, r, _ = sat.sgp4(jd, fr)
            if e != 0:
                return None
            positions.append(r)
        return np.array(positions, dtype=np.float64)
    except Exception:
        return None

def get_current_position(tle1, tle2):
    """Get single current position for globe display."""
    now = datetime.now(timezone.utc)
    jd, fr = dt_to_jd(now)
    try:
        sat = Satrec.twoline2rv(tle1, tle2)
        e, r, _ = sat.sgp4(jd, fr)
        return list(r) if e == 0 else None
    except:
        return None

def orbital_altitude_band(tle1, tle2):
    """Get approximate perigee/apogee altitude (km) for pre-filtering."""
    try:
        sat = Satrec.twoline2rv(tle1, tle2)
        n = sat.no_kozai / 60.0          # rad/s
        mu = 398600.4418                 # km^3/s^2
        a = (mu / n**2) ** (1/3)
        e = sat.ecco
        R_E = 6371.0
        return a * (1 - e) - R_E, a * (1 + e) - R_E
    except:
        return 200.0, 2000.0

# ============================================================
# CONJUNCTION SCREENING
# ============================================================

def altitudes_overlap(p1, a1, p2, a2, margin=100.0):
    return not (a1 + margin < p2 or a2 + margin < p1)

# Known co-located / docked object groups — skip these pairs
COLOCATED_GROUPS = [
    {25544, 49044, 37820, 43205, 47813, 48915, 40239, 41765,  # ISS & modules
     33153, 39090, 43476, 47025, 49044, 52389},
]

def is_colocated(id1, id2):
    for group in COLOCATED_GROUPS:
        if id1 in group and id2 in group:
            return True
    return False

def get_orbital_elements(tle1, tle2):
    """Extract key orbital elements from TLE for display."""
    try:
        sat = Satrec.twoline2rv(tle1, tle2)
        n   = sat.no_kozai / 60.0          # rad/s
        mu  = 398600.4418
        a   = (mu / n**2) ** (1/3)         # km, semi-major axis
        e   = sat.ecco
        inc = math.degrees(sat.inclo)      # inclination deg
        R_E = 6371.0
        alt_perigee  = round(a*(1-e) - R_E, 1)
        alt_apogee   = round(a*(1+e) - R_E, 1)
        alt_mean     = round(a - R_E, 1)
        # velocity at mean altitude (vis-viva, circular approx)
        velocity     = round(math.sqrt(mu / a), 3)
        period_min   = round(2*math.pi*math.sqrt(a**3/mu)/60, 1)
        return {
            "inclination":  round(inc, 2),
            "altitude_km":  alt_mean,
            "perigee_km":   alt_perigee,
            "apogee_km":    alt_apogee,
            "velocity_kms": velocity,
            "period_min":   period_min,
            "eccentricity": round(e, 6),
        }
    except:
        return {}


def screen_conjunctions(tles, time_grid, threshold_km=50.0):
    """
    Screen all pairs for conjunctions.
    Uses altitude band pre-filter to avoid O(n^2) full propagation.
    Returns list of conjunction event dicts.
    """
    print(f"  Propagating {len(tles)} objects...")
    objects = []
    for t in tles:
        pos = propagate(t["tle1"], t["tle2"], time_grid)
        if pos is not None:
            p, a = orbital_altitude_band(t["tle1"], t["tle2"])
            is_geo = (p > GEO_ALT_MIN and a < GEO_ALT_MAX)
        objects.append({**t, "positions": pos, "perigee": p, "apogee": a, "is_geo": is_geo})

    print(f"  Propagated {len(objects)} successfully. Screening pairs...")
    n = len(objects)
    conjunctions = []
    pairs_checked = 0

    for i in range(n):
        for j in range(i + 1, n):
            o1, o2 = objects[i], objects[j]
            if not altitudes_overlap(o1["perigee"], o1["apogee"],
                                     o2["perigee"], o2["apogee"]):
                continue
            if is_colocated(o1["norad_id"], o2["norad_id"]):
                continue
            pairs_checked += 1
            # Use tighter threshold for GEO pairs
            pair_threshold = (GEO_CONJ_THRESHOLD_KM
                              if o1.get("is_geo") and o2.get("is_geo")
                              else threshold_km)
            diff = o1["positions"] - o2["positions"]
            dists = np.linalg.norm(diff, axis=1)
            min_d = float(np.min(dists))

            if min_d < pair_threshold:
                tca_idx = int(np.argmin(dists))
                tca_min = tca_idx * CONFIG["TIME_STEP_MINUTES"]
                if tca_idx < 2:
                    continue
                pc = compute_pc(min_d)
                # Use ML model if available, otherwise formula
                ml_result = ml_predict_risk(min_d, pc, tca_min,
                                            o1["name"], o2["name"])
                risk = ml_result["risk_level"] if ml_result else classify_risk(min_d, pc)
                ml_confidence = ml_result["confidence"] if ml_result else None
                is_geo_pair = o1.get("is_geo", False) and o2.get("is_geo", False)
                # ── Skip co-located / docked objects ──
                name1 = o1["name"].upper()
                name2 = o2["name"].upper()

                # ISS module families — all physically docked to same station
                ISS_TOKENS = ["NAUKA","ZARYA","ZVEZDA","POISK","PIRS","RASSVET",
                              "HARMONY","UNITY","DESTINY","QUEST","TRANQUILITY",
                              "LEONARDO","CUPOLA","BIGELOW","PROGRESS","SOYUZ",
                              "ISS (","ISS("]
                CSS_TOKENS = ["TIANZHOU","SHENZHOU","TIANHE","WENTIAN","MENGTIAN","CSS"]

                def in_family(n, tokens):
                    return any(t in n for t in tokens)

                # Both in ISS family → docked modules
                if in_family(name1, ISS_TOKENS) and in_family(name2, ISS_TOKENS) and min_d < 5.0:
                    continue
                # Both in CSS family → docked modules
                if in_family(name1, CSS_TOKENS) and in_family(name2, CSS_TOKENS) and min_d < 5.0:
                    continue
                # Same constellation (Starlink, OneWeb) very close → station-keeping, not collision
                for constel in ["STARLINK","ONEWEB","IRIDIUM"]:
                    if constel in name1 and constel in name2 and min_d < 2.0:
                        continue

                conj = {
                    "sat1_id":    o1["norad_id"],
                    "sat1_name":  o1["name"],
                    "sat1_type":  o1["object_type"],
                    "sat2_id":    o2["norad_id"],
                    "sat2_name":  o2["name"],
                    "sat2_type":  o2["object_type"],
                    "miss_distance":             min_d,
                    "tca_minutes_from_now":      tca_min,
                    "probability_of_collision":  pc,
                    "risk_level":                risk,
                    "tca_idx":                   tca_idx,
                    "sat1_path":                 o1["positions"].tolist(),
                    "sat2_path":                 o2["positions"].tolist(),
                    "sat1_pos_at_tca":           o1["positions"][tca_idx].tolist(),
                    "sat2_pos_at_tca":           o2["positions"][tca_idx].tolist(),
                    "sat1_orbital":              get_orbital_elements(o1["tle1"], o1["tle2"]),
                    "sat2_orbital":              get_orbital_elements(o2["tle1"], o2["tle2"]),
                    "relative_velocity":         round(float(np.linalg.norm(
                        o1["positions"][tca_idx] - o1["positions"][max(0,tca_idx-1)] -
                        (o2["positions"][tca_idx] - o2["positions"][max(0,tca_idx-1)])
                    ) / (CONFIG["TIME_STEP_MINUTES"]*60)), 3),
                    "is_synthetic":              False,
                    "is_geo":                    is_geo_pair,
                    "orbit_regime":              "GEO" if is_geo_pair else "LEO",
                    "ml_confidence":             ml_confidence,
                }
                conjunctions.append(conj)
                try:
                    log_conjunction(conj)
                except:
                    pass

    print(f"  Checked {pairs_checked} altitude-compatible pairs → {len(conjunctions)} conjunctions found")
    return sorted(conjunctions, key=lambda x: x["miss_distance"])

# ============================================================
# PROBABILITY OF COLLISION  (Foster 1992, simplified)
# ============================================================

def compute_pc(miss_km, sigma_km=1.0, hard_body_km=0.02):
    """
    1D Gaussian approximation of Pc.
    sigma: combined position uncertainty (typical LEO TLE error ≈ 1 km)
    hard_body: combined hard-body radius (satellites ≈ 10-20 m)
    """
    if sigma_km <= 0:
        return 0.0
    x = miss_km / sigma_km
    # Area under Gaussian within hard-body radius
    pc = (hard_body_km / sigma_km) * math.exp(-0.5 * x * x) * math.sqrt(2 * math.pi) * 0.5
    return min(float(pc), 1.0)

def classify_risk(miss_km, pc):
    # Don't classify physically impossible near-zero distances as real risks
    if miss_km < 0.01:              return "LOW"   # same object / docked
    if miss_km < 1.0  or pc > 1e-4: return "CRITICAL"
    if miss_km < 5.0  or pc > 1e-5: return "HIGH"
    if miss_km < 20.0 or pc > 1e-6: return "MEDIUM"
    return "LOW"

# ============================================================
# GA MANEUVER PLANNER
# ============================================================

# Module-level state for PyGAD's fitness function
_ga_state = {"sat": None, "deb": None, "step": 0}

def _fitness(ga_instance, solution, solution_idx):
    try:
        dv   = np.array(solution, dtype=np.float64)
        fuel = float(np.linalg.norm(dv))

        sat = _ga_state["sat"]
        deb = _ga_state["deb"]

        # Guard: paths must be 2D arrays with shape (N, 3)
        if sat is None or deb is None:
            return -1e9
        if sat.ndim != 2 or deb.ndim != 2:
            return -1e9
        if sat.shape[0] == 0 or deb.shape[0] == 0:
            return -1e9

        new_pos = sat.copy()
        m = _ga_state["step"]
        n = len(new_pos)
        dt_per_step = CONFIG["TIME_STEP_MINUTES"] * 60.0

        for k in range(m, n):
            dt = (k - m) * dt_per_step
            new_pos[k] = new_pos[k] + dv * dt

        # Align lengths
        end = min(len(new_pos), len(deb))
        if end <= m:
            return -1e9

        diff  = new_pos[m:end] - deb[m:end]
        dists = np.linalg.norm(diff, axis=1)
        if len(dists) == 0:
            return -1e9

        min_d = float(np.min(dists))
        return min_d * 10.0 - fuel * 500.0
    except Exception:
        return -1e9

def plan_maneuver(sat_path, deb_path, tca_idx):
    """
    Run GA to find optimal avoidance delta-v.
    Maneuver is applied 30 minutes (6 steps) before closest approach.
    Returns dict with delta_v, safe_path, and metadata.
    """
    # Guard against empty paths (CDM conjunctions with no matching TLE)
    if not sat_path or not deb_path:
        raise ValueError("Cannot plan maneuver: orbit paths are empty. "
                         "No matching TLE found for this CDM object.")

    sat = np.array(sat_path, dtype=np.float64)
    deb = np.array(deb_path, dtype=np.float64)

    # Ensure 2D shape (N, 3)
    if sat.ndim != 2 or sat.shape[1] != 3:
        raise ValueError(f"Invalid sat_path shape: {sat.shape}")
    if deb.ndim != 2 or deb.shape[1] != 3:
        raise ValueError(f"Invalid deb_path shape: {deb.shape}")

    # Align lengths
    min_len = min(len(sat), len(deb))
    sat = sat[:min_len]
    deb = deb[:min_len]

    m = max(0, min(tca_idx - 6, min_len - 10))

    _ga_state["sat"] = sat
    _ga_state["deb"] = deb
    _ga_state["step"] = m

    lim = CONFIG["GA_DV_LIMIT"]
    ga = pygad.GA(
        num_generations=CONFIG["GA_GENERATIONS"],
        num_parents_mating=6,
        fitness_func=_fitness,
        sol_per_pop=CONFIG["GA_POPULATION"],
        num_genes=3,
        gene_space={"low": -lim, "high": lim},
        parent_selection_type="tournament",
        crossover_type="scattered",
        mutation_type="adaptive",
        mutation_percent_genes=[25, 10],
        stop_criteria=["saturate_20"],
        suppress_warnings=True,
    )
    ga.run()
    best, best_fitness, _ = ga.best_solution()

    # Build safe path
    dv = np.array(best, dtype=np.float64)
    dt_per_step = CONFIG["TIME_STEP_MINUTES"] * 60.0
    safe = sat.copy()
    for k in range(m, len(safe)):
        dt = (k - m) * dt_per_step
        safe[k] = safe[k] + dv * dt

    # Verify improvement
    new_dists = np.linalg.norm(safe - deb, axis=1)
    new_min = float(np.min(new_dists))

    return {
        "delta_v":           best.tolist(),
        "delta_v_magnitude": float(np.linalg.norm(best)),
        "apply_at_minutes":  m * CONFIG["TIME_STEP_MINUTES"],
        "achieved_safe_dist": new_min,
        "safe_path":         safe.tolist(),
    }

# ============================================================
# DATA PIPELINE  (runs in background thread)
# ============================================================

def enrich_cdm_paths(conjunctions, tles):
    """
    For CDM-sourced conjunctions, find matching TLEs and generate
    orbit paths for globe visualisation. CDMs don't include paths.
    """
    # Build lookup: norad_id -> tle dict
    tle_lookup = {t["norad_id"]: t for t in tles}
    time_grid  = get_time_grid(CONFIG["PROPAGATION_HOURS"], CONFIG["TIME_STEP_MINUTES"])

    enriched = []
    for conj in conjunctions:
        c = dict(conj)
        s1 = tle_lookup.get(c["sat1_id"])
        s2 = tle_lookup.get(c["sat2_id"])

        p1 = propagate(s1["tle1"], s1["tle2"], time_grid) if s1 else None
        p2 = propagate(s2["tle1"], s2["tle2"], time_grid) if s2 else None

        if p1 is not None:
            c["sat1_path"] = p1.tolist()
            # Use orbital elements from TLE for accuracy
            c["sat1_orbital"] = get_orbital_elements(s1["tle1"], s1["tle2"])
            tca_idx = min(c["tca_idx"], len(p1)-1)
            c["sat1_pos_at_tca"] = p1[tca_idx].tolist()

        if p2 is not None:
            c["sat2_path"] = p2.tolist()
            c["sat2_orbital"] = get_orbital_elements(s2["tle1"], s2["tle2"])
            tca_idx = min(c["tca_idx"], len(p2)-1)
            c["sat2_pos_at_tca"] = p2[tca_idx].tolist()

        enriched.append(c)
    return enriched


class Pipeline:
    def __init__(self):
        self.conjunctions   = []
        self.objects        = []
        self.last_update    = None
        self.is_loading     = True
        self.stats = {
            "objects_tracked": 0,
            "conjunctions_found": 0,
            "pairs_checked": 0,
            "scan_duration_s": 0,
            "data_source": "Initializing...",
        }

    def refresh(self):
        self.is_loading = True
        t0 = time.time()
        print("\n=== PIPELINE: Starting data refresh ===")

        # ── Try Space-Track first ──
        st_user = CONFIG.get("SPACETRACK_USER","").strip()
        st_pass = CONFIG.get("SPACETRACK_PASS","").strip()
        using_spacetrack = bool(st_user and st_pass)

        conjunctions = []
        data_source  = "Celestrak.org"

        if using_spacetrack:
            print("  Space-Track credentials found — fetching official CDMs...")
            cdms = fetch_cdms_from_spacetrack()
            if cdms:
                # Filter out None results
                conjunctions = [c for c in cdms if c is not None]
                data_source  = "Space-Track.org (CDMs)"
                print(f"  Using {len(conjunctions)} official CDMs")

                # Also get better TLEs from Space-Track for path generation
                st_tles = fetch_tles_from_spacetrack()
                if st_tles:
                    tles = st_tles
                    print(f"  Using Space-Track TLEs: {len(tles)} objects")
                else:
                    tles = fetch_tles(CONFIG["MAX_OBJECTS"])
            else:
                print("  No CDMs returned — falling back to Celestrak screening")
                using_spacetrack = False

        if not using_spacetrack:
            tles = fetch_tles(CONFIG["MAX_OBJECTS"])
            data_source = "Celestrak.org"

        # Always store TLEs for globe display
        store_tles(tles)
        self.objects = tles

        # ── If no CDMs, run our own conjunction screening ──
        if not conjunctions:
            time_grid = get_time_grid(CONFIG["PROPAGATION_HOURS"], CONFIG["TIME_STEP_MINUTES"])
            conjunctions = screen_conjunctions(tles, time_grid, CONFIG["CONJUNCTION_THRESHOLD"])

        # ── For CDM-sourced conjunctions, generate paths from TLEs ──
        if using_spacetrack and conjunctions:
            print("  Generating orbit paths for CDM conjunctions...")
            conjunctions = enrich_cdm_paths(conjunctions, tles)

        # ── Fallback synthetic demo ──
        if not conjunctions:
            print("  No conjunctions found — injecting synthetic demo event")
            syn = _synthetic_conjunction(tles)
            if syn:
                conjunctions = [syn]

        self.conjunctions = conjunctions
        self.last_update  = datetime.now(timezone.utc).isoformat()
        elapsed = round(time.time() - t0, 1)

        self.stats = {
            "objects_tracked":    len(tles),
            "conjunctions_found": len(conjunctions),
            "scan_duration_s":    elapsed,
            "data_source":        data_source,
            "last_updated":       self.last_update,
            "source_type":        "cdm" if using_spacetrack else "screened",
        }
        self.is_loading = False
        print(f"=== PIPELINE: Done in {elapsed}s — {len(conjunctions)} conjunctions ({data_source}) ===\n")

    def start(self):
        def _loop():
            while True:
                self.refresh()
                time.sleep(CONFIG["DATA_REFRESH_SECONDS"])
        threading.Thread(target=_loop, daemon=True).start()

def _synthetic_conjunction(tles):
    """
    Create a believable synthetic conjunction for demo when real data has no close pairs.
    Takes two real objects but artificially brings their paths together.
    """
    if len(tles) < 2:
        return None
    time_grid = get_time_grid(12, 5)
    for i in range(len(tles)):
        for j in range(i+1, min(i+20, len(tles))):
            s = tles[i]; d = tles[j]
            sp = propagate(s["tle1"], s["tle2"], time_grid)
            dp = propagate(d["tle1"], d["tle2"], time_grid)
            if sp is None or dp is None:
                continue

            # Place debris 4 km from satellite at step 48 (4 hours from now)
            tca_idx = 48
            dp[tca_idx] = sp[tca_idx] + np.array([2.5, 1.5, 1.0])

            miss = float(np.linalg.norm(sp[tca_idx] - dp[tca_idx]))
            pc   = compute_pc(miss)
            risk = classify_risk(miss, pc)

            return {
                "sat1_id": s["norad_id"], "sat1_name": s["name"] + " ★DEMO",
                "sat1_type": s["object_type"],
                "sat2_id": d["norad_id"], "sat2_name": d["name"] + " (DEBRIS)",
                "sat2_type": "DEBRIS",
                "miss_distance": miss, "tca_minutes_from_now": tca_idx * 5,
                "probability_of_collision": pc, "risk_level": risk,
                "tca_idx": tca_idx,
                "sat1_path": sp.tolist(), "sat2_path": dp.tolist(),
                "sat1_pos_at_tca": sp[tca_idx].tolist(),
                "sat2_pos_at_tca": dp[tca_idx].tolist(),
                "sat1_orbital": get_orbital_elements(s["tle1"], s["tle2"]),
                "sat2_orbital": get_orbital_elements(d["tle1"], d["tle2"]),
                "relative_velocity": 0.0,
                "is_synthetic": True,
            }
    return None

# ============================================================
# INIT + FLASK ROUTES
# ============================================================

# ── Load ML risk model if available ──
import os as _os
_os.environ.setdefault("MPLBACKEND", "Agg")  # suppress matplotlib GUI
import pickle as _pickle

_ML_MODEL = None
_ML_MODEL_PATH = "risk_model.pkl"

def load_ml_model():
    global _ML_MODEL
    try:
        with open(_ML_MODEL_PATH, "rb") as f:
            _ML_MODEL = _pickle.load(f)
        print(f"  ML Risk Model loaded from {_ML_MODEL_PATH}")
        return True
    except FileNotFoundError:
        print(f"  No ML model found at {_ML_MODEL_PATH} — using formula-based risk.")
        print(f"  Run: python train_model.py  to train the model.")
        return False

def ml_predict_risk(miss_km, pc, tca_min, sat1_name, sat2_name):
    """Use ML model to predict risk if available, else fall back to formula."""
    if _ML_MODEL is None:
        return None
    try:
        import math
        log_pc    = math.log10(max(pc, 1e-15))
        urgency   = max(0, 1 - tca_min / (24*60))
        is_debris = int("DEB" in str(sat1_name).upper() or "DEB" in str(sat2_name).upper())
        feat = [[miss_km, log_pc, urgency, is_debris, min(miss_km/50.0, 1.0)]]
        pred = _ML_MODEL.predict(feat)[0]
        proba = _ML_MODEL.predict_proba(feat)[0]
        labels = {0:"LOW", 1:"MEDIUM", 2:"HIGH", 3:"CRITICAL"}
        return {
            "risk_level":   labels.get(pred, "MEDIUM"),
            "confidence":   round(float(max(proba)) * 100, 1),
            "probabilities": {labels[i]: round(float(p)*100,1) for i,p in enumerate(proba)},
        }
    except Exception as e:
        print(f"  ML predict error: {e}")
        return None

load_ml_model()

init_db()
pipeline = Pipeline()
pipeline.start()

# -- Status --

@app.route("/")
def serve_index():
    from flask import make_response
    resp = make_response(send_from_directory(".", "index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp

@app.route("/api/status")
def api_status():
    return jsonify({
        "online": True,
        "is_loading": pipeline.is_loading,
        "stats": pipeline.stats,
        "last_update": pipeline.last_update,
    })

# -- Objects for globe display (current positions only) --

@app.route("/api/globe")
def api_globe():
    if pipeline.is_loading:
        return jsonify({"status": "loading"}), 202
    objects = []
    for t in pipeline.objects[:200]:   # Cap at 200 for globe performance
        pos = get_current_position(t["tle1"], t["tle2"])
        if pos:
            objects.append({
                "id":    t["norad_id"],
                "name":  t["name"],
                "type":  t["object_type"],
                "pos":   pos,
                "is_geo": t.get("is_geo", False),
            })
    return jsonify({"count": len(objects), "objects": objects})

# -- Conjunction list (no path data) --

@app.route("/api/conjunctions")
def api_conjunctions():
    if pipeline.is_loading:
        return jsonify({"status": "loading", "message": "Initial orbital scan in progress..."}), 202
    slim = []
    for c in pipeline.conjunctions:
        slim.append({k: v for k, v in c.items() if k not in ("sat1_path", "sat2_path")})
    return jsonify({
        "status": "ok",
        "count": len(slim),
        "conjunctions": slim,
        "stats": pipeline.stats,
    })

# -- Full detail + AI maneuver for a single conjunction --

@app.route("/api/conjunction/<int:idx>")
def api_conjunction_detail(idx):
    if idx >= len(pipeline.conjunctions):
        return jsonify({"error": "Not found"}), 404

    conj = pipeline.conjunctions[idx]
    print(f"\nRunning AI maneuver planner for conjunction [{idx}]: "
          f"{conj['sat1_name']} vs {conj['sat2_name']}")

    # Guard: paths must exist before running GA
    if not conj.get("sat1_path") or not conj.get("sat2_path"):
        return jsonify({
            "error": "no_paths",
            "message": ("Orbit paths unavailable — NORAD ID not found in TLE catalog. "
                        "Try selecting a different conjunction."),
            "sat1_name": conj["sat1_name"],
            "sat2_name": conj["sat2_name"],
            "miss_distance": conj["miss_distance"],
            "tca_minutes_from_now": conj["tca_minutes_from_now"],
            "risk_level": conj["risk_level"],
            "probability_of_collision": conj["probability_of_collision"],
            "sat1_orbital": conj.get("sat1_orbital", {}),
            "sat2_orbital": conj.get("sat2_orbital", {}),
            "relative_velocity": conj.get("relative_velocity", 0),
            "sat1_pos_at_tca": conj.get("sat1_pos_at_tca", [0,0,0]),
            "sat2_pos_at_tca": conj.get("sat2_pos_at_tca", [0,0,0]),
            "paths": {"satellite": [], "debris": [], "safe": []},
            "maneuver": None,
            "source_type": pipeline.stats.get("source_type","screened"),
            "data_source": pipeline.stats.get("data_source","Celestrak.org"),
        }), 200

    maneuver = plan_maneuver(
        conj["sat1_path"],
        conj["sat2_path"],
        conj["tca_idx"],
    )

    result = {k: v for k, v in conj.items() if k not in ("sat1_path", "sat2_path", "tca_idx")}
    result["paths"] = {
        "satellite": conj["sat1_path"],
        "debris":    conj["sat2_path"],
        "safe":      maneuver["safe_path"],
    }
    result["maneuver"] = {
        "delta_v":            maneuver["delta_v"],
        "delta_v_magnitude":  maneuver["delta_v_magnitude"],
        "apply_at_minutes":   maneuver["apply_at_minutes"],
        "achieved_safe_dist": maneuver["achieved_safe_dist"],
    }
    result["data_source"] = pipeline.stats.get("data_source", "Celestrak.org")
    result["source_type"] = pipeline.stats.get("source_type", "screened")
    return jsonify(result)

# -- Historical conjunction log --

@app.route("/api/history")
def api_history():
    return jsonify({"history": get_conjunction_history(50)})

# -- Reload ML model (call after training) --

@app.route("/api/reload_model")
def api_reload_model():
    success = load_ml_model()
    return jsonify({"status": "ok" if success else "no_model",
                    "model_loaded": _ML_MODEL is not None})

# -- Manual rescan trigger --

@app.route("/api/rescan")
def api_rescan():
    def _bg():
        pipeline.refresh()
    threading.Thread(target=_bg, daemon=True).start()
    return jsonify({"status": "rescan_started"})

# -- Risk summary for dashboard chart --

@app.route("/api/risk_summary")
def api_risk_summary():
    if pipeline.is_loading:
        return jsonify({"status": "loading"}), 202
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for c in pipeline.conjunctions:
        counts[c.get("risk_level","LOW")] = counts.get(c.get("risk_level","LOW"),0) + 1
    closest = pipeline.conjunctions[0] if pipeline.conjunctions else None
    return jsonify({
        "counts": counts,
        "total":  len(pipeline.conjunctions),
        "closest_miss": closest["miss_distance"] if closest else None,
        "closest_names": f"{closest['sat1_name']} / {closest['sat2_name']}" if closest else None,
    })

# -- Stats --

@app.route("/api/stats")
def api_stats():
    return jsonify(pipeline.stats)

# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ORBITAL WATCH SYSTEM  |  http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, port=5000, threaded=True)
