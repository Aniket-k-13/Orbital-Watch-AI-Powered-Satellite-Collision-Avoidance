#!/usr/bin/env python3
"""
ORBITAL WATCH — ML Risk Classifier
===================================
Trains a Random Forest classifier to predict conjunction risk escalation.

Features used:
  - miss_distance_km
  - probability_of_collision (log scale)
  - relative_velocity_kms
  - sat1_altitude_km, sat2_altitude_km
  - inclination_diff (crossing angle proxy)
  - tca_minutes (urgency)
  - is_debris (object type)
  - orbit_regime (LEO vs GEO)

Target:
  - risk_label: 0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL

Run this AFTER collecting at least a few hours of conjunction data:
  python train_model.py

Output:
  risk_model.pkl  (used by app.py automatically)
"""

import sqlite3
import numpy as np
import pickle
import math
import os
from datetime import datetime

DB_PATH    = "satellite_watch.db"
MODEL_PATH = "risk_model.pkl"

RISK_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
RISK_INV = {v: k for k, v in RISK_MAP.items()}

# ── 1. Load data from SQLite conjunction log ──
def load_training_data():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found. Run app.py first to collect data.")
        return None, None

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT * FROM conjunction_log")
    rows = c.fetchall()
    conn.close()

    if len(rows) < 10:
        print(f"Only {len(rows)} records found. Need at least 10 to train.")
        print("Let the system run for a few hours to collect more data.")
        return None, None

    print(f"Loaded {len(rows)} conjunction records from database")

    X, y = [], []
    for row in rows:
        try:
            # row: (id, sat1_name, sat2_name, miss_dist, pc, risk_level, tca_min, detected_at)
            miss_km    = float(row[3])
            pc         = float(row[4]) if row[4] else 1e-10
            risk_label = RISK_MAP.get(row[5], 0)
            tca_min    = float(row[6]) if row[6] else 60

            # Derived features
            log_pc     = math.log10(max(pc, 1e-15))
            urgency    = max(0, 1 - tca_min / (24 * 60))  # 0-1 scale
            is_debris  = int("DEB" in str(row[1]).upper() or "DEB" in str(row[2]).upper())

            features = [
                miss_km,
                log_pc,
                urgency,
                is_debris,
                min(miss_km / 50.0, 1.0),   # normalized miss distance
            ]
            X.append(features)
            y.append(risk_label)
        except Exception as e:
            continue

    return np.array(X), np.array(y)

# ── 2. Generate synthetic training data if real data is sparse ──
def generate_synthetic_data(n=2000):
    """
    Generate physically-plausible synthetic conjunction records
    for bootstrapping the model before enough real data is collected.
    """
    np.random.seed(42)
    X, y = [], []

    for _ in range(n):
        miss_km   = np.random.exponential(20.0)         # most conjunctions are far
        miss_km   = max(0.1, min(miss_km, 50.0))
        pc        = max(1e-15, 10 ** np.random.uniform(-15, -3))
        tca_min   = np.random.uniform(10, 1440)
        is_debris = np.random.choice([0, 1], p=[0.4, 0.6])

        # Physics-based labeling
        if miss_km < 1.0 or pc > 1e-4:
            label = 3   # CRITICAL
        elif miss_km < 5.0 or pc > 1e-5:
            label = 2   # HIGH
        elif miss_km < 20.0 or pc > 1e-6:
            label = 1   # MEDIUM
        else:
            label = 0   # LOW

        log_pc   = math.log10(max(pc, 1e-15))
        urgency  = max(0, 1 - tca_min / (24*60))

        X.append([miss_km, log_pc, urgency, is_debris, miss_km/50.0])
        y.append(label)

    print(f"Generated {n} synthetic training samples")
    return np.array(X), np.array(y)

# ── 3. Train model ──
def train(X, y):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print(f"Training on {len(X)} samples...")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # Cross-validation
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted")
    print(f"Cross-val F1: {scores.mean():.3f} (+/- {scores.std():.3f})")

    pipeline.fit(X, y)
    return pipeline

# ── 4. Save model ──
def save_model(pipeline):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {MODEL_PATH}")
    print("Restart app.py to use the trained model.")

# ── 5. Main ──
if __name__ == "__main__":
    print("=" * 50)
    print("  ORBITAL WATCH — ML Risk Classifier Training")
    print("=" * 50)

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("Installing scikit-learn...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "scikit-learn", "--quiet"])
        from sklearn.ensemble import RandomForestClassifier

    # Try real data first, fall back to synthetic
    X_real, y_real = load_training_data()

    if X_real is None or len(X_real) < 50:
        print("\nNot enough real data yet — using synthetic training data.")
        print("(This is normal for a new deployment. Re-run after 24h of data collection.)")
        X_syn, y_syn = generate_synthetic_data(3000)
        if X_real is not None and len(X_real) >= 10:
            X = np.vstack([X_real, X_syn])
            y = np.concatenate([y_real, y_syn])
            print(f"Combined: {len(X_real)} real + {len(X_syn)} synthetic = {len(X)} total")
        else:
            X, y = X_syn, y_syn
    else:
        X, y = X_real, y_real

    model = train(X, y)
    save_model(model)
    print("\nDone! The ML model will now enhance risk predictions.")
