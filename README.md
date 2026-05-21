# 🛰️ Orbital Watch — AI-Powered Satellite Collision Avoidance
> Real orbital data · SGP4 propagation · Genetic Algorithm planning · Live 3D globe · Mission Control UI

---

## Quick Start (Windows)
```

1. Double-click  setup_and_run.bat
2. Wait ~60–90 seconds for the orbital scan
3. Open browser → http://localhost:5000

```

## Manual Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
---

## What This System Does


| Layer | What happens |
|---|---|
| **Data** | Fetches live TLEs from Celestrak (stations, active, Iridium debris, Cosmos debris) |
| **Physics** | SGP4 propagates 300 objects × 24h × 5-min steps = 72,000 position vectors |
| **Screening** | Altitude pre-filter + pairwise distance check on all compatible pairs |
| **Pc Math** | Foster 1992 1D Gaussian probability of collision |
| **AI Planner** | Genetic Algorithm (PyGAD) finds optimal 3D delta-v maneuver |
| **Database** | SQLite logs every detected conjunction for historical analysis |
| **Visualisation** | CesiumJS 3D globe with ECI→ECEF conversion, orbit paths, TCA markers |

---

## Full Feature List

### Backend (app.py)
- Live TLE ingestion from 4 Celestrak groups
- SGP4 orbital propagation (python-sgp4)
- Altitude-band pre-filter (avoids brute-force O(n²))
- Conjunction screening with configurable threshold (default 50 km)
- Foster 1992 Probability of Collision
- Risk classification: CRITICAL / HIGH / MEDIUM / LOW
- Orbital elements extraction (inclination, altitude, velocity, period, eccentricity)
- Relative velocity at TCA
- Genetic Algorithm maneuver planner (80 generations, 30 population, 3D Δv)
- Multi-burn point support (burn applied 30 min before TCA)
- SQLite persistence for conjunction history
- Synthetic demo event injection when real data has no close pairs
- Background pipeline thread with hourly refresh
- 7 REST API endpoints

### Frontend (index.html)
- Animated loading screen with orbital matrix grid
- CesiumJS 3D globe with Bing Maps Aerial imagery (darkened)
- ECI → ECEF coordinate conversion with GMST rotation
- Satellite dot cloud (cyan = active, red = debris)
- Orbit path rendering: yellow dashed (initial), red glow (debris), green glow (safe)
- TCA danger sphere with ⚠ label
- BURN point marker with label
- **+/− zoom buttons** (smooth flyTo transitions)
- **⌂ reset view** — returns to full Earth + restarts auto-rotate
- **⟳ auto-rotate toggle** — slow globe spin when idle
- **Filter bar** — ALL / CRITICAL / HIGH / MEDIUM / LOW
- **Risk distribution bar** in topbar (live colour-coded segments)
- **Live TCA countdown** (seconds precision, turns red under 1h)
- **2D conjunction geometry diagram** (canvas, not-to-scale visual)
- **Orbital elements grid** — inclination, altitude, period for both objects
- **Orbital context bars** — altitude, velocity, urgency
- **Conjunction history drawer** (slides up from left panel)
- **📋 Mission Report modal** — full printable/copyable summary
- **Keyboard shortcuts**: +/− zoom · R rotate · ←/→ navigate · Enter plan · M report · Esc reset
- **↺ SCAN button** — triggers fresh scan without server restart
- Auto-selects highest-risk conjunction on load

### API Endpoints
| Endpoint | Description |
|---|---|
| `GET /` | Serves dashboard HTML |
| `GET /api/status` | System health + scan progress |
| `GET /api/globe` | Current positions of all 300 objects |
| `GET /api/conjunctions` | All detected conjunctions (no path data) |
| `GET /api/conjunction/<idx>` | Full detail + AI maneuver plan |
| `GET /api/risk_summary` | Risk counts + closest miss for chart |
| `GET /api/history` | Historical conjunction log (last 50) |
| `GET /api/rescan` | Trigger fresh pipeline run |

---

## Configuration (CONFIG dict in app.py)
| Key | Default | Effect |
|---|---|---|
| MAX_OBJECTS | 300 | More = better coverage, slower scan |
| PROPAGATION_HOURS | 24 | Forecast window |
| TIME_STEP_MINUTES | 5 | Resolution (lower = slower) |
| CONJUNCTION_THRESHOLD | 50 km | Flag pairs closer than this |
| GA_GENERATIONS | 80 | More = better maneuver quality |
| GA_POPULATION | 30 | GA population size |
| GA_DV_LIMIT | 0.05 km/s | Max thrust per axis |
| DATA_REFRESH_SECONDS | 3600 | TLE refresh interval |

---

## Project Structure
```
orbital-watch/
├── app.py                ← Full backend (physics + AI + API)
├── index.html            ← Mission control dashboard
├── requirements.txt      ← Python deps (6 packages)
├── setup_and_run.bat     ← Windows one-click launcher
├── satellite_watch.db    ← SQLite (auto-created)
└── README.md
```
---

## Tech Stack
| Component | Library | Why |
|---|---|---|
| Orbital propagation | python-sgp4 | Industry-standard SGP4/SDP4 |
| Maneuver planning | PyGAD | Genetic algorithm, no GPU needed |
| Backend | Flask | Lightweight, easy to deploy |
| 3D globe | CesiumJS 1.121 | Used by real aerospace firms |
| Database | SQLite | Zero-setup persistent storage |
| Data source | Celestrak | Free, real, reliable |

---

## Known Limitations (important for interviews)
- TLE propagation degrades past ~7 days
- Pc values use 1D Gaussian (not full 6×6 covariance)
- No atmospheric drag correction beyond SGP4 built-in
- Objects under ~10cm not in any public catalog

---
## Portfolio Talking Points
> "Ingests live TLE data for 300+ objects, propagates 24h forward with SGP4,
> screens ~25,000 altitude-compatible pairs for close approaches, computes
> Probability of Collision using Foster 1992, and runs a Genetic Algorithm
> to find an optimal 3D avoidance maneuver — all in a real-time pipeline
> with a mission-control dashboard featuring a live 3D globe, countdown timers,
> geometry diagrams, and exportable mission reports."

Resume keywords: Space Situational Awareness · SGP4 · Orbital Mechanics ·
Genetic Algorithm · Probability of Collision · Real-Time Data Pipeline ·
CesiumJS · Flask · SQLite · Python

---
*MIT License · Built with Python 3.10+ · CesiumJS 1.121 · SGP4 · PyGAD*
