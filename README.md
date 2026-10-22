# Coral Reef Health Monitoring

A synthetic-data demo of the coral-reef monitoring loop from our project diagram:

```
Measurement  →  Prediction  →  Output  →  Correction
(5 probes)      (LightGBM)     (CWCHI)     (Kalman / hard write)
     ↑                                         │
     └────────── corrected state ──────────────┘
```

The repository contains **two separate web demos**:

| Demo | Files | Port | What it shows |
| --- | --- | --- | --- |
| **Upper half (done earlier)** | `server/app.py`, `server/web/index.html` | 8000 | Sliders for yesterday's sensor readings → LightGBM predicts today's CoralWatch health index. Kept untouched as its own demo. |
| **Full loop (new)** | `server/loop_app.py`, `server/loop_web/index.html` | 8001 | The complete closed loop on simulated live sensors, including the Correction stage. |

The correction stage supports **two selectable algorithms** (version 2):

* **Kalman filter** (default) — a 1-D Kalman filter over the CWCHI state. The
  LightGBM prediction is the process model; each diver measurement is fused with
  gain `K = P / (P + R)`. Uncertainty `P` grows while the loop is open-loop and
  shrinks on every update, so the state is corrected smoothly instead of jumping.
  The page plots the ±1σ state-uncertainty band (the card also shows ±2σ).
* **Hard write** — the version 1 behaviour: `state := measured CWCHI`.

In both cases the user types the diver's measured CoralWatch health index (CWCHI)
and presses the button once; nothing is corrected automatically and the loop keeps
running on predictions between button presses. The algorithm can be switched live
in the Correction card.

## Setup

The demos need Python 3 with `lightgbm`, `pandas` and `numpy`:

```bash
python3 -m venv .venv
.venv/bin/pip install lightgbm pandas numpy
```

Everything else uses the Python standard library.

## Run the full-loop demo

```bash
.venv/bin/python server/loop_app.py
# open http://127.0.0.1:8001
```

Options:

```bash
.venv/bin/python server/loop_app.py --port 8001 --seed 7 --sites 12 --days 1500 --interval 1.0 --open
```

* `--seed`, `--sites`, `--days` control the fabricated reef trajectory (same
  generator as the training data, replayed live).
* `--condition-phi` (default `0.997`) and `--noise-scale` (default `0.3`) control
  how calm the simulated reef is. Higher `--condition-phi` and lower
  `--noise-scale` make consecutive ticks change less, so the loop looks like
  realistic day-to-day drift instead of jumping around.
* `--algorithm kalman|hard` selects the initial correction algorithm (default
  `kalman`).
* `--process-noise` (default `0.05`) and `--measurement-noise` (default `0.20`)
  set the Kalman filter's daily process-noise std and diver measurement-noise std
  (both in CWCHI units).
* `--interval` is the wall-clock seconds per simulated day at 1× speed; the page
  also has 0.5× / 2× / 4× speed controls.

### What happens each tick

1. **Measurement** — the five simulated probes produce fresh readings
   (dissolved oxygen, pH, temperature, zooxanthellae density, nitrate).
2. **Prediction** — the trained LightGBM model (`server/model.txt`) takes those
   five readings plus the loop-carried CWCHI state and predicts the next daily
   CWCHI.
3. **Output** — the prediction becomes the new loop state (and is plotted live).
4. **Correction** — only when the user presses the correction button, the diver's
   measured CWCHI is fused into the state using the selected algorithm: a Kalman
   update (`state ← prediction + K·(measured − prediction)`, `K = P/(P+R)`) or a
   hard write (`state := measured`). The **Use live diver read** toggle follows
   the simulated diver reading automatically every tick, so you don't have to
   re-sample before each correction; typing a value switches back to manual and
   the typed value is kept for subsequent corrections. Otherwise the next
   prediction simply continues from the predicted state.

The page also shows the simulated ground truth (dashed line) so you can see the
prediction drift away from the reef and how each correction pulls it back. In
Kalman mode the shaded teal band is the filter's ±1σ state uncertainty (the
correction card also shows the ±2σ value).

### API (used by the page)

```
GET  /api/state                        snapshot: state, sensors, history, events,
                                       current simulated diver_reading
POST /api/correct {"measured": 3.42}   hard-set state ← measured CWCHI
POST /api/control {"running": true,    pause/resume, speed, reset, algorithm
                   "speed": 1.5,
                   "reset": false,
                   "algorithm": "kalman"}
GET  /api/diver-sample?noise=0.12      simulated diver reading helper
```

## Run the upper-half demo (kept separate)

```bash
.venv/bin/python server/app.py --open
# open http://127.0.0.1:8000
```

Both servers can run at the same time.

## Data and model

* `server/fabricate.py` — generates the synthetic coral-reef dataset used for
  training (`server/data/coral_health_training.csv`). The health index has
  inertia: recovery is slow, decline is faster.
* `server/model.ipynb` — trains the LightGBM regressor and writes
  `server/model.txt`.
* `server/model.txt` — the trained model, loaded by both demos.

Regenerate data / retrain:

```bash
.venv/bin/python server/fabricate.py        # writes data/coral_health_training.csv
.venv/bin/jupyter nbconvert --to notebook --execute --inplace server/model.ipynb
```

## Notes and limitations

* All reef data here is **synthetic** — for demonstration and exploration only,
  not a scientific conclusion about real reefs.
* The "true" CWCHI shown in the full-loop demo is part of the simulation; in a
  real deployment only the model prediction and the diver measurements would be
  available.
* The Kalman filter is a deliberately small 1-D filter over the CWCHI state; the
  five probe parameters are treated as direct measurements rather than filter
  states. A full state-space model could be the next step.
* The hard-write algorithm is kept as the v1 baseline so the two correction
  strategies can be compared side by side.
