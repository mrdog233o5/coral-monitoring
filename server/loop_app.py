#!/usr/bin/env python3
"""Closed-loop coral-health monitoring demo (the "whole loop" version).

The upper half of the project (sensor sliders -> LightGBM prediction) already
lives in ``server/app.py`` and stays there as a standalone demo.  This app adds
the lower half of the block diagram and runs the **entire loop** on simulated
live data:

    Measurement  ->  Prediction  ->  Output  ->  Correction
    (5 probes)       (LightGBM)      (CWCHI)      (hard write)
         ^                                          |
         +----------- corrected state --------------+

What the loop does, tick after tick:

* **Measurement** - the five probe parameters are read from the simulated reef
  (this demo fabricates a plausible reef trajectory with ``fabricate.py``).
* **Prediction** - the trained LightGBM model (``model.txt``) takes the five
  measured probe values plus the loop-carried health index and predicts the
  next daily CoralWatch health index (CWCHI, 0-6).
* **Output** - the predicted CWCHI becomes the new loop state.
* **Correction** - instead of a Kalman filter (future work), this demo does a
  **hard write**: the diver's manually typed CWCHI measurement is written
  straight into the loop state.  It is *not* applied automatically: the loop
  keeps running on predictions until the user presses the button.

Endpoints
---------
``GET  /api/state``          snapshot: loop state, sensors, history, events
``POST /api/correct``        ``{"measured": 3.42}`` hard-set state <- measured
``POST /api/control``        ``{"running": true, "speed": 1.5, "reset": false}``
``GET  /api/diver-sample``   helper for the demo: simulated diver reading

Run it from the repository root with the project virtualenv::

    .venv/bin/python server/loop_app.py
    # then open http://127.0.0.1:8001

Only ``lightgbm`` is required (plus ``pandas`` for the training metrics
readout, which is optional).  The reef simulation itself is standard library.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import lightgbm as lgb

from fabricate import (
    _normal_cdf,
    _simulate_health_series,
    make_profile,
    simulate_sensor_series,
)

# --------------------------------------------------------------------------- #
# Paths, schema and presentation metadata
# --------------------------------------------------------------------------- #

SERVER_DIR = Path(__file__).resolve().parent
WEB_DIR = SERVER_DIR / "loop_web"
DEFAULT_MODEL = SERVER_DIR / "model.txt"
DEFAULT_CSV = SERVER_DIR / "data" / "coral_health_training.csv"

HEALTH_MIN = 0.0
HEALTH_MAX = 6.0

# The five probe parameters, in the same order the model was trained on.
# ``sim_key`` is the key used by fabricate.simulate_sensor_series().
FEATURES = [
    {
        "key": "dissolved_oxygen",
        "sim_key": "dissolved_oxygen_mg_l",
        "label": "Dissolved oxygen",
        "unit": "mg/L",
        "decimals": 2,
    },
    {
        "key": "ph",
        "sim_key": "ph",
        "label": "pH",
        "unit": "",
        "decimals": 3,
    },
    {
        "key": "temperature",
        "sim_key": "temperature_c",
        "label": "Water temperature",
        "unit": "°C",
        "decimals": 2,
    },
    {
        "key": "zooxanthellae",
        "sim_key": "zooxanthellae_million_cells_cm2",
        "label": "Zooxanthellae density",
        "unit": "million cells/cm²",
        "decimals": 2,
    },
    {
        "key": "nitrate",
        "sim_key": "nitrate_umol_l",
        "label": "Nitrate",
        "unit": "µmol/L",
        "decimals": 2,
    },
]

TARGET_LABEL = "CoralWatch health index (CWCHI)"

# Smoothness of the simulated reef.  The latent reef condition is a very slow
# AR(1) process and the probes only carry gentle noise, so consecutive loop
# ticks differ by realistic day-to-day amounts instead of large jumps.  Tune
# with the --condition-phi / --noise-scale CLI flags if you want it smoother or
# choppier.
DEFAULT_CONDITION_PHI = 0.997
DEFAULT_SENSOR_NOISE_SCALE = 0.3


# --------------------------------------------------------------------------- #
# Reef trajectory: plausible measured sensors + hidden "true" health
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Frame:
    """One simulated day at one reef site."""

    site: int
    day: int
    sensors: tuple[float, ...]  # order matches FEATURES
    true_health: float


def _slow_condition_series(days: int, rng: random.Random, phi: float) -> list[float]:
    """Latent reef condition in [0, 1] that changes only slowly day to day.

    Same AR(1) process as the training-data generator, but with a much larger
    persistence (phi close to 1) so the reef does not lurch between states.
    """
    latent = rng.gauss(0.0, 1.0)
    innovation = math.sqrt(1.0 - phi * phi)
    conditions: list[float] = []
    for _ in range(days):
        latent = phi * latent + rng.gauss(0.0, innovation)
        conditions.append(_normal_cdf(latent))
    return conditions


def build_trajectory(
    seed: int,
    sites: int,
    days: int,
    condition_phi: float = DEFAULT_CONDITION_PHI,
    sensor_noise_scale: float = DEFAULT_SENSOR_NOISE_SCALE,
) -> list[Frame]:
    """Fabricate a long, model-compatible reef trajectory to replay live.

    The trajectory follows the same physics as ``fabricate.py``: probes are
    noisy observations of a latent condition and the true health index follows
    the trained inertia dynamics.  Two differences keep the live loop calm and
    believable:

    * the latent condition uses a very slow AR(1) process, so the reef trends
      gradually instead of jumping around every tick;
    * the probe noise is scaled down, so readings move by realistic day-to-day
      amounts (see ``DEFAULT_CONDITION_PHI`` / ``DEFAULT_SENSOR_NOISE_SCALE``).

    The health index keeps its full 0-6 range because the stationary condition
    process is approximately uniform; no rank calibration is needed, which
    avoids stretching slow moves into visible jumps.
    """
    frames: list[Frame] = []
    start_date = dt.date(2024, 1, 1)

    for site_index in range(sites):
        rng = random.Random(seed + 1009 * site_index)
        profile = make_profile(rng, site_index)
        if sensor_noise_scale != 1.0:
            profile = replace(
                profile,
                sensor_noise_scale=profile.sensor_noise_scale * sensor_noise_scale,
            )

        conditions = _slow_condition_series(days, rng, condition_phi)
        readings = simulate_sensor_series(
            profile, conditions, start_date, random.Random(seed + 2003 * site_index + 11)
        )
        health = _simulate_health_series(conditions, random.Random(seed + 4001 * site_index + 17))

        for day_index in range(days):
            reading = readings[day_index]
            frames.append(
                Frame(
                    site=site_index,
                    day=day_index,
                    sensors=tuple(float(reading[feature["sim_key"]]) for feature in FEATURES),
                    true_health=float(health[day_index]),
                )
            )
    return frames


def _round(value: float, ndigits: int = 4) -> float:
    return round(value, ndigits)


def _clamp_health(value: float) -> float:
    return min(HEALTH_MAX, max(HEALTH_MIN, value))


def load_model(path: Path) -> lgb.Booster:
    if not path.exists():
        raise SystemExit(
            f"model file not found: {path}\n"
            "Train the model with server/model.ipynb first (it saves model.txt)."
        )
    return lgb.Booster(model_file=str(path))


def training_metrics(model: lgb.Booster, csv_path: Path) -> dict[str, object] | None:
    """In-sample MAE / R² of the trained model, for the footer readout."""
    if not csv_path.exists():
        return None
    try:
        import numpy as np
        import pandas as pd

        frame = pd.read_csv(csv_path)
        feature_columns = [
            "prev_dissolved_oxygen_mg_l",
            "prev_ph",
            "prev_temperature_c",
            "prev_zooxanthellae_million_cells_cm2",
            "prev_nitrate_umol_l",
            "prev_coral_health_index",
        ]
        target = frame["coral_health_index"].to_numpy(dtype=float)
        features = frame[feature_columns].to_numpy(dtype=float)
        predictions = np.asarray(model.predict(features), dtype=float)
        errors = predictions - target
        denominator = float(np.sum((target - target.mean()) ** 2))
        return {
            "mae": round(float(np.mean(np.abs(errors))), 4),
            "r2": round(1.0 - float(np.sum(errors**2)) / denominator, 4) if denominator else None,
            "rows": int(len(frame)),
        }
    except Exception as exc:  # noqa: BLE001 - optional readout, never fatal
        print(f"[loop] could not compute training metrics: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# The live loop
# --------------------------------------------------------------------------- #

class LoopSim:
    """Background simulation of the measurement -> prediction -> correction loop.

    The only loop-carried *state* is the CoralWatch health index; the five probe
    parameters are fresh measurements every tick.  A correction (hard write)
    overwrites that state with the diver's measured value; the loop otherwise
    keeps feeding predictions forward.
    """

    HISTORY_LIMIT = 1200
    EVENT_LIMIT = 60

    def __init__(
        self,
        model: lgb.Booster,
        frames: list[Frame],
        sensor_ranges: dict[str, tuple[float, float]],
        metrics: dict[str, object] | None,
        interval: float = 1.0,
    ) -> None:
        self._model = model
        self._frames = frames
        self._sensor_ranges = sensor_ranges
        self._metrics = metrics
        self._lock = threading.RLock()
        self._base_interval = max(0.15, float(interval))

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="coral-loop", daemon=True)

        self._running = True
        self._speed = 1.0
        self._interval = self._base_interval
        self._next_at = time.monotonic()
        self._sites = len({frame.site for frame in frames})

        self._reset_locked(log_event=False)

    # -- lifecycle -------------------------------------------------------- #

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    # -- internal state ---------------------------------------------------- #

    def _reset_locked(self, log_event: bool = True) -> None:
        first = self._frames[0]
        self._cursor = 0
        self._tick = 0
        self._state = _round(first.true_health, 3)
        self._source = "init"
        self._last_prediction: float | None = None
        self._true_health = first.true_health
        self._sensors = list(first.sensors)
        self._corrections = 0
        self._last_correction: dict[str, object] | None = None
        self._ticks_since_correction = 0
        self._error_sum = 0.0
        self._error_count = 0
        self._history: deque[dict[str, object]] = deque(maxlen=self.HISTORY_LIMIT)
        self._history.append(
            {
                "t": 0,
                "state": self._state,
                "true": _round(self._true_health, 3),
                "corrected": False,
                "init": True,
            }
        )
        self._events: deque[dict[str, object]] = deque(maxlen=self.EVENT_LIMIT)
        if log_event:
            self._events.append(
                {
                    "tick": 0,
                    "kind": "reset",
                    "text": f"Loop reset - state initialised from first diver transect ({self._state:.2f})",
                }
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                running = self._running
                interval = self._interval
            if running:
                now = time.monotonic()
                if now >= self._next_at:
                    self._step()
                    with self._lock:
                        self._next_at = time.monotonic() + self._interval
                remaining = self._next_at - time.monotonic()
                time.sleep(min(0.02, max(0.001, remaining)))
            else:
                with self._lock:
                    self._next_at = time.monotonic()
                time.sleep(0.05)

    def _step(self) -> None:
        with self._lock:
            previous = self._frames[self._cursor]
            next_index = (self._cursor + 1) % len(self._frames)
            following = self._frames[next_index]
            tick = self._tick + 1

            if following.site != previous.site:
                # New simulated reef site: re-initialise the state from the
                # first diver transect, like swapping to a new deployment.
                state = _round(following.true_health, 3)
                self._source = "init"
                self._last_prediction = None
                self._ticks_since_correction = 0
                self._error_sum = 0.0
                self._error_count = 0
                self._events.append(
                    {
                        "tick": tick,
                        "kind": "init",
                        "text": (
                            f"New reef site {following.site + 1} - state initialised from "
                            f"first diver transect ({state:.2f})"
                        ),
                    }
                )
                corrected = True
                init = True
            else:
                inputs = list(previous.sensors) + [self._state]
                prediction = _clamp_health(float(self._model.predict([inputs])[0]))
                state = _round(prediction, 3)
                self._source = "prediction"
                self._last_prediction = state
                corrected = False
                init = False

            self._cursor = next_index
            self._tick = tick
            self._state = state
            self._true_health = following.true_health
            self._sensors = list(following.sensors)
            self._ticks_since_correction += 1
            self._error_sum += abs(state - following.true_health)
            self._error_count += 1
            self._history.append(
                {
                    "t": tick,
                    "state": state,
                    "true": _round(following.true_health, 3),
                    "corrected": corrected,
                    "init": init,
                }
            )

    # -- public API -------------------------------------------------------- #

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            frame = self._frames[self._cursor]
            error = self._state - frame.true_health
            mean_abs_error = self._error_sum / self._error_count if self._error_count else None
            sensors = []
            for index, feature in enumerate(FEATURES):
                low, high = self._sensor_ranges[feature["key"]]
                sensors.append(
                    {
                        "key": feature["key"],
                        "label": feature["label"],
                        "unit": feature["unit"],
                        "decimals": feature["decimals"],
                        "value": _round(self._sensors[index], 3),
                        "min": low,
                        "max": high,
                    }
                )
            return {
                "tick": self._tick,
                "running": self._running,
                "speed": self._speed,
                "interval": _round(self._interval, 3),
                "site": frame.site + 1,
                "sites": self._sites,
                "day": frame.day + 1,
                "state": {
                    "cwchi": self._state,
                    "source": self._source,
                    "ticks_since_correction": self._ticks_since_correction,
                    "corrections": self._corrections,
                },
                "last_prediction": self._last_prediction,
                "true_health": _round(self._true_health, 3),
                "error": _round(error, 3),
                "mean_abs_error": _round(mean_abs_error, 3) if mean_abs_error is not None else None,
                "sensors": sensors,
                "history": list(self._history)[-400:],
                "last_correction": self._last_correction,
                "events": list(self._events)[-12:][::-1],
                "model": {
                    "trees": int(self._model.num_trees()),
                    "metrics": self._metrics,
                },
            }

    def correct(self, measured: float) -> dict[str, object]:
        """Hard write: loop state <- diver's measured CWCHI (no Kalman yet)."""
        with self._lock:
            before = self._state
            state = _round(_clamp_health(float(measured)), 3)
            self._state = state
            self._source = "measurement"
            self._corrections += 1
            self._last_correction = {
                "tick": self._tick,
                "measured": state,
                "before": before,
            }
            self._ticks_since_correction = 0
            self._error_sum = 0.0
            self._error_count = 0
            if self._history and self._history[-1]["t"] == self._tick:
                self._history[-1]["state"] = state
                self._history[-1]["corrected"] = True
            else:
                self._history.append(
                    {
                        "t": self._tick,
                        "state": state,
                        "true": _round(self._true_health, 3),
                        "corrected": True,
                        "init": False,
                    }
                )
            self._events.append(
                {
                    "tick": self._tick,
                    "kind": "correction",
                    "text": f"Hard write - state set to measured CWCHI {state:.2f} (was {before:.2f})",
                }
            )
            return self.snapshot()

    def control(
        self,
        running: bool | None = None,
        speed: float | None = None,
        reset: bool = False,
    ) -> dict[str, object]:
        with self._lock:
            if reset:
                self._reset_locked()
                self._next_at = time.monotonic()
            if speed is not None:
                self._speed = min(8.0, max(0.25, float(speed)))
                self._interval = self._base_interval / self._speed
            if running is not None and bool(running) != self._running:
                self._running = bool(running)
                self._next_at = time.monotonic()
                self._events.append(
                    {
                        "tick": self._tick,
                        "kind": "control",
                        "text": "Loop resumed" if self._running else "Loop paused",
                    }
                )
            return self.snapshot()

    def diver_sample(self, noise: float = 0.12) -> dict[str, object]:
        """Simulation helper: what a diver's manual check might read today."""
        with self._lock:
            reading = _clamp_health(self._true_health + random.gauss(0.0, max(0.0, noise)))
            return {"measured": _round(reading, 2)}


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #

class LoopHandler(SimpleHTTPRequestHandler):
    server_version = "CoralLoopDemo/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    # -- helpers ----------------------------------------------------------- #

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ValueError("invalid Content-Length") from None
        if length <= 0:
            raise ValueError("empty request body")
        if length > 100_000:
            raise ValueError("request body too large")
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from None
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    @staticmethod
    def _number(payload: dict[str, object], key: str, *, required: bool = True) -> float | None:
        if key not in payload:
            if required:
                raise ValueError(f"missing field: {key}")
            return None
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            raise ValueError(f"non-numeric field: {key}") from None
        if not math.isfinite(value):
            raise ValueError(f"non-finite field: {key}")
        return value

    @staticmethod
    def _boolean(payload: dict[str, object], key: str) -> bool | None:
        if key not in payload:
            return None
        return bool(payload[key])

    # -- routes ------------------------------------------------------------ #

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.startswith("/api/"):
            route, _, query = self.path.partition("?")
            if route == "/api/state":
                self._send_json(loop_sim.snapshot())
                return
            if route == "/api/diver-sample":
                noise = 0.12
                for part in query.split("&"):
                    key, _, value = part.partition("=")
                    if key == "noise":
                        try:
                            noise = max(0.0, min(0.6, float(value)))
                        except ValueError:
                            pass
                self._send_json(loop_sim.diver_sample(noise))
                return
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        if self.path in ("/", ""):
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path == "/api/correct":
            try:
                payload = self._read_json()
                measured = self._number(payload, "measured")
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            assert measured is not None
            snapshot = loop_sim.correct(measured)
            self._send_json({"ok": True, **snapshot})
            return

        if self.path == "/api/control":
            try:
                payload = self._read_json()
                running = self._boolean(payload, "running")
                speed = self._number(payload, "speed", required=False)
                reset = self._boolean(payload, "reset") or False
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            snapshot = loop_sim.control(running=running, speed=speed, reset=reset)
            self._send_json({"ok": True, **snapshot})
            return

        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write(f"[loop-web] {self.address_string()} - {fmt % args}\n")


# Filled in by main(); declared here so the handler can reference them.
loop_sim: LoopSim
model: lgb.Booster


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    global loop_sim, model

    parser = argparse.ArgumentParser(
        description="Serve the closed-loop coral-health monitoring demo page.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="port (default: 8001)")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL, help=f"model file (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_CSV,
        help=f"training CSV for metrics readout (default: {DEFAULT_CSV})",
    )
    parser.add_argument("--seed", type=int, default=7, help="simulation seed (default: 7)")
    parser.add_argument("--sites", type=int, default=12, help="simulated reef sites (default: 12)")
    parser.add_argument("--days", type=int, default=1500, help="days per site (default: 1500)")
    parser.add_argument(
        "--condition-phi",
        type=float,
        default=DEFAULT_CONDITION_PHI,
        help=(
            "reef-condition persistence, closer to 1 = smoother day-to-day change "
            f"(default: {DEFAULT_CONDITION_PHI})"
        ),
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=DEFAULT_SENSOR_NOISE_SCALE,
        help=(
            "multiplier on probe measurement noise, 0 = perfectly calm readings "
            f"(default: {DEFAULT_SENSOR_NOISE_SCALE})"
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="seconds per simulated day at 1x speed (default: 1.0)",
    )
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args(argv)

    if args.sites < 1:
        parser.error("--sites must be >= 1")
    if args.days < 2:
        parser.error("--days must be >= 2")
    if not 0.0 < args.condition_phi < 1.0:
        parser.error("--condition-phi must be in (0, 1)")
    if args.noise_scale < 0.0:
        parser.error("--noise-scale must be >= 0")

    print(f"[loop] loading model: {args.model}")
    model = load_model(args.model)
    metrics = training_metrics(model, args.data)

    print(
        f"[loop] fabricating reef trajectory: {args.sites} sites x {args.days} days "
        f"(condition phi={args.condition_phi}, noise scale={args.noise_scale}) ..."
    )
    frames = build_trajectory(
        args.seed,
        args.sites,
        args.days,
        condition_phi=args.condition_phi,
        sensor_noise_scale=args.noise_scale,
    )
    sensor_ranges = {
        feature["key"]: (
            round(min(frame.sensors[index] for frame in frames), 3),
            round(max(frame.sensors[index] for frame in frames), 3),
        )
        for index, feature in enumerate(FEATURES)
    }
    loop_sim = LoopSim(model, frames, sensor_ranges, metrics, interval=args.interval)
    loop_sim.start()

    url = f"http://{args.host}:{args.port}/"
    try:
        server = ThreadingHTTPServer((args.host, args.port), LoopHandler)
    except OSError as exc:
        loop_sim.close()
        raise SystemExit(f"could not start server on {args.host}:{args.port}: {exc}") from exc

    print(f"closed-loop demo running at {url}")
    print(
        f"model: {args.model} ({model.num_trees()} trees) · "
        f"{len(frames)} simulated days · {args.interval:.2f}s per day"
    )
    if metrics:
        print(f"training metrics: MAE={metrics['mae']}  R2={metrics['r2']}  rows={metrics['rows']}")
    print("press Ctrl+C to stop")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        loop_sim.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
