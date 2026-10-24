#!/usr/bin/env python3
"""Web demo for the CoralWatch coral-health prediction model.

Serves a single-page slider UI and exposes the trained LightGBM model
(``server/model.txt``) through a tiny JSON API, so the page can show a live
prediction while the user drags the sliders.

Run it from the repository root with the project virtualenv::

    .venv/bin/python server/app.py
    # then open http://127.0.0.1:8000

No web framework is required: the server uses only the Python standard
library plus ``lightgbm`` (and ``pandas`` if the training CSV is present, to
expose realistic slider ranges and model metrics).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import lightgbm as lgb

# --------------------------------------------------------------------------- #
# Paths and model schema
# --------------------------------------------------------------------------- #

SERVER_DIR = Path(__file__).resolve().parent
WEB_DIR = SERVER_DIR / "web"
DEFAULT_MODEL = SERVER_DIR / "model.txt"
DEFAULT_CSV = SERVER_DIR / "data" / "coral_health_training.csv"

# Order must match the training CSV (the target is first there, then these).
FEATURE_KEYS = [
    "prev_temperature_c",
    "prev_ph",
    "prev_nitrate_umol_l",
    "prev_par_umol_m2_s",
    "prev_coral_health_index",
]

# Presentation metadata for the sliders.  Min/max/default are filled in from
# the training CSV at startup when it is available.
FEATURE_META = {
    "prev_temperature_c": {
        "label": "Temperature",
        "unit": "°C", "decimals": 1, "step": 0.1,
        "fallback": (24.0, 32.8, 28.4),
    },
    "prev_ph": {
        "label": "pH",
        "unit": "", "decimals": 3, "step": 0.001,
        "fallback": (7.65, 8.27, 7.96),
    },
    "prev_nitrate_umol_l": {
        "label": "Nitrate concentration",
        "unit": "µmol/L", "decimals": 2, "step": 0.01,
        "fallback": (0.15, 8.0, 3.38),
    },
    "prev_par_umol_m2_s": {
        "label": "PAR",
        "unit": "µmol m⁻² s⁻¹", "decimals": 0, "step": 1,
        "fallback": (260, 690, 480),
    },
    "prev_coral_health_index": {
        "label": "Yesterday's health index",
        "unit": "0–6", "decimals": 2, "step": 0.01,
        "fallback": (0.0, 6.0, 3.0),
    },
}

TARGET_MIN = 0.0
TARGET_MAX = 6.0


# --------------------------------------------------------------------------- #
# Startup: load model, build metadata
# --------------------------------------------------------------------------- #

def load_model(path: Path) -> lgb.Booster:
    if not path.exists():
        raise SystemExit(
            f"model file not found: {path}\n"
            "Train the model with server/model.ipynb first (it saves model.txt)."
        )
    return lgb.Booster(model_file=str(path))


def _round_outward(value: float, step: float, direction: int) -> float:
    scaled = value / step
    rounded = math.floor(scaled) * step if direction < 0 else math.ceil(scaled) * step
    return round(rounded, 10)


def build_features_meta(csv_path: Path) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Return slider metadata (ranges from the training data) and model metrics."""
    stats: dict[str, dict[str, float]] = {}
    metrics = None

    if csv_path.exists():
        try:
            import pandas as pd

            frame = pd.read_csv(csv_path)
            missing = [key for key in FEATURE_KEYS if key not in frame.columns]
            if missing:
                raise ValueError(f"training CSV is missing columns: {missing}")
            for key in FEATURE_KEYS:
                column = frame[key].astype(float)
                stats[key] = {
                    "min": float(column.min()),
                    "max": float(column.max()),
                    "mean": float(column.mean()),
                }
            metrics = _training_metrics(frame)
        except ImportError:
            print("[app] pandas not installed; using fallback slider ranges", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - demo server, keep running
            print(f"[app] could not read training CSV ({exc}); using fallback ranges",
                  file=sys.stderr)

    features = []
    for key in FEATURE_KEYS:
        meta = FEATURE_META[key]
        step = float(meta["step"])
        if key in stats:
            low = _round_outward(stats[key]["min"], step, -1)
            high = _round_outward(stats[key]["max"], step, +1)
            default = round(stats[key]["mean"] / step) * step
        else:
            low, high, default = meta["fallback"]
        features.append(
            {
                "key": key,
                "label": meta["label"],
                "unit": meta["unit"],
                "decimals": meta["decimals"],
                "step": step,
                "min": low,
                "max": high,
                "default": round(min(max(default, low), high), 10),
            }
        )
    return features, metrics


def _training_metrics(frame) -> dict[str, object] | None:
    """In-sample MAE / R^2, so the page can show how good the demo model is."""
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        features = frame[FEATURE_KEYS].to_numpy(dtype=float)
    except KeyError:
        return None

    target = frame["coral_health_index"].to_numpy(dtype=float)
    predictions = np.asarray(model.predict(features), dtype=float)
    errors = predictions - target
    mae = float(np.mean(np.abs(errors)))
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum(errors**2)) / denominator if denominator else float("nan")
    return {"mae": round(mae, 4), "r2": round(r2, 4), "rows": int(len(frame))}


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #

class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "CoralHealthDemo/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    # -- helpers ---------------------------------------------------------- #

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
        if length > 1_000_000:
            raise ValueError("request body too large")
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from None
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    # -- routes ----------------------------------------------------------- #

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path == "/api/meta":
            self._send_json(meta_payload)
            return
        if self.path == "/api/health":
            self._send_json({"status": "ok"})
            return
        if self.path in ("/", ""):
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path != "/api/predict":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        values: list[float] = []
        for key in FEATURE_KEYS:
            if key not in payload:
                self._send_json({"error": f"missing field: {key}"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                value = float(payload[key])
            except (TypeError, ValueError):
                self._send_json({"error": f"non-numeric field: {key}"}, HTTPStatus.BAD_REQUEST)
                return
            if not math.isfinite(value):
                self._send_json({"error": f"non-finite field: {key}"}, HTTPStatus.BAD_REQUEST)
                return
            values.append(value)

        prediction = float(model.predict([values])[0])
        prediction = min(TARGET_MAX, max(TARGET_MIN, prediction))
        prev_health = values[FEATURE_KEYS.index("prev_coral_health_index")]
        self._send_json(
            {
                "prediction": round(prediction, 4),
                "prev_health": round(prev_health, 4),
                "change": round(prediction - prev_health, 4),
                "scale": {"min": TARGET_MIN, "max": TARGET_MAX},
            }
        )

    def end_headers(self) -> None:
        # Demo app: always serve fresh HTML/JS while iterating.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # keep the console tidy
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write(f"[web] {self.address_string()} - {fmt % args}\n")


# Filled in by main(); declared here so the handler can reference it.
meta_payload: dict[str, object] = {}
model: lgb.Booster


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    global meta_payload, model

    parser = argparse.ArgumentParser(description="Serve the coral-health prediction demo page.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL, help=f"model file (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_CSV,
        help=f"training CSV used for slider ranges/metrics (default: {DEFAULT_CSV})",
    )
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args(argv)

    model = load_model(args.model)
    features, metrics = build_features_meta(args.data)
    meta_payload = {
        "features": features,
        "model": {
            "num_trees": int(model.num_trees()),
            "num_features": int(model.num_feature()),
            "metrics": metrics,
        },
        "scale": {"min": TARGET_MIN, "max": TARGET_MAX},
    }

    url = f"http://{args.host}:{args.port}/"
    try:
        server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    except OSError as exc:
        raise SystemExit(f"could not start server on {args.host}:{args.port}: {exc}") from exc

    print(f"coral-health demo running at {url}")
    print(f"model: {args.model}  ({model.num_trees()} trees, {model.num_feature()} features)")
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
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
