#!/usr/bin/env python3
"""Fabricate synthetic coral-reef sensor data for training a small model.

The script simulates a set of reef sites that are measured every day by
seabed sensors.  For each site and each day it records the daily averages of

    * dissolved oxygen  (mg/L)
    * pH
    * temperature       (deg C)
    * zooxanthellae concentration (million cells / cm^2)
    * nitrate           (umol/L)
    * coral health index (0 = fully bleached / dead, 6 = pristine reef)

The generated training file has one row per simulated site/day, where the
inputs are the *previous* day's readings (plus the previous day's health index)
and the target is *today's* health index.  The target is fractional because it
is the average over a reef transect.

The health index follows an extended 0-6 CoralWatch-inspired scale.  The real
CoralWatch colour chart scores colonies from 1 (white / bleached) to 6 (dark,
healthy); here the scale is stretched down to 0.  After simulation the health
values are rank-calibrated across the whole dataset, so the target covers 0-6
almost perfectly evenly and a small model sees just as many low-health
examples (0.x, 1.x, ...) as healthy ones.

Site and date are simulation details only and are **not** written to the file.
The CSV contains exactly the output followed by the six inputs, in this order:

    coral_health_index                    OUTPUT (today, 0-6, fractional)
    prev_dissolved_oxygen_mg_l            input (yesterday)
    prev_ph                               input (yesterday)
    prev_temperature_c                    input (yesterday)
    prev_zooxanthellae_million_cells_cm2  input (yesterday)
    prev_nitrate_umol_l                   input (yesterday)
    prev_coral_health_index               input (yesterday, 0-6)

The file is plain CSV, so it can be loaded directly with

    import pandas as pd
    df = pd.read_csv("server/data/coral_health_training.csv")
    y = df.iloc[:, 0]          # target: coral_health_index
    X = df.iloc[:, 1:]         # six input columns

or, for convenience, with ``load_training_data(path)`` from this module.

Health dynamics (why the index moves slowly)
--------------------------------------------
Each site has a slow-moving latent "reef condition" u(t) in [0, 1]: an
autocorrelated Gaussian process mapped through the normal CDF and then
rank-calibrated to be uniform over the dataset.  The five sensor readings are
noisy observations of the same-day condition.  As u falls (reef gets sicker)
temperature and nitrate go up, while pH, dissolved oxygen and zooxanthellae
density go down; the opposite happens as u rises.

The health index is *not* recomputed from scratch every day: it has inertia.
Today's index only moves a limited step toward the level implied by yesterday's
condition::

    h(t) = h(t-1) + step
    step = up_gain   * (target - h(t-1))   if target > h(t-1)   # recovery
    step = down_gain * (target - h(t-1))   if target < h(t-1)   # decline

where ``target = 6 * u(t-1)``.  Recovery is deliberately slow (up_gain 0.18,
capped at +0.22 per day) while decline is faster (down_gain 0.45, capped at
-0.92 per day), mirroring real reefs where bleaching can happen within days
but recovery takes weeks or months.  As a result:

    * yesterday's health index is by far the strongest predictor,
    * the sensor readings only nudge the predicted *change*,
    * the day-to-day moves in the target stay small (typically < 0.45, and at
      most about +0.4 / -1.0 even after the final rank calibration) instead of
      jumping across the whole 0-6 scale in a single day.

The data is synthetic.  It is statistically plausible but must not be used as
scientific evidence.

Usage
-----
    python3 server/fabricate.py                       # defaults: 8 sites x 365 days
    python3 server/fabricate.py --sites 12 --days 730 --seed 7
    python3 server/fabricate.py --out data/train.csv

Rows from one simulated site are consecutive days (site after site, in
chronological order).  If you later split the data into train/validation sets,
be aware that a plain random split can leak neighbouring days into both sides.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

FEATURE_COLUMNS = [
    "prev_dissolved_oxygen_mg_l",
    "prev_ph",
    "prev_temperature_c",
    "prev_zooxanthellae_million_cells_cm2",
    "prev_nitrate_umol_l",
    "prev_coral_health_index",
]

TARGET_COLUMN = "coral_health_index"
# The output (y) goes first, followed by the six inputs (X).
CSV_COLUMNS = [TARGET_COLUMN] + FEATURE_COLUMNS

HEALTH_MIN = 0.0
HEALTH_MAX = 6.0

# Health-index inertia.  Recovery upwards is slow; decline can be faster.
HEALTH_UP_GAIN = 0.18
HEALTH_DOWN_GAIN = 0.45
HEALTH_UP_CAP = 0.22       # max raw daily rise, health-index units
HEALTH_DOWN_CAP = -0.92    # max raw daily fall, health-index units
HEALTH_STEP_NOISE = 0.02

# Persistence of the latent reef-condition process (higher = slower).
CONDITION_PHI = 0.96

DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "coral_health_training.csv"

# Plausible bounds used by the self-check after writing the file.
_BOUNDS = {
    "prev_dissolved_oxygen_mg_l": (2.0, 10.0),
    "prev_ph": (7.4, 8.6),
    "prev_temperature_c": (20.0, 36.0),
    "prev_zooxanthellae_million_cells_cm2": (0.05, 6.0),
    "prev_nitrate_umol_l": (0.0, 8.0),
    "prev_coral_health_index": (HEALTH_MIN, HEALTH_MAX),
    TARGET_COLUMN: (HEALTH_MIN, HEALTH_MAX),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round(value: float, ndigits: int = 3) -> float:
    """Round and normalise -0.0 to 0.0 so the CSV stays pretty."""
    rounded = round(value, ndigits)
    return 0.0 if rounded == 0 else rounded


def _normal_cdf(value: float) -> float:
    """Standard normal CDF, used to turn the latent process into [0, 1]."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _rank_to_unit(values: Sequence[float]) -> list[float]:
    """Map values to (0, 1) by rank, giving an exactly uniform distribution.

    The transform is monotone, so each site's day-to-day ordering and the
    autocorrelation of the series are preserved.
    """
    count = len(values)
    if count == 0:
        return []
    order = sorted(range(count), key=values.__getitem__)
    mapped = [0.0] * count
    for position, index in enumerate(order):
        mapped[index] = (position + 0.5) / count
    return mapped


# --------------------------------------------------------------------------- #
# Site profiles
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SiteProfile:
    """Static characteristics of one simulated reef / sensor station."""

    base_temperature_c: float
    seasonal_amplitude_c: float
    base_ph: float
    base_do_mg_l: float
    nitrate_baseline_umol_l: float
    base_zoox_million_cells_cm2: float
    stress_sensitivity: float
    sensor_noise_scale: float


def make_profile(rng: random.Random, index: int) -> SiteProfile:
    """Create a deterministic, site-specific set of baselines.

    Every third site is slightly warmer / more nutrient-loaded, so the dataset
    mixes pristine and chronically stressed reefs (the condition process still
    pushes both types across the full 0-6 health range).
    """
    stressed = index % 3 == 2
    return SiteProfile(
        base_temperature_c=rng.uniform(26.2, 27.9) + (0.5 if stressed else 0.0),
        seasonal_amplitude_c=rng.uniform(0.4, 0.9),
        base_ph=rng.uniform(7.98, 8.12) - (0.03 if stressed else 0.0),
        base_do_mg_l=rng.uniform(6.2, 7.2) - (0.3 if stressed else 0.0),
        nitrate_baseline_umol_l=rng.uniform(0.15, 0.8) + (0.3 if stressed else 0.0),
        base_zoox_million_cells_cm2=rng.uniform(2.2, 3.4) * (0.9 if stressed else 1.0),
        stress_sensitivity=rng.uniform(0.85, 1.15) * (1.08 if stressed else 1.0),
        sensor_noise_scale=rng.uniform(0.8, 1.2),
    )


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

def simulate_condition_series(days: int, rng: random.Random) -> list[float]:
    """Slow latent reef condition for one site, in [0, 1] (0 = bleached)."""
    latent = rng.gauss(0.0, 1.0)
    innovation = math.sqrt(1.0 - CONDITION_PHI * CONDITION_PHI)
    conditions: list[float] = []
    for _ in range(days):
        latent = CONDITION_PHI * latent + rng.gauss(0.0, innovation)
        conditions.append(_normal_cdf(latent))
    return conditions


def simulate_sensor_series(
    profile: SiteProfile,
    conditions: Sequence[float],
    start: date,
    rng: random.Random,
) -> list[dict[str, float]]:
    """Daily sensor averages derived from the same-day reef condition."""
    sensors: list[dict[str, float]] = []
    noise_scale = profile.sensor_noise_scale

    for offset, condition in enumerate(conditions):
        day = start + timedelta(days=offset)
        season = math.sin(2.0 * math.pi * (day.timetuple().tm_yday - 45) / 365.25)
        stress = _clip((1.0 - condition) * profile.stress_sensitivity, 0.0, 1.2)

        temperature = (
            profile.base_temperature_c
            + profile.seasonal_amplitude_c * season
            + 4.6 * stress
            - 1.2
            + rng.gauss(0.0, 0.35 * noise_scale)
        )
        ph = (
            profile.base_ph
            + 0.10
            - 0.36 * stress
            + rng.gauss(0.0, 0.018 * noise_scale)
        )
        dissolved_oxygen = (
            profile.base_do_mg_l
            + 1.20
            - 3.60 * stress
            + rng.gauss(0.0, 0.22 * noise_scale)
        )
        nitrate = (
            profile.nitrate_baseline_umol_l
            + 0.12
            + 6.0 * stress**1.3
            + rng.gauss(0.0, 0.18 * noise_scale)
        )
        zooxanthellae = (
            profile.base_zoox_million_cells_cm2 * (1.18 - 0.85 * stress)
            + rng.gauss(0.0, 0.08 * noise_scale)
        )

        sensors.append(
            {
                "dissolved_oxygen_mg_l": _clip(dissolved_oxygen, 1.5, 11.0),
                "ph": _clip(ph, 7.4, 8.5),
                "temperature_c": _clip(temperature, 20.0, 36.0),
                "zooxanthellae_million_cells_cm2": _clip(zooxanthellae, 0.05, 6.0),
                "nitrate_umol_l": _clip(nitrate, 0.01, 8.0),
            }
        )
    return sensors


def _simulate_health_series(
    conditions: Sequence[float],
    rng: random.Random,
) -> list[float]:
    """Health index with inertia toward yesterday's condition-implied level."""
    health = [
        _clip(HEALTH_MAX * conditions[0] + rng.gauss(0.0, 0.05), HEALTH_MIN, HEALTH_MAX)
    ]
    for index in range(1, len(conditions)):
        target = HEALTH_MAX * conditions[index - 1]
        difference = target - health[-1]
        if difference > 0.0:
            step = HEALTH_UP_GAIN * difference
        else:
            step = HEALTH_DOWN_GAIN * difference
        step = _clip(step, HEALTH_DOWN_CAP, HEALTH_UP_CAP)
        health.append(
            _clip(
                health[-1] + step + rng.gauss(0.0, HEALTH_STEP_NOISE),
                HEALTH_MIN,
                HEALTH_MAX,
            )
        )
    return health


def build_health_series(
    conditions_by_site: Sequence[Sequence[float]],
    seed: int,
) -> list[list[float]]:
    """Simulate health for every site, then spread it evenly over 0-6.

    The inertia in :func:`_simulate_health_series` keeps day-to-day moves small.
    The final rank calibration only re-scales the values monotonically, so the
    ordering (and therefore the slow dynamics) of each site's series is
    preserved while the dataset-wide target distribution becomes flat.
    """
    raw: list[list[float]] = []
    for index, conditions in enumerate(conditions_by_site):
        rng = random.Random(seed + 4001 * index + 17)
        raw.append(_simulate_health_series(conditions, rng))

    flat = [value for site_health in raw for value in site_health]
    unit = _rank_to_unit(flat)
    if not unit:
        return [[] for _ in raw]

    calibrated: list[list[float]] = []
    cursor = 0
    for site_health in raw:
        length = len(site_health)
        calibrated.append([HEALTH_MAX * value for value in unit[cursor : cursor + length]])
        cursor += length
    return calibrated


# --------------------------------------------------------------------------- #
# Row building and file IO
# --------------------------------------------------------------------------- #

def build_training_rows(
    sensors: Sequence[dict[str, float]],
    health: Sequence[float],
) -> list[dict[str, object]]:
    """One row per day: yesterday's six inputs -> today's health index."""
    rows: list[dict[str, object]] = []
    for index in range(1, len(sensors)):
        previous = sensors[index - 1]
        rows.append(
            {
                TARGET_COLUMN: _round(health[index]),
                "prev_dissolved_oxygen_mg_l": _round(previous["dissolved_oxygen_mg_l"]),
                "prev_ph": _round(previous["ph"]),
                "prev_temperature_c": _round(previous["temperature_c"]),
                "prev_zooxanthellae_million_cells_cm2": _round(
                    previous["zooxanthellae_million_cells_cm2"]
                ),
                "prev_nitrate_umol_l": _round(previous["nitrate_umol_l"]),
                "prev_coral_health_index": _round(health[index - 1]),
            }
        )
    return rows


def write_csv(rows: Iterable[dict[str, object]], columns: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def validate_csv(path: Path, columns: Sequence[str], expected_rows: int) -> None:
    """Re-read the file and check schema, row count, missing values and ranges."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != list(columns):
            raise SystemExit(
                f"self-check failed: unexpected header in {path}: {reader.fieldnames}"
            )
        rows = list(reader)

    if len(rows) != expected_rows:
        raise SystemExit(
            f"self-check failed: expected {expected_rows} rows, found {len(rows)}"
        )

    for line_number, row in enumerate(rows, start=2):
        for column in columns:
            if row[column] is None or row[column] == "":
                raise SystemExit(
                    f"self-check failed: missing value in column {column!r} "
                    f"on line {line_number}"
                )
        for column, (low, high) in _BOUNDS.items():
            if column not in columns:
                continue
            try:
                value = float(row[column])
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"self-check failed: non-numeric value in column {column!r} "
                    f"on line {line_number}"
                ) from exc
            if not low <= value <= high:
                raise SystemExit(
                    f"self-check failed: {column}={value} outside [{low}, {high}] "
                    f"on line {line_number}"
                )


# --------------------------------------------------------------------------- #
# Convenience loader for downstream training scripts
# --------------------------------------------------------------------------- #

def load_training_data(path: str | Path):
    """Load the generated CSV with pandas.

    Returns ``(X, y)`` where ``X`` is the 6-feature DataFrame and ``y`` is the
    target Series.
    """
    import pandas as pd  # imported lazily so data generation needs no pandas

    frame = pd.read_csv(path)
    missing = [column for column in CSV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    frame = frame[CSV_COLUMNS]
    return frame[FEATURE_COLUMNS], frame[TARGET_COLUMN]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic coral-reef dataset for training a small model "
            "that predicts tomorrow's coral health index from yesterday's sensor "
            "averages and yesterday's index.  The target covers the full 0-6 range "
            "and changes slowly from day to day."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"training CSV path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--sites", type=int, default=8, help="number of reef sites (default: 8)")
    parser.add_argument(
        "--days", type=int, default=365, help="days simulated per site (default: 365)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="first simulated date, ISO format (default: 2024-01-01)",
    )
    args = parser.parse_args(argv)

    if args.sites < 1:
        parser.error("--sites must be >= 1")
    if args.days < 2:
        parser.error("--days must be >= 2 (one day is needed to form the first input row)")
    try:
        start = date.fromisoformat(args.start_date)
    except ValueError:
        parser.error(f"--start-date must be ISO format (YYYY-MM-DD), got {args.start_date!r}")

    profiles: list[SiteProfile] = []
    raw_conditions: list[list[float]] = []
    for index in range(args.sites):
        rng = random.Random(args.seed + 1009 * index)
        profiles.append(make_profile(rng, index))
        raw_conditions.append(simulate_condition_series(args.days, rng))

    # Calibrate the condition values across all sites so their distribution is
    # exactly uniform; then turn them into sensors and health.
    flat_conditions = [value for conditions in raw_conditions for value in conditions]
    unit_conditions = _rank_to_unit(flat_conditions)
    conditions_by_site: list[list[float]] = []
    cursor = 0
    for conditions in raw_conditions:
        conditions_by_site.append(unit_conditions[cursor : cursor + len(conditions)])
        cursor += len(conditions)

    sensors_by_site: list[list[dict[str, float]]] = []
    for index, (profile, conditions) in enumerate(zip(profiles, conditions_by_site)):
        rng = random.Random(args.seed + 2003 * index + 11)
        sensors_by_site.append(
            simulate_sensor_series(profile, conditions, start, rng)
        )

    health_by_site = build_health_series(conditions_by_site, args.seed)

    training_rows: list[dict[str, object]] = []
    for sensors, health in zip(sensors_by_site, health_by_site):
        training_rows.extend(build_training_rows(sensors, health))

    write_csv(training_rows, CSV_COLUMNS, args.out)
    expected_rows = args.sites * (args.days - 1)
    validate_csv(args.out, CSV_COLUMNS, expected_rows)

    targets = [float(row[TARGET_COLUMN]) for row in training_rows]
    previous = [float(row["prev_coral_health_index"]) for row in training_rows]
    deltas = [target - prev for target, prev in zip(targets, previous)]
    bins = [0] * 6
    for value in targets:
        bins[min(5, int(value))] += 1
    print(f"wrote {expected_rows} rows x {len(CSV_COLUMNS)} columns -> {_display_path(args.out)}")
    print(
        f"sites={args.sites} days/site={args.days} seed={args.seed} "
        f"start={args.start_date}"
    )
    print(
        "target coral_health_index: "
        f"min={min(targets):.3f} mean={sum(targets) / len(targets):.3f} "
        f"max={max(targets):.3f}"
    )
    print("target counts per unit bin [0-1, 1-2, ..., 5-6]: " + ", ".join(str(b) for b in bins))
    print(
        "day-to-day change: "
        f"mean|d|={sum(abs(d) for d in deltas) / len(deltas):.3f} "
        f"max rise={max(deltas):.3f} max fall={min(deltas):.3f}"
    )
    print("self-check: OK (schema, row count, missing values and ranges all valid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
