import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = RELEASE_ROOT / "config" / "paired_samples_50.csv"
DAY = "2018-11-12"
REQUESTS_PER_SAMPLE = 10_000
PERIOD_SECONDS = 1_800
FORECAST_SECONDS = 900
GRID_EDGE_M = 1_500.0
RADIUS_M = GRID_EDGE_M / 4.0
VERTICAL_BOUNDARIES_M = (585_000.0, 586_500.0, 588_000.0)
HORIZONTAL_BOUNDARIES_M = (4_510_500.0, 4_512_000.0, 4_513_500.0)
FULL_WITH_FARE_SHA256 = "066f45403fe2d86c37db3c89531f4df9d248584eb2f6af2c417f6061958b9d18"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def splitmix64(values, salt):
    values = values.astype(np.uint64, copy=True)
    with np.errstate(over="ignore"):
        values += np.uint64(0x6A09E667F3BCC909 + salt) + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
    return values


def deterministic_pick(candidates, count, request_ids, salt):
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count > len(candidates):
        raise ValueError("Cannot select more requests than the candidate pool contains.")
    scores = splitmix64(request_ids[candidates], salt)
    order = np.lexsort((request_ids[candidates], scores))
    return candidates[order[:count]]


def stable_uniform(request_ids, salt):
    ids = request_ids.astype(np.uint64, copy=False)
    x = ids + np.uint64(salt)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return ((x >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def add_max_fare(source):
    ids = pd.to_numeric(source["request_id"], errors="raise").to_numpy(dtype=np.int64, copy=False)
    uniform_1 = np.clip(stable_uniform(ids, 42), 1e-12, 1.0 - 1e-12)
    uniform_2 = stable_uniform(ids, 43)
    standard_normal = np.sqrt(-2.0 * np.log(uniform_1)) * np.cos(2.0 * np.pi * uniform_2)
    result = source.copy()
    result["max_fare"] = np.clip(np.exp(np.log(3600.0) + 0.5 * standard_normal), 100.0, None).astype(int)
    return result


def corridor_mask(source, node_xy):
    origins = pd.to_numeric(source["start"], errors="raise").to_numpy(dtype=np.int64)
    xy = node_xy.reindex(origins)[["pos_x", "pos_y"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("A request origin has no network coordinate.")
    vertical = np.any(np.abs(xy[:, 0, None] - np.asarray(VERTICAL_BOUNDARIES_M)) <= RADIUS_M, axis=1)
    horizontal = np.any(np.abs(xy[:, 1, None] - np.asarray(HORIZONTAL_BOUNDARIES_M)) <= RADIUS_M, axis=1)
    return vertical | horizontal


def select_sample(source, quotas, sample_id, node_xy):
    expected_columns = ["request_id", "rq_time", "start", "end", "max_fare"]
    if list(source.columns) != expected_columns:
        raise ValueError(f"Expected source columns {expected_columns}, found {list(source.columns)}")
    if not source["request_id"].is_unique:
        raise ValueError("The source demand file has duplicate request IDs.")
    ids = pd.to_numeric(source["request_id"], errors="raise").to_numpy(dtype=np.int64)
    periods = pd.to_numeric(source["rq_time"], errors="raise").to_numpy(dtype=np.int64) // PERIOD_SECONDS
    is_corridor = corridor_mask(source, node_xy)
    selected_indices = []
    for period in range(48):
        quota = int(quotas[str(period)])
        period_indices = np.flatnonzero(periods == period)
        corridor_indices = period_indices[is_corridor[period_indices]]
        background_indices = period_indices[~is_corridor[period_indices]]
        target_corridor = int(np.floor(0.70 * quota + 0.5))
        chosen_corridor = deterministic_pick(
            corridor_indices,
            min(target_corridor, len(corridor_indices)),
            ids,
            sample_id * 10_000 + period,
        )
        remaining = quota - len(chosen_corridor)
        if len(background_indices) >= remaining:
            chosen_background = deterministic_pick(
                background_indices,
                remaining,
                ids,
                1_000_000 + sample_id * 10_000 + period,
            )
        else:
            unused_corridor = corridor_indices[~np.isin(corridor_indices, chosen_corridor)]
            chosen_background = np.concatenate(
                (
                    background_indices,
                    deterministic_pick(
                        unused_corridor,
                        remaining - len(background_indices),
                        ids,
                        2_000_000 + sample_id * 10_000 + period,
                    ),
                )
            )
        selected_indices.extend(chosen_corridor.tolist())
        selected_indices.extend(chosen_background.tolist())
    selected = source.iloc[np.asarray(selected_indices, dtype=np.int64)].copy()
    selected = selected.sort_values(["rq_time", "request_id"]).reset_index(drop=True)
    if len(selected) != REQUESTS_PER_SAMPLE or not selected["request_id"].is_unique:
        raise AssertionError("The selected trace is not a 10,000-request sample with unique IDs.")
    return selected


def build_forecast(sample, taxi_zones, travel_times):
    origins = sample["start"].to_numpy(dtype=np.int64, copy=False)
    destinations = sample["end"].to_numpy(dtype=np.int64, copy=False)
    request_times = sample["rq_time"].to_numpy(dtype=float, copy=False)
    trip_times = np.asarray(travel_times[origins, destinations], dtype=float)
    origin_zones = taxi_zones.reindex(origins).to_numpy()
    destination_zones = taxi_zones.reindex(destinations).to_numpy()
    valid = np.isfinite(trip_times) & pd.notna(origin_zones) & pd.notna(destination_zones)
    if not valid.all():
        raise ValueError(f"{int((~valid).sum())} requests cannot be mapped to a forecast OD pair.")
    source = pd.DataFrame(
        {
            "o_zone": origin_zones.astype(np.int64),
            "d_zone": destination_zones.astype(np.int64),
            "o_time": (np.ceil(request_times / FORECAST_SECONDS) * FORECAST_SECONDS).astype(np.int64),
            "d_time": (np.ceil((request_times + trip_times) / FORECAST_SECONDS) * FORECAST_SECONDS).astype(np.int64),
            "nr_trips": 1,
            "nr_pax": 1,
        }
    )
    origin = source.groupby(["o_time", "o_zone"]).agg({"nr_trips": "sum", "nr_pax": "sum"})
    origin.index.names = ["time", "zone_id"]
    origin = origin.rename(columns={"nr_trips": "out perfect_trips", "nr_pax": "out perfect_pax"})
    destination = source.groupby(["d_time", "d_zone"]).agg({"nr_trips": "sum", "nr_pax": "sum"})
    destination.index.names = ["time", "zone_id"]
    destination = destination.rename(columns={"nr_trips": "in perfect_trips", "nr_pax": "in perfect_pax"})
    return origin.merge(destination, left_index=True, right_index=True).fillna(0).reset_index()


def ensure_file(frame, path, expected_hash, verify_hashes):
    if path.exists():
        existing = pd.read_csv(path, keep_default_na=False)
        pd.testing.assert_frame_equal(
            existing.astype(str),
            frame.fillna("").astype(str),
            check_dtype=False,
        )
        actual_hash = sha256(path)
        if verify_hashes and actual_hash != expected_hash:
            raise ValueError(f"Hash mismatch for {path.name}: {actual_hash} != {expected_hash}")
        return "validated"
    write_csv(frame, path)
    actual_hash = sha256(path)
    if verify_hashes and actual_hash != expected_hash:
        raise ValueError(f"Hash mismatch for {path.name}: {actual_hash} != {expected_hash}")
    return "created"


def prepare(fleetpy_root, samples, verify_hashes):
    fleetpy_root = fleetpy_root.resolve()
    demand_dir = fleetpy_root / "data" / "demand" / "Manhattan_2018" / "matched" / "Manhattan_2019_corrected"
    forecast_dir = fleetpy_root / "data" / "demand" / "Manhattan_2018" / "aggregated" / "Manhattan_Taxi_Zones" / "900"
    nodes_path = fleetpy_root / "data" / "networks" / "Manhattan_2019_corrected" / "base" / "nodes.csv"
    taxi_zone_path = fleetpy_root / "data" / "zones" / "Manhattan_Taxi_Zones" / "Manhattan_2019_corrected" / "node_zone_info.csv"
    matrix_path = fleetpy_root / "data" / "networks" / "Manhattan_2019_corrected" / "ff" / "tables" / "nn_fastest_tt.npy"
    raw_path = demand_dir / f"{DAY}.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    raw = pd.read_csv(raw_path)
    source = add_max_fare(raw)
    full_path = demand_dir / f"{DAY}_with_fare.csv"
    ensure_file(source, full_path, FULL_WITH_FARE_SHA256, verify_hashes)

    records = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))[:samples]
    if len(records) != samples:
        raise ValueError(f"Only {len(records)} paired samples are available.")
    node_xy = pd.read_csv(nodes_path).set_index("node_index")[["pos_x", "pos_y"]]
    taxi_zones = pd.read_csv(taxi_zone_path).set_index("node_index")["zone_id"]
    travel_times = np.load(matrix_path, mmap_mode="r")
    for record in records:
        sample_id = int(record["sample_id"])
        selected = select_sample(source, json.loads(record["quota_by_period"]), sample_id, node_xy)
        demand_name = f"{DAY}_accepted_daily10000_boundarycorridor70_sample{sample_id:03d}_with_fare.csv"
        demand_path = demand_dir / demand_name
        forecast_path = forecast_dir / f"agg_{demand_name}"
        demand_status = ensure_file(selected, demand_path, record["demand_sha256"], verify_hashes)
        forecast = build_forecast(selected, taxi_zones, travel_times)
        forecast_status = ensure_file(forecast, forecast_path, record["forecast_sha256"], verify_hashes)
        print(f"sample {sample_id:03d}: demand={demand_status}, forecast={forecast_status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleetpy-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.samples <= 50:
        raise ValueError("samples must be between 1 and 50.")
    prepare(args.fleetpy_root, args.samples, args.verify_hashes)


if __name__ == "__main__":
    main()
