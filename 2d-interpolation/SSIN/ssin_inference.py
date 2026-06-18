#!/usr/bin/env python
"""Train SSIN from a long-table CSV, then interpolate selected timestamps."""

from __future__ import print_function

import argparse
import json
import math
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from geographiclib.geodesic import Geodesic
from pyproj import CRS, Transformer

ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from SSIN.networks.Models import SpaFormer


MODEL_CFG = dict(
    n_layers=3, n_head=2, d_k=16, d_v=16,
    d_model=16, d_inner=256, dropout=0.1,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and run SSIN")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--time-field", required=True)
    parser.add_argument("--station-field", required=True)
    parser.add_argument("--x-field", required=True)
    parser.add_argument("--y-field", required=True)
    parser.add_argument("--value-field", required=True)
    parser.add_argument("--input-crs", default="EPSG:4326")
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--extent", nargs=4, type=float)
    parser.add_argument("--cell-size", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("AUTO", "CUDA", "CPU"), default="AUTO")
    parser.add_argument("--min-valid-stations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-log", default="")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def station_key(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text.split(".", 1)[0] if re.match(r"^-?\d+\.0+$", text) else text


def safe_name(value, index):
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("._")
    return name or "time_{:04d}".format(index)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "CPU":
        return torch.device("cpu")
    if requested == "CUDA":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_long_csv(args):
    if not os.path.isfile(args.input_csv):
        raise ValueError("Input CSV does not exist: {}".format(args.input_csv))
    source = pd.read_csv(args.input_csv)
    requested = [
        args.time_field, args.station_field, args.x_field,
        args.y_field, args.value_field,
    ]
    missing = [name for name in requested if name not in source.columns]
    if missing:
        raise ValueError("CSV is missing fields: {}".format(", ".join(missing)))

    frame = source[requested].copy()
    frame.columns = ["time", "station", "lon", "lat", "value"]
    frame["station_key"] = frame["station"].map(station_key)
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame.loc[frame["value"] < 0, "value"] = np.nan
    if frame.empty or frame["time"].isna().any() or (frame["station_key"] == "").any():
        raise ValueError("CSV is empty or contains an empty time/station value.")
    if frame[["lon", "lat"]].isna().any().any():
        raise ValueError("Longitude and latitude must be numeric for every row.")
    if frame.duplicated(["time", "station_key"], keep=False).any():
        raise ValueError("Duplicate time/station rows were found.")

    coordinate_counts = frame.groupby("station_key")[["lon", "lat"]].nunique()
    if (coordinate_counts > 1).any().any():
        raise ValueError("A station has inconsistent coordinates across timestamps.")
    stations = (
        frame[["station_key", "station", "lon", "lat"]]
        .drop_duplicates("station_key")
        .sort_values("station_key")
        .reset_index(drop=True)
    )
    if len(stations) < args.min_valid_stations:
        raise ValueError("The CSV contains too few unique stations.")

    times = list(pd.unique(frame["time"]))
    time_order = {value: index for index, value in enumerate(times)}
    frame["_time_order"] = frame["time"].map(time_order)
    frame = frame.sort_values(["_time_order", "station_key"])
    return frame, stations, times


def transform_coordinates(stations, extent, input_crs):
    source = CRS.from_user_input(input_crs)
    target = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(source, target, always_xy=True)
    x_values, y_values = transformer.transform(
        stations["lon"].to_numpy(dtype=np.float64),
        stations["lat"].to_numpy(dtype=np.float64),
    )
    stations = stations.copy()
    stations["lon"] = x_values
    stations["lat"] = y_values
    if not np.isfinite(stations[["lon", "lat"]].to_numpy()).all():
        raise ValueError("Input coordinates could not be transformed to WGS84.")
    if (
        (stations["lon"].abs() > 180).any()
        or (stations["lat"].abs() > 90).any()
    ):
        raise ValueError("Transformed station coordinates are outside WGS84 bounds.")

    transformed_extent = None
    if extent is not None:
        xmin, ymin, xmax, ymax = map(float, extent)
        corner_x = [xmin, xmin, xmax, xmax]
        corner_y = [ymin, ymax, ymin, ymax]
        lon, lat = transformer.transform(corner_x, corner_y)
        values = np.asarray(list(lon) + list(lat), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Interpolation extent could not be transformed to WGS84.")
        transformed_extent = [min(lon), min(lat), max(lon), max(lat)]
    return stations, transformed_extent, source.to_string()


def build_sequences(frame, stations, times):
    station_index = {
        key: index for index, key in enumerate(stations["station_key"])
    }
    time_index = {value: index for index, value in enumerate(times)}
    values = np.full((len(times), len(stations)), np.nan, dtype=np.float32)
    for row in frame.itertuples(index=False):
        values[time_index[row.time], station_index[row.station_key]] = row.value
    return values


def selected_indices(times, start_time, end_time):
    if not start_time and not end_time:
        return list(range(len(times)))
    text = [str(value) for value in times]
    try:
        start = text.index(start_time) if start_time else 0
        end = text.index(end_time) if end_time else len(times) - 1
    except ValueError:
        raise ValueError("Start or end time is not present in the selected time field.")
    if start > end:
        raise ValueError("Start time must not be after end time.")
    return list(range(start, end + 1))


def make_grid(extent, cell_size, stations):
    if cell_size <= 0:
        raise ValueError("Cell size must be greater than zero.")
    if extent is None:
        xmin, xmax = stations["lon"].min(), stations["lon"].max()
        ymin, ymax = stations["lat"].min(), stations["lat"].max()
        xpad, ypad = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
        extent = [xmin - xpad, ymin - ypad, xmax + xpad, ymax + ypad]
    xmin, ymin, xmax, ymax = map(float, extent)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid interpolation extent.")
    ncols = int(math.ceil((xmax - xmin) / cell_size))
    nrows = int(math.ceil((ymax - ymin) / cell_size))
    if nrows * ncols > 2000000:
        raise ValueError("Grid exceeds 2,000,000 cells; increase cell size.")
    lons = xmin + (np.arange(ncols) + 0.5) * cell_size
    lats = ymin + (np.arange(nrows) + 0.5) * cell_size
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    snapped = [xmin, ymin, xmin + ncols * cell_size, ymin + nrows * cell_size]
    return snapped, points, nrows, ncols


def distance_angle(src_lats, src_lons, dst_lats, dst_lons):
    result = np.zeros((len(src_lats), len(dst_lats), 2), dtype=np.float32)
    for i, (src_lat, src_lon) in enumerate(zip(src_lats, src_lons)):
        for j, (dst_lat, dst_lon) in enumerate(zip(dst_lats, dst_lons)):
            inverse = Geodesic.WGS84.Inverse(
                float(src_lat), float(src_lon), float(dst_lat), float(dst_lon)
            )
            result[i, j] = inverse["s12"] / 1000.0, inverse["azi1"]
    return result


def normalize_rpe(rpe):
    result = rpe.copy()
    stats = {}
    for index, name in enumerate(("distance", "angle")):
        mean = float(result[:, :, index].mean())
        std = float(result[:, :, index].std())
        if std < 1e-8:
            std = 1.0
        result[:, :, index] = (result[:, :, index] - mean) / std
        stats[name + "_mean"] = mean
        stats[name + "_std"] = std
    return result, stats


def create_model(device):
    return SpaFormer(
        d_feat=1, d_pos=2, n_layers=MODEL_CFG["n_layers"],
        n_head=MODEL_CFG["n_head"], d_k=MODEL_CFG["d_k"],
        d_v=MODEL_CFG["d_v"], d_model=MODEL_CFG["d_model"],
        d_inner=MODEL_CFG["d_inner"], dropout=MODEL_CFG["dropout"],
        scale_emb=True,
    ).to(device)


def training_example(values, mask_ratio, device):
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < 2:
        return None
    masked_count = min(len(valid) - 1, max(1, int(round(len(valid) * mask_ratio))))
    masked = np.random.choice(valid, masked_count, replace=False)
    visible = np.setdiff1d(valid, masked)
    mean = float(values[visible].mean())
    std = float(values[visible].std())
    if std < 1e-8:
        std = 1.0
    sequence = np.zeros((len(values), 1), dtype=np.float32)
    sequence[valid, 0] = (values[valid] - mean) / std
    sequence[masked, 0] = 0.0
    labels = ((values[masked] - mean) / std).astype(np.float32)
    attention = np.zeros((len(values), len(values)), dtype=np.float32)
    attention[:, visible] = 1.0
    np.fill_diagonal(attention, 1.0)
    return (
        torch.from_numpy(sequence).unsqueeze(0).to(device),
        torch.from_numpy(masked.astype(np.int64)).unsqueeze(0).to(device),
        torch.from_numpy(labels).to(device),
        torch.from_numpy(attention).unsqueeze(0).to(device),
    )


def train_model(model, sequences, rpe, args, device):
    usable = [row for row in sequences if np.isfinite(row).sum() >= args.min_valid_stations]
    if not usable:
        raise ValueError("No timestamp has enough valid stations for training.")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    rpe_tensor = torch.from_numpy(rpe).to(device)
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(usable)
        losses = []
        for values in usable:
            example = training_example(values, args.mask_ratio, device)
            if example is None:
                continue
            sequence, masked, labels, attention = example
            optimizer.zero_grad()
            prediction, _, _ = model(
                sequence, rpe_tensor, masked, attn_mask=attention
            )
            loss = loss_function(prediction.reshape(-1), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise RuntimeError("Training produced no valid batches.")
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            print(
                "[SSIN] Training epoch {}/{} loss={:.6f}".format(
                    epoch, args.epochs, float(np.mean(losses))
                ),
                flush=True,
            )
    model.eval()


def shielded_mask(observations, queries):
    total = observations + queries
    mask = np.zeros((total, total), dtype=np.float32)
    mask[:observations, :observations] = 1
    np.fill_diagonal(mask, 1)
    mask[observations:, :observations] = 1
    return mask


def interpolate(model, device, values, station_lats, station_lons,
                station_rpe, stats, points, batch_size):
    valid = np.isfinite(values)
    mean = float(values[valid].mean())
    std = float(values[valid].std())
    if std < 1e-8:
        std = 1.0
    filled = values.copy()
    filled[~valid] = mean
    normalized = (filled - mean) / std
    output = np.zeros(len(points), dtype=np.float32)
    station_count = len(values)

    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        batch = points[start:end]
        query_count = len(batch)
        total = station_count + query_count
        query_rpe = distance_angle(
            batch[:, 0], batch[:, 1], station_lats, station_lons
        )
        query_rpe[:, :, 0] = (
            query_rpe[:, :, 0] - stats["distance_mean"]
        ) / stats["distance_std"]
        query_rpe[:, :, 1] = (
            query_rpe[:, :, 1] - stats["angle_mean"]
        ) / stats["angle_std"]
        positions = np.zeros((total, total, 2), dtype=np.float32)
        positions[:station_count, :station_count] = station_rpe
        positions[station_count:, :station_count] = query_rpe
        positions[:station_count, station_count:] = query_rpe.transpose(1, 0, 2)
        sequence = np.concatenate([
            normalized.reshape(-1, 1),
            np.zeros((query_count, 1), dtype=np.float32),
        ])
        with torch.no_grad():
            prediction, _, _ = model(
                torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0).to(device),
                torch.from_numpy(positions).to(device),
                torch.arange(station_count, total, device=device).unsqueeze(0),
                attn_mask=torch.from_numpy(
                    shielded_mask(station_count, query_count)
                ).unsqueeze(0).to(device),
            )
        output[start:end] = np.maximum(
            prediction.detach().cpu().numpy().reshape(-1) * std + mean, 0
        )
    return output


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def run(args):
    if args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs and learning rate must be positive.")
    if not 0 < args.mask_ratio < 1:
        raise ValueError("Mask ratio must be between 0 and 1.")
    if args.batch_size < 1 or args.min_valid_stations < 2:
        raise ValueError("Batch size and minimum valid stations are invalid.")

    frame, stations, times = read_long_csv(args)
    stations, transformed_extent, source_crs = transform_coordinates(
        stations, args.extent, args.input_crs
    )
    sequences = build_sequences(frame, stations, times)
    outputs = selected_indices(times, args.start_time, args.end_time)
    outputs = [
        index for index in outputs
        if np.isfinite(sequences[index]).sum() >= args.min_valid_stations
    ]
    if not outputs:
        raise ValueError("No selected timestamp has enough valid stations.")
    extent, points, nrows, ncols = make_grid(
        transformed_extent, args.cell_size, stations
    )
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "ssin_manifest.json")
    model_path = os.path.join(args.output_dir, "ssin_trained_model.pyt")
    manifest = {
        "format_version": 2,
        "coordinate_system": "EPSG:4326",
        "input_coordinate_system": source_crs,
        "extent": extent,
        "cell_size": args.cell_size,
        "nrows": nrows,
        "ncols": ncols,
        "station_count": len(stations),
        "training_timestamp_count": len(times),
        "results": [],
        "model_path": os.path.abspath(model_path),
        "run_log": os.path.abspath(args.run_log) if args.run_log else "",
    }
    if args.validate_only:
        manifest["validation_only"] = True
        manifest["output_timestamps"] = [str(times[index]) for index in outputs]
        write_json(manifest_path, manifest)
        print("[SSIN] Validation passed. Manifest: {}".format(manifest_path))
        return
    if not args.overwrite:
        existing = [path for path in (manifest_path, model_path) if os.path.exists(path)]
        if existing:
            raise ValueError("Output exists: {}".format(existing[0]))

    set_seed(args.seed)
    device = resolve_device(args.device)
    print("[SSIN] Device: {}".format(device), flush=True)
    station_lats = stations["lat"].to_numpy(dtype=np.float64)
    station_lons = stations["lon"].to_numpy(dtype=np.float64)
    station_rpe, stats = normalize_rpe(
        distance_angle(station_lats, station_lons, station_lats, station_lons)
    )
    model = create_model(device)
    train_model(model, sequences, station_rpe, args, device)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": MODEL_CFG,
        "station_keys": list(stations["station_key"]),
        "station_lats": station_lats,
        "station_lons": station_lons,
        "relative_position_stats": stats,
        "training_parameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "mask_ratio": args.mask_ratio,
            "seed": args.seed,
        },
    }, model_path)
    print("[SSIN] Model saved: {}".format(model_path), flush=True)

    for number, index in enumerate(outputs):
        timestamp = times[index]
        print("[SSIN] Interpolating {}/{}: {}".format(
            number + 1, len(outputs), timestamp
        ), flush=True)
        prediction = interpolate(
            model, device, sequences[index], station_lats, station_lons,
            station_rpe, stats, points, args.batch_size,
        )
        array_path = os.path.join(
            args.output_dir, "ssin_{}.npz".format(safe_name(timestamp, number))
        )
        np.savez(
            array_path,
            grid=np.flipud(prediction.reshape(nrows, ncols)).astype(np.float32),
            timestamp=str(timestamp),
        )
        manifest["results"].append({
            "timestamp": str(timestamp),
            "array_path": os.path.abspath(array_path),
            "valid_station_count": int(np.isfinite(sequences[index]).sum()),
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
            "prediction_mean": float(prediction.mean()),
        })
        write_json(manifest_path, manifest)
    print("[SSIN] Completed. Manifest: {}".format(manifest_path), flush=True)


def main(argv=None):
    try:
        run(parse_args(argv))
        return 0
    except Exception as error:
        print("[SSIN ERROR] {}".format(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

