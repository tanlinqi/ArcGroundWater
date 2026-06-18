#!/usr/bin/env python
"""Train DeepKriging models from a long-table CSV and interpolate rasters."""

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
from pyproj import CRS, Transformer

from model.deepkriging_model import DeepKrigingMLP


MAX_GRID_CELLS = 2000000
MANIFEST_NAME = "deepkriging_manifest.json"
MODEL_NAME = "deepkriging_trained_model.pyt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and run DeepKriging")
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--basis-resolutions", default="10,19,37")
    parser.add_argument("--support-multiplier", type=float, default=2.5)
    parser.add_argument("--hidden-units", type=int, default=100)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("AUTO", "CUDA", "CPU"), default="AUTO")
    parser.add_argument("--min-valid-stations", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--run-log", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def parse_resolutions(text):
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError:
        raise ValueError("Basis resolutions must be comma-separated integers.")
    if not values or any(value < 2 for value in values):
        raise ValueError("Each basis resolution must be at least 2.")
    if len(set(values)) != len(values):
        raise ValueError("Basis resolutions must not contain duplicates.")
    return values


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


def safe_name(value, index):
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("._")
    return name or "time_{:04d}".format(index)


def station_key(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text.split(".", 1)[0] if re.match(r"^-?\d+\.0+$", text) else text


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
    frame.columns = ["time", "station", "x", "y", "value"]
    frame["time"] = frame["time"].map(lambda value: str(value).strip())
    frame["station_key"] = frame["station"].map(station_key)
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame.empty or (frame["time"] == "").any() or (frame["station_key"] == "").any():
        raise ValueError("CSV is empty or contains an empty time/station value.")
    if frame[["x", "y"]].isna().any().any():
        raise ValueError("X and Y coordinates must be numeric for every row.")
    if frame.duplicated(["time", "station_key"], keep=False).any():
        raise ValueError("Duplicate time/station rows were found.")

    coordinate_counts = frame.groupby("station_key")[["x", "y"]].nunique()
    if (coordinate_counts > 1).any().any():
        raise ValueError("A station has inconsistent coordinates across timestamps.")
    stations = (
        frame[["station_key", "station", "x", "y"]]
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
    lon, lat = transformer.transform(
        stations["x"].to_numpy(dtype=np.float64),
        stations["y"].to_numpy(dtype=np.float64),
    )
    stations = stations.copy()
    stations["lon"] = lon
    stations["lat"] = lat
    coordinates = stations[["lon", "lat"]].to_numpy()
    if not np.isfinite(coordinates).all():
        raise ValueError("Input coordinates could not be transformed to WGS84.")
    if (stations["lon"].abs() > 180).any() or (stations["lat"].abs() > 90).any():
        raise ValueError("Transformed station coordinates are outside WGS84 bounds.")

    transformed_extent = None
    if extent is not None:
        xmin, ymin, xmax, ymax = map(float, extent)
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Invalid interpolation extent.")
        corner_x = [xmin, xmin, xmax, xmax]
        corner_y = [ymin, ymax, ymin, ymax]
        out_x, out_y = transformer.transform(corner_x, corner_y)
        values = np.asarray(list(out_x) + list(out_y), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Interpolation extent could not be transformed to WGS84.")
        transformed_extent = [min(out_x), min(out_y), max(out_x), max(out_y)]
    return stations, transformed_extent, source.to_string()


def build_values(frame, stations, times):
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
    try:
        start = times.index(start_time) if start_time else 0
        end = times.index(end_time) if end_time else len(times) - 1
    except ValueError:
        raise ValueError("Start or end time is not present in the selected field.")
    if start > end:
        raise ValueError("Start time must not be after end time.")
    return list(range(start, end + 1))


def make_grid(extent, cell_size, stations):
    if cell_size <= 0:
        raise ValueError("Cell size must be greater than zero.")
    if extent is None:
        xmin = float(stations["lon"].min())
        xmax = float(stations["lon"].max())
        ymin = float(stations["lat"].min())
        ymax = float(stations["lat"].max())
        xpad = max((xmax - xmin) * 0.05, cell_size)
        ypad = max((ymax - ymin) * 0.05, cell_size)
        extent = [xmin - xpad, ymin - ypad, xmax + xpad, ymax + ypad]
    xmin, ymin, xmax, ymax = map(float, extent)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid interpolation extent.")
    ncols = int(math.ceil((xmax - xmin) / cell_size))
    nrows = int(math.ceil((ymax - ymin) / cell_size))
    if nrows < 1 or ncols < 1:
        raise ValueError("Interpolation grid has no cells.")
    if nrows * ncols > MAX_GRID_CELLS:
        raise ValueError(
            "Grid exceeds {:,} cells; increase cell size.".format(MAX_GRID_CELLS)
        )
    snapped = [xmin, ymin, xmin + ncols * cell_size, ymin + nrows * cell_size]
    return snapped, nrows, ncols


def grid_points(extent, cell_size, nrows, ncols):
    xmin, ymin, _, _ = extent
    x_values = xmin + (np.arange(ncols) + 0.5) * cell_size
    y_values = ymin + (np.arange(nrows) + 0.5) * cell_size
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return np.column_stack((x_grid.ravel(), y_grid.ravel()))


def normalize_points(points, extent):
    xmin, ymin, xmax, ymax = extent
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        raise ValueError("Normalization extent must have positive width and height.")
    result = np.empty_like(points, dtype=np.float32)
    result[:, 0] = (points[:, 0] - xmin) / width
    result[:, 1] = (points[:, 1] - ymin) / height
    return result


def build_knots(resolutions):
    knot_blocks = []
    theta_blocks = []
    for resolution in resolutions:
        axis = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
        knot_x, knot_y = np.meshgrid(axis, axis)
        knot_blocks.append(
            np.column_stack((knot_x.ravel(), knot_y.ravel())).astype(np.float32)
        )
        theta_blocks.append(resolution)
    return knot_blocks, theta_blocks


def wendland_features(points, resolutions, support_multiplier):
    blocks = []
    knots, _ = build_knots(resolutions)
    for resolution, level_knots in zip(resolutions, knots):
        theta = support_multiplier / float(resolution)
        delta = points[:, None, :] - level_knots[None, :, :]
        distance = np.sqrt(np.sum(delta * delta, axis=2)) / theta
        inside = distance <= 1.0
        clipped = np.minimum(distance, 1.0)
        values = (
            (1.0 - clipped) ** 6
            * (35.0 * clipped ** 2 + 18.0 * clipped + 3.0)
            / 3.0
        )
        values[~inside] = 0.0
        blocks.append(values.astype(np.float32))
    return np.concatenate(blocks, axis=1)


def train_model(features, target, args, device, timestamp_index):
    mean = float(target.mean())
    std = float(target.std())
    if std < 1e-8:
        return None, mean, 1.0, []
    normalized = ((target - mean) / std).astype(np.float32)
    model = DeepKrigingMLP(
        features.shape[1], args.hidden_units, args.hidden_layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    x_tensor = torch.from_numpy(features)
    y_tensor = torch.from_numpy(normalized.reshape(-1, 1))
    batch_size = min(args.train_batch_size, len(target))
    generator = torch.Generator()
    generator.manual_seed(args.seed + timestamp_index)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    losses = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(epoch_losses))
        losses.append(epoch_loss)
        report_every = max(1, args.epochs // 10)
        if epoch == 1 or epoch == args.epochs or epoch % report_every == 0:
            print(
                "[DeepKriging] Epoch {}/{} loss={:.6f}".format(
                    epoch, args.epochs, epoch_loss
                ),
                flush=True,
            )
    model.eval()
    return model, mean, std, losses


def predict_grid(model, mean, std, points, extent, args, device):
    if model is None:
        return np.full(len(points), mean, dtype=np.float32)
    output = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), args.inference_batch_size):
        end = min(start + args.inference_batch_size, len(points))
        print(
            "[DeepKriging] Inference cells {}-{} of {}".format(
                start + 1, end, len(points)
            ),
            flush=True,
        )
        normalized = normalize_points(points[start:end], extent)
        features = wendland_features(
            normalized, args.resolutions, args.support_multiplier
        )
        with torch.no_grad():
            prediction = model(
                torch.from_numpy(features).to(device)
            ).detach().cpu().numpy().reshape(-1)
        output[start:end] = prediction * std + mean
    return output


def write_json(path, data):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def validate_args(args):
    if args.epochs < 1:
        raise ValueError("Training epochs must be at least 1.")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be greater than zero.")
    if args.support_multiplier <= 0:
        raise ValueError("Support multiplier must be greater than zero.")
    if args.hidden_units < 1 or args.hidden_layers < 1:
        raise ValueError("Hidden units and hidden layers must be at least 1.")
    if args.min_valid_stations < 2:
        raise ValueError("Minimum valid stations must be at least 2.")
    if args.train_batch_size < 1 or args.inference_batch_size < 1:
        raise ValueError("Batch sizes must be at least 1.")
    args.resolutions = parse_resolutions(args.basis_resolutions)


def run(args):
    validate_args(args)
    frame, stations, times = read_long_csv(args)
    stations, transformed_extent, source_crs = transform_coordinates(
        stations, args.extent, args.input_crs
    )
    if transformed_extent is not None:
        xmin, ymin, xmax, ymax = transformed_extent
        outside = (
            (stations["lon"] < xmin)
            | (stations["lon"] > xmax)
            | (stations["lat"] < ymin)
            | (stations["lat"] > ymax)
        )
        if outside.any():
            raise ValueError(
                "Interpolation extent excludes {} station(s). "
                "Leave extent empty or use an extent containing all stations.".format(
                    int(outside.sum())
                )
            )
    values = build_values(frame, stations, times)
    output_indices = selected_indices(times, args.start_time, args.end_time)
    skipped = []
    usable_indices = []
    for index in output_indices:
        valid_count = int(np.isfinite(values[index]).sum())
        if valid_count < args.min_valid_stations:
            skipped.append({
                "timestamp": times[index],
                "reason": "Too few valid stations",
                "valid_station_count": valid_count,
            })
        else:
            usable_indices.append(index)
    if not usable_indices:
        raise ValueError("No selected timestamp has enough valid stations.")

    extent, nrows, ncols = make_grid(
        transformed_extent, args.cell_size, stations
    )
    points = grid_points(extent, args.cell_size, nrows, ncols)
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, MANIFEST_NAME)
    model_path = os.path.join(args.output_dir, MODEL_NAME)
    manifest = {
        "format_version": 1,
        "model_name": "DeepKriging",
        "coordinate_system": "EPSG:4326",
        "input_coordinate_system": source_crs,
        "extent": extent,
        "cell_size": args.cell_size,
        "nrows": nrows,
        "ncols": ncols,
        "station_count": len(stations),
        "training_timestamp_count": len(usable_indices),
        "model_path": os.path.abspath(model_path),
        "run_log": os.path.abspath(args.run_log) if args.run_log else "",
        "training_parameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "basis_resolutions": args.resolutions,
            "support_multiplier": args.support_multiplier,
            "hidden_units": args.hidden_units,
            "hidden_layers": args.hidden_layers,
            "seed": args.seed,
        },
        "results": [],
        "skipped": skipped,
    }
    if args.validate_only:
        manifest["validation_only"] = True
        manifest["output_timestamps"] = [times[index] for index in usable_indices]
        write_json(manifest_path, manifest)
        print(
            "[DeepKriging] Validation passed. Manifest: {}".format(manifest_path),
            flush=True,
        )
        return
    if not args.overwrite:
        existing = [
            path for path in (manifest_path, model_path) if os.path.exists(path)
        ]
        if existing:
            raise ValueError("Output exists: {}".format(existing[0]))

    set_seed(args.seed)
    device = resolve_device(args.device)
    print("[DeepKriging] Device: {}".format(device), flush=True)
    station_points = stations[["lon", "lat"]].to_numpy(dtype=np.float64)
    station_features_all = wendland_features(
        normalize_points(station_points, extent),
        args.resolutions,
        args.support_multiplier,
    )
    saved_models = {}

    for number, index in enumerate(usable_indices):
        timestamp = times[index]
        valid = np.isfinite(values[index])
        target = values[index, valid].astype(np.float32)
        features = station_features_all[valid]
        print(
            "[DeepKriging] Training {}/{}: {} ({} stations)".format(
                number + 1, len(usable_indices), timestamp, len(target)
            ),
            flush=True,
        )
        model, target_mean, target_std, losses = train_model(
            features, target, args, device, index
        )
        prediction = predict_grid(
            model, target_mean, target_std, points, extent, args, device
        )
        array_path = os.path.join(
            args.output_dir,
            "deepkriging_{}.npz".format(safe_name(timestamp, number)),
        )
        if os.path.exists(array_path) and not args.overwrite:
            raise ValueError("Output exists: {}".format(array_path))
        np.savez(
            array_path,
            grid=np.flipud(prediction.reshape(nrows, ncols)).astype(np.float32),
            timestamp=str(timestamp),
        )
        print(
            "[DeepKriging] Saved array: {}".format(os.path.abspath(array_path)),
            flush=True,
        )
        saved_models[str(timestamp)] = {
            "state_dict": None if model is None else {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "target_mean": target_mean,
            "target_std": target_std,
            "valid_station_count": int(valid.sum()),
        }
        manifest["results"].append({
            "timestamp": str(timestamp),
            "array_path": os.path.abspath(array_path),
            "valid_station_count": int(valid.sum()),
            "final_training_loss": None if not losses else losses[-1],
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
            "prediction_mean": float(prediction.mean()),
        })
        write_json(manifest_path, manifest)

    torch.save({
        "format_version": 1,
        "model_name": "DeepKriging",
        "input_dimension": int(station_features_all.shape[1]),
        "basis_resolutions": args.resolutions,
        "support_multiplier": args.support_multiplier,
        "hidden_units": args.hidden_units,
        "hidden_layers": args.hidden_layers,
        "normalization_extent": extent,
        "coordinate_system": "EPSG:4326",
        "models": saved_models,
    }, model_path)
    write_json(manifest_path, manifest)
    print("[DeepKriging] Model saved: {}".format(model_path), flush=True)
    print("[DeepKriging] Completed. Manifest: {}".format(manifest_path), flush=True)


def main(argv=None):
    try:
        run(parse_args(argv))
        return 0
    except Exception as error:
        print("[DeepKriging ERROR] {}".format(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
