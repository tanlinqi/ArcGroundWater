from __future__ import print_function

import argparse
import csv
import json
import math
import os
import random
import sys
import traceback

import numpy as np
import torch
from pyproj import CRS, Transformer
from torch import nn
from torch import optim

from model.basic_structure import IGNNK


FORMAT_VERSION = 1
MODEL_NAME = "IGNNK"
OUTPUT_CRS = "EPSG:4326"


def parse_args():
    parser = argparse.ArgumentParser(description="IGNNK spatial interpolation backend")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--time-field", required=True)
    parser.add_argument("--station-field", required=True)
    parser.add_argument("--x-field", required=True)
    parser.add_argument("--y-field", required=True)
    parser.add_argument("--value-field", required=True)
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--input-crs", default=OUTPUT_CRS)
    parser.add_argument("--extent", default="")
    parser.add_argument("--cell-size", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--time-window", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-order", type=int, default=1)
    parser.add_argument("--masked-stations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--distance-scale-km", type=float, default=100.0)
    parser.add_argument("--adjacency-threshold", type=float, default=0.01)
    parser.add_argument("--value-scale", type=float, default=0.0)
    parser.add_argument("--anchor-weight", type=float, default=0.85)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--missing-zero", action="store_true")
    parser.add_argument("--device", choices=["AUTO", "CUDA", "CPU"], default="AUTO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grid-cells", type=int, default=2000000)
    parser.add_argument("--run-log", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def fail(message):
    raise ValueError(message)


def ensure_finite(value, label):
    if not math.isfinite(float(value)):
        fail("%s must be finite" % label)


def validate_args(args):
    if args.cell_size <= 0:
        fail("cell-size must be greater than zero")
    if args.epochs < 1:
        fail("epochs must be at least 1")
    if args.time_window < 2:
        fail("time-window must be at least 2")
    if args.hidden_dim < 1 or args.diffusion_order < 1:
        fail("hidden-dim and diffusion-order must be positive")
    if args.masked_stations < 1:
        fail("masked-stations must be at least 1")
    if args.learning_rate <= 0 or args.batch_size < 1:
        fail("learning-rate and batch-size must be positive")
    if args.query_batch_size < 1 or args.distance_scale_km <= 0:
        fail("query-batch-size and distance-scale-km must be positive")
    if not 0 <= args.adjacency_threshold < 1:
        fail("adjacency-threshold must be in [0, 1)")
    if args.value_scale < 0:
        fail("value-scale cannot be negative")
    if not 0 <= args.anchor_weight <= 1:
        fail("anchor-weight must be in [0, 1]")
    if args.idw_power <= 0:
        fail("idw-power must be greater than zero")
    if args.max_grid_cells < 1:
        fail("max-grid-cells must be positive")


def read_long_csv(args):
    required = [
        args.time_field,
        args.station_field,
        args.x_field,
        args.y_field,
        args.value_field,
    ]
    records = []
    station_coords = {}
    seen = set()
    with open(args.input_csv, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            fail("input CSV has no header")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            fail("input CSV is missing fields: %s" % ", ".join(missing))
        for line_number, row in enumerate(reader, 2):
            timestamp = str(row[args.time_field]).strip()
            station = str(row[args.station_field]).strip()
            if not timestamp or not station:
                fail("empty time or station at CSV line %d" % line_number)
            key = (timestamp, station)
            if key in seen:
                fail("duplicate time and station at CSV line %d" % line_number)
            seen.add(key)
            try:
                x_value = float(row[args.x_field])
                y_value = float(row[args.y_field])
            except Exception:
                fail("invalid coordinate at CSV line %d" % line_number)
            ensure_finite(x_value, "x coordinate")
            ensure_finite(y_value, "y coordinate")
            old_coord = station_coords.get(station)
            if old_coord is not None and (
                abs(old_coord[0] - x_value) > 1e-9
                or abs(old_coord[1] - y_value) > 1e-9
            ):
                fail("station %s has inconsistent coordinates" % station)
            station_coords[station] = (x_value, y_value)
            text = str(row[args.value_field]).strip()
            try:
                value = float(text) if text else float("nan")
            except Exception:
                value = float("nan")
            if args.missing_zero and value == 0:
                value = float("nan")
            if not np.isfinite(value):
                value = float("nan")
            records.append((timestamp, station, value))
    if not records:
        fail("input CSV has no data rows")
    return records, station_coords


def build_matrix(records, station_coords, args):
    all_times = sorted(set(item[0] for item in records))
    if args.start_time and args.start_time not in all_times:
        fail("start-time was not found in the input CSV")
    if args.end_time and args.end_time not in all_times:
        fail("end-time was not found in the input CSV")
    start_index = all_times.index(args.start_time) if args.start_time else 0
    end_index = all_times.index(args.end_time) if args.end_time else len(all_times) - 1
    if end_index < start_index:
        fail("end-time cannot be earlier than start-time")
    times = all_times[start_index : end_index + 1]
    stations = sorted(station_coords)
    time_index = dict((value, index) for index, value in enumerate(times))
    station_index = dict((value, index) for index, value in enumerate(stations))
    values = np.full((len(times), len(stations)), np.nan, dtype=np.float32)
    for timestamp, station, value in records:
        if timestamp in time_index:
            values[time_index[timestamp], station_index[station]] = value
    if len(times) < 2:
        fail("at least two timestamps are required")
    if len(stations) < 2:
        fail("at least two stations are required")
    usable = np.sum(np.isfinite(values), axis=0) > 0
    if not np.all(usable):
        stations = [station for station, keep in zip(stations, usable) if keep]
        values = values[:, usable]
    if len(stations) < 2:
        fail("at least two stations must contain valid values")
    coords = np.asarray([station_coords[name] for name in stations], dtype=np.float64)
    return times, stations, coords, values


def transform_coordinates(coords, input_crs):
    source = CRS.from_user_input(input_crs)
    target = CRS.from_epsg(4326)
    if source == target:
        result = coords.copy()
    else:
        transformer = Transformer.from_crs(source, target, always_xy=True)
        x_values, y_values = transformer.transform(coords[:, 0], coords[:, 1])
        result = np.column_stack([x_values, y_values])
    if not np.all(np.isfinite(result)):
        fail("coordinate transformation produced invalid values")
    if (
        np.min(result[:, 0]) < -180
        or np.max(result[:, 0]) > 180
        or np.min(result[:, 1]) < -90
        or np.max(result[:, 1]) > 90
    ):
        fail("transformed station coordinates are outside WGS84 bounds")
    return result


def parse_extent(text, coords, input_crs):
    if not text or text.upper() in ("#", "MAXOF", "MINOF", "DISPLAY", "DEFAULT"):
        xmin = float(np.min(coords[:, 0]))
        ymin = float(np.min(coords[:, 1]))
        xmax = float(np.max(coords[:, 0]))
        ymax = float(np.max(coords[:, 1]))
    else:
        parts = text.replace(",", " ").split()
        if len(parts) != 4:
            fail("extent must contain xmin ymin xmax ymax")
        raw = [float(value) for value in parts]
        source = CRS.from_user_input(input_crs)
        if source == CRS.from_epsg(4326):
            xmin, ymin, xmax, ymax = raw
        else:
            transformer = Transformer.from_crs(source, CRS.from_epsg(4326), always_xy=True)
            corner_x = [raw[0], raw[0], raw[2], raw[2]]
            corner_y = [raw[1], raw[3], raw[1], raw[3]]
            out_x, out_y = transformer.transform(corner_x, corner_y)
            xmin, xmax = min(out_x), max(out_x)
            ymin, ymax = min(out_y), max(out_y)
    if not (-180 <= xmin < xmax <= 180 and -90 <= ymin < ymax <= 90):
        fail("WGS84 output extent is invalid")
    return [xmin, ymin, xmax, ymax]


def build_grid(extent, cell_size, max_grid_cells):
    xmin, ymin, xmax, ymax = extent
    ncols = int(math.ceil((xmax - xmin) / cell_size))
    nrows = int(math.ceil((ymax - ymin) / cell_size))
    cell_count = nrows * ncols
    if nrows < 1 or ncols < 1:
        fail("extent and cell-size produce an empty grid")
    if cell_count > max_grid_cells:
        fail(
            "grid has %d cells; increase cell-size or max-grid-cells"
            % cell_count
        )
    x_centers = xmin + (np.arange(ncols) + 0.5) * cell_size
    y_centers = ymax - (np.arange(nrows) + 0.5) * cell_size
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    raster_extent = [
        xmin,
        ymax - nrows * cell_size,
        xmin + ncols * cell_size,
        ymax,
    ]
    return points, nrows, ncols, raster_extent


def haversine_matrix(coords_a, coords_b):
    lon_a = np.radians(coords_a[:, 0])[:, None]
    lat_a = np.radians(coords_a[:, 1])[:, None]
    lon_b = np.radians(coords_b[:, 0])[None, :]
    lat_b = np.radians(coords_b[:, 1])[None, :]
    dlon = lon_b - lon_a
    dlat = lat_b - lat_a
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2.0) ** 2
    )
    value = np.clip(value, 0.0, 1.0)
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(value))


def adjacency(coords, distance_scale_km, threshold):
    distance = haversine_matrix(coords, coords)
    matrix = np.exp(-distance / distance_scale_km).astype(np.float32)
    matrix[matrix < threshold] = 0.0
    np.fill_diagonal(matrix, 1.0)
    return matrix


def random_walk(matrix):
    row_sum = matrix.sum(axis=1)
    row_sum[row_sum <= 1e-12] = 1.0
    return matrix / row_sum[:, None]


def idw_predict(points, station_coords, station_values, power):
    valid = np.isfinite(station_values)
    if not np.any(valid):
        return np.full(points.shape[0], np.nan, dtype=np.float32)
    coords = station_coords[valid]
    values = station_values[valid].astype(np.float64)
    distances = haversine_matrix(points, coords)
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(distances.shape[0]), nearest]
    result = np.empty(points.shape[0], dtype=np.float64)
    exact = nearest_distance < 1e-9
    if np.any(exact):
        result[exact] = values[nearest[exact]]
    if np.any(~exact):
        work = distances[~exact]
        weights = 1.0 / np.power(np.maximum(work, 1e-6), power)
        result[~exact] = np.sum(weights * values[None, :], axis=1) / np.sum(
            weights, axis=1
        )
    return result.astype(np.float32)


def select_device(name):
    if name == "CPU":
        return torch.device("cpu")
    if name == "CUDA":
        if not torch.cuda.is_available():
            fail("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def padded_window(values, start, window):
    indices = np.arange(start, start + window)
    indices = np.clip(indices, 0, values.shape[0] - 1)
    return values[indices]


def train_model(values, station_adj, args, device, scale):
    time_window = min(args.time_window, values.shape[0])
    if args.masked_stations >= values.shape[1]:
        fail("masked-stations must be smaller than station count")
    model = IGNNK(time_window, args.hidden_dim, args.diffusion_order).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss(reduction="sum")
    max_start = max(1, values.shape[0] - time_window + 1)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        for unused in range(args.batch_size):
            start = np.random.randint(0, max_start)
            sample = padded_window(values, start, time_window)
            observed = np.isfinite(sample)
            candidates = np.where(np.any(observed, axis=0))[0]
            if len(candidates) <= args.masked_stations:
                continue
            hidden = np.random.choice(
                candidates, size=args.masked_stations, replace=False
            )
            input_mask = observed.copy()
            input_mask[:, hidden] = False
            model_input = np.where(input_mask, sample / scale, 0.0)
            target = np.where(observed, sample / scale, 0.0)
            loss_mask = np.zeros(sample.shape, dtype=np.float32)
            loss_mask[:, hidden] = observed[:, hidden].astype(np.float32)
            if np.sum(loss_mask) <= 0:
                continue
            input_tensor = torch.from_numpy(model_input[None].astype(np.float32)).to(device)
            target_tensor = torch.from_numpy(target[None].astype(np.float32)).to(device)
            mask_tensor = torch.from_numpy(loss_mask[None]).to(device)
            a_q = torch.from_numpy(random_walk(station_adj).T.copy()).to(device)
            a_h = torch.from_numpy(random_walk(station_adj.T).T.copy()).to(device)
            optimizer.zero_grad()
            output = model(input_tensor, a_q, a_h)
            denominator = torch.clamp(mask_tensor.sum(), min=1.0)
            loss = criterion(output * mask_tensor, target_tensor * mask_tensor) / denominator
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            valid_batches += 1
        if valid_batches == 0:
            fail("training found no usable batches")
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(
                "[IGNNK] epoch %d/%d loss %.8f"
                % (epoch + 1, args.epochs, epoch_loss / valid_batches),
                flush=True,
            )
    return model, time_window


def save_checkpoint(path, model, args, time_window, scale):
    payload = {
        "format_version": FORMAT_VERSION,
        "model_name": MODEL_NAME,
        "state_dict": model.state_dict(),
        "time_window": time_window,
        "hidden_dim": args.hidden_dim,
        "diffusion_order": args.diffusion_order,
        "value_scale": scale,
    }
    torch.save(payload, path)


def make_time_windows(values, time_window):
    windows = []
    positions = []
    left = time_window // 2
    for index in range(values.shape[0]):
        start = index - left
        windows.append(padded_window(values, start, time_window))
        positions.append(index - start)
    return windows, positions


def interpolate(
    model,
    values,
    station_coords,
    grid_points,
    nrows,
    ncols,
    args,
    device,
    scale,
    time_window,
    times,
    output_dir,
    manifest,
    manifest_path,
    input_min,
):
    windows, positions = make_time_windows(values, time_window)
    flat_results = [np.full(len(grid_points), np.nan, dtype=np.float32) for unused in times]
    model.eval()
    station_count = len(station_coords)
    with torch.no_grad():
        for batch_start in range(0, len(grid_points), args.query_batch_size):
            batch_points = grid_points[
                batch_start : batch_start + args.query_batch_size
            ]
            combined = np.vstack([station_coords, batch_points])
            graph = adjacency(
                combined, args.distance_scale_km, args.adjacency_threshold
            )
            a_q = torch.from_numpy(random_walk(graph).T.copy()).to(device)
            a_h = torch.from_numpy(random_walk(graph.T).T.copy()).to(device)
            for time_index, window in enumerate(windows):
                model_values = np.zeros((time_window, len(combined)), dtype=np.float32)
                model_values[:, :station_count] = np.where(
                    np.isfinite(window), window / scale, 0.0
                )
                tensor = torch.from_numpy(model_values[None]).to(device)
                prediction = model(tensor, a_q, a_h)[0, positions[time_index]]
                predicted = prediction[station_count:].detach().cpu().numpy() * scale
                if args.anchor_weight > 0:
                    anchored = idw_predict(
                        batch_points,
                        station_coords,
                        values[time_index],
                        args.idw_power,
                    )
                    predicted = (
                        (1.0 - args.anchor_weight) * predicted
                        + args.anchor_weight * anchored
                    )
                if input_min >= 0:
                    predicted = np.maximum(predicted, 0.0)
                flat_results[time_index][
                    batch_start : batch_start + len(batch_points)
                ] = predicted.astype(np.float32)
            print(
                "[IGNNK] grid %d/%d"
                % (min(batch_start + len(batch_points), len(grid_points)), len(grid_points)),
                flush=True,
            )
    for index, timestamp in enumerate(times):
        grid = flat_results[index].reshape(nrows, ncols)
        safe_time = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in timestamp
        )
        array_path = os.path.abspath(
            os.path.join(output_dir, "interpolation_%s.npz" % safe_time)
        )
        np.savez_compressed(array_path, grid=grid, timestamp=np.asarray(timestamp))
        print("[IGNNK] array %s" % array_path, flush=True)
        finite = grid[np.isfinite(grid)]
        result = {
            "timestamp": timestamp,
            "array_path": array_path,
            "valid_station_count": int(np.sum(np.isfinite(values[index]))),
            "prediction_min": float(np.min(finite)) if finite.size else None,
            "prediction_max": float(np.max(finite)) if finite.size else None,
            "prediction_mean": float(np.mean(finite)) if finite.size else None,
        }
        manifest["results"].append(result)
        write_json(manifest_path, manifest)
        print("[IGNNK] wrote %s" % timestamp, flush=True)


def write_json(path, content):
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=True)
    os.replace(temp_path, path)


def check_outputs(output_dir, overwrite):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    names = os.listdir(output_dir)
    conflicts = [
        name
        for name in names
        if name == "result_manifest.json"
        or name == "trained_model.pth"
        or (name.startswith("interpolation_") and name.endswith((".npz", ".tif")))
    ]
    if conflicts and not overwrite:
        fail("output files already exist; enable overwrite or use another folder")


def run(args):
    validate_args(args)
    set_seed(args.seed)
    records, station_coords = read_long_csv(args)
    times, stations, raw_coords, values = build_matrix(
        records, station_coords, args
    )
    coords = transform_coordinates(raw_coords, args.input_crs)
    extent = parse_extent(args.extent, coords, args.input_crs)
    grid_points, nrows, ncols, extent = build_grid(
        extent, args.cell_size, args.max_grid_cells
    )
    if args.validate_only:
        print(
            "[IGNNK] validation OK: %d times, %d stations, %d x %d grid"
            % (len(times), len(stations), nrows, ncols)
        )
        return
    check_outputs(args.output_dir, args.overwrite)
    device = select_device(args.device)
    print("[IGNNK] device %s" % device, flush=True)
    station_adj = adjacency(
        coords, args.distance_scale_km, args.adjacency_threshold
    )
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        fail("input CSV has no valid numeric values")
    scale = (
        args.value_scale
        if args.value_scale > 0
        else max(float(np.max(np.abs(finite_values))), 1e-6)
    )
    model_path = os.path.abspath(
        os.path.join(args.output_dir, "trained_model.pth")
    )
    model, time_window = train_model(values, station_adj, args, device, scale)
    save_checkpoint(model_path, model, args, time_window, scale)
    manifest_path = os.path.abspath(
        os.path.join(args.output_dir, "result_manifest.json")
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "model_name": MODEL_NAME,
        "coordinate_system": OUTPUT_CRS,
        "input_coordinate_system": args.input_crs,
        "extent": extent,
        "cell_size": args.cell_size,
        "nrows": nrows,
        "ncols": ncols,
        "station_count": len(stations),
        "training_timestamp_count": len(times),
        "training_time_range": [times[0], times[-1]],
        "model_path": model_path,
        "run_log": os.path.abspath(args.run_log) if args.run_log else "",
        "training_parameters": {
            "workflow": "train_then_interpolate",
            "epochs": args.epochs,
            "time_window": time_window,
            "hidden_dim": args.hidden_dim,
            "diffusion_order": args.diffusion_order,
            "masked_stations": args.masked_stations,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "distance_scale_km": args.distance_scale_km,
            "adjacency_threshold": args.adjacency_threshold,
            "value_scale": scale,
            "anchor_weight": args.anchor_weight,
            "idw_power": args.idw_power,
            "seed": args.seed,
            "device": str(device),
        },
        "results": [],
        "skipped": [],
    }
    write_json(manifest_path, manifest)
    interpolate(
        model,
        values,
        coords,
        grid_points,
        nrows,
        ncols,
        args,
        device,
        scale,
        time_window,
        times,
        args.output_dir,
        manifest,
        manifest_path,
        float(np.min(finite_values)),
    )
    print("[IGNNK] manifest %s" % manifest_path, flush=True)


def main():
    try:
        run(parse_args())
    except Exception as exc:
        print("[MODEL ERROR] %s" % exc, file=sys.stderr, flush=True)
        if os.environ.get("IGNNK_DEBUG") == "1":
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

