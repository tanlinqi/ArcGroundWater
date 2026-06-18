import argparse
import csv
import json
import math
import os
import random
import re
import sys
import traceback

import numpy as np


FORMAT_VERSION = 1
MODEL_NAME = "KCN"
EXTENT_KEYWORDS = set(["", "#", "MAXOF", "MINOF", "DISPLAY", "DEFAULT"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="KCN ArcMap interpolation backend")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--time-field", required=True)
    parser.add_argument("--station-field", required=True)
    parser.add_argument("--x-field", required=True)
    parser.add_argument("--y-field", required=True)
    parser.add_argument("--value-field", required=True)
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--input-crs", default="EPSG:4326")
    parser.add_argument("--extent", default="")
    parser.add_argument("--cell-size", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=["kcn", "kcn_gat", "kcn_sage"], default="kcn")
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--hidden-sizes", default="8,8,8")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--es-patience", type=int, default=20)
    parser.add_argument("--length-scale", default="auto")
    parser.add_argument("--device", choices=["AUTO", "CUDA", "CPU"], default="AUTO")
    parser.add_argument("--random-seed", type=int, default=5)
    parser.add_argument("--prediction-batch-size", type=int, default=4096)
    parser.add_argument("--max-grid-cells", type=int, default=5000000)
    parser.add_argument("--run-log", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def parse_hidden_sizes(text):
    try:
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError:
        raise ValueError("hidden sizes must be comma-separated integers")
    if not values or any(value <= 0 for value in values):
        raise ValueError("hidden sizes must contain positive integers")
    return values


def read_long_csv(args):
    required = [
        args.time_field,
        args.station_field,
        args.x_field,
        args.y_field,
        args.value_field,
    ]
    rows = []
    skipped = []
    with open(args.input_csv, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("input CSV has no header")
        missing = [field for field in required if field not in reader.fieldnames]
        if missing:
            raise ValueError("input CSV is missing fields: " + ", ".join(missing))
        for line_number, row in enumerate(reader, 2):
            timestamp = str(row[args.time_field]).strip()
            station = str(row[args.station_field]).strip()
            if not timestamp or not station:
                raise ValueError("empty time or station at CSV line %d" % line_number)
            try:
                x_value = float(row[args.x_field])
                y_value = float(row[args.y_field])
            except (TypeError, ValueError):
                raise ValueError("non-numeric coordinate at CSV line %d" % line_number)
            value_text = row[args.value_field]
            if value_text is None or str(value_text).strip() == "":
                skipped.append(
                    {
                        "line": line_number,
                        "time": timestamp,
                        "station": station,
                        "reason": "missing value",
                    }
                )
                continue
            try:
                value = float(value_text)
            except (TypeError, ValueError):
                skipped.append(
                    {
                        "line": line_number,
                        "time": timestamp,
                        "station": station,
                        "reason": "non-numeric value",
                    }
                )
                continue
            if not all(math.isfinite(item) for item in [x_value, y_value]):
                raise ValueError("non-finite coordinate at CSV line %d" % line_number)
            if not math.isfinite(value):
                skipped.append(
                    {
                        "line": line_number,
                        "time": timestamp,
                        "station": station,
                        "reason": "non-finite value",
                    }
                )
                continue
            rows.append(
                {
                    "time": timestamp,
                    "station": station,
                    "x": x_value,
                    "y": y_value,
                    "value": value,
                }
            )
    if not rows:
        raise ValueError("input CSV has no data rows")
    return rows, skipped


def validate_input(rows, args):
    if args.cell_size <= 0:
        raise ValueError("cell size must be greater than zero")
    if args.n_neighbors < 1:
        raise ValueError("n-neighbors must be at least 1")
    if args.epochs < 1 or args.batch_size < 1 or args.prediction_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    if args.max_grid_cells < 1:
        raise ValueError("max grid cells must be positive")
    parse_hidden_sizes(args.hidden_sizes)

    seen = set()
    station_coords = {}
    for row in rows:
        key = (row["time"], row["station"])
        if key in seen:
            raise ValueError("duplicate time and station: %s, %s" % key)
        seen.add(key)
        coord = (row["x"], row["y"])
        previous = station_coords.get(row["station"])
        if previous is not None and previous != coord:
            raise ValueError("station coordinates change over time: %s" % row["station"])
        station_coords[row["station"]] = coord

    timestamps = sorted(set(row["time"] for row in rows))
    start_time = args.start_time or timestamps[0]
    end_time = args.end_time or timestamps[-1]
    if start_time not in timestamps:
        raise ValueError("start time is not present in the CSV: %s" % start_time)
    if end_time not in timestamps:
        raise ValueError("end time is not present in the CSV: %s" % end_time)
    if timestamps.index(end_time) < timestamps.index(start_time):
        raise ValueError("end time must not be earlier than start time")
    selected = timestamps[timestamps.index(start_time) : timestamps.index(end_time) + 1]
    return selected


def is_wgs84(input_crs):
    normalized = input_crs.strip().upper().replace(" ", "")
    return normalized in ["EPSG:4326", "4326", "WGS84", "WGS_1984"]


def make_transformer(input_crs):
    try:
        from pyproj import Transformer
    except ImportError:
        raise RuntimeError("pyproj is required when input CRS is not EPSG:4326")
    return Transformer.from_crs(input_crs, "EPSG:4326", always_xy=True)


def transform_coordinates(rows, input_crs):
    if is_wgs84(input_crs):
        transformed = [dict(row) for row in rows]
    else:
        transformer = make_transformer(input_crs)
        transformed = []
        for row in rows:
            item = dict(row)
            item["x"], item["y"] = transformer.transform(row["x"], row["y"])
            transformed.append(item)
    for row in transformed:
        if not (-180.0 <= row["x"] <= 180.0 and -90.0 <= row["y"] <= 90.0):
            raise ValueError("transformed coordinate is outside WGS84 bounds")
    return transformed


def parse_extent(text, rows, input_crs, cell_size):
    value = (text or "").strip()
    if value.upper() in EXTENT_KEYWORDS:
        xs = [row["x"] for row in rows]
        ys = [row["y"] for row in rows]
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        if xmin == xmax:
            xmin = max(-180.0, xmin - cell_size / 2.0)
            xmax = min(180.0, xmax + cell_size / 2.0)
        if ymin == ymax:
            ymin = max(-90.0, ymin - cell_size / 2.0)
            ymax = min(90.0, ymax + cell_size / 2.0)
        return [xmin, ymin, xmax, ymax]
    parts = [part for part in re.split(r"[\s,;]+", value) if part]
    if len(parts) != 4:
        raise ValueError("extent must contain xmin ymin xmax ymax")
    extent = [float(part) for part in parts]
    xmin, ymin, xmax, ymax = extent
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("extent minimums must be smaller than maximums")
    if not is_wgs84(input_crs):
        transformer = make_transformer(input_crs)
        corners = [
            transformer.transform(xmin, ymin),
            transformer.transform(xmin, ymax),
            transformer.transform(xmax, ymin),
            transformer.transform(xmax, ymax),
        ]
        xs = [item[0] for item in corners]
        ys = [item[1] for item in corners]
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        extent = [xmin, ymin, xmax, ymax]
    if not (-180.0 <= xmin < xmax <= 180.0 and -90.0 <= ymin < ymax <= 90.0):
        raise ValueError("transformed extent must be valid EPSG:4326 coordinates")
    return extent


def build_grid(extent, cell_size, max_grid_cells):
    xmin, ymin, xmax, ymax = extent
    ncols = int(math.ceil((xmax - xmin) / cell_size))
    nrows = int(math.ceil((ymax - ymin) / cell_size))
    if ncols < 1 or nrows < 1:
        raise ValueError("extent and cell size produce an empty grid")
    if nrows * ncols > max_grid_cells:
        raise ValueError(
            "grid has %d cells; increase cell size or max-grid-cells" % (nrows * ncols)
        )
    snapped_xmax = xmin + ncols * cell_size
    snapped_ymin = ymax - nrows * cell_size
    snapped_extent = [xmin, snapped_ymin, snapped_xmax, ymax]
    x_centers = xmin + (np.arange(ncols, dtype=np.float32) + 0.5) * cell_size
    y_centers = ymax - (np.arange(nrows, dtype=np.float32) + 0.5) * cell_size
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    query_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    return query_coords, nrows, ncols, snapped_extent


def make_model_args(args, torch):
    class ModelArgs(object):
        pass

    model_args = ModelArgs()
    model_args.n_neighbors = args.n_neighbors
    model_args.length_scale = args.length_scale
    if args.length_scale != "auto":
        model_args.length_scale = float(args.length_scale)
    model_args.hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    model_args.dropout = args.dropout
    model_args.last_activation = "none"
    model_args.model = args.model
    if args.device == "AUTO":
        model_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "CUDA":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        model_args.device = torch.device("cuda")
    else:
        model_args.device = torch.device("cpu")
    return model_args


def train_and_interpolate(timestamp, time_rows, query_coords, args):
    import torch

    from model import data
    from model import kcn

    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    coords = np.asarray([[row["x"], row["y"]] for row in time_rows], dtype=np.float32)
    values = np.asarray([[row["value"]] for row in time_rows], dtype=np.float32)
    if coords.shape[0] <= args.n_neighbors:
        raise ValueError(
            "time %s has %d stations; n-neighbors must be smaller"
            % (timestamp, coords.shape[0])
        )

    feature_mean = coords.mean(axis=0, keepdims=True)
    feature_std = coords.std(axis=0, keepdims=True)
    train_features = (coords - feature_mean) / (feature_std + 0.01)
    query_features = (query_coords - feature_mean) / (feature_std + 0.01)
    trainset = data.SpatialDataset(coords=coords, features=train_features, y=values)
    model_args = make_model_args(args, torch)
    model = kcn.KCN(trainset, model_args).to(model_args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_func = torch.nn.MSELoss(reduction="mean")

    indices = np.arange(len(trainset))
    random_state = np.random.RandomState(args.random_seed)
    random_state.shuffle(indices)
    valid_count = int(round(len(indices) * args.validation_fraction))
    if valid_count >= len(indices):
        valid_count = len(indices) - 1
    valid_indices = indices[-valid_count:] if valid_count else np.asarray([], dtype=int)
    train_indices = indices[:-valid_count] if valid_count else indices

    best_loss = None
    best_state = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        for offset in range(0, len(train_indices), args.batch_size):
            batch_indices = train_indices[offset : offset + args.batch_size]
            batch_coords, batch_features, batch_y = trainset[batch_indices]
            prediction = model(batch_coords, batch_features, batch_indices)
            loss = loss_func(prediction, batch_y.to(model_args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            if len(valid_indices):
                valid_coords, valid_features, valid_y = trainset[valid_indices]
                valid_prediction = model(valid_coords, valid_features, valid_indices)
                monitor_loss = loss_func(
                    valid_prediction, valid_y.to(model_args.device)
                ).item()
            else:
                monitor_loss = float(np.mean(batch_losses))
        print(
            "[KCN] time=%s epoch=%d train_loss=%.8f monitor_loss=%.8f"
            % (timestamp, epoch + 1, float(np.mean(batch_losses)), monitor_loss),
            flush=True,
        )
        if best_loss is None or monitor_loss < best_loss:
            best_loss = monitor_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.es_patience:
                print("[KCN] early stopping at epoch %d" % (epoch + 1), flush=True)
                break

    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(query_coords), args.prediction_batch_size):
            end = offset + args.prediction_batch_size
            print(
                "[KCN] predicting cells %d-%d of %d for %s"
                % (offset + 1, min(end, len(query_coords)), len(query_coords), timestamp),
                flush=True,
            )
            batch_coords = torch.from_numpy(query_coords[offset:end])
            batch_features = torch.from_numpy(query_features[offset:end].astype(np.float32))
            batch_prediction = model(batch_coords, batch_features)
            predictions.append(batch_prediction.detach().cpu().numpy())

    checkpoint = {
        "timestamp": timestamp,
        "state_dict": best_state,
        "coords": coords,
        "values": values,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "model": args.model,
        "n_neighbors": args.n_neighbors,
        "hidden_sizes": parse_hidden_sizes(args.hidden_sizes),
        "length_scale": model.length_scale,
    }
    return np.concatenate(predictions, axis=0)[:, 0], checkpoint, str(model_args.device)


def safe_timestamp(timestamp):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", timestamp)
    return value or "time"


def write_manifest(path, manifest):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, path)


def ensure_outputs(args, selected_times):
    os.makedirs(args.output_dir, exist_ok=True)
    paths = [
        os.path.join(args.output_dir, "trained_model.pyt"),
        os.path.join(args.output_dir, "result_manifest.json"),
    ]
    paths.extend(
        os.path.join(args.output_dir, "interpolation_%s.npz" % safe_timestamp(item))
        for item in selected_times
    )
    existing = [path for path in paths if os.path.exists(path)]
    if existing and not args.overwrite:
        raise ValueError("output exists and overwrite is disabled: %s" % existing[0])


def run(args):
    rows, skipped = read_long_csv(args)
    selected_times = validate_input(rows, args)
    transformed_rows = transform_coordinates(rows, args.input_crs)
    selected_rows = [row for row in transformed_rows if row["time"] in selected_times]
    extent = parse_extent(args.extent, selected_rows, args.input_crs, args.cell_size)
    query_coords, nrows, ncols, extent = build_grid(
        extent, args.cell_size, args.max_grid_cells
    )
    if args.validate_only:
        print(
            "[KCN] validation OK: rows=%d times=%d grid=%dx%d"
            % (len(rows), len(selected_times), nrows, ncols)
        )
        return 0

    ensure_outputs(args, selected_times)
    model_path = os.path.abspath(os.path.join(args.output_dir, "trained_model.pyt"))
    manifest_path = os.path.abspath(
        os.path.join(args.output_dir, "result_manifest.json")
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "model_name": MODEL_NAME,
        "coordinate_system": "EPSG:4326",
        "input_coordinate_system": args.input_crs,
        "extent": extent,
        "cell_size": args.cell_size,
        "nrows": nrows,
        "ncols": ncols,
        "station_count": len(set(row["station"] for row in selected_rows)),
        "training_timestamp_count": len(selected_times),
        "model_path": model_path,
        "run_log": os.path.abspath(args.run_log) if args.run_log else "",
        "training_parameters": {
            "model": args.model,
            "n_neighbors": args.n_neighbors,
            "hidden_sizes": parse_hidden_sizes(args.hidden_sizes),
            "dropout": args.dropout,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "validation_fraction": args.validation_fraction,
            "es_patience": args.es_patience,
            "length_scale": args.length_scale,
            "device": args.device,
            "random_seed": args.random_seed,
        },
        "results": [],
        "skipped": skipped,
    }
    write_manifest(manifest_path, manifest)

    import torch

    checkpoints = {}
    for position, timestamp in enumerate(selected_times, 1):
        print(
            "[KCN] interpolating time %d/%d: %s"
            % (position, len(selected_times), timestamp),
            flush=True,
        )
        time_rows = [row for row in selected_rows if row["time"] == timestamp]
        prediction, checkpoint, device = train_and_interpolate(
            timestamp, time_rows, query_coords, args
        )
        grid = prediction.reshape(nrows, ncols).astype(np.float32)
        array_path = os.path.abspath(
            os.path.join(
                args.output_dir,
                "interpolation_%s.npz" % safe_timestamp(timestamp),
            )
        )
        np.savez_compressed(array_path, grid=grid, timestamp=np.asarray(timestamp))
        print("[KCN] wrote array: %s" % array_path, flush=True)
        checkpoints[timestamp] = checkpoint
        torch.save(
            {
                "format_version": FORMAT_VERSION,
                "model_name": MODEL_NAME,
                "checkpoints": checkpoints,
                "training_parameters": manifest["training_parameters"],
            },
            model_path,
        )
        manifest["results"].append(
            {
                "timestamp": timestamp,
                "array_path": array_path,
                "valid_station_count": len(time_rows),
                "prediction_min": float(np.min(grid)),
                "prediction_max": float(np.max(grid)),
                "prediction_mean": float(np.mean(grid)),
                "device": device,
            }
        )
        write_manifest(manifest_path, manifest)
    write_manifest(manifest_path, manifest)
    print("[KCN] manifest: %s" % manifest_path, flush=True)
    return 0


def main(argv=None):
    try:
        args = parse_args(argv)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        return run(args)
    except Exception as exc:
        print("[KCN ERROR] %s" % exc, file=sys.stderr, flush=True)
        if os.environ.get("KCN_DEBUG") == "1":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
