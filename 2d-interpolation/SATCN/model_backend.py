# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from loss import masked_mae
from model import general_satcn
from sample_gene import ZerooneScaler, sadj_transform, sample_processing, test_processing


MODEL_NAME = "SATCN_TIME_SERIES_SPATIAL_INTERPOLATION"
DEFAULT_AGGREGATORS = ["mean", "softmin", "softmax", "normalised_mean", "std"]
DEFAULT_SCALERS = ["identity", "amplification", "attenuation"]


def parse_args():
    parser = argparse.ArgumentParser(description="SATCN time series spatial interpolation backend")
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
    parser.add_argument("--cell-size", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="AUTO", choices=["AUTO", "CUDA", "CPU"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--t-kernel", type=int, default=2)
    parser.add_argument("--least-k", type=int, default=8)
    parser.add_argument("--masked-nodes", type=int, default=8)
    parser.add_argument("--distance-bandwidth", type=float, default=0.0)
    parser.add_argument("--max-cells", type=int, default=250000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def fail(message):
    print("[MODEL ERROR] " + message, file=sys.stderr)
    return 1


def choose_device(requested):
    if requested == "CPU":
        return "cpu"
    if requested == "CUDA":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_long_csv(args):
    if not os.path.isfile(args.input_csv):
        raise ValueError("Input CSV does not exist: " + args.input_csv)
    df = pd.read_csv(args.input_csv)
    mapping = {
        args.time_field: "time",
        args.station_field: "station",
        args.x_field: "x",
        args.y_field: "y",
        args.value_field: "value",
    }
    missing = [name for name in mapping if name not in df.columns]
    if missing:
        raise ValueError("Missing CSV fields: " + ", ".join(missing))
    df = df.rename(columns=mapping)[["time", "station", "x", "y", "value"]]
    df["time"] = df["time"].astype(str)
    df["station"] = df["station"].astype(str)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df[["time", "station", "x", "y"]].isnull().any().any():
        raise ValueError("Input CSV has empty time, station, x, or y values")
    duplicated = df.duplicated(["time", "station"])
    if duplicated.any():
        row = int(np.where(duplicated.values)[0][0]) + 2
        raise ValueError("Duplicate time and station pair near CSV row " + str(row))
    coord_counts = df.groupby("station")[["x", "y"]].nunique()
    bad_coord = coord_counts[(coord_counts["x"] > 1) | (coord_counts["y"] > 1)]
    if len(bad_coord) > 0:
        raise ValueError("Station coordinates are inconsistent: " + str(bad_coord.index[0]))
    times = sorted(df["time"].unique().tolist())
    if args.start_time:
        if args.start_time not in times:
            raise ValueError("Start time is not present in CSV: " + args.start_time)
        times = [item for item in times if item >= args.start_time]
    if args.end_time:
        if args.end_time not in df["time"].unique().tolist():
            raise ValueError("End time is not present in CSV: " + args.end_time)
        times = [item for item in times if item <= args.end_time]
    if len(times) == 0:
        raise ValueError("No timestamps remain after time filtering")
    stations = sorted(df["station"].unique().tolist())
    coords = df.drop_duplicates("station").set_index("station").loc[stations][["x", "y"]].values
    pivot = df.pivot(index="station", columns="time", values="value").reindex(index=stations, columns=times)
    values = pivot.values.astype(np.float32)
    observed = np.isfinite(values)
    values = np.where(observed, values, 0.0).astype(np.float32)
    if observed.sum() == 0:
        raise ValueError("No valid numeric values were found in the selected time range")
    return {
        "df": df,
        "times": times,
        "stations": stations,
        "station_coords": coords.astype(np.float32),
        "values": values,
        "observed": observed,
    }


def parse_extent(extent_text, coords, cell_size):
    if extent_text.strip() and extent_text.strip().upper() not in ["MAXOF", "MINOF", "DISPLAY", "DEFAULT", "#"]:
        parts = extent_text.replace(",", " ").split()
        if len(parts) != 4:
            raise ValueError("Extent must contain xmin ymin xmax ymax")
        extent = [float(item) for item in parts]
    else:
        xmin = float(np.min(coords[:, 0]))
        xmax = float(np.max(coords[:, 0]))
        ymin = float(np.min(coords[:, 1]))
        ymax = float(np.max(coords[:, 1]))
        pad = float(cell_size)
        extent = [xmin - pad, ymin - pad, xmax + pad, ymax + pad]
    xmin, ymin, xmax, ymax = extent
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("Invalid extent: xmin/ymin must be smaller than xmax/ymax")
    if not (-180.0 <= xmin < xmax <= 180.0 and -90.0 <= ymin < ymax <= 90.0):
        raise ValueError("Extent must be WGS84 lon/lat bounds")
    return extent


def build_grid(extent, cell_size, max_cells):
    if cell_size <= 0:
        raise ValueError("Cell size must be greater than zero")
    xmin, ymin, xmax, ymax = extent
    ncols = int(math.floor((xmax - xmin) / cell_size)) + 1
    nrows = int(math.floor((ymax - ymin) / cell_size)) + 1
    if nrows <= 0 or ncols <= 0:
        raise ValueError("Grid has no cells")
    if nrows * ncols > max_cells:
        raise ValueError("Grid has too many cells; increase cell size or reduce extent")
    xs = xmin + np.arange(ncols, dtype=np.float32) * cell_size
    ys = ymax - np.arange(nrows, dtype=np.float32) * cell_size
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32)
    return coords, nrows, ncols


def pairwise_distance(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)


def build_similarity(coords, bandwidth):
    dist = pairwise_distance(coords)
    positive = dist[dist > 0]
    if len(positive) == 0:
        raise ValueError("At least two distinct locations are required")
    if bandwidth <= 0:
        bandwidth = float(np.median(positive))
    bandwidth = max(float(bandwidth), 1e-6)
    adj = np.exp(-dist / bandwidth).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj, bandwidth


def make_time_mask_adj(adj, values_window, least_k):
    seq_len = values_window.shape[1]
    base = np.expand_dims(adj, 0).repeat(seq_len, axis=0)
    missing = values_window == 0
    for t in range(seq_len):
        base[t, :, missing[:, t]] = 0.0
    return test_processing(base, least_k)


def output_manifest(path, manifest):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fobj:
        json.dump(manifest, fobj, indent=2, sort_keys=True)
    if os.path.isfile(path):
        os.remove(path)
    os.rename(tmp_path, path)


def create_model(avg_d, args, device):
    model = general_satcn(
        avg_d,
        device,
        in_variables=1,
        layers=args.layers,
        channels=args.channels,
        t_kernel=args.t_kernel,
        aggragators=DEFAULT_AGGREGATORS,
        scalers=DEFAULT_SCALERS,
        masking=True,
        dropout=0.0,
    )
    return model.to(device)


def train_model(data, adj_station, args, device):
    values = data["values"]
    observed_values = np.where(data["observed"], values, 0.0).astype(np.float32)
    t_reduce = (args.layers + 1) * (args.t_kernel - 1)
    seq_len = t_reduce + 1
    if observed_values.shape[1] < seq_len:
        raise ValueError("Not enough timestamps for the selected layers and temporal kernel")
    least_k = min(max(1, args.least_k), max(1, adj_station.shape[0] - 1))
    masked_nodes = min(max(1, args.masked_nodes), max(1, adj_station.shape[0] - 1))
    adj_sparse = sadj_transform(adj_station.copy(), least_k)
    avg_d = {"log": torch.tensor(np.mean(np.log(np.sum(adj_sparse, axis=0) + 1.0)), device=device)}
    scaler_max = float(np.max(np.abs(observed_values)))
    if scaler_max <= 0:
        scaler_max = 1.0
    scaler = ZerooneScaler(scaler_max)
    model = create_model(avg_d, args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    adj_static = torch.tensor(adj_sparse, dtype=torch.float32, device=device)
    window_starts = list(range(0, observed_values.shape[1] - seq_len + 1))
    if len(window_starts) == 0:
        raise ValueError("No training windows are available")
    for epoch in range(1, args.epochs + 1):
        random.shuffle(window_starts)
        losses = []
        for offset in range(0, len(window_starts), args.batch_size):
            starts = window_starts[offset:offset + args.batch_size]
            x_list = []
            adj_list = []
            for start in starts:
                window = observed_values[:, start:start + seq_len]
                x_list.append(window)
                adj_list.append(make_time_mask_adj(adj_station.copy(), window, least_k))
            x_np = np.stack(x_list, axis=0)
            x_np = np.expand_dims(x_np, 1).transpose([0, 1, 3, 2])
            adj_np = np.stack(adj_list, axis=0)
            x_np, y_np, adj_np = sample_processing(x_np, adj_np, least_k, masked_nodes, t_reduce)
            x_np = scaler.transform(x_np)
            y_np = scaler.transform(y_np)
            train_x = torch.tensor(x_np, dtype=torch.float32, device=device)
            train_y = torch.tensor(y_np, dtype=torch.float32, device=device)
            adj_mask = torch.tensor(adj_np, dtype=torch.float32, device=device)
            model.train()
            optimizer.zero_grad()
            pred = model(train_x, adj_static, adj_mask)
            loss = masked_mae(pred, train_y, 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print("Epoch {0}/{1}, loss {2:.6f}".format(epoch, args.epochs, float(np.mean(losses))), flush=True)
    return model, scaler, adj_sparse, seq_len, t_reduce, least_k


def interpolate(data, grid_coords, adj_all, args, device, model, scaler, seq_len, t_reduce, least_k, nrows, ncols, manifest):
    station_count = len(data["stations"])
    values = np.where(data["observed"], data["values"], 0.0).astype(np.float32)
    all_count = adj_all.shape[0]
    adj_sparse = sadj_transform(adj_all.copy(), least_k)
    adj_static = torch.tensor(adj_sparse, dtype=torch.float32, device=device)
    results = []
    times = data["times"]
    model.eval()
    for out_idx, timestamp in enumerate(times):
        start = out_idx - t_reduce
        window = np.zeros((all_count, seq_len), dtype=np.float32)
        for local_t in range(seq_len):
            src_idx = start + local_t
            if src_idx < 0:
                src_idx = 0
            if src_idx >= len(times):
                src_idx = len(times) - 1
            window[:station_count, local_t] = values[:, src_idx]
        adj_mask = make_time_mask_adj(adj_all.copy(), window, least_k)
        x_np = np.expand_dims(np.expand_dims(window, 0), 1).transpose([0, 1, 3, 2])
        x_np = scaler.transform(x_np)
        test_x = torch.tensor(x_np, dtype=torch.float32, device=device)
        test_adj = torch.tensor(np.expand_dims(adj_mask, 0), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(test_x, adj_static, test_adj)
            pred = scaler.inverse_transform(pred).detach().cpu().numpy()
        grid_values = pred[0, 0, 0, station_count:].reshape(nrows, ncols).astype(np.float32)
        safe_time = "".join([c if c.isalnum() or c in ["-", "_"] else "_" for c in timestamp])
        npz_path = os.path.abspath(os.path.join(args.output_dir, "interpolation_" + safe_time + ".npz"))
        np.savez_compressed(npz_path, grid=grid_values, timestamp=timestamp, x=grid_coords[:, 0], y=grid_coords[:, 1])
        item = {
            "timestamp": timestamp,
            "array_path": npz_path,
            "valid_station_count": int(np.sum(data["observed"][:, out_idx])),
            "prediction_min": float(np.nanmin(grid_values)),
            "prediction_max": float(np.nanmax(grid_values)),
            "prediction_mean": float(np.nanmean(grid_values)),
        }
        results.append(item)
        manifest["results"] = results
        output_manifest(os.path.join(args.output_dir, "result_manifest.json"), manifest)
        print("Interpolated {0} ({1}/{2})".format(timestamp, out_idx + 1, len(times)), flush=True)
    return results


def main():
    args = parse_args()
    try:
        set_seed(args.seed)
        device = choose_device(args.device)
        os.makedirs(args.output_dir, exist_ok=True)
        manifest_path = os.path.join(args.output_dir, "result_manifest.json")
        if os.path.exists(manifest_path) and not args.overwrite and not args.validate_only:
            raise ValueError("Output manifest exists; enable overwrite to replace results")
        data = read_long_csv(args)
        extent = parse_extent(args.extent, data["station_coords"], args.cell_size)
        grid_coords, nrows, ncols = build_grid(extent, args.cell_size, args.max_cells)
        if args.validate_only:
            print("Validation OK", flush=True)
            return 0
        all_coords = np.vstack([data["station_coords"], grid_coords]).astype(np.float32)
        adj_station, bandwidth = build_similarity(data["station_coords"], args.distance_bandwidth)
        adj_all, _ = build_similarity(all_coords, bandwidth)
        start_clock = time.time()
        model, scaler, adj_sparse, seq_len, t_reduce, least_k = train_model(data, adj_station, args, device)
        model_path = os.path.abspath(os.path.join(args.output_dir, "trained_model.pth"))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler_max": scaler.max,
                "args": vars(args),
                "distance_bandwidth": bandwidth,
            },
            model_path,
        )
        manifest = {
            "format_version": 1,
            "model_name": MODEL_NAME,
            "coordinate_system": "EPSG:4326",
            "input_coordinate_system": args.input_crs,
            "extent": [float(item) for item in extent],
            "cell_size": float(args.cell_size),
            "nrows": int(nrows),
            "ncols": int(ncols),
            "station_count": int(len(data["stations"])),
            "training_timestamp_count": int(len(data["times"])),
            "model_path": model_path,
            "training_parameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "layers": args.layers,
                "channels": args.channels,
                "t_kernel": args.t_kernel,
                "least_k": least_k,
                "masked_nodes": args.masked_nodes,
                "sequence_length": seq_len,
                "device": device,
                "seed": args.seed,
                "distance_bandwidth": bandwidth,
                "training_time_seconds": round(time.time() - start_clock, 3),
            },
            "results": [],
            "skipped": [],
        }
        output_manifest(manifest_path, manifest)
        results = interpolate(data, grid_coords, adj_all, args, device, model, scaler, seq_len, t_reduce, least_k, nrows, ncols, manifest)
        manifest["results"] = results
        output_manifest(manifest_path, manifest)
        print("Done. Manifest: " + os.path.abspath(manifest_path), flush=True)
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    sys.exit(main())
