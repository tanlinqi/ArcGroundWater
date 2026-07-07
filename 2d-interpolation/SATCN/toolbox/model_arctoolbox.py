# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import os
import subprocess
import sys

import arcpy
import numpy as np


def text_param(index):
    value = arcpy.GetParameterAsText(index)
    return value if value is not None else ""


def bool_param(index):
    value = arcpy.GetParameter(index)
    return bool(value)


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def spatial_reference_text(sr):
    if not sr:
        return "EPSG:4326"
    try:
        code = int(sr.factoryCode)
        if code > 0:
            return "EPSG:{0}".format(code)
    except Exception:
        pass
    return "EPSG:4326"


def extent_text(extent):
    if not extent:
        return ""
    text = str(extent).strip()
    if text.upper() in ["MAXOF", "MINOF", "DISPLAY", "DEFAULT", "#"]:
        return text
    try:
        return "{0} {1} {2} {3}".format(extent.XMin, extent.YMin, extent.XMax, extent.YMax)
    except Exception:
        return text


def arcground_root():
    return os.path.abspath(os.path.join(project_root(), os.pardir, os.pardir))


def model_python_path(value):
    if value and value != "#":
        return value
    return os.path.join(arcground_root(), "envs", "ssin", "python.exe")

def run_backend(command, cwd):
    arcpy.AddMessage("Starting SATCN backend...")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in iter(process.stdout.readline, b""):
        if not line:
            break
        try:
            arcpy.AddMessage(line.decode("utf-8", "replace").rstrip())
        except TypeError:
            arcpy.AddMessage(line.rstrip())
    code = process.wait()
    if code != 0:
        raise arcpy.ExecuteError("SATCN backend failed with exit code {0}".format(code))


def create_rasters(manifest_path, output_dir, cell_size):
    with open(manifest_path, "r") as fobj:
        manifest = json.load(fobj)
    xmin, ymin, xmax, ymax = manifest["extent"]
    sr = arcpy.SpatialReference(4326)
    rasters = []
    for item in manifest.get("results", []):
        npz = np.load(item["array_path"])
        arr = npz["grid"].astype("float32")
        name = "interpolation_{0}.tif".format(safe_name(item["timestamp"]))
        raster_path = os.path.join(output_dir, name)
        lower_left = arcpy.Point(float(xmin), float(ymin))
        raster = arcpy.NumPyArrayToRaster(arr, lower_left, float(cell_size), float(cell_size))
        raster.save(raster_path)
        arcpy.DefineProjection_management(raster_path, sr)
        try:
            mxd = arcpy.mapping.MapDocument("CURRENT")
            data_frame = arcpy.mapping.ListDataFrames(mxd)[0]
            layer = arcpy.mapping.Layer(raster_path)
            arcpy.mapping.AddLayer(data_frame, layer, "TOP")
        except Exception:
            pass
        rasters.append(raster_path)
    return rasters


def safe_name(value):
    chars = []
    for char in str(value):
        if char.isalnum() or char in ["-", "_"]:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars)


def main():
    input_csv = text_param(0)
    time_field = text_param(1)
    station_field = text_param(2)
    x_field = text_param(3)
    y_field = text_param(4)
    value_field = text_param(5)
    start_time = text_param(6)
    end_time = text_param(7)
    input_sr = arcpy.GetParameter(8)
    extent = arcpy.GetParameter(9)
    cell_size = text_param(10)
    output_dir = text_param(11)
    epochs = text_param(12) or "20"
    batch_size = text_param(13) or "8"
    learning_rate = text_param(14) or "0.001"
    channels = text_param(15) or "64"
    layers = text_param(16) or "1"
    t_kernel = text_param(17) or "2"
    least_k = text_param(18) or "8"
    masked_nodes = text_param(19) or "8"
    device = (text_param(20) or "AUTO").upper()
    seed = text_param(21) or "0"
    model_python = model_python_path(text_param(22))
    overwrite = bool_param(23)
    root = project_root()
    backend = os.path.join(root, "model_backend.py")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    command = [
        model_python,
        "-u",
        backend,
        "--input-csv",
        input_csv,
        "--time-field",
        time_field,
        "--station-field",
        station_field,
        "--x-field",
        x_field,
        "--y-field",
        y_field,
        "--value-field",
        value_field,
        "--input-crs",
        spatial_reference_text(input_sr),
        "--extent",
        extent_text(extent),
        "--cell-size",
        cell_size,
        "--output-dir",
        output_dir,
        "--epochs",
        epochs,
        "--batch-size",
        batch_size,
        "--learning-rate",
        learning_rate,
        "--channels",
        channels,
        "--layers",
        layers,
        "--t-kernel",
        t_kernel,
        "--least-k",
        least_k,
        "--masked-nodes",
        masked_nodes,
        "--device",
        device,
        "--seed",
        seed,
    ]
    if start_time:
        command.extend(["--start-time", start_time])
    if end_time:
        command.extend(["--end-time", end_time])
    if overwrite:
        command.append("--overwrite")
    run_backend(command, root)
    manifest_path = os.path.join(output_dir, "result_manifest.json")
    rasters = create_rasters(manifest_path, output_dir, cell_size)
    if len(rasters) > 0:
        arcpy.SetParameterAsText(24, rasters[0])
    arcpy.SetParameterAsText(25, os.path.join(output_dir, "trained_model.pth"))
    arcpy.SetParameterAsText(26, manifest_path)
    arcpy.AddMessage("Created {0} raster(s).".format(len(rasters)))


if __name__ == "__main__":
    main()


