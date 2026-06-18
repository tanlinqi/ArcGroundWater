from __future__ import print_function

import json
import os
import subprocess
import sys
import ctypes
import datetime
import traceback

import arcpy
import numpy as np


FS_ENCODING = sys.getfilesystemencoding() or "mbcs"
MODEL_KEY = "ignnk"
CURRENT_LOG_HANDLE = None


def to_unicode(value):
    if value is None:
        return u""
    if isinstance(value, unicode):
        return value
    return str(value).decode(FS_ENCODING)


def to_native(value):
    if value is None:
        return ""
    if isinstance(value, unicode):
        return value.encode(FS_ENCODING)
    return str(value)


def short_path(path):
    try:
        value = to_unicode(path)
        buffer_size = 1024
        buffer = ctypes.create_unicode_buffer(buffer_size)
        length = ctypes.windll.kernel32.GetShortPathNameW(
            value, buffer, buffer_size
        )
        if length:
            return buffer.value
    except Exception:
        pass
    return path


def text(index):
    return arcpy.GetParameterAsText(index)


def boolean(index):
    return bool(arcpy.GetParameter(index))


def normalize_extent(value):
    if not value:
        return ""
    upper = value.strip().upper()
    if upper in ("#", "MAXOF", "MINOF", "DISPLAY", "DEFAULT"):
        return ""
    return value


def spatial_reference_text(spatial_reference):
    if spatial_reference.factoryCode:
        return "EPSG:%d" % spatial_reference.factoryCode
    return spatial_reference.exportToString()


def add(command, name, value):
    command.extend([name, to_native(value)])


def log_line(log_handle, message):
    text_value = to_unicode(message)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = u"[%s] %s" % (stamp, text_value)
    log_handle.write(line.encode("utf-8") + "\n")
    log_handle.flush()


def build_log_path(project_root):
    log_dir = os.path.abspath(
        os.path.join(project_root, os.pardir, "log", MODEL_KEY + "-log")
    )
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, MODEL_KEY + "_run_" + stamp + ".log")


def run_backend(command, project_root, log_handle):
    native_command = [to_native(item) for item in command]
    native_cwd = to_native(short_path(project_root))
    log_line(log_handle, "Backend command: %s" % " ".join(native_command))
    process = subprocess.Popen(
        native_command,
        cwd=native_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    for line in iter(process.stdout.readline, ""):
        if line:
            clean = line.rstrip()
            arcpy.AddMessage(clean)
            log_line(log_handle, clean)
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError("IGNNK backend failed with exit code %d" % return_code)


def create_rasters(manifest_path, output_dir, add_to_map=True):
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    xmin, ymin, xmax, ymax = manifest["extent"]
    cell_size = float(manifest["cell_size"])
    spatial_reference = arcpy.SpatialReference(4326)
    raster_paths = []
    for result in manifest["results"]:
        data = np.load(result["array_path"])
        grid = data["grid"].astype(np.float32)
        timestamp = result["timestamp"]
        safe_time = "".join(
            character if character.isalnum() or character in "-_."
            else "_"
            for character in timestamp
        )
        if isinstance(safe_time, unicode):
            safe_time = safe_time.encode("ascii", "replace")
        raster_path = os.path.join(
            output_dir, "interpolation_%s.tif" % safe_time
        )
        raster = arcpy.NumPyArrayToRaster(
            grid,
            arcpy.Point(xmin, ymin),
            cell_size,
            cell_size,
            -9999,
        )
        raster.save(raster_path)
        arcpy.DefineProjection_management(raster_path, spatial_reference)
        result["raster_path"] = os.path.abspath(raster_path)
        raster_paths.append(os.path.abspath(raster_path))
        if add_to_map:
            try:
                document = arcpy.mapping.MapDocument("CURRENT")
                data_frame = arcpy.mapping.ListDataFrames(document)[0]
                layer = arcpy.mapping.Layer(raster_path)
                arcpy.mapping.AddLayer(data_frame, layer, "TOP")
            except Exception as exc:
                arcpy.AddWarning("Could not add raster to current map: %s" % exc)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return raster_paths, manifest["model_path"]


def create_rasters_logged(manifest_path, output_dir, log_handle, add_to_map=True):
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    xmin, ymin, xmax, ymax = manifest["extent"]
    cell_size = float(manifest["cell_size"])
    spatial_reference = arcpy.SpatialReference(4326)
    raster_paths = []
    for result in manifest["results"]:
        log_line(log_handle, "Reading array: %s" % result["array_path"])
        data = np.load(result["array_path"])
        grid = data["grid"].astype(np.float32)
        timestamp = result["timestamp"]
        safe_time = "".join(
            character if character.isalnum() or character in "-_."
            else "_"
            for character in timestamp
        )
        if isinstance(safe_time, unicode):
            safe_time = safe_time.encode("ascii", "replace")
        raster_path = os.path.join(
            output_dir, "interpolation_%s.tif" % safe_time
        )
        raster = arcpy.NumPyArrayToRaster(
            grid,
            arcpy.Point(xmin, ymin),
            cell_size,
            cell_size,
            -9999,
        )
        raster.save(raster_path)
        arcpy.DefineProjection_management(raster_path, spatial_reference)
        result["raster_path"] = os.path.abspath(raster_path)
        raster_paths.append(os.path.abspath(raster_path))
        log_line(log_handle, "GeoTIFF: %s" % os.path.abspath(raster_path))
        if add_to_map:
            try:
                document = arcpy.mapping.MapDocument("CURRENT")
                data_frame = arcpy.mapping.ListDataFrames(document)[0]
                layer = arcpy.mapping.Layer(raster_path)
                arcpy.mapping.AddLayer(data_frame, layer, "TOP")
                log_line(log_handle, "ArcMap load OK: %s" % raster_path)
            except Exception as exc:
                warning = "Could not add raster to current map: %s" % exc
                arcpy.AddWarning(warning)
                log_line(log_handle, "WARNING: %s" % warning)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return raster_paths, manifest["model_path"]


def main():
    global CURRENT_LOG_HANDLE
    status = "FAILED"
    log_handle = None
    log_path = None
    start_time = datetime.datetime.now()
    input_csv = text(0)
    output_dir = text(11)
    arcpy.env.overwriteOutput = boolean(31)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_script = os.path.join(project_root, "model_backend.py")
    log_path = build_log_path(project_root)
    log_handle = open(log_path, "ab")
    CURRENT_LOG_HANDLE = log_handle
    log_line(log_handle, "Start time: %s" % start_time.isoformat())
    log_line(log_handle, "ArcMap wrapper: %s" % os.path.abspath(__file__))
    log_line(log_handle, "Python 3 backend: %s" % backend_script)
    model_python = text(30)
    if not model_python or model_python == "#":
        model_python = r"C:\Users\Lenovo\Miniconda3\envs\ssin\python.exe"
    log_line(log_handle, "Python 3 interpreter: %s" % model_python)
    log_line(log_handle, "Input CSV: %s" % input_csv)
    log_line(
        log_handle,
        "Field mapping: time=%s station=%s x=%s y=%s value=%s"
        % (text(1), text(2), text(3), text(4), text(5)),
    )
    log_line(log_handle, "Time range: %s to %s" % (text(6), text(7)))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    input_spatial_reference = arcpy.GetParameter(8)
    log_line(
        log_handle,
        "Input coordinate system: %s"
        % spatial_reference_text(input_spatial_reference),
    )
    log_line(log_handle, "Extent: %s" % (text(9) or ""))
    log_line(log_handle, "Cell size: %s" % text(10))
    log_line(log_handle, "Output folder: %s" % output_dir)
    log_line(
        log_handle,
        "Training parameters: epochs=%s time_window=%s hidden_dim=%s diffusion_order=%s masked_stations=%s learning_rate=%s batch_size=%s seed=%s device=%s"
        % (
            text(12),
            text(13),
            text(14),
            text(15),
            text(16),
            text(17),
            text(18),
            text(25),
            text(24),
        ),
    )
    log_line(
        log_handle,
        "Inference parameters: query_batch_size=%s distance_scale_km=%s adjacency_threshold=%s value_scale=%s max_grid_cells=%s missing_zero=%s overwrite=%s"
        % (
            text(19),
            text(20),
            text(21),
            text(22),
            text(26),
            boolean(23),
            boolean(31),
        ),
    )
    command = [
        to_native(short_path(model_python)),
        "-u",
        to_native(short_path(backend_script)),
    ]
    add(command, "--input-csv", input_csv)
    add(command, "--time-field", text(1))
    add(command, "--station-field", text(2))
    add(command, "--x-field", text(3))
    add(command, "--y-field", text(4))
    add(command, "--value-field", text(5))
    if text(6):
        add(command, "--start-time", text(6))
    if text(7):
        add(command, "--end-time", text(7))
    add(command, "--input-crs", spatial_reference_text(input_spatial_reference))
    extent = normalize_extent(text(9))
    if extent:
        add(command, "--extent", extent)
    add(command, "--cell-size", text(10))
    add(command, "--output-dir", output_dir)
    add(command, "--epochs", text(12))
    add(command, "--time-window", text(13))
    add(command, "--hidden-dim", text(14))
    add(command, "--diffusion-order", text(15))
    add(command, "--masked-stations", text(16))
    add(command, "--learning-rate", text(17))
    add(command, "--batch-size", text(18))
    add(command, "--query-batch-size", text(19))
    add(command, "--distance-scale-km", text(20))
    add(command, "--adjacency-threshold", text(21))
    add(command, "--value-scale", text(22))
    if boolean(23):
        command.append("--missing-zero")
    add(command, "--device", text(24).upper())
    add(command, "--seed", text(25))
    add(command, "--max-grid-cells", text(26))
    add(command, "--run-log", log_path)
    if boolean(31):
        command.append("--overwrite")
    run_backend(command, project_root, log_handle)
    manifest_path = os.path.join(output_dir, "result_manifest.json")
    raster_paths, model_path = create_rasters_logged(
        manifest_path, output_dir, log_handle
    )
    arcpy.SetParameterAsText(27, ";".join(raster_paths))
    arcpy.SetParameterAsText(28, model_path)
    arcpy.SetParameterAsText(29, manifest_path)
    log_line(log_handle, "Manifest: %s" % manifest_path)
    log_line(log_handle, "Model output: %s" % model_path)
    arcpy.AddMessage("Created %d GeoTIFF rasters" % len(raster_paths))
    status = "SUCCESS"
    log_line(log_handle, "Status: %s" % status)
    log_line(log_handle, "End time: %s" % datetime.datetime.now().isoformat())
    log_handle.close()
    CURRENT_LOG_HANDLE = None


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        arcpy.AddError(str(exc))
        try:
            if CURRENT_LOG_HANDLE:
                log_line(CURRENT_LOG_HANDLE, "ERROR: %s" % exc)
                log_line(CURRENT_LOG_HANDLE, traceback.format_exc())
                log_line(CURRENT_LOG_HANDLE, "Status: FAILED")
                log_line(CURRENT_LOG_HANDLE, "End time: %s" % datetime.datetime.now().isoformat())
                CURRENT_LOG_HANDLE.close()
        except Exception:
            pass
        raise
