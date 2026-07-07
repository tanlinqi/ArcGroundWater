from __future__ import print_function

import codecs
import datetime
import json
import os
import subprocess
import traceback

import arcpy
import numpy as np


MODEL_KEY = "kcn"
CURRENT_LOG_PATH = ""


def now_text():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_run_log(project_root):
    log_dir = os.path.join(os.path.dirname(os.path.dirname(project_root)), "log", MODEL_KEY + "-log")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, MODEL_KEY + "_run_" + stamp + ".log")


def log_write(log_path, message):
    if not log_path:
        return
    try:
        if not isinstance(message, unicode):
            message = unicode(message)
        with codecs.open(log_path, "a", "utf-8") as handle:
            handle.write(u"[%s] %s\n" % (now_text(), message))
    except Exception:
        pass


def log_value(value):
    try:
        if isinstance(value, unicode):
            return value
    except NameError:
        pass
    try:
        return unicode(value, "mbcs")
    except Exception:
        try:
            return unicode(value)
        except Exception:
            return u"<unprintable>"


def log_section(log_path, title):
    log_write(log_path, "")
    log_write(log_path, "==== " + title + " ====")


def short_path(path):
    try:
        import ctypes

        unicode_path = unicode(os.path.abspath(path))
        buffer_size = ctypes.windll.kernel32.GetShortPathNameW(
            unicode_path, None, 0
        )
        if buffer_size:
            buffer = ctypes.create_unicode_buffer(buffer_size)
            ctypes.windll.kernel32.GetShortPathNameW(
                unicode_path, buffer, buffer_size
            )
            value = buffer.value
            if value:
                return value.encode("mbcs")
    except Exception:
        pass
    try:
        return os.path.abspath(path).encode("mbcs")
    except Exception:
        return os.path.abspath(path)


def process_arg(value):
    try:
        if isinstance(value, unicode):
            return value.encode("mbcs")
    except NameError:
        pass
    return value


def text(index):
    value = arcpy.GetParameterAsText(index)
    return value.strip() if value else ""


def boolean(index):
    value = text(index).lower()
    return value in ("true", "1", "yes")


def default_model_python(project_root):
    home = os.path.expanduser("~")
    arcground_root = os.path.dirname(os.path.dirname(project_root))
    candidates = [
        os.path.join(arcground_root, "envs", "ssin", "python.exe"),
        os.path.join(home, "Miniconda3", "envs", "ssin", "python.exe"),
        os.path.join(home, "miniconda3", "envs", "ssin", "python.exe"),
        os.path.join(home, ".conda", "envs", "ssin", "python.exe"),
        os.path.join(home, "Anaconda3", "envs", "ssin", "python.exe"),
        os.path.join(home, "anaconda3", "envs", "ssin", "python.exe"),
        os.path.join(project_root, "venv", "Scripts", "python.exe"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return ""

def check_model_python(value):
    code = (
        "import importlib.util,sys;"
        "mods=['numpy','torch','torch_geometric','sklearn','pyproj'];"
        "missing=[m for m in mods if importlib.util.find_spec(m) is None];"
        "sys.stdout.write(','.join(missing));"
        "raise SystemExit(1 if missing else 0)"
    )
    process = subprocess.Popen(
        [value, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    missing = (stdout or "").strip()
    if process.returncode:
        if missing:
            raise ValueError(
                "Model Python is missing required packages: %s. "
                "Create the kcn_env environment from environment.yml or choose another python.exe."
                % missing
            )
        raise ValueError(
            "Model Python dependency check failed: %s" % ((stderr or "").strip())
        )


def resolve_model_python(value, project_root):
    if value in ("", "#"):
        candidate = default_model_python(project_root)
        if candidate:
            arcpy.AddMessage("Using model Python: %s" % candidate)
            check_model_python(candidate)
            return candidate
        raise ValueError(
            "Model Python executable is required. No kcn_env or kcn environment "
            "was found. Create it with 'conda env create -f environment.yml', "
            "then set this parameter to its python.exe."
        )
    if not os.path.isfile(value):
        if value.replace(".", "", 1).isdigit():
            raise ValueError(
                "Model Python executable received a numeric value (%s). "
                "The ArcToolbox parameter order is probably out of date; "
                "recreate the Script Tool from README_ArcMap.md." % value
            )
        raise ValueError("Model Python executable was not found: %s" % value)
    check_model_python(value)
    return value


def spatial_reference_text(spatial_reference):
    if not spatial_reference:
        return "EPSG:4326"
    factory_code = getattr(spatial_reference, "factoryCode", 0)
    if factory_code:
        return "EPSG:%d" % factory_code
    return spatial_reference.exportToString()


def append_option(command, name, value):
    if value != "":
        command.extend([name, value])


def run_process(command, project_root, log_path):
    process_command = [process_arg(item) for item in command]
    process_cwd = process_arg(project_root)
    log_section(log_path, "Backend command")
    log_write(log_path, "cwd: %s" % project_root)
    log_write(log_path, u"command: %s" % u" ".join([log_value(item) for item in command]))
    process = subprocess.Popen(
        process_command,
        cwd=process_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    for line in iter(process.stdout.readline, ""):
        if line:
            clean_line = line.rstrip()
            arcpy.AddMessage(clean_line)
            log_write(log_path, clean_line)
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError("KCN backend failed with exit code %d" % return_code)


def create_rasters(manifest, output_dir, overwrite, log_path):
    xmin, ymin, xmax, ymax = manifest["extent"]
    cell_size = float(manifest["cell_size"])
    lower_left = arcpy.Point(xmin, ymin)
    output_paths = []
    for result in manifest["results"]:
        array_data = np.load(result["array_path"])
        grid = array_data["grid"]
        timestamp = result["timestamp"]
        safe_time = "".join(
            char if char.isalnum() or char in "._-" else "_" for char in timestamp
        )
        raster_path = os.path.join(output_dir, "interpolation_%s.tif" % safe_time)
        if arcpy.Exists(raster_path) and not overwrite:
            raise RuntimeError(
                "Output raster exists and overwrite is disabled: %s" % raster_path
            )
        arcpy.env.overwriteOutput = overwrite
        raster = arcpy.NumPyArrayToRaster(
            grid, lower_left, cell_size, cell_size, -9999.0
        )
        raster.save(raster_path)
        arcpy.DefineProjection_management(raster_path, arcpy.SpatialReference(4326))
        result["raster_path"] = os.path.abspath(raster_path)
        output_paths.append(raster_path)
        log_write(log_path, "GeoTIFF created: %s" % raster_path)
        try:
            map_doc = arcpy.mapping.MapDocument("CURRENT")
            data_frame = map_doc.activeDataFrame
            arcpy.mapping.AddLayer(
                data_frame,
                arcpy.mapping.Layer(raster_path),
                "TOP",
            )
            log_write(log_path, "ArcMap loaded raster: %s" % raster_path)
        except Exception as exc:
            arcpy.AddWarning("Could not add raster to current map: %s" % exc)
            log_write(log_path, "warning: Could not add raster to current map: %s" % exc)
    return output_paths


def main():
    global CURRENT_LOG_PATH
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(script_path))
    backend_script = os.path.join(project_root, "model_backend.py")
    if not os.path.isfile(backend_script):
        backend_script = os.path.join(project_root, "kcn_backend.py")
    log_path = make_run_log(project_root)
    CURRENT_LOG_PATH = log_path
    log_section(log_path, "Run start")
    log_write(log_path, "status: STARTED")
    log_write(log_path, "start_time: %s" % now_text())
    log_write(log_path, "ArcMap wrapper path: %s" % script_path)
    log_write(log_path, "Python 3 backend path: %s" % backend_script)
    project_root_for_process = short_path(project_root)
    backend_script_for_process = short_path(backend_script)
    model_python = resolve_model_python(text(30), project_root)
    log_write(log_path, "Python 3 interpreter path: %s" % model_python)

    input_csv = text(0)
    output_dir = text(11)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    spatial_reference = arcpy.GetParameter(8)
    log_section(log_path, "Parameters")
    log_write(log_path, "input_csv: %s" % input_csv)
    log_write(log_path, "field_mapping: time=%s station=%s x=%s y=%s value=%s" % (text(1), text(2), text(3), text(4), text(5)))
    log_write(log_path, "time_range: %s to %s" % (text(6), text(7)))
    log_write(log_path, "input_crs: %s" % spatial_reference_text(spatial_reference))
    log_write(log_path, "extent: %s" % text(9))
    log_write(log_path, "cell_size: %s" % text(10))
    log_write(log_path, "output_dir: %s" % output_dir)
    log_write(log_path, "training_parameters: model=%s n_neighbors=%s hidden_sizes=%s dropout=%s epochs=%s lr=%s weight_decay=%s batch_size=%s validation_fraction=%s es_patience=%s length_scale=%s device=%s random_seed=%s" % (text(12), text(13), text(14), text(15), text(16), text(17), text(18), text(19), text(20), text(21), text(22), text(23), text(24)))
    log_write(log_path, "inference_parameters: max_grid_cells=%s prediction_batch_size=%s" % (text(25), text(26)))
    command = [
        model_python,
        "-u",
        backend_script_for_process,
        "--input-csv",
        input_csv,
        "--time-field",
        text(1),
        "--station-field",
        text(2),
        "--x-field",
        text(3),
        "--y-field",
        text(4),
        "--value-field",
        text(5),
        "--input-crs",
        spatial_reference_text(spatial_reference),
        "--cell-size",
        text(10),
        "--output-dir",
        output_dir,
        "--model",
        text(12),
        "--n-neighbors",
        text(13),
        "--hidden-sizes",
        text(14),
        "--dropout",
        text(15),
        "--epochs",
        text(16),
        "--lr",
        text(17),
        "--weight-decay",
        text(18),
        "--batch-size",
        text(19),
        "--validation-fraction",
        text(20),
        "--es-patience",
        text(21),
        "--length-scale",
        text(22),
        "--device",
        text(23),
        "--random-seed",
        text(24),
        "--max-grid-cells",
        text(25),
        "--prediction-batch-size",
        text(26),
        "--run-log",
        log_path,
    ]
    append_option(command, "--start-time", text(6))
    append_option(command, "--end-time", text(7))
    append_option(command, "--extent", text(9))
    overwrite = boolean(31)
    if overwrite:
        command.append("--overwrite")
    log_write(log_path, "overwrite: %s" % overwrite)

    arcpy.AddMessage("Starting KCN backend")
    log_write(log_path, "Starting KCN backend")
    run_process(command, project_root_for_process, log_path)
    manifest_path = os.path.join(output_dir, "result_manifest.json")
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    manifest["run_log"] = os.path.abspath(log_path)
    raster_paths = create_rasters(manifest, output_dir, overwrite, log_path)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    arcpy.SetParameterAsText(27, ";".join(raster_paths))
    arcpy.SetParameterAsText(28, manifest["model_path"])
    arcpy.SetParameterAsText(29, manifest_path)
    log_section(log_path, "Run end")
    log_write(log_path, "manifest: %s" % manifest_path)
    log_write(log_path, "rasters: %s" % ";".join(raster_paths))
    log_write(log_path, "status: SUCCESS")
    log_write(log_path, "end_time: %s" % now_text())
    arcpy.AddMessage("KCN interpolation completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        arcpy.AddError(str(exc))
        if CURRENT_LOG_PATH:
            log_section(CURRENT_LOG_PATH, "Run error")
            log_write(CURRENT_LOG_PATH, "status: FAILED")
            log_write(CURRENT_LOG_PATH, "error: %s" % exc)
            log_write(CURRENT_LOG_PATH, traceback.format_exc())
            log_write(CURRENT_LOG_PATH, "end_time: %s" % now_text())
        raise


