"""ArcMap 10.8 Script Tool wrapper for DeepKriging."""

from __future__ import print_function

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime

import arcpy
import numpy


DEFAULT_MODEL_PYTHON = r"C:\Users\Lenovo\Miniconda3\envs\ssin\python.exe"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
ARC_GROUND_DIR = os.path.dirname(REPO_DIR)
LOG_DIR = os.path.join(ARC_GROUND_DIR, "log", "deepkriging-log")
BACKEND = os.path.join(REPO_DIR, "deepkriging_backend.py")
CURRENT_LOG = None


def unicode_path(value):
    if isinstance(value, unicode):
        return value
    encoding = sys.getfilesystemencoding() or "mbcs"
    try:
        return value.decode(encoding)
    except UnicodeDecodeError:
        return value.decode("mbcs")


def subprocess_argument(value):
    if isinstance(value, unicode):
        return value.encode("mbcs")
    return value


def subprocess_command(command):
    return [subprocess_argument(value) for value in command]


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_run_log():
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    name = "deepkriging_run_{0}.log".format(
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    return os.path.join(LOG_DIR, name)


def append_log(path, message):
    if not path:
        return
    with open(path, "ab") as stream:
        text = u"[{0}] {1}\n".format(timestamp(), unicode_path(message))
        stream.write(text.encode("utf-8"))


def log_message(path, message):
    append_log(path, message)
    try:
        arcpy.AddMessage(message)
    except UnicodeEncodeError:
        arcpy.AddMessage(unicode_path(message))


def parameter(index):
    value = arcpy.GetParameterAsText(index)
    return value.strip() if value else ""


def optional_parameter(index):
    try:
        value = parameter(index)
        if value.upper() in ("#", "DEFAULT"):
            return ""
        return value
    except Exception:
        return ""


def fail(message):
    append_log(CURRENT_LOG, "ERROR: {0}".format(message))
    arcpy.AddError(message)
    raise RuntimeError(message)


def parse_extent(value):
    if not value:
        return None
    if value.strip().upper() in ("#", "DEFAULT", "DISPLAY", "MAXOF", "MINOF"):
        log_message(
            CURRENT_LOG,
            "WARNING: ArcGIS extent keyword '{0}' will use the CSV station extent.".format(value)
        )
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 4:
        fail("Interpolation extent requires xmin ymin xmax ymax.")
    try:
        return [float(part) for part in parts]
    except ValueError:
        fail("Interpolation extent contains a non-numeric value: {0}".format(value))


def normalize_extent(extent, spatial_reference):
    if extent is None:
        return None
    if spatial_reference.type == "Geographic":
        xmin, ymin, xmax, ymax = extent
        if xmin < -180 or xmax > 180 or ymin < -90 or ymax > 90:
            message = "WARNING: Extent is outside geographic coordinate bounds and will be ignored."
            append_log(CURRENT_LOG, message)
            arcpy.AddWarning(
                "Extent is outside geographic coordinate bounds and will be ignored."
            )
            return None
    return extent


def model_environment():
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def validate_model_python(model_python, environment, log_path):
    command = subprocess_command([
        model_python,
        "-c",
        (
            "import sys, torch; "
            "print(sys.executable); "
            "print(torch.__version__); "
            "print(torch.cuda.is_available())"
        ),
    ])
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    output = process.communicate()[0]
    if process.returncode:
        try:
            details = output.decode("utf-8", "replace")
        except AttributeError:
            details = output
        fail(
            u"Model Python cannot import PyTorch: {0}\n{1}".format(
                unicode_path(model_python), details
            )
        )
    lines = output.decode("utf-8", "replace").splitlines()
    if len(lines) >= 3:
        log_message(log_path, u"Model Python: {0}".format(lines[0]))
        log_message(log_path, u"PyTorch: {0}; CUDA available: {1}".format(
            lines[1], lines[2]
        ))


def run_backend(command, environment, log_path):
    log_message(
        log_path,
        u"Starting DeepKriging environment: {0}".format(unicode_path(command[0]))
    )
    append_log(log_path, "Backend command: {0}".format(" ".join([
        unicode_path(item) for item in command
    ])))
    process = subprocess.Popen(
        subprocess_command(command),
        cwd=subprocess_argument(REPO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=environment,
    )
    for line in iter(process.stdout.readline, ""):
        line = line.rstrip()
        if line:
            try:
                text = line.decode("utf-8")
            except (AttributeError, UnicodeDecodeError):
                text = line
            log_message(log_path, text)
    process.stdout.close()
    code = process.wait()
    if code:
        fail("DeepKriging process failed with exit code: {0}".format(code))


def load_manifest(path):
    if not os.path.isfile(path):
        fail(u"DeepKriging manifest was not created: {0}".format(unicode_path(path)))
    with open(path, "rb") as stream:
        return json.loads(stream.read().decode("utf-8"))


def raster_filename(timestamp, index):
    safe = []
    for char in unicode(timestamp):
        safe.append(char if char.isalnum() or char in "-_." else "_")
    name = "".join(safe).strip("._")
    if not name:
        name = "time_{0:04d}".format(index)
    return ("deepkriging_" + name)[:100] + ".tif"


def create_rasters(manifest, output_dir, overwrite):
    output_dir = unicode_path(output_dir)
    extent = manifest["extent"]
    cell_size = float(manifest["cell_size"])
    lower_left = arcpy.Point(float(extent[0]), float(extent[1]))
    wgs84 = arcpy.SpatialReference(4326)
    outputs = []
    arcpy.env.overwriteOutput = overwrite
    for index, result in enumerate(manifest.get("results", [])):
        with numpy.load(result["array_path"]) as archive:
            grid = archive["grid"].astype(numpy.float32)
        output = os.path.join(
            output_dir, raster_filename(result["timestamp"], index)
        )
        if arcpy.Exists(output) and not overwrite:
            fail(u"Output raster already exists: {0}".format(output))
        raster = arcpy.NumPyArrayToRaster(
            grid, lower_left, cell_size, cell_size
        )
        raster.save(output)
        arcpy.DefineProjection_management(output, wgs84)
        outputs.append(output)
        log_message(
            CURRENT_LOG,
            u"Created GeoTIFF {0}: {1}".format(result["timestamp"], output),
        )
    if not outputs:
        fail("No interpolation arrays were found in the result manifest.")
    return outputs


def add_to_map(raster_paths):
    document = arcpy.mapping.MapDocument("CURRENT")
    data_frames = arcpy.mapping.ListDataFrames(document)
    if not data_frames:
        fail("The current map has no data frame.")
    for path in raster_paths:
        arcpy.mapping.AddLayer(
            data_frames[0], arcpy.mapping.Layer(path), "TOP"
        )
        log_message(CURRENT_LOG, u"Loaded into ArcMap: {0}".format(path))
    arcpy.RefreshTOC()
    arcpy.RefreshActiveView()


def main():
    global CURRENT_LOG
    CURRENT_LOG = create_run_log()
    log_message(CURRENT_LOG, "Run status: STARTED")
    log_message(CURRENT_LOG, u"ArcMap wrapper path: {0}".format(__file__))
    log_message(CURRENT_LOG, u"Python 3 backend path: {0}".format(BACKEND))
    input_csv = parameter(0)
    time_field = parameter(1)
    station_field = parameter(2)
    x_field = parameter(3)
    y_field = parameter(4)
    value_field = parameter(5)
    start_time = parameter(6)
    end_time = parameter(7)
    input_spatial_reference = arcpy.GetParameter(8)
    if not input_spatial_reference:
        input_spatial_reference = arcpy.SpatialReference(4326)
    extent = normalize_extent(
        parse_extent(parameter(9)), input_spatial_reference
    )
    cell_size = parameter(10)
    output_dir = parameter(11)
    output_dir = unicode_path(output_dir)
    epochs = parameter(12) or "200"
    learning_rate = parameter(13) or "0.001"
    basis_resolutions = parameter(14) or "10,19,37"
    support_multiplier = parameter(15) or "2.5"
    hidden_units = parameter(16) or "100"
    hidden_layers = parameter(17) or "3"
    seed = parameter(18) or "42"
    device = (parameter(19) or "AUTO").upper()
    minimum_valid = parameter(20) or "5"
    train_batch_size = parameter(21) or "64"
    inference_batch_size = parameter(22) or "4096"
    model_python = optional_parameter(26) or DEFAULT_MODEL_PYTHON
    overwrite = optional_parameter(27).lower() in ("true", "1", "yes")
    log_message(CURRENT_LOG, u"Input CSV: {0}".format(input_csv))
    log_message(CURRENT_LOG, "Field mapping: time={0}, station={1}, x={2}, y={3}, value={4}".format(
        time_field, station_field, x_field, y_field, value_field
    ))
    log_message(CURRENT_LOG, "Time range: {0} to {1}".format(start_time, end_time))
    log_message(CURRENT_LOG, "Input coordinate system: {0}".format(
        input_spatial_reference.exportToString()
    ))
    log_message(CURRENT_LOG, "Interpolation extent: {0}".format(extent))
    log_message(CURRENT_LOG, "Cell size: {0}".format(cell_size))
    log_message(CURRENT_LOG, u"Output folder: {0}".format(output_dir))
    log_message(CURRENT_LOG, "Training parameters: epochs={0}, learning_rate={1}, basis_resolutions={2}, support_multiplier={3}, hidden_units={4}, hidden_layers={5}, seed={6}, device={7}".format(
        epochs, learning_rate, basis_resolutions, support_multiplier,
        hidden_units, hidden_layers, seed, device
    ))
    log_message(CURRENT_LOG, "Inference parameters: min_valid_stations={0}, train_batch_size={1}, inference_batch_size={2}".format(
        minimum_valid, train_batch_size, inference_batch_size
    ))

    required = [
        (input_csv, "Input time-series CSV"),
        (time_field, "Time field"),
        (station_field, "Station field"),
        (x_field, "X field"),
        (y_field, "Y field"),
        (value_field, "Value field"),
        (cell_size, "Cell size"),
        (output_dir, "Output folder"),
    ]
    missing = [name for value, name in required if not value]
    if missing:
        fail("Missing required parameters: " + ", ".join(missing))
    if not os.path.isfile(model_python):
        fail(u"Model Python was not found: {0}".format(unicode_path(model_python)))
    if not os.path.isfile(BACKEND):
        fail(u"DeepKriging backend was not found: {0}".format(unicode_path(BACKEND)))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    environment = model_environment()
    validate_model_python(model_python, environment, CURRENT_LOG)

    command = [
        model_python, "-u", BACKEND,
        "--input-csv", input_csv,
        "--time-field", time_field,
        "--station-field", station_field,
        "--x-field", x_field,
        "--y-field", y_field,
        "--value-field", value_field,
        "--input-crs", input_spatial_reference.exportToString(),
        "--cell-size", cell_size,
        "--output-dir", output_dir,
        "--epochs", epochs,
        "--learning-rate", learning_rate,
        "--basis-resolutions", basis_resolutions,
        "--support-multiplier", support_multiplier,
        "--hidden-units", hidden_units,
        "--hidden-layers", hidden_layers,
        "--seed", seed,
        "--device", device,
        "--min-valid-stations", minimum_valid,
        "--train-batch-size", train_batch_size,
        "--inference-batch-size", inference_batch_size,
        "--run-log", CURRENT_LOG,
    ]
    if start_time:
        command.extend(["--start-time", start_time])
    if end_time:
        command.extend(["--end-time", end_time])
    if extent:
        command.append("--extent")
        command.extend([str(item) for item in extent])
    if overwrite:
        command.append("--overwrite")

    run_backend(command, environment, CURRENT_LOG)
    manifest_path = os.path.join(output_dir, "deepkriging_manifest.json")
    rasters = create_rasters(load_manifest(manifest_path), output_dir, overwrite)
    add_to_map(rasters)
    arcpy.SetParameterAsText(23, ";".join(rasters))
    arcpy.SetParameterAsText(
        24, os.path.join(output_dir, "deepkriging_trained_model.pyt")
    )
    arcpy.SetParameterAsText(25, manifest_path)
    arcpy.AddMessage(
        "Completed. Created and loaded {0} raster(s).".format(len(rasters))
    )
    log_message(CURRENT_LOG, "Manifest: {0}".format(manifest_path))
    log_message(CURRENT_LOG, "Run status: SUCCESS")
    log_message(CURRENT_LOG, "Run completed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        details = traceback.format_exc()
        append_log(CURRENT_LOG, "Run status: FAILED")
        append_log(CURRENT_LOG, details)
        arcpy.AddError(details)
        raise
