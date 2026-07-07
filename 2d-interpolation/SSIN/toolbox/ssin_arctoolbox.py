"""ArcMap 10.8 Script Tool wrapper for SSIN."""

from __future__ import print_function

import datetime
import json
import os
import subprocess
import traceback

import arcpy
import numpy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
BACKEND = os.path.join(REPO_DIR, "ssin_inference.py")
PROJECT_ROOT = os.path.dirname(os.path.dirname(REPO_DIR))
DEFAULT_SSIN_PYTHON = os.path.join(PROJECT_ROOT, "envs", "ssin", "python.exe")
LOG_DIR = os.path.join(PROJECT_ROOT, "log", "ssin-log")
LOG_PATH = None


def parameter(index):
    value = arcpy.GetParameterAsText(index)
    return value.strip() if value else ""


def optional_parameter(index):
    try:
        return parameter(index)
    except Exception:
        return ""


def ensure_text(value):
    if isinstance(value, unicode):
        return value
    try:
        return value.decode("utf-8")
    except Exception:
        return unicode(value)


def create_log_file():
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, "ssin_run_{0}.log".format(stamp))
    with open(path, "wb") as stream:
        stream.write((u"SSIN ArcMap run log\n").encode("utf-8"))
        stream.write((u"Started: {0}\n".format(datetime.datetime.now())).encode("utf-8"))
        stream.write((u"Repository: {0}\n".format(REPO_DIR)).encode("utf-8"))
        stream.write((u"\n").encode("utf-8"))
    return path


def log_line(message):
    if not LOG_PATH:
        return
    try:
        text = ensure_text(message)
        with open(LOG_PATH, "ab") as stream:
            stream.write((text + u"\n").encode("utf-8"))
    except Exception:
        pass


def message(text):
    arcpy.AddMessage(text)
    log_line(text)


def warning(text):
    arcpy.AddWarning(text)
    log_line("WARNING: " + text)


def fail(message_text):
    arcpy.AddError(message_text)
    log_line("ERROR: " + message_text)
    raise RuntimeError(message_text)


def parse_extent(value):
    if not value:
        return None
    if value.strip().upper() in (
        "#", "DEFAULT", "DISPLAY", "MAXOF", "MINOF"
    ):
        message(
            "ArcGIS extent keyword '{0}' will use the CSV station extent.".format(
                value
            )
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
            warning(
                "Extent is outside geographic coordinate bounds and will be ignored."
            )
            return None
    return extent


def log_parameters(items):
    log_line("Parameters:")
    for name, value in items:
        log_line("  {0}: {1}".format(name, value))
    log_line("")


def run_backend(command):
    message("Run log: {0}".format(LOG_PATH))
    message("Starting SSIN environment: {0}".format(command[0]))
    log_line("Backend command:")
    log_line("  " + " ".join(['"{0}"'.format(item) if " " in item else item for item in command]))
    log_line("")
    process = subprocess.Popen(
        command,
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    for line in iter(process.stdout.readline, ""):
        line = line.rstrip()
        if line:
            try:
                decoded = line.decode("utf-8")
            except (AttributeError, UnicodeDecodeError):
                decoded = line
            arcpy.AddMessage(decoded)
            log_line(decoded)
    process.stdout.close()
    code = process.wait()
    log_line("Backend exit code: {0}".format(code))
    if code:
        fail("SSIN process failed with exit code: {0}".format(code))


def load_manifest(path):
    if not os.path.isfile(path):
        fail("SSIN result manifest was not created: {0}".format(path))
    with open(path, "rb") as stream:
        return json.loads(stream.read().decode("utf-8"))


def raster_filename(timestamp, index):
    safe = []
    for char in unicode(timestamp):
        safe.append(char if char.isalnum() or char in "-_." else "_")
    name = "".join(safe).strip("._")
    if not name:
        name = "time_{0:04d}".format(index)
    return ("ssin_" + name)[:100] + ".tif"


def create_rasters(manifest, output_dir, overwrite):
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
            fail("Output raster already exists: {0}".format(output))
        raster = arcpy.NumPyArrayToRaster(
            grid, lower_left, cell_size, cell_size
        )
        raster.save(output)
        arcpy.DefineProjection_management(output, wgs84)
        outputs.append(output)
        message(
            "Created {0}: {1}".format(result["timestamp"], output)
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
        log_line("Loaded raster into ArcMap: {0}".format(path))
    arcpy.RefreshTOC()
    arcpy.RefreshActiveView()


def main():
    global LOG_PATH
    LOG_PATH = create_log_file()

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
    epochs = parameter(12) or "100"
    learning_rate = parameter(13) or "0.001"
    mask_ratio = parameter(14) or "0.2"
    seed = parameter(15) or "42"
    device = (parameter(16) or "AUTO").upper()
    minimum_valid = parameter(17) or "5"
    batch_size = parameter(18) or "200"
    ssin_python = optional_parameter(22) or DEFAULT_SSIN_PYTHON
    overwrite = optional_parameter(23).lower() in ("true", "1", "yes")

    log_parameters([
        ("input_csv", input_csv),
        ("time_field", time_field),
        ("station_field", station_field),
        ("x_field", x_field),
        ("y_field", y_field),
        ("value_field", value_field),
        ("start_time", start_time),
        ("end_time", end_time),
        ("input_spatial_reference", input_spatial_reference.exportToString()),
        ("extent", extent),
        ("cell_size", cell_size),
        ("output_dir", output_dir),
        ("epochs", epochs),
        ("learning_rate", learning_rate),
        ("mask_ratio", mask_ratio),
        ("seed", seed),
        ("device", device),
        ("minimum_valid", minimum_valid),
        ("batch_size", batch_size),
        ("ssin_python", ssin_python),
        ("overwrite", overwrite),
        ("log_path", LOG_PATH),
    ])

    required = [
        (input_csv, "Input time-series CSV"),
        (time_field, "Time field"),
        (station_field, "Station field"),
        (x_field, "Longitude field"),
        (y_field, "Latitude field"),
        (value_field, "Value field"),
        (cell_size, "Cell size"),
        (output_dir, "Output folder"),
        (epochs, "Training epochs"),
        (learning_rate, "Learning rate"),
        (mask_ratio, "Mask ratio"),
    ]
    missing = [name for value, name in required if not value]
    if missing:
        fail("Missing required parameters: " + ", ".join(missing))
    if not os.path.isfile(ssin_python):
        fail("SSIN Python was not found: {0}".format(ssin_python))
    if not os.path.isfile(BACKEND):
        fail("SSIN backend script was not found: {0}".format(BACKEND))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    command = [
        ssin_python, "-u", BACKEND,
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
        "--mask-ratio", mask_ratio,
        "--seed", seed,
        "--device", device,
        "--min-valid-stations", minimum_valid,
        "--batch-size", batch_size,
        "--run-log", LOG_PATH,
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

    run_backend(command)
    manifest_path = os.path.join(output_dir, "ssin_manifest.json")
    rasters = create_rasters(
        load_manifest(manifest_path), output_dir, overwrite
    )
    add_to_map(rasters)
    arcpy.SetParameterAsText(19, ";".join(rasters))
    arcpy.SetParameterAsText(
        20, os.path.join(output_dir, "ssin_trained_model.pyt")
    )
    arcpy.SetParameterAsText(21, manifest_path)
    message(
        "Completed. Created and loaded {0} raster(s).".format(len(rasters))
    )
    log_line("Finished: {0}".format(datetime.datetime.now()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_text = traceback.format_exc()
        arcpy.AddError(error_text)
        log_line(error_text)
        raise
