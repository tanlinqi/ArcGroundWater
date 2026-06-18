# SSIN ArcMap Runtime Package

This folder is the deployment copy for the SSIN ArcMap Script Tool. It contains only runtime code and minimal documentation. It does not include input data or model outputs.

## Folder Layout

```text
SSIN/
|-- ssin_inference.py
|-- requirements-ssin-model.txt
|-- README.md
|-- networks/
|   |-- Models.py
|   |-- Layers.py
|   `-- RelativeAttentionLayer.py
`-- toolbox/
    |-- ssin_arctoolbox.py
    |-- ToolValidator.py
    `-- SSIN Tools.tbx.README.txt
```

The folder name should remain `SSIN`, because the backend imports model code with `from SSIN.networks.Models import SpaFormer`.

## ArcMap Tool

Create `SSIN Tools.tbx` in ArcCatalog or ArcMap and bind the Script Tool to:

```text
SSIN\toolbox\ssin_arctoolbox.py
```

Copy the full content of `SSIN\toolbox\ToolValidator.py` into the Script Tool validation editor.

ArcMap runs the wrapper with ArcGIS Python 2.7. The wrapper starts the SSIN Python 3 environment directly:

```text
C:\Users\Lenovo\Miniconda3\envs\ssin\python.exe
```

Do not import PyTorch in ArcMap Python and do not use `conda activate` inside ArcMap.

## Required Python Packages

The SSIN Python 3 environment needs:

```text
torch
numpy
pandas
pyproj
geographiclib
```

ArcMap Python 2.7 only needs ArcMap built-ins:

```text
arcpy
numpy
standard library
```

## Data

Input CSV files are selected by the user in ArcToolbox. They are not stored in this deployment package.

Expected long-table CSV fields:

```text
time, station, lon/x, lat/y, value
```

For the BW data, use:

```text
time field: time
station field: station
x/lon field: lon
y/lat field: lat
value field: value
input coordinate system: GCS_WGS_1984
```

## Run Logs

Each run creates one log file under the parent ArcGround folder:

```text
1-ArcGround\log\ssin-log\ssin_run_YYYYMMDD_HHMMSS.log
```

The code builds this path relatively from the `SSIN` package location:

```text
SSIN\..\log\ssin-log
```

The log records parameters, backend command, SSIN training output, interpolation output, GeoTIFF creation, ArcMap loading messages, warnings, and tracebacks.

The output folder still receives the model outputs:

```text
ssin_manifest.json
ssin_trained_model.pyt
ssin_<time>.npz
ssin_<time>.tif
```

The manifest also records the run log path in `run_log`.
