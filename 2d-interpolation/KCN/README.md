# KCN ArcMap 10.8 Deployment

This package deploys Kriging Convolutional Networks (KCN) as an ArcMap 10.8
Script Tool. ArcMap Python 2.7 only handles toolbox parameters, process launch,
logging, NPZ-to-GeoTIFF conversion, and map loading. Model training and
interpolation run in a separate Python 3 environment.

## Files

```text
KCN/
|-- README.md
|-- requirements-kcn.txt
|-- kcn_backend.py
|-- model/
|   |-- __init__.py
|   |-- data.py
|   `-- kcn.py
`-- toolbox/
    |-- kcn_arctoolbox.py
    |-- ToolValidator.py
    `-- KCN Tools.tbx.README.txt
```

## Python 3 Environment

The tested environment is `C:\Users\Lenovo\Miniconda3\envs\ssin\python.exe`.
Install required packages with:

```bat
python -m pip install -r requirements-kcn.txt
```

Required packages are NumPy, PyTorch, PyTorch Geometric, scikit-learn, and
pyproj.

## ArcMap Script Tool

Create a normal ArcMap Script Tool and bind it to:

```text
toolbox\kcn_arctoolbox.py
```

Do not create a `.pyt` Python toolbox.

Parameter order:

| Index | Name | Type | Direction | Required | Default |
|---:|---|---|---|---|---|
| 0 | Input time-series CSV | File | Input | Yes | |
| 1 | Time field | String | Input | Yes | time |
| 2 | Station field | String | Input | Yes | station |
| 3 | X field | String | Input | Yes | lon |
| 4 | Y field | String | Input | Yes | lat |
| 5 | Value field | String | Input | Yes | value |
| 6 | Start time | String | Input | No | |
| 7 | End time | String | Input | No | |
| 8 | Input coordinate system | Spatial Reference | Input | Yes | WGS 1984 |
| 9 | Interpolation extent | Extent | Input | No | |
| 10 | Cell size | Double | Input | Yes | 0.05 |
| 11 | Output folder | Folder | Input | Yes | |
| 12 | Model type | String | Input | Yes | kcn |
| 13 | Neighbor count | Long | Input | Yes | 5 |
| 14 | Hidden sizes | String | Input | Yes | 10,8,8 |
| 15 | Dropout | Double | Input | Yes | 0.1 |
| 16 | Epochs | Long | Input | Yes | 100 |
| 17 | Learning rate | Double | Input | Yes | 0.001 |
| 18 | Weight decay | Double | Input | Yes | 0.0005 |
| 19 | Batch size | Long | Input | Yes | 64 |
| 20 | Validation fraction | Double | Input | Yes | 0.2 |
| 21 | Early-stop patience | Long | Input | Yes | 20 |
| 22 | Length scale | String | Input | Yes | auto |
| 23 | Device | String | Input | Yes | AUTO |
| 24 | Random seed | Long | Input | Yes | 5 |
| 25 | Maximum grid cells | Long | Input | Yes | 5000000 |
| 26 | Prediction batch size | Long | Input | Yes | 4096 |
| 27 | Output rasters | Raster Dataset | Output | No | MultiValue, Derived |
| 28 | Output trained model | File | Output | No | Derived |
| 29 | Output result manifest | File | Output | No | Derived |
| 30 | Model Python executable | File | Input | Yes | # |
| 31 | Overwrite existing results | Boolean | Input | No | False |

Model type choices:

- `kcn`: default KCN with GCNConv.
- `kcn_gat`: KCN with GATConv.
- `kcn_sage`: KCN with SAGEConv.

Use `ToolValidator.py` as the Script Tool validation code. ArcMap copies
validator code into the toolbox, so paste it again after editing the file.

## Logs

Each run creates one independent log file under:

```text
KCN\..\log\kcn-log\
```

The filename format is:

```text
kcn_run_YYYYMMDD_HHMMSS.log
```

The log records start and end time, run status, wrapper path, backend path,
Python 3 interpreter path, input CSV, field mapping, time range, CRS, extent,
cell size, output folder, training parameters, inference parameters, backend
command, backend progress, NPZ paths, GeoTIFF paths, ArcMap loading status,
warnings, and error traceback.

The backend manifest JSON also contains `run_log`, pointing to the run log.

## Backend Validation

```bat
python kcn_backend.py --input-csv path\to\input.csv ^
  --time-field time --station-field station --x-field lon --y-field lat ^
  --value-field value --input-crs EPSG:4326 --cell-size 0.05 ^
  --output-dir path\to\output --model kcn --n-neighbors 5 ^
  --hidden-sizes 10,8,8 --device CPU --validate-only
```

Blank or non-numeric `value` cells are skipped and recorded in the manifest
`skipped` list. Coordinates must be numeric.
