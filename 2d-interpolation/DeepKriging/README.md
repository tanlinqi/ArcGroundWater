# DeepKriging ArcMap 10.8 Deployment

This package contains the ArcMap-callable DeepKriging spatial interpolation
tool. It is designed to live under:

```text
C:\Users\Lenovo\Desktop\1-ArcGround\DeepKriging
```

Runtime logs are written outside the model folder:

```text
C:\Users\Lenovo\Desktop\1-ArcGround\log\deepkriging-log
```

Each run creates a separate log file:

```text
deepkriging_run_YYYYMMDD_HHMMSS.log
```

## Files

```text
DeepKriging/
|-- README.md
|-- requirements-deepkriging.txt
|-- deepkriging_backend.py
|-- model/
|   |-- __init__.py
|   `-- deepkriging_model.py
`-- toolbox/
    |-- deepkriging_arctoolbox.py
    |-- ToolValidator.py
    `-- DeepKriging Tools.tbx.README.txt
```

## Responsibilities

ArcMap Python 2.7 runs only:

- `toolbox/deepkriging_arctoolbox.py`
- ArcToolbox parameter reading
- log creation
- Python 3 backend launch
- backend stdout/stderr forwarding
- NPZ to GeoTIFF conversion
- ArcMap layer loading

Python 3 runs:

- `deepkriging_backend.py`
- CSV reading
- coordinate conversion
- DeepKriging training
- grid interpolation
- NPZ output
- manifest JSON output

The ArcMap wrapper does not import PyTorch, pandas, TensorFlow, or any other
deep-learning dependency.

## Python Environments

ArcMap wrapper:

```text
C:\Python27\ArcGIS10.8\python.exe
```

Model backend default:

```text
C:\Users\Lenovo\Miniconda3\envs\ssin\python.exe
```

The backend environment must include the packages listed in
`requirements-deepkriging.txt`.

## ArcToolbox Parameters

Create a normal Script Tool, not a `.pyt`, and bind it to:

```text
toolbox\deepkriging_arctoolbox.py
```

Paste the full content of `toolbox\ToolValidator.py` into the tool validation
editor. Keep this exact parameter order:

| Index | Parameter | Type | Direction | Default |
|---:|---|---|---|---|
| 0 | Input CSV | File | Input | csv file |
| 1 | Time field | String | Input | from CSV |
| 2 | Station field | String | Input | from CSV |
| 3 | X/longitude field | String | Input | from CSV |
| 4 | Y/latitude field | String | Input | from CSV |
| 5 | Value field | String | Input | from CSV |
| 6 | Start time | String | Input | optional |
| 7 | End time | String | Input | optional |
| 8 | Input coordinate system | Spatial Reference | Input | WGS84 |
| 9 | Interpolation extent | Extent | Input | optional |
| 10 | Cell size | Double | Input | required |
| 11 | Output folder | Folder | Input | required |
| 12 | Epochs | Long | Input | 200 |
| 13 | Learning rate | Double | Input | 0.001 |
| 14 | Basis resolutions | String | Input | 10,19,37 |
| 15 | Support multiplier | Double | Input | 2.5 |
| 16 | Hidden units | Long | Input | 100 |
| 17 | Hidden layers | Long | Input | 3 |
| 18 | Seed | Long | Input | 42 |
| 19 | Device | String | Input | AUTO/CUDA/CPU |
| 20 | Minimum valid stations | Long | Input | 5 |
| 21 | Training batch size | Long | Input | 64 |
| 22 | Inference batch size | Long | Input | 4096 |
| 23 | Output rasters | Raster Dataset | Derived Output | multi-value |
| 24 | Output model | File | Derived Output | pyt file |
| 25 | Output manifest | File | Derived Output | json file |
| 26 | Model Python | File | Input | optional |
| 27 | Overwrite outputs | Boolean | Input | False |

## Output

The selected output folder receives:

- `deepkriging_trained_model.pyt`
- `deepkriging_manifest.json`
- one `deepkriging_<timestamp>.npz` per output time
- one `deepkriging_<timestamp>.tif` per output time

The manifest includes a `run_log` field pointing to the log file for that run.

## Notes

The interpolation extent is also used as the coordinate normalization extent.
It must contain all training stations. Leave it empty to use the station extent
expanded by 5 percent.

The support multiplier controls the local Wendland basis support radius. It is
not the same as the output interpolation extent.
