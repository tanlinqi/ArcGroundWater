# ArcGroundWater 1D 到 2D 数据流说明

## 总体流程

```text
原始单站点时间序列 CSV
  -> 1D 缺失值补齐工具
  -> 每个站点一个 *_filled.csv
  -> 数据适配工具
  -> 2D 长表 CSV: time, station, x, y, value
  -> SSIN/KCN/DeepKriging/IGNNK/SATCN 空间插值工具
  -> NPZ / GeoTIFF / manifest / log
```

## 1D 输入 CSV

1D 工具处理的是单站点时间序列。每个 CSV 通常表示一个监测站。

必需字段：

| 字段 | 含义 | 示例 |
|---|---|---|
| `Month` | 时间字段，可在工具参数中指定 | `2000-01-01` |
| `Qm` | 待补齐的水文/地下水数值字段，可在工具参数中指定 | `12.35` |

允许有缺失值，缺失值由 1D 工具参数指定，例如空值或 `-99999`。

## 1D 输出 CSV

1D 工具输出文件名通常为：

```text
站点名_lr_filled.csv
站点名_rf_filled.csv
站点名_arima_filled.csv
```

典型字段：

| 字段 | 含义 |
|---|---|
| `Month` | 时间 |
| `Qm` | 原始数值 |
| `_arcwater_time` | 内部排序时间 |
| `is_missing` | 原始值是否缺失 |
| `Qm_filled` | 补齐后的数值 |

后续 2D 空间插值应优先使用 `Qm_filled`。

## 站点坐标表

因为 1D 输出通常没有空间坐标，所以需要单独提供一个站点坐标表。

推荐格式：

```csv
station,x,y
白家川,110.123,35.456
河津,110.712,35.598
```

其中：

| 字段 | 含义 |
|---|---|
| `station` | 站点名，必须与 1D 输出文件名前缀一致 |
| `x` | 经度或投影 X 坐标 |
| `y` | 纬度或投影 Y 坐标 |

## 2D 输入 CSV

2D 工具统一使用长表格式：

```csv
time,station,x,y,value
2000-01-01,白家川,110.123,35.456,12.35
2000-01-01,河津,110.712,35.598,11.80
2000-02-01,白家川,110.123,35.456,12.42
```

必需字段：

| 字段 | 含义 |
|---|---|
| `time` | 时间 |
| `station` | 站点 |
| `x` | X/经度 |
| `y` | Y/纬度 |
| `value` | 用于空间插值的数值 |

## 数据适配工具

脚本位置：

```text
data-adapter/one_d_to_2d_long_table.py
```

示例命令：

```bat
python data-adapter\one_d_to_2d_long_table.py ^
  --input-dir 1d-interpolation\python2 ^
  --station-meta station_meta.csv ^
  --output-csv output\arcgroundwater_2d_input.csv ^
  --time-field Month ^
  --value-field Qm_filled
```

输出文件即可作为 2D 插值工具的输入 CSV。
