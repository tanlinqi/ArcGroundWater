# SATCN ArcMap 脚本工具说明

本工具用于将 SATCN 转换为 ArcMap 10.8 可调用的时间序列空间插值工具。工具从长表 CSV 中读取站点时间序列数据，先训练 SATCN 模型，再对选定时间范围内的每个时间片输出规则 WGS84 网格插值结果。

## 参数表

| 序号 | 参数名称 | 是否必填 | 说明 |
|---|---|---|---|
| 0 | 输入时序 CSV | 是 | 长表格式输入文件。 |
| 1 | 时间字段 | 是 | 时间戳字段。 |
| 2 | 站点字段 | 是 | 站点编号字段。 |
| 3 | X/经度字段 | 是 | WGS84 经度字段。 |
| 4 | Y/纬度字段 | 是 | WGS84 纬度字段。 |
| 5 | 数值字段 | 是 | 需要插值的观测值字段。 |
| 6 | 开始时间 | 否 | 默认使用第一个时间。 |
| 7 | 结束时间 | 否 | 默认使用最后一个时间。 |
| 8 | 输入坐标系 | 是 | 默认 WGS84。 |
| 9 | 插值范围 | 否 | 留空或使用 ArcGIS 范围关键字时，采用站点坐标范围。 |
| 10 | 像元大小 | 是 | 输出网格像元大小，单位为度。 |
| 11 | 输出文件夹 | 是 | 保存模型、NPZ、栅格和 manifest 文件。 |
| 12 | 训练轮数 | 是 | 默认 20。 |
| 13 | 批大小 | 是 | 默认 8。 |
| 14 | 学习率 | 是 | 默认 0.001。 |
| 15 | 隐藏通道数 | 是 | 默认 64。 |
| 16 | 网络层数 | 是 | 默认 1。 |
| 17 | 时间卷积核大小 | 是 | 默认 2。 |
| 18 | 邻居数量 | 是 | 默认 8。 |
| 19 | 遮挡站点数 | 是 | 自监督训练时随机隐藏的站点数量。 |
| 20 | 运行设备 | 是 | 可选 AUTO、CUDA 或 CPU。 |
| 21 | 随机种子 | 是 | 用于复现实验结果。 |
| 22 | 模型 Python | 否 | Python 3 环境路径，需要安装 torch、numpy、pandas。 |
| 23 | 覆盖已有结果 | 否 | 允许替换已有的 result_manifest.json。 |
| 24 | 输出栅格 | 派生 | 返回第一个生成的栅格。 |
| 25 | 输出训练模型 | 派生 | 返回 trained_model.pth。 |
| 26 | 输出结果清单 | 派生 | 返回 result_manifest.json。 |

## 输入 CSV 格式

推荐字段如下：

```csv
time,station,x,y,value
20120104-00,20,9.9129,48.9219,1.3
20120104-00,71,8.9784,48.2156,0.4
20120104-01,20,9.9129,48.9219,0.8
```

输入规则：

- 每一行表示一个站点在一个时间的观测值。
- `time + station` 组合必须唯一。
- 同一站点在所有时间的坐标必须一致。
- 缺失值可以留空或写成无法转换为数值的内容，后端会按缺失处理。
- 工具支持字段映射，CSV 字段名不必固定为示例中的名称。

## 输出文件

输出文件夹中会生成：

```text
trained_model.pth
interpolation_<time>.npz
interpolation_<time>.tif
result_manifest.json
```

其中：

- `trained_model.pth` 是本次训练得到的模型。
- `interpolation_<time>.npz` 保存每个时间片的插值数组。
- `interpolation_<time>.tif` 是 ArcMap 可加载的 GeoTIFF 栅格。
- `result_manifest.json` 记录参数、范围、模型路径和每个时间片的输出信息。

## 后端运行方式

ArcMap 脚本工具只负责读取参数、启动子进程、生成 GeoTIFF 和加载地图图层。深度学习模型由 `model_backend.py` 在独立 Python 3 进程中运行，ArcMap Python 2.7 不会导入 PyTorch。

后端命令示例：

```bat
python model_backend.py ^
  --input-csv input.csv ^
  --time-field time ^
  --station-field station ^
  --x-field x ^
  --y-field y ^
  --value-field value ^
  --cell-size 0.05 ^
  --output-dir output ^
  --device AUTO ^
  --overwrite
```

## ArcMap 绑定说明

创建 Script Tool 时，请保持参数编号与上方参数表完全一致。脚本文件选择：

```text
toolbox/model_arctoolbox.py
```

验证器代码使用：

```text
toolbox/ToolValidator.py
```

修改磁盘上的 `ToolValidator.py` 后，需要重新复制到 ArcMap 工具属性页的“验证”编辑器中。
