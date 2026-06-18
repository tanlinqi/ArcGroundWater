# IGNNK ArcMap 10.8 空间插值工具

[ArcMap 深度学习空间插值统一接入规范](../../../../SSIN/ARCMAP_DL_INTERPOLATION_INTEGRATION_GUIDE.md)

本工具保留 IGNNK 的归纳式图神经网络机制。运行时固定采用“先训练后插值”流程：
从所选时间范围中抽取时间窗口，随机遮蔽一部分站点并重建完整信号；训练完成后将
规则网格点作为未观测新节点分批加入图，由已观测站点恢复每个网格点的值。

每次运行都会使用当前输入 CSV 和参数重新训练，并保存本次得到的 `trained_model.pth`。
工具不会读取项目自带旧模型，也不提供“仅加载已有模型”的运行模式。

## ArcToolbox 参数

| 编号 | 参数 | 类型 | 默认值 |
|---:|---|---|---|
| 0 | 输入时序 CSV | File | 必填 |
| 1 | 时间字段 | String | 必填 |
| 2 | 站点字段 | String | 必填 |
| 3 | X/经度字段 | String | 必填 |
| 4 | Y/纬度字段 | String | 必填 |
| 5 | 数值字段 | String | 必填 |
| 6 | 开始时间 | String | 首个时间 |
| 7 | 结束时间 | String | 最后时间 |
| 8 | 输入坐标系 | Spatial Reference | WGS84 |
| 9 | 插值范围 | Extent | 站点范围 |
| 10 | 像元大小（WGS84 度） | Double | 建议 0.05 |
| 11 | 输出文件夹 | Folder | 必填 |
| 12 | 训练轮数 | Long | 100 |
| 13 | 时间窗口长度 | Long | 12 |
| 14 | 隐藏维度 | Long | 64 |
| 15 | 扩散阶数 | Long | 1 |
| 16 | 每批遮蔽站点数 | Long | 20 |
| 17 | 学习率 | Double | 0.001 |
| 18 | 每轮训练批数 | Long | 4 |
| 19 | 网格查询批大小 | Long | 512 |
| 20 | 距离衰减尺度（km） | Double | 100 |
| 21 | 邻接阈值 | Double | 0.01 |
| 22 | 数值缩放（0=自动） | Double | 0 |
| 23 | 将 0 视为缺失 | Boolean | False |
| 24 | 计算设备 | String | AUTO |
| 25 | 随机种子 | Long | 42 |
| 26 | 最大网格数 | Long | 2000000 |
| 27 | 输出栅格 | Raster Dataset, MultiValue, Derived | |
| 28 | 输出训练模型 | File, Derived | |
| 29 | 输出结果清单 | File, Derived | |
| 30 | 模型 Python | File | `C:\Users\Lenovo\Miniconda3\envs\ssin\python.exe` |
| 31 | 覆盖已有结果 | Boolean | False |

## 创建 ArcToolbox

1. 在 ArcCatalog 或 ArcMap Catalog 窗口中新建一个 Legacy Toolbox（`.tbx`）。
2. 在工具箱中新建 Script Tool，名称可设为 `IGNNK Spatial Interpolation`。
3. Script File 指向 `toolbox\ignnk_arctoolbox.py`，取消后台运行，确保前台执行。
4. 按上表从 0 到 31 添加参数，编号、方向和数据类型必须一致。
5. 参数 27 设置为 `Raster Dataset`、`Derived`、`MultiValue`；28 和 29 设置为
   `File`、`Derived`。
6. 在工具属性的 Validation 页粘贴 `ToolValidator.py` 全部内容。

## 输入和输出

输入采用长表 CSV，`time + station` 必须唯一，同一站点坐标必须始终一致。空白、
`NaN` 和非数值内容作为缺失值；参数 23 可额外把 0 作为缺失值。

后端把输入坐标转换为 WGS84。自定义范围按输入坐标系解释并转换为 WGS84 包络，
像元大小按 WGS84 度解释。每个时间输出一个北向在数组第 0 行的 NPZ 和 GeoTIFF，
并生成增量更新的 `result_manifest.json`。

对于 BW 132 站点这类数据，建议“每批遮蔽站点数”设为 10-30。设为 1 时训练信号太弱，
模型容易只学到已知站点重建，网格插值会收缩到接近全局均值的小范围。

降水数据零值多、局部峰值强，纯 IGNNK 容易过度平滑。后端默认使用观测锚定融合：
每个输出时刻先用站点观测计算 IDW 锚定场，再与 IGNNK 输出融合。默认锚定权重为
0.85，IDW 幂次为 2.0；这些参数记录在 `result_manifest.json` 的
`training_parameters` 中。输入值全部非负时，输出会自动裁剪到 0 以上。

## 命令行验证

```bat
C:\Users\Lenovo\Miniconda3\envs\ssin\python.exe model_backend.py ^
  --input-csv toolbox\sample_input.csv ^
  --time-field time --station-field station ^
  --x-field x --y-field y --value-field value ^
  --input-crs EPSG:4326 --cell-size 0.05 ^
  --output-dir output_test --epochs 2 --time-window 3 ^
  --hidden-dim 8 --masked-stations 1 --device CPU --overwrite
```
