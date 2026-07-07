# Linear Regression ArcToolbox Parameters

| 顺序 | 参数 | 中文数据类型 | 要求 | 默认值 |
|---:|---|---|---|---|
| 0 | 输入时序文件 | 文件（多值） | 必选，CSV/Excel，多文件用分号分隔 | 无 |
| 1 | 时间字段 | 字符串 | 必选，验证器读取 CSV 字段并生成下拉框 | Month |
| 2 | 数值字段 | 字符串 | 必选，验证器读取 CSV 字段并生成下拉框 | Qm |
| 3 | 时间频率 | 字符串 | 可选，下拉：Auto/Daily/Monthly/Yearly/Hourly/Custom | Auto |
| 4 | 缺失值标记 | 双精度/字符串 | 可选，用于识别缺失值；不是下拉框 | -99999 |
| 5 | 按时间排序 | 布尔值 | 可选，true/false | true |
| 6 | 重复时间处理 | 字符串 | 可选，error/mean/first/last | error |
| 7 | 输出 HTML 报告 | 文件 | 可选，验证器根据输入文件自动填充 | 输入文件夹/Multiple_Interpolation_Result.html |
| 8 | 输出文件夹 | 文件夹 | 必选，验证器根据输入文件自动填充 | 输入文件所在文件夹 |
| 9 | 打开 HTML 报告 | 布尔值 | 可选，true/false | true |
| 10 | 滞后阶数 | 长整型 | 必选，生成 lag_1 到 lag_n 特征 | 3 |
| 11 | 验证策略 | 字符串 | 可选，time_split/rolling | time_split |
| 12 | 训练测试比例 | 字符串 | 可选，例如 8:2 或 0.2；Filter 必须为 None | 8:2 |
| 13 | 交叉验证折数 | 长整型 | 可选，GridSearchCV/TimeSeriesSplit 使用 | 3 |
| 14 | 随机种子 | 长整型 | 可选，用于可复现 | 42 |
| 15 | 插补最小值 | 双精度 | 可选，插补结果下限 | 0 |
| 16 | 插补最大值 | 双精度 | 可选，插补结果上限 | 空 |
| 17 | 加入趋势特征 | 布尔值 | 可选，true/false | false |
| 18 | 加入季节特征 | 布尔值 | 可选，true/false | false |
| 19 | 拟合截距 | 布尔值 | 可选，true/false | true |
| 20 | 正系数约束 | 布尔值 | 可选，true/false | false |
| 21 | Python3 路径 | 文件 | 可选，python.exe；留空使用默认环境或 ARCWATER_PY3 | 空 |


