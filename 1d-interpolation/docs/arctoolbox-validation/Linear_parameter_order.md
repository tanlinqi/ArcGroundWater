# Linear ArcToolbox Parameter Order

Use this order for the Linear script tool. The validation code in
`Linear_ToolValidator_minimal.py` is written for this exact order.

| Index | Parameter | Type / UI |
|---:|---|---|
| 0 | Input time-series CSV | File, multi-value allowed |
| 1 | Time field | String, dropdown from CSV fields |
| 2 | Value field | String, dropdown from CSV fields |
| 3 | Missing-value marker | Double/String |
| 4 | Time frequency | String dropdown: Auto, Daily, Monthly, Yearly, Hourly, Custom |
| 5 | Sort by time | Boolean/String dropdown |
| 6 | Duplicate-time handling | String dropdown: error, mean, first, last |
| 7 | Output HTML report | File, optional, auto-filled from input CSV |
| 8 | Output folder | Folder, auto-filled from input CSV folder |
| 9 | Open HTML report | Boolean/String dropdown |
| 10 | Lag steps | Long |
| 11 | Validation strategy | String dropdown: time_split, rolling |
| 12 | Train/test ratio | String, for example 8:2 |
| 13 | Cross-validation folds | Long |
| 14 | Random seed | Long |
| 15 | Minimum imputation value | Double, optional |
| 16 | Maximum imputation value | Double, optional |
| 17 | Add trend feature | Boolean/String dropdown |
| 18 | Add month/season feature | Boolean/String dropdown |
| 19 | Fit intercept | Boolean/String dropdown |
| 20 | Positive coefficients only | Boolean/String dropdown |

If ArcToolbox still shows shifted values, run the tool window once after
pasting the validator. The validator writes this diagnostic file:

`%TEMP%\arcwater_linear_validator_params.txt`

That file records the real order ArcMap is passing to the validator. Compare it
with the table above; if the order differs, the script tool parameter list in
ArcToolbox must be reordered to match this table, or the validator indexes must
be adjusted to the actual order.
