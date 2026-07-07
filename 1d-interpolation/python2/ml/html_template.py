# -*- coding: utf-8 -*-

def generate_report_html(page_title_json, all_data_json, y_field_json):
    """
    Scientific Dashboard v10.0:
    - Advanced Data Diagnostics: Added Max, Min, Variance, Skewness, and Median calculations.
    - Professional dual-column comparative layout for Original vs Imputed datasets.
    - Cleaned up UI interactions.
    """
    html_content = u'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>''' + page_title_json + u'''</title>
    <script src="echarts.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body { 
            font-family: "Segoe UI", sans-serif; 
            background: #f4f7f9; 
            color: #333; 
            display: flex; 
            height: 100vh; 
            overflow: hidden; 
        }

        .main-content { 
            flex: 0 0 78%; 
            padding: 20px 30px; 
            overflow-y: auto; 
            height: 100%;
            scroll-behavior: smooth;
        }

        .sidebar { 
            flex: 0 0 22%; 
            background: #ffffff; 
            border-left: 1px solid #dcdfe6; 
            padding: 20px; 
            display: flex; 
            flex-direction: column; 
            gap: 15px; 
            box-shadow: -4px 0 10px rgba(0,0,0,0.05);
            z-index: 100;
            overflow-y: auto;
        }

        .chart-container { 
            width: 100%; 
            background: #ffffff; 
            border-radius: 10px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.06); 
            margin-bottom: 40px;
            padding: 20px;
        }

        #mainChart { height: 550px; }
        #trainChart, #testChart, #residualChart { height: 500px; }
        #detailPanel { min-height: 260px; }

        .sidebar-header {
            font-size: 13px;
            font-weight: 700;
            color: #1a3a6c;
            border-bottom: 2px solid #4a90d9;
            padding-bottom: 5px;
            margin-top: 5px;
            text-transform: uppercase;
        }

        .control-group { display: flex; flex-direction: column; gap: 6px; }
        .flex-row-align { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
        label { font-weight: 600; color: #5a5e66; font-size: 12px; }

        select, input[type="text"] { 
            padding: 6px 8px; 
            font-size: 12px; 
            border: 1px solid #dcdfe6; 
            border-radius: 4px; 
            outline: none; 
            background: #fff;
            width: 100%;
        }

        input[type="range"] { flex: 1; cursor: pointer; }

        .info-box {
            background: #f8f9fb;
            border-radius: 6px;
            padding: 12px;
            font-size: 11.5px;
            color: #444;
            line-height: 1.6;
            border: 1px solid #eef1f6;
        }

        /* Styles for the new Diagnostics Table */
        .diag-table { width: 100%; border-collapse: collapse; text-align: right; font-size: 11px; }
        .diag-table th { border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #1a3a6c; }
        .diag-table td { padding: 4px 0; border-bottom: 1px dashed #eee; }
        .diag-table th:first-child, .diag-table td:first-child { text-align: left; font-weight: bold; color: #555; }
        .val-orig { color: #888; }
        .val-full { color: #2c4a7c; font-weight: bold; }
        .warn-text { color: #e05252; font-weight: bold; }

        .section-title {
            color: #1a3a6c;
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .detail-note {
            color: #687385;
            font-size: 12px;
            margin-bottom: 12px;
            line-height: 1.5;
        }

        .detail-table-wrap {
            max-height: 360px;
            overflow: auto;
            border: 1px solid #e3e8f0;
            border-radius: 8px;
        }

        .detail-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }

        .detail-table th {
            position: sticky;
            top: 0;
            background: #f2f6fc;
            color: #1a3a6c;
            z-index: 1;
            text-align: left;
            padding: 9px 10px;
            border-bottom: 1px solid #dbe3ef;
        }

        .detail-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #eef1f6;
            color: #3f4652;
        }

        .detail-table td.num {
            text-align: right;
            font-family: Consolas, monospace;
        }

        .detail-table tbody tr:nth-child(even) { background: #fafbfd; }
        .detail-table tbody tr:hover { background: #fff7ec; }
        .detail-table tbody tr.selected-row { background: #fff0d9; outline: 2px solid #f5a623; }
        .status-badge { display: inline-block; padding: 2px 7px; border-radius: 10px; font-size: 10px; font-weight: 700; }
        .status-filled { background: #e8f4ff; color: #1a6fb0; }
        .status-unfilled { background: #ffecec; color: #c23b3b; }
        .toolbar-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; }
        .small-button { border: 1px solid #cfd8e6; background: #ffffff; color: #1a3a6c; border-radius: 4px; padding: 6px 10px; font-size: 12px; cursor: pointer; }
        .small-button:hover { background: #f2f6fc; }
        .risk-list { display: flex; flex-direction: column; gap: 6px; }
        .risk-item { padding: 7px 8px; border-radius: 5px; background: #fff4e8; color: #9a5a00; border: 1px solid #f3d2a7; font-size: 11px; line-height: 1.4; }
        .risk-ok { padding: 8px; border-radius: 5px; background: #edf8f1; color: #287346; border: 1px solid #ccebd7; font-size: 11px; }
        .summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 10px; }
        .summary-card { background: #f8f9fb; border: 1px solid #eef1f6; border-radius: 6px; padding: 8px; }
        .summary-card b { display: block; color: #1a3a6c; font-size: 14px; }
        .summary-card span { color: #687385; font-size: 10px; }
        .empty-state { padding: 28px; color: #8792a2; text-align: center; }

        #loadError { display: none; padding: 15px; background: #fff4e5; color: #8a5000; border: 1px solid #f1c27d; border-radius: 6px; text-align: center; }
    </style>
</head>
<body>

    <div class="main-content" id="leftPanel">
        <div id="mainChart" class="chart-container"></div>
        <div id="trainChart" class="chart-container"></div>
        <div id="testChart" class="chart-container"></div>
        <div id="residualChart" class="chart-container"></div>
        <div id="detailPanel" class="chart-container">
            <div class="toolbar-row">
                <div>
                    <div class="section-title">Imputation Detail Inspector</div>
                    <div class="detail-note">Each row corresponds to one missing value in the current dataset. Next observed value is shown only as a post-hoc reference, not as model input.</div>
                </div>
                <button id="exportDetailsBtn" class="small-button">Export Details CSV</button>
            </div>
            <div id="detailTableHost"></div>
        </div>
        <div id="loadError">ECharts failed to load.</div>
    </div>

    <div class="sidebar">
        <div class="sidebar-header">Global Data</div>
        <div class="control-group">
            <label>Select Dataset</label>
            <select id="csvSelector"></select>
        </div>

        <div class="sidebar-header">Customize Chart</div>
        <div class="control-group">
            <label>Target Chart</label>
            <select id="targetChartSelect">
                <option value="main">Main Imputation Chart</option>
                <option value="train">Train Fitting Chart</option>
                <option value="test">Test Fitting Chart</option>
                <option value="residual">Residual Diagnostics</option>
            </select>
        </div>
        <div class="control-group"><label>Chart Title</label><input type="text" id="editChartTitle" placeholder="Enter title"></div>
        <div class="control-group"><label>X-Axis Title</label><input type="text" id="editXTitle"></div>
        <div class="control-group"><label>Y-Axis Title</label><input type="text" id="editYTitle"></div>

        <div class="control-group" id="styleControlGroup">
            <label>Chart Style (Main Only)</label>
            <select id="editChartStyle">
                <option value="line">Line Chart</option>
                <option value="scatter">Scatter Plot</option>
                <option value="bar">Bar Chart</option>
                <option value="step">Step Line</option>
            </select>
        </div>

        <div class="control-group" id="sizeControlGroup">
            <div class="flex-row-align">
                <label id="symbolSizeLabel" style="width: 45px;">Size: 8</label>
                <input type="range" id="symbolSizeSlider" min="1" max="25" value="8">
            </div>
        </div>

        <div class="sidebar-header">Model Context</div>
        <div id="paramInfo" class="info-box"></div>

        <div class="sidebar-header">Data Diagnostics</div>
        <div class="info-box" id="calcResultBox">
            </div>

        <div class="sidebar-header">Risk Flags</div>
        <div class="info-box" id="riskFlagBox"></div>

        <div class="sidebar-header">Selected Imputation</div>
        <div class="info-box" id="selectedPointBox">Click an interpolated point to inspect it.</div>

        <div style="margin-top: auto; font-size: 10px; color: #b0b4bb; text-align: center;">
            Scientific Dashboard v10.0
        </div>
    </div>

    <script>
        if (typeof echarts === 'undefined') {
            document.querySelectorAll('.chart-container').forEach(function(el) { el.style.display = 'none'; });
            document.getElementById('loadError').style.display = 'block';
            throw new Error('ECharts core not found.');
        }

        var charts = {
            main: echarts.init(document.getElementById('mainChart')),
            train: echarts.init(document.getElementById('trainChart')),
            test: echarts.init(document.getElementById('testChart')),
            residual: echarts.init(document.getElementById('residualChart'))
        };

        var allData = ''' + all_data_json + u''';
        var yFieldGlobal = ''' + y_field_json + u''';

        var chartStates = {
            main: { title: '', xTitle: 'Date', yTitle: yFieldGlobal, style: 'line', symbolSize: 8 },
            train: { title: 'Train Set: Predicted vs. True', xTitle: 'True Value', yTitle: 'Predicted Value', style: 'scatter', symbolSize: 6 },
            test: { title: 'Test Set: Predicted vs. True', xTitle: 'True Value', yTitle: 'Predicted Value', style: 'scatter', symbolSize: 6 },
            residual: { title: 'Test Residual Diagnostics', xTitle: 'Sample Index', yTitle: 'Residual', style: 'scatter', symbolSize: 7 }
        };

        var dsSelector = document.getElementById('csvSelector');
        var targetSelect = document.getElementById('targetChartSelect');
        var editTitle = document.getElementById('editChartTitle');
        var editX = document.getElementById('editXTitle');
        var editY = document.getElementById('editYTitle');
        var editStyle = document.getElementById('editChartStyle');
        var styleControl = document.getElementById('styleControlGroup');
        var sizeControl = document.getElementById('sizeControlGroup');
        var sizeSlider = document.getElementById('symbolSizeSlider');
        var sizeLabel = document.getElementById('symbolSizeLabel');
        var paramBox = document.getElementById('paramInfo');
        var calcResultBox = document.getElementById('calcResultBox');
        var detailTableHost = document.getElementById('detailTableHost');
        var riskFlagBox = document.getElementById('riskFlagBox');
        var selectedPointBox = document.getElementById('selectedPointBox');
        var exportDetailsBtn = document.getElementById('exportDetailsBtn');

        Object.keys(allData).forEach(function(name) {
            var opt = document.createElement('option');
            opt.value = name; opt.innerHTML = name; dsSelector.appendChild(opt);
        });

        function markAreasFromSegments(data) {
            if (!data.missingSegments || !data.dates) return [];
            return data.missingSegments.map(function(seg) {
                return [
                    { xAxis: data.dates[seg.start], itemStyle: { color: 'rgba(224,82,82,0.08)' } },
                    { xAxis: data.dates[seg.end] }
                ];
            });
        }

        function renderMainChart(data, fileName) {
            var state = chartStates.main;
            var isStep = state.style === 'step' ? 'middle' : false;
            var baseType = state.style === 'step' ? 'line' : state.style;
            var showSymbol = state.style === 'scatter';

            var seriesList = [];
            var missingIndices = data.scatterPts.map(function(pt) { return pt[0]; });
            var originalOnlyValues = [];
            var interpolatedOnlyValues = [];

            data.lineValues.forEach(function(val, idx) {
                if (missingIndices.indexOf(idx) !== -1) {
                    originalOnlyValues.push("-");
                    interpolatedOnlyValues.push(val);
                } else {
                    originalOnlyValues.push(val);
                    interpolatedOnlyValues.push("-");
                }
            });

            var markLineData = { data: [{yAxis: data.meanLine, label: {formatter: function(p) { return 'Mean: ' + formatDetailNumber(p.value); }}}] };
            var markAreaData = markAreasFromSegments(data);
            var markArea = markAreaData.length ? { silent: true, data: markAreaData } : undefined;

            if (baseType === 'line') {
                seriesList.push({ 
                    name: 'Original', type: baseType, step: isStep, symbol: 'none', 
                    data: data.lineValues, itemStyle: { color: '#4a90d9' }, lineStyle: { color: '#4a90d9', width: 2 }, markLine: markLineData, markArea: markArea
                });
                seriesList.push({ 
                    name: 'Interpolated', type: 'scatter', symbol: 'circle', symbolSize: state.symbolSize + 2, 
                    data: data.scatterPts, itemStyle: { color: '#e05252' } 
                });
            } else {
                seriesList.push({ 
                    name: 'Original', type: baseType, symbol: showSymbol ? 'circle' : 'none', symbolSize: state.symbolSize, 
                    data: originalOnlyValues, itemStyle: { color: '#4a90d9' }, markLine: markLineData, markArea: markArea
                });
                seriesList.push({ 
                    name: 'Interpolated', type: baseType, symbol: showSymbol ? 'circle' : 'none', symbolSize: state.symbolSize + 2, 
                    data: interpolatedOnlyValues, itemStyle: { color: '#e05252' } 
                });
            }

            var option = {
                animationDurationUpdate: 500, backgroundColor: '#ffffff',
                title: { text: state.title || (fileName + ' - Imputation Results'), left: 'center', top: 10, textStyle: { color: '#1a3a6c', fontSize: 16 } },
                tooltip: { 
                    trigger: 'axis',
                    formatter: function(params) {
                        var dataIdx = params[0].dataIndex;
                        var isMissing = missingIndices.indexOf(dataIdx) !== -1;
                        var res = '<b>' + params[0].axisValue + '</b><br/>';
                        params.forEach(function(p) {
                            if (p.seriesName === 'Original' && isMissing) return;
                            if (p.seriesName === 'Interpolated' && !isMissing) return;
                            var val = Array.isArray(p.value) ? p.value[1] : p.value;
                            if (val !== '-' && val != null) {
                                res += '<span style="color:' + p.color + '">&#9679;</span> ' + p.seriesName + ': <b>' + Number(val).toFixed(2) + '</b><br/>';
                            }
                        });
                        return res;
                    }
                },
                toolbox: { feature: { saveAsImage: {} }, right: 20 },
                legend: { data: ['Original', 'Interpolated'], bottom: 10, left: 'center', itemGap: 50, textStyle: { fontSize: 14 } },
                dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 50, height: 20 }],
                grid: { left: '5%', right: '5%', bottom: '25%', containLabel: true },
                xAxis: { type: 'category', data: data.dates, name: state.xTitle, nameLocation: 'middle', nameGap: 35 },
                yAxis: { type: 'value', min: data.yAxisMin, max: data.yAxisMax, scale: true, name: state.yTitle },
                series: seriesList
            };
            charts.main.setOption(option, true);
            charts.main.off('click');
            charts.main.on('click', function(params) {
                if (params.seriesName !== 'Interpolated') return;
                var idx = Array.isArray(params.value) ? params.value[0] : params.dataIndex;
                selectImputationRow(idx, true);
            });
        }

        function renderFittingChart(chartKey, dataObj) {
            var state = chartStates[chartKey];
            var fitPoints = dataObj.y_true.map(function(val, i) { return [val, dataObj.y_pred[i]]; })
                .filter(function(pt) { return pt[0] !== null && pt[1] !== null; });

            var allVals = dataObj.y_true.concat(dataObj.y_pred).filter(function(v) { return v !== null && !isNaN(Number(v)); });
            if (allVals.length === 0) {
                charts[chartKey].clear();
                return;
            }
            var min = Math.floor(Math.min.apply(null, allVals));
            var max = Math.ceil(Math.max.apply(null, allVals));
            var padding = (max - min) * 0.05;
            var axisMin = min - padding;
            var axisMax = max + padding;

            var option = {
                title: { text: state.title + ' (R2: ' + formatDetailNumber(dataObj.r2) + ')', left: 'center', top: 10, textStyle: { fontSize: 15 } },
                tooltip: { formatter: function(params) { 
                    if(params.seriesName === 'y=x') return '';
                    return 'True: ' + formatDetailNumber(params.value[0]) + '<br/>Pred: ' + formatDetailNumber(params.value[1]); 
                }},
                toolbox: { feature: { saveAsImage: {} }, right: 20 },
                grid: { left: '10%', right: '10%', bottom: '20%', top: '15%', containLabel: true },
                xAxis: { type: 'value', name: state.xTitle, min: axisMin, max: axisMax, nameLocation: 'middle', nameGap: 30, axisLabel: { formatter: function(value) { return formatDetailNumber(value); } } },
                yAxis: { type: 'value', name: state.yTitle, min: axisMin, max: axisMax, axisLabel: { formatter: function(value) { return formatDetailNumber(value); } } },
                dataZoom: [
                    { type: 'inside', xAxisIndex: 0, yAxisIndex: 0, filterMode: 'none' }, 
                    { type: 'slider', xAxisIndex: 0, yAxisIndex: 0, bottom: 10, height: 20, filterMode: 'none' }
                ],
                series: [
                    {
                        name: 'Fitting', type: 'scatter', data: fitPoints, symbolSize: state.symbolSize,
                        itemStyle: { color: chartKey === 'train' ? '#4a90d9' : '#e05252', opacity: 0.7 }
                    },
                    {
                        name: 'y=x', type: 'line', data: [[axisMin-1000, axisMin-1000], [axisMax+1000, axisMax+1000]],
                        symbol: 'none', lineStyle: { type: 'dashed', color: '#666', width: 1.5 },
                        silent: true, animation: false
                    }
                ]
            };
            charts[chartKey].setOption(option, true);
        }

        function renderResidualChart(data) {
            var state = chartStates.residual;
            var testData = data.testData || {};
            var residualRows = [];
            if (testData.indices && testData.residual) {
                testData.indices.forEach(function(idx, i) {
                    var residual = testData.residual[i];
                    if (residual !== null && residual !== undefined) {
                        residualRows.push([idx, residual, testData.y_true[i], testData.y_pred[i]]);
                    }
                });
            }

            var option = {
                title: { text: state.title, left: 'center', top: 10, textStyle: { color: '#1a3a6c', fontSize: 15 } },
                tooltip: { trigger: 'item', formatter: function(params) {
                    return 'Index: ' + params.value[0] +
                        '<br/>Residual: <b>' + Number(params.value[1]).toFixed(2) + '</b>' +
                        '<br/>True: ' + formatDetailNumber(params.value[2]) +
                        '<br/>Pred: ' + formatDetailNumber(params.value[3]);
                }},
                toolbox: { feature: { saveAsImage: {} }, right: 20 },
                grid: { left: '7%', right: '5%', bottom: '18%', top: '16%', containLabel: true },
                dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 15, height: 20 }],
                xAxis: { type: 'value', name: state.xTitle, nameLocation: 'middle', nameGap: 30, axisLabel: { formatter: function(value) { return formatDetailNumber(value); } } },
                yAxis: { type: 'value', name: state.yTitle, scale: true, axisLabel: { formatter: function(value) { return formatDetailNumber(value); } } },
                series: [
                    { name: 'Residual', type: 'scatter', data: residualRows, symbolSize: state.symbolSize, itemStyle: { color: '#8e63c7', opacity: 0.78 } },
                    { name: 'Zero Line', type: 'line', data: residualRows.length ? [[residualRows[0][0], 0], [residualRows[residualRows.length - 1][0], 0]] : [], symbol: 'none', lineStyle: { type: 'dashed', color: '#777' }, silent: true }
                ]
            };
            charts.residual.setOption(option, true);
        }

        // --- MATH UTILS FOR DIAGNOSTICS ---
        function getStatistics(arr) {
            if(arr.length === 0) return { max: 0, min: 0, median: 0, variance: 0, skew: 0 };
            var sum = 0, max = -Infinity, min = Infinity, n = arr.length;
            var sorted = arr.slice().sort(function(a,b){return a-b;});
            for(var i=0; i<n; i++) {
                sum += arr[i];
                if(arr[i] > max) max = arr[i];
                if(arr[i] < min) min = arr[i];
            }
            var mean = sum / n;
            var median = n % 2 === 0 ? (sorted[n/2 - 1] + sorted[n/2]) / 2 : sorted[Math.floor(n/2)];

            var sumSq = 0, sumCub = 0;
            for(var j=0; j<n; j++) {
                var diff = arr[j] - mean;
                sumSq += diff * diff;
                sumCub += diff * diff * diff;
            }
            var variance = sumSq / n;
            var stdDev = Math.sqrt(variance);
            var skew = stdDev === 0 ? 0 : (sumCub / n) / Math.pow(stdDev, 3);

            return { max: max, min: min, median: median, variance: variance, skew: skew };
        }

        function runCalculationDiagnostics(data) {
            var origValues = [];
            var fullValues = []; // Combined original + imputed

            var missingIndices = data.scatterPts.map(function(pt) { return pt[0]; });
            var interpMap = {};
            data.scatterPts.forEach(function(pt) { interpMap[pt[0]] = pt[1]; });

            data.lineValues.forEach(function(val, idx) {
                if (missingIndices.indexOf(idx) === -1 && val !== null) {
                    origValues.push(val);
                    fullValues.push(val);
                } else if (interpMap[idx] !== undefined) {
                    fullValues.push(interpMap[idx]);
                }
            });

            var total = fullValues.length;
            var missingCount = data.scatterPts.length;
            var ratio = total === 0 ? '0.00' : ((missingCount / total) * 100).toFixed(2);

            var stOrig = getStatistics(origValues);
            var stFull = getStatistics(fullValues);

            // Calculate percentage changes
            var varDiff = stOrig.variance === 0 ? 0 : ((stFull.variance - stOrig.variance) / stOrig.variance * 100);
            var varColor = varDiff < -5 ? '#e05252' : '#2c4a7c'; // Red if variance drops significantly
            var varArrow = varDiff > 0 ? ' up' : (varDiff < 0 ? ' down' : '');

            var html = '<div style="margin-bottom:8px; border-bottom:1px solid #ccc; padding-bottom:4px;">';
            html += '<strong>Missing Ratio:</strong> <span style="color:#e05252">' + missingCount + ' / ' + total + ' (' + ratio + '%)</span></div>';

            html += '<table class="diag-table">';
            html += '<tr><th>Metric</th><th>Orig</th><th>Full Set</th></tr>';
            html += '<tr><td>Max</td><td class="val-orig">' + stOrig.max.toFixed(2) + '</td><td class="val-full">' + stFull.max.toFixed(2) + '</td></tr>';
            html += '<tr><td>Min</td><td class="val-orig">' + stOrig.min.toFixed(2) + '</td><td class="val-full">' + stFull.min.toFixed(2) + '</td></tr>';
            html += '<tr><td>Median</td><td class="val-orig">' + stOrig.median.toFixed(2) + '</td><td class="val-full">' + stFull.median.toFixed(2) + '</td></tr>';

            // Variance with dynamic indicator
            html += '<tr><td>Variance</td><td class="val-orig">' + stOrig.variance.toFixed(2) + '</td><td class="val-full" style="color:' + varColor + '">';
            html += stFull.variance.toFixed(2) + ' <span style="font-size:9px">(' + (varDiff>0?'+':'') + varDiff.toFixed(2) + '%' + varArrow + ')</span></td></tr>';

            html += '<tr><td style="border:none;">Skewness</td><td class="val-orig" style="border:none;">' + stOrig.skew.toFixed(2) + '</td><td class="val-full" style="border:none;">' + stFull.skew.toFixed(2) + '</td></tr>';
            html += '</table>';

            calcResultBox.innerHTML = html;
        }

        function escapeHtml(value) {
            return String(value)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        function formatDetailNumber(value) {
            if (value === null || value === undefined || value === '-' || isNaN(Number(value))) return '-';
            return Number(value).toFixed(2);
        }

        function getCurrentData() {
            return allData[dsSelector.value];
        }

        function selectImputationRow(index, scrollToTable) {
            var data = getCurrentData();
            if (!data || !data.imputationDetails) return;
            var detail = null;
            data.imputationDetails.forEach(function(item) {
                if (Number(item.index) === Number(index)) detail = item;
            });
            if (!detail) return;

            document.querySelectorAll('.detail-table tbody tr').forEach(function(row) {
                row.classList.remove('selected-row');
            });
            var targetRow = document.querySelector('.detail-table tbody tr[data-index="' + index + '"]');
            if (targetRow) {
                targetRow.classList.add('selected-row');
                targetRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }

            selectedPointBox.innerHTML =
                '<strong>Time:</strong> ' + escapeHtml(detail.time) + '<br/>' +
                '<strong>Index:</strong> ' + detail.index + '<br/>' +
                '<strong>Status:</strong> ' + escapeHtml(detail.status) + '<br/>' +
                '<strong>Imputed:</strong> ' + formatDetailNumber(detail.imputed_value) + '<br/>' +
                '<strong>Previous observed:</strong> ' + formatDetailNumber(detail.previous_observed) + '<br/>' +
                '<strong>Next observed reference:</strong> ' + formatDetailNumber(detail.next_observed_reference) + '<br/>' +
                '<strong>Missing segment:</strong> #' + detail.segment_id + ' (' + detail.segment_length + ' point(s))';

            if (scrollToTable) {
                document.getElementById('detailPanel').scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }

        function renderImputationDetails(data) {
            var details = data.imputationDetails || [];
            if (details.length === 0) {
                detailTableHost.innerHTML = '<div class="empty-state">No imputed points in the current dataset.</div>';
                return;
            }

            var html = '<div class="detail-table-wrap"><table class="detail-table">';
            html += '<thead><tr>';
            html += '<th>#</th><th>Time</th><th>Index</th><th>Status</th><th>Imputed Value</th><th>Previous Observed</th><th>Next Reference</th><th>Segment</th><th>Segment Length</th>';
            html += '</tr></thead><tbody>';

            details.forEach(function(item) {
                var statusClass = item.status === 'filled' ? 'status-filled' : 'status-unfilled';
                html += '<tr data-index="' + item.index + '">';
                html += '<td>' + item.row + '</td>';
                html += '<td>' + escapeHtml(item.time) + '</td>';
                html += '<td class="num">' + item.index + '</td>';
                html += '<td><span class="status-badge ' + statusClass + '">' + escapeHtml(item.status) + '</span></td>';
                html += '<td class="num">' + formatDetailNumber(item.imputed_value) + '</td>';
                html += '<td class="num">' + formatDetailNumber(item.previous_observed) + '</td>';
                html += '<td class="num">' + formatDetailNumber(item.next_observed_reference) + '</td>';
                html += '<td class="num">' + (item.segment_id || '-') + '</td>';
                html += '<td class="num">' + (item.segment_length || '-') + '</td>';
                html += '</tr>';
            });

            html += '</tbody></table></div>';
            detailTableHost.innerHTML = html;
            document.querySelectorAll('.detail-table tbody tr').forEach(function(row) {
                row.addEventListener('click', function() {
                    selectImputationRow(row.getAttribute('data-index'), false);
                });
            });
        }

        function renderRiskFlags(data) {
            var summary = data.qualitySummary || {};
            var flags = summary.risk_flags || [];
            var html = '<div class="summary-grid">';
            html += '<div class="summary-card"><b>' + (summary.missing_count || 0) + '</b><span>Missing</span></div>';
            html += '<div class="summary-card"><b>' + (summary.filled_count || 0) + '</b><span>Filled</span></div>';
            html += '<div class="summary-card"><b>' + (((summary.missing_rate || 0) * 100).toFixed(2)) + '%</b><span>Missing Rate</span></div>';
            html += '<div class="summary-card"><b>' + (summary.longest_missing_segment || 0) + '</b><span>Longest Segment</span></div>';
            html += '</div>';
            if (!flags.length) {
                html += '<div class="risk-ok">No major automatic risk flag was detected.</div>';
            } else {
                html += '<div class="risk-list">';
                flags.forEach(function(flag) { html += '<div class="risk-item">' + escapeHtml(flag) + '</div>'; });
                html += '</div>';
            }
            riskFlagBox.innerHTML = html;
        }

        function exportDetailsCsv() {
            var data = getCurrentData();
            if (!data || !data.imputationDetails || !data.imputationDetails.length) return;
            var headers = ['row','time','index','status','imputed_value','previous_observed','next_observed_reference','segment_id','segment_length'];
            var lines = [headers.join(',')];
            data.imputationDetails.forEach(function(item) {
                lines.push(headers.map(function(key) {
                    var value = item[key];
                    if (value === null || value === undefined) value = '';
                    value = String(value).replace(/"/g, '""');
                    return '"' + value + '"';
                }).join(','));
            });
            var blob = new Blob(['\ufeff' + lines.join('\\n')], { type: 'text/csv;charset=utf-8;' });
            var link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = dsSelector.value + '_imputation_details.csv';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        function updateContextBox(chartKey) {
            var data = allData[dsSelector.value];
            if (!data) return;
            if (chartKey === 'main') {
                paramBox.innerHTML = '<strong style="color:#2c4a7c">Evaluation:</strong><br/>' + data.parameterText + '<br/><br/><strong style="color:#2c4a7c">Chosen Params:</strong><br/>' + data.bestParamsText;
            } else if (chartKey === 'train') {
                paramBox.innerHTML = '<strong style="color:#4a90d9">Train Set Metrics:</strong><br/>R虏 Score: ' + formatDetailNumber(data.trainData.r2) + '<br/>RMSE: ' + formatDetailNumber(data.trainData.rmse);
            } else if (chartKey === 'test') {
                paramBox.innerHTML = '<strong style="color:#e05252">Test Set Metrics:</strong><br/>R虏 Score: ' + formatDetailNumber(data.testData.r2) + '<br/>RMSE: ' + formatDetailNumber(data.testData.rmse);
            }
        }

        function updateContextBoxV2(chartKey) {
            var data = allData[dsSelector.value];
            if (!data) return;
            if (chartKey === 'main') {
                paramBox.innerHTML = '<strong style="color:#2c4a7c">Evaluation:</strong><br/>' + data.parameterText + '<br/><br/><strong style="color:#2c4a7c">Chosen Params:</strong><br/>' + data.bestParamsText;
            } else if (chartKey === 'train') {
                paramBox.innerHTML = '<strong style="color:#4a90d9">Train Set Metrics:</strong><br/>R2 Score: ' + formatDetailNumber(data.trainData.r2) + '<br/>RMSE: ' + formatDetailNumber(data.trainData.rmse) + '<br/>MAE: ' + formatDetailNumber(data.trainData.mae) + '<br/>Bias: ' + formatDetailNumber(data.trainData.bias);
            } else if (chartKey === 'test') {
                paramBox.innerHTML = '<strong style="color:#e05252">Test Set Metrics:</strong><br/>R2 Score: ' + formatDetailNumber(data.testData.r2) + '<br/>RMSE: ' + formatDetailNumber(data.testData.rmse) + '<br/>MAE: ' + formatDetailNumber(data.testData.mae) + '<br/>Bias: ' + formatDetailNumber(data.testData.bias);
            } else if (chartKey === 'residual') {
                var rows = data.testData.top_residuals || [];
                var html = '<strong style="color:#8e63c7">Top Test Residuals:</strong><br/>';
                if (!rows.length) {
                    html += 'No residual data available.';
                } else {
                    rows.forEach(function(row) {
                        html += 'Index ' + row.index + ': ' + formatDetailNumber(row.residual) + '<br/>';
                    });
                }
                paramBox.innerHTML = html;
            }
        }

        var isManualScroll = false;
        var scrollTimeout = null;
        if (typeof IntersectionObserver !== 'undefined') {
        var observer = new IntersectionObserver(function(entries) {
            if (isManualScroll) return;
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    var id = entry.target.id;
                    var key = id.replace('Chart', '');
                    if (targetSelect.value !== key) {
                        targetSelect.value = key;
                        updateUIFromState(true);
                    }
                }
            });
        }, { root: document.getElementById('leftPanel'), threshold: 0.5 });

        observer.observe(document.getElementById('mainChart'));
        observer.observe(document.getElementById('trainChart'));
        observer.observe(document.getElementById('testChart'));
        observer.observe(document.getElementById('residualChart'));
        }

        function updateUIFromState(skipScroll) {
            var key = targetSelect.value;
            var state = chartStates[key];

            editTitle.value = state.title;
            editX.value = state.xTitle;
            editY.value = state.yTitle;

            if (key === 'main') {
                styleControl.style.display = 'flex';
                editStyle.value = state.style;
                sizeControl.style.display = (state.style === 'scatter' || state.style === 'line') ? 'flex' : 'none';
            } else {
                styleControl.style.display = 'none';
                sizeControl.style.display = 'flex';
            }

            sizeSlider.value = state.symbolSize;
            sizeLabel.innerText = "Size: " + state.symbolSize;

            updateContextBoxV2(key);

            if (!skipScroll) {
                isManualScroll = true;
                clearTimeout(scrollTimeout);
                var targetId = key === 'main' ? 'mainChart' : (key + 'Chart');
                document.getElementById(targetId).scrollIntoView({ behavior: 'smooth', block: 'center' });
                scrollTimeout = setTimeout(function(){ isManualScroll = false; }, 800);
            }
        }

        function updateAll() {
            var data = allData[dsSelector.value];
            if (!data) return;

            renderMainChart(data, dsSelector.value);
            renderFittingChart('train', data.trainData);
            renderFittingChart('test', data.testData);
            renderResidualChart(data);
            updateContextBoxV2(targetSelect.value);
            runCalculationDiagnostics(data);
            renderImputationDetails(data);
            renderRiskFlags(data);
            selectedPointBox.innerHTML = 'Click an interpolated point to inspect it.';
        }

        targetSelect.addEventListener('change', function() { updateUIFromState(false); });
        dsSelector.addEventListener('change', updateAll);
        exportDetailsBtn.addEventListener('click', exportDetailsCsv);

        [editTitle, editX, editY, editStyle, sizeSlider].forEach(function(el) {
            el.addEventListener('input', function() {
                var key = targetSelect.value;
                chartStates[key].title = editTitle.value;
                chartStates[key].xTitle = editX.value;
                chartStates[key].yTitle = editY.value;
                if (key === 'main') chartStates[key].style = editStyle.value;
                chartStates[key].symbolSize = parseInt(sizeSlider.value);

                sizeLabel.innerText = "Size: " + sizeSlider.value;
                if (key === 'main') sizeControl.style.display = (editStyle.value === 'scatter' || editStyle.value === 'line') ? 'flex' : 'none';
                updateAll();
            });
        });

        if (Object.keys(allData).length > 0) { 
            chartStates.main.title = dsSelector.value + " - Imputation Result";
            updateUIFromState(true);
            updateAll(); 
        }

        window.addEventListener('resize', function() {
            Object.keys(charts).forEach(function(k) { charts[k].resize(); });
        });
    </script>
</body>
</html>
'''
    return html_content


