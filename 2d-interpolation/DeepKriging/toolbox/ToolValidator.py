"""Paste this class into the Script Tool Validation tab."""

import csv
import os

import arcpy


class ToolValidator(object):
    def __init__(self):
        self.params = arcpy.GetParameterInfo()
        self._signature = None

    def _set_filter_list(self, index, values):
        if index >= len(self.params):
            return False
        parameter_filter = self.params[index].filter
        if parameter_filter is None:
            return False
        parameter_filter.list = values
        return True

    def _filter_values(self, index):
        if index >= len(self.params):
            return []
        parameter_filter = self.params[index].filter
        if parameter_filter is None or parameter_filter.list is None:
            return []
        return list(parameter_filter.list)

    def initializeParameters(self):
        if len(self.params) > 8 and not self.params[8].value:
            self.params[8].value = arcpy.SpatialReference(4326)
        self._set_filter_list(19, ["AUTO", "CUDA", "CPU"])
        return

    def _encoding(self, path):
        with open(path, "rb") as stream:
            prefix = stream.read(3)
        return "utf-8-sig" if prefix == "\xef\xbb\xbf" else "utf-8"

    def _utf8_lines(self, path):
        encoding = self._encoding(path)
        with open(path, "rb") as stream:
            for line in stream:
                yield line.decode(encoding).encode("utf-8")

    def _read_header(self, path):
        return next(csv.reader(self._utf8_lines(path)))

    def _read_times(self, path, field):
        values = []
        seen = set()
        reader = csv.DictReader(self._utf8_lines(path))
        for row in reader:
            value = row.get(field, "").strip()
            if value and value not in seen:
                seen.add(value)
                values.append(value)
        return values

    def updateParameters(self):
        path = self.params[0].valueAsText
        time_field = self.params[1].valueAsText
        signature = (path, time_field)
        if signature == self._signature:
            return
        self._signature = signature
        if path and os.path.isfile(path):
            try:
                fields = self._read_header(path)
                for index in range(1, 6):
                    self._set_filter_list(index, fields)
                if time_field in fields:
                    times = self._read_times(path, time_field)
                    self._set_filter_list(6, times)
                    self._set_filter_list(7, times)
                    if times:
                        if not self.params[6].altered:
                            self.params[6].value = times[0]
                        if not self.params[7].altered:
                            self.params[7].value = times[-1]
            except Exception:
                pass
        return

    def updateMessages(self):
        path = self.params[0].valueAsText
        if path and not os.path.isfile(path):
            self.params[0].setErrorMessage("Input CSV file does not exist.")

        start = self.params[6].valueAsText
        end = self.params[7].valueAsText
        values = self._filter_values(6)
        if start and end and start in values and end in values:
            if values.index(start) > values.index(end):
                self.params[7].setErrorMessage(
                    "End time must occur after start time."
                )

        spatial_reference = self.params[8].value
        extent_text = self.params[9].valueAsText
        extent_keyword = extent_text.strip().upper() if extent_text else ""
        if extent_keyword in ("DEFAULT", "DISPLAY", "MAXOF", "MINOF"):
            self.params[9].setWarningMessage(
                "This extent keyword will use the CSV station extent."
            )
        elif spatial_reference and extent_text:
            try:
                extent = [
                    float(item)
                    for item in extent_text.replace(",", " ").split()
                ]
                if (
                    len(extent) == 4
                    and spatial_reference.type == "Geographic"
                    and (
                        extent[0] < -180
                        or extent[2] > 180
                        or extent[1] < -90
                        or extent[3] > 90
                    )
                ):
                    self.params[9].setWarningMessage(
                        "Extent is outside geographic bounds and will be ignored."
                    )
            except Exception:
                pass

        if len(self.params) <= 27:
            self.params[0].setErrorMessage(
                "The tool requires parameters 0 through 27. Check the parameter table."
            )
            return
        if self.params[19].filter is None:
            self.params[19].setErrorMessage(
                "Parameter 19 (device) must use the String data type."
            )

        checks = [
            (10, 0, None, False, "Cell size must be greater than 0."),
            (12, 1, None, True, "Training epochs must be at least 1."),
            (13, 0, None, False, "Learning rate must be greater than 0."),
            (15, 0, None, False, "Support multiplier must be greater than 0."),
            (16, 1, None, True, "Hidden units must be at least 1."),
            (17, 1, None, True, "Hidden layers must be at least 1."),
            (20, 2, None, True, "Minimum valid stations must be at least 2."),
            (21, 1, None, True, "Training batch size must be at least 1."),
            (22, 1, None, True, "Inference batch size must be at least 1."),
        ]
        for index, minimum, maximum, inclusive, message in checks:
            text = self.params[index].valueAsText
            if not text:
                continue
            try:
                value = float(text)
                invalid = value < minimum if inclusive else value <= minimum
                if maximum is not None:
                    invalid = invalid or value > maximum
                if invalid:
                    self.params[index].setErrorMessage(message)
            except ValueError:
                self.params[index].setErrorMessage("Enter a valid number.")

        resolutions = self.params[14].valueAsText
        if resolutions:
            try:
                values = [
                    int(item.strip())
                    for item in resolutions.split(",")
                    if item.strip()
                ]
                if not values or any(value < 2 for value in values):
                    raise ValueError()
            except ValueError:
                self.params[14].setErrorMessage(
                    "Use comma-separated integers of at least 2."
                )
        return
