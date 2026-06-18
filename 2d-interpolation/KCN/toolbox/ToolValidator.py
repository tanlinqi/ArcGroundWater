from __future__ import print_function

import csv

import arcpy


FIELD_INDICES = [1, 2, 3, 4, 5]
NUMERIC_LIMITS = {
    10: (0.0, None),
    13: (1.0, None),
    15: (0.0, 1.0),
    16: (1.0, None),
    17: (0.0, None),
    18: (0.0, None),
    19: (1.0, None),
    20: (0.0, 1.0),
    21: (1.0, None),
    24: (0.0, None),
    25: (1.0, None),
    26: (1.0, None),
}
EXTENT_KEYWORDS = set(["", "#", "MAXOF", "MINOF", "DISPLAY", "DEFAULT"])


class ToolValidator(object):
    def __init__(self):
        self.params = arcpy.GetParameterInfo()

    def initializeParameters(self):
        self.params[12].filter.list = ["kcn", "kcn_gat", "kcn_sage"]
        self.params[23].filter.list = ["AUTO", "CUDA", "CPU"]
        if not self.params[8].value:
            self.params[8].value = arcpy.SpatialReference(4326)
        return

    def _read_header(self, path):
        with open(path, "rb") as handle:
            reader = csv.reader(handle)
            return next(reader)

    def _read_times(self, path, field):
        values = []
        seen = set()
        with open(path, "rb") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = row.get(field, "").strip()
                if value and value not in seen:
                    seen.add(value)
                    values.append(value)
        return sorted(values)

    def updateParameters(self):
        csv_path = self.params[0].valueAsText
        if not csv_path:
            return
        try:
            fields = self._read_header(csv_path)
            for index in FIELD_INDICES:
                self.params[index].filter.list = fields
            time_field = self.params[1].valueAsText
            if time_field in fields:
                times = self._read_times(csv_path, time_field)
                self.params[6].filter.list = times
                self.params[7].filter.list = times
                if times and not self.params[6].altered:
                    self.params[6].value = times[0]
                if times and not self.params[7].altered:
                    self.params[7].value = times[-1]
        except Exception:
            pass
        return

    def updateMessages(self):
        start_time = self.params[6].valueAsText
        end_time = self.params[7].valueAsText
        start_values = list(self.params[6].filter.list or [])
        if start_time and end_time and start_values:
            try:
                if start_values.index(end_time) < start_values.index(start_time):
                    self.params[7].setErrorMessage(
                        "End time must not be earlier than start time."
                    )
            except ValueError:
                pass

        for index, limits in NUMERIC_LIMITS.items():
            value_text = self.params[index].valueAsText
            if value_text in (None, ""):
                continue
            try:
                value = float(value_text)
                minimum, maximum = limits
                if minimum is not None and value <= minimum and index in (10, 17):
                    self.params[index].setErrorMessage("Value must be greater than zero.")
                elif minimum is not None and value < minimum:
                    self.params[index].setErrorMessage("Value is below the allowed minimum.")
                elif maximum is not None and value >= maximum:
                    self.params[index].setErrorMessage("Value is above the allowed maximum.")
            except ValueError:
                self.params[index].setErrorMessage("A numeric value is required.")

        extent = (self.params[9].valueAsText or "").strip()
        if extent.upper() not in EXTENT_KEYWORDS and extent:
            parts = extent.replace(",", " ").replace(";", " ").split()
            if len(parts) != 4:
                self.params[9].setWarningMessage(
                    "Extent should contain xmin ymin xmax ymax."
                )
        return
