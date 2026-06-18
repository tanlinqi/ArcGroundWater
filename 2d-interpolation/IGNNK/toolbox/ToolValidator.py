from __future__ import print_function

import csv
import os

import arcpy


class ToolValidator(object):
    def __init__(self):
        self.params = arcpy.GetParameterInfo()

    def initializeParameters(self):
        self.params[24].filter.list = ["AUTO", "CUDA", "CPU"]
        if not self.params[8].value:
            self.params[8].value = arcpy.SpatialReference(4326)
        return

    def _read_header(self, path):
        if not path or not os.path.isfile(path):
            return []
        try:
            with open(path, "rb") as handle:
                reader = csv.reader(handle)
                return next(reader)
        except Exception:
            return []

    def _read_times(self, path, field):
        if not path or not field or not os.path.isfile(path):
            return []
        values = set()
        try:
            with open(path, "rb") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    value = row.get(field, "").strip()
                    if value:
                        values.add(value)
        except Exception:
            return []
        return sorted(values)

    def updateParameters(self):
        csv_path = self.params[0].valueAsText
        fields = self._read_header(csv_path)
        for index in range(1, 6):
            self.params[index].filter.list = fields
        time_field = self.params[1].valueAsText
        times = self._read_times(csv_path, time_field)
        self.params[6].filter.list = times
        self.params[7].filter.list = times
        if times:
            if not self.params[6].altered:
                self.params[6].value = times[0]
            if not self.params[7].altered:
                self.params[7].value = times[-1]
        return

    def updateMessages(self):
        start_time = self.params[6].valueAsText
        end_time = self.params[7].valueAsText
        times = self.params[6].filter.list
        if start_time in times and end_time in times:
            if times.index(end_time) < times.index(start_time):
                self.params[7].setErrorMessage(
                    "End time cannot be earlier than start time."
                )
        positive = [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 26]
        for index in positive:
            value = self.params[index].value
            if value is not None and float(value) <= 0:
                self.params[index].setErrorMessage(
                    "This value must be greater than zero."
                )
        threshold = self.params[21].value
        if threshold is not None and not 0 <= float(threshold) < 1:
            self.params[21].setErrorMessage(
                "Adjacency threshold must be in [0, 1)."
            )
        extent = (self.params[9].valueAsText or "").strip().upper()
        if extent in ("MAXOF", "MINOF", "DISPLAY", "DEFAULT", "#"):
            self.params[9].setWarningMessage(
                "Station coordinate extent will be used."
            )
        return
