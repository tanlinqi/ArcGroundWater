# -*- coding: utf-8 -*-
import csv
import os

import arcpy


class ToolValidator(object):
    def __init__(self):
        self.params = arcpy.GetParameterInfo()

    def initializeParameters(self):
        self.params[20].filter.list = ["AUTO", "CUDA", "CPU"]
        if not self.params[8].value:
            self.params[8].value = arcpy.SpatialReference(4326)
        defaults = {
            10: "0.05",
            12: "20",
            13: "8",
            14: "0.001",
            15: "64",
            16: "1",
            17: "2",
            18: "8",
            19: "8",
            20: "AUTO",
            21: "0",
            23: False,
        }
        for index, value in defaults.items():
            if not self.params[index].value:
                self.params[index].value = value
        return

    def updateParameters(self):
        csv_path = self._text(0)
        if csv_path and os.path.isfile(csv_path):
            fields = self._read_header(csv_path)
            for index in [1, 2, 3, 4, 5]:
                self.params[index].filter.list = fields
            self._set_default_field(fields, 1, ["time", "timestamp", "date"])
            self._set_default_field(fields, 2, ["station", "site", "id"])
            self._set_default_field(fields, 3, ["x", "lon", "longitude"])
            self._set_default_field(fields, 4, ["y", "lat", "latitude"])
            self._set_default_field(fields, 5, ["value", "pm25", "pm2_5", "target"])
            time_field = self._text(1)
            if time_field:
                times = self._read_times(csv_path, time_field)
                self.params[6].filter.list = times
                self.params[7].filter.list = times
                if times and not self.params[6].value:
                    self.params[6].value = times[0]
                if times and not self.params[7].value:
                    self.params[7].value = times[-1]
        return

    def updateMessages(self):
        self._check_positive_float(10, "Cell size")
        self._check_positive_int(12, "Epochs")
        self._check_positive_int(13, "Batch size")
        self._check_positive_float(14, "Learning rate")
        self._check_positive_int(15, "Channels")
        self._check_positive_int(16, "Layers")
        self._check_positive_int(17, "Temporal kernel")
        self._check_positive_int(18, "Neighbor count")
        self._check_positive_int(19, "Masked nodes")
        start_time = self._text(6)
        end_time = self._text(7)
        times = self.params[6].filter.list
        if start_time and end_time and start_time in times and end_time in times:
            if times.index(end_time) < times.index(start_time):
                self.params[7].setErrorMessage("End time must not be earlier than start time.")
        return

    def _text(self, index):
        value = self.params[index].valueAsText
        return value if value else ""

    def _read_header(self, path):
        try:
            with open(path, "rb") as fobj:
                reader = csv.reader(fobj)
                row = next(reader)
                return [item.strip() for item in row]
        except Exception:
            return []

    def _read_times(self, path, field):
        values = []
        seen = set()
        try:
            with open(path, "rb") as fobj:
                reader = csv.DictReader(fobj)
                for row in reader:
                    value = row.get(field, "")
                    if value not in seen:
                        values.append(value)
                        seen.add(value)
        except Exception:
            return []
        return sorted(values)

    def _set_default_field(self, fields, index, names):
        if self.params[index].value:
            return
        lower = dict((field.lower(), field) for field in fields)
        for name in names:
            if name in lower:
                self.params[index].value = lower[name]
                return
        if fields:
            self.params[index].value = fields[0]

    def _check_positive_float(self, index, label):
        value = self._text(index)
        try:
            if float(value) <= 0:
                self.params[index].setErrorMessage(label + " must be greater than zero.")
        except Exception:
            self.params[index].setErrorMessage(label + " must be numeric.")

    def _check_positive_int(self, index, label):
        value = self._text(index)
        try:
            if int(value) <= 0:
                self.params[index].setErrorMessage(label + " must be greater than zero.")
        except Exception:
            self.params[index].setErrorMessage(label + " must be an integer.")
