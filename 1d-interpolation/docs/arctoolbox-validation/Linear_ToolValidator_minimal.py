"""Paste this class into the Linear Script Tool Validation tab."""

import csv
import os

import arcpy


FIELD_INDEXES = [1, 2]
TIME_FREQUENCY_INDEX = 3
DUPLICATE_INDEX = 6
OUTPUT_HTML_INDEX = 7
OUTPUT_FOLDER_INDEX = 8
VALIDATION_STRATEGY_INDEX = 11
CLEAR_FILTER_INDEXES = [4, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
METHOD_VALUE_LISTS = [

]
FREQUENCY_VALUES = ["Auto", "Daily", "Monthly", "Yearly", "Hourly", "Custom"]
DUPLICATE_VALUES = ["error", "mean", "first", "last"]
VALIDATION_VALUES = ["time_split", "rolling"]


class ToolValidator(object):
    def __init__(self):
        self.params = arcpy.GetParameterInfo()

    def initializeParameters(self):
        for index in CLEAR_FILTER_INDEXES:
            self._set_filter(index, [])
        self._set_filter(TIME_FREQUENCY_INDEX, FREQUENCY_VALUES)
        self._set_filter(DUPLICATE_INDEX, DUPLICATE_VALUES)
        self._set_filter(VALIDATION_STRATEGY_INDEX, VALIDATION_VALUES)
        for index, values in METHOD_VALUE_LISTS:
            self._set_filter(index, values)
        return

    def _set_filter(self, index, values):
        try:
            self.params[index].filter.list = values
        except Exception:
            pass

    def _encoding(self, path):
        with open(path, "rb") as stream:
            prefix = stream.read(3)
        return "utf-8-sig" if prefix == "\xef\xbb\xbf" else "utf-8"

    def _utf8_lines(self, path):
        encoding = self._encoding(path)
        with open(path, "rb") as stream:
            for line in stream:
                try:
                    yield line.decode(encoding).encode("utf-8")
                except Exception:
                    yield line

    def _read_header(self, path):
        return next(csv.reader(self._utf8_lines(path)))

    def _path_exists(self, path):
        if not path:
            return False
        if os.path.isfile(path):
            return True
        try:
            return os.path.isfile(path.encode("mbcs"))
        except Exception:
            pass
        try:
            return arcpy.Exists(path)
        except Exception:
            return False

    def _first_path(self, value):
        if not value:
            return ""
        for item in value.split(";"):
            path = item.strip().strip("'\"")
            if path and self._path_exists(path):
                return path
        return value.strip().strip("'\"")

    def _open_path(self, path):
        if os.path.isfile(path):
            return path
        try:
            encoded = path.encode("mbcs")
            if os.path.isfile(encoded):
                return encoded
        except Exception:
            pass
        return path

    def updateParameters(self):
        path = self._first_path(self.params[0].valueAsText)
        if path and self._path_exists(path):
            try:
                fields = self._read_header(self._open_path(path))
                if fields:
                    for index in FIELD_INDEXES:
                        self.params[index].filter.list = fields
                    if not self.params[1].altered:
                        self.params[1].value = fields[0]
                    if len(fields) > 1 and not self.params[2].altered:
                        self.params[2].value = fields[1]
            except Exception:
                pass
            self._fill_output_paths(path, self.params[0].valueAsText)
        return

    def _fill_output_paths(self, path, path_text):
        folder = os.path.dirname(path)
        if not folder:
            return
        if path_text and ";" in path_text:
            html_name = "Multiple_Interpolation_Result.html"
        else:
            base = os.path.splitext(os.path.basename(path))[0]
            html_name = base + "_Interpolation_Report.html"
        html_path = os.path.join(folder, html_name)
        try:
            if not self.params[OUTPUT_HTML_INDEX].altered or not self.params[OUTPUT_HTML_INDEX].valueAsText:
                self.params[OUTPUT_HTML_INDEX].value = html_path
        except Exception:
            pass
        try:
            if not self.params[OUTPUT_FOLDER_INDEX].altered or not self.params[OUTPUT_FOLDER_INDEX].valueAsText:
                self.params[OUTPUT_FOLDER_INDEX].value = folder
        except Exception:
            pass

    def updateMessages(self):
        return
