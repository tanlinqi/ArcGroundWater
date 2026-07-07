"""Paste this class into the Ridge Regression Script Tool Validation tab."""

import csv
import os

import arcpy


INPUT_INDEX = 0
TIME_FIELD_INDEX = 1
VALUE_FIELD_INDEX = 2
DUPLICATE_INDEX = 5
OUTPUT_HTML_INDEX = 6
OUTPUT_FOLDER_INDEX = 7
FIELD_INDEXES = [TIME_FIELD_INDEX, VALUE_FIELD_INDEX]
DUPLICATE_VALUES = ["error", "mean", "first", "last"]
ENUM_FILTERS = {18: ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']}
DEFAULT_VALUES = {3: '-99999', 4: 'true', 5: 'error', 8: 'true', 9: '3', 10: '8:2', 11: '3', 12: '0', 14: 'false', 15: 'false', 16: '1', 17: 'true', 18: 'auto', 19: '0.001', 20: 'false'}
CLEAR_FILTER_INDEXES = [3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21]


class ToolValidator(object):
    def __init__(self):
        self.params = arcpy.GetParameterInfo()
        self._last_auto_html = ""
        self._last_auto_folder = ""

    def initializeParameters(self):
        for index in CLEAR_FILTER_INDEXES:
            self._set_filter(index, [])
        self._set_filter(DUPLICATE_INDEX, DUPLICATE_VALUES)
        for index, values in ENUM_FILTERS.items():
            self._set_filter(index, values)
        for index, value in DEFAULT_VALUES.items():
            self._set_default(index, value)
        return

    def _set_filter(self, index, values):
        try:
            self.params[index].filter.list = values
        except Exception:
            pass

    def _set_default(self, index, value):
        try:
            if not self.params[index].altered and not self.params[index].valueAsText:
                self.params[index].value = value
        except Exception:
            pass

    def _encoding(self, path):
        try:
            with open(path, "rb") as stream:
                prefix = stream.read(3)
            if prefix == b"\xef\xbb\xbf":
                return "utf-8-sig"
        except Exception:
            pass
        return "utf-8"

    def _decode_bytes(self, value, preferred):
        for encoding in (preferred, "utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return value.decode(encoding)
            except Exception:
                pass
        return value.decode("utf-8", "ignore")

    def _to_unicode(self, value):
        try:
            unicode_type = unicode
        except NameError:
            unicode_type = str
        if isinstance(value, unicode_type):
            return value
        return self._decode_bytes(value, "utf-8")

    def _utf8_lines(self, path):
        encoding = self._encoding(path)
        with open(path, "rb") as stream:
            for line in stream:
                yield self._decode_bytes(line, encoding).encode("utf-8")

    def _read_header(self, path):
        fields = next(csv.reader(self._utf8_lines(path)))
        return [self._to_unicode(item).strip() for item in fields]

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
            path = item.strip().strip(chr(34)).strip(chr(39))
            if path and self._path_exists(path):
                return path
        return value.strip().strip(chr(34)).strip(chr(39))

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
        path_text = self.params[INPUT_INDEX].valueAsText
        path = self._first_path(path_text)
        if path and self._path_exists(path):
            self._update_field_lists(path)
            self._fill_output_paths(path, path_text)
        return

    def _update_field_lists(self, path):
        try:
            fields = self._read_header(self._open_path(path))
            fields = [item for item in fields if item]
            if not fields:
                return
            for index in FIELD_INDEXES:
                self.params[index].filter.list = fields
            current_time = self.params[TIME_FIELD_INDEX].valueAsText
            current_value = self.params[VALUE_FIELD_INDEX].valueAsText
            if (not self.params[TIME_FIELD_INDEX].altered) or current_time not in fields:
                self.params[TIME_FIELD_INDEX].value = fields[0]
            if len(fields) > 1 and ((not self.params[VALUE_FIELD_INDEX].altered) or current_value not in fields):
                self.params[VALUE_FIELD_INDEX].value = fields[1]
        except Exception:
            pass

    def _should_auto_replace(self, index, old_auto_value):
        try:
            current = self.params[index].valueAsText
            return (not self.params[index].altered) or (not current) or (old_auto_value and current == old_auto_value)
        except Exception:
            return False

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
            if self._should_auto_replace(OUTPUT_HTML_INDEX, self._last_auto_html):
                self.params[OUTPUT_HTML_INDEX].value = html_path
                self._last_auto_html = html_path
        except Exception:
            pass
        try:
            if self._should_auto_replace(OUTPUT_FOLDER_INDEX, self._last_auto_folder):
                self.params[OUTPUT_FOLDER_INDEX].value = folder
                self._last_auto_folder = folder
        except Exception:
            pass

    def updateMessages(self):
        return
