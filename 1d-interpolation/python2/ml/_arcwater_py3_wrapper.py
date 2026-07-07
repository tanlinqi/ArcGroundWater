# -*- coding: utf-8 -*-
"""ArcMap Python 2 wrapper that delegates ML imputation to Python 3."""

from __future__ import print_function

import datetime
import json
import os
import subprocess
import sys
import traceback

import arcpy


DEFAULT_PY3 = r"C:\Users\Lenovo\Miniconda3\envs\ssin\python.exe"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BACKEND = os.path.join(PROJECT_ROOT, "python3", "ml", "gwl_fill_runner.py")
LOG_ROOT = os.path.join(PROJECT_ROOT, "log")
LOG_PATH = None


def ensure_text(value):
    try:
        if isinstance(value, unicode):
            return value
    except NameError:
        pass
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            return value.decode(encoding)
        except Exception:
            pass
    try:
        return unicode(value)
    except Exception:
        return str(value)


def get_parameter(index):
    try:
        value = arcpy.GetParameterAsText(index)
        return ensure_text(value).strip() if value else u""
    except Exception:
        return u""


def filesystem_path(path):
    try:
        open(path, "ab").close()
        return path
    except UnicodeEncodeError:
        return ensure_text(path).encode("mbcs")
    except IOError:
        return path


def windows_arg(value):
    text = ensure_text(value)
    try:
        return text.encode("mbcs")
    except Exception:
        try:
            return text.encode("utf-8")
        except Exception:
            return str(value)


def ensure_dir(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except UnicodeEncodeError:
            os.makedirs(ensure_text(path).encode("mbcs"))


def method_log_dir(method):
    method_name = ensure_text(method).lower().strip() or u"unknown"
    path = os.path.join(ensure_text(LOG_ROOT), method_name)
    ensure_dir(path)
    return path


def create_log_file(method):
    log_dir = method_log_dir(method)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, "{0}_{1}.log".format(ensure_text(method).lower(), stamp))
    open_path = filesystem_path(path)
    with open(open_path, "wb") as stream:
        stream.write((u"ArcWater ML ArcMap run log\n").encode("utf-8"))
        stream.write((u"Started: {0}\n".format(datetime.datetime.now())).encode("utf-8"))
        stream.write((u"Method: {0}\n".format(ensure_text(method))).encode("utf-8"))
        stream.write((u"Backend: {0}\n\n".format(ensure_text(BACKEND))).encode("utf-8"))
    return path


def log_line(message):
    if not LOG_PATH:
        return
    try:
        with open(filesystem_path(LOG_PATH), "ab") as stream:
            stream.write((ensure_text(message) + u"\n").encode("utf-8"))
    except Exception:
        pass


def message(text):
    arcpy.AddMessage(text)
    log_line(text)


def fail(text):
    arcpy.AddError(text)
    log_line("ERROR: " + ensure_text(text))
    raise RuntimeError(text)


def write_params_file(method, params):
    log_dir = method_log_dir(method)
    path = os.path.join(log_dir, "{0}_params.json".format(ensure_text(method).lower()))
    payload = {
        "method": method,
        "params": params,
        "project_root": ensure_text(PROJECT_ROOT),
        "script_dir": ensure_text(SCRIPT_DIR),
    }
    open_path = filesystem_path(path)
    with open(open_path, "wb") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    return path


def looks_like_python_executable(value):
    text = ensure_text(value).strip().strip("'\"")
    if not text:
        return False
    lower = text.lower().replace("/", "\\")
    return lower.endswith("python.exe") or lower.endswith("pythonw.exe")


def find_python_executable(params):
    env_value = os.environ.get("ARCWATER_PY3", "")
    if env_value:
        return env_value
    for value in reversed(params):
        if looks_like_python_executable(value):
            return ensure_text(value).strip().strip("'\"")
    legacy = get_parameter(29)
    if legacy:
        return legacy
    return DEFAULT_PY3


def run_process(command):
    message(u"Run log: {0}".format(ensure_text(LOG_PATH)))
    message(u"Starting Python 3 environment: {0}".format(ensure_text(command[0])))
    log_line("Backend command:")
    log_line(u"  " + u" ".join([u'"{0}"'.format(ensure_text(item)) if u" " in ensure_text(item) else ensure_text(item) for item in command]))
    popen_command = [windows_arg(item) for item in command]
    process = subprocess.Popen(
        popen_command,
        cwd=filesystem_path(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    for line in iter(process.stdout.readline, ""):
        line = line.rstrip()
        if not line:
            continue
        try:
            decoded = line.decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            decoded = line
        arcpy.AddMessage(decoded)
        log_line(decoded)
    process.stdout.close()
    code = process.wait()
    log_line("Backend exit code: {0}".format(code))
    if code:
        fail(u"Python 3 backend failed with exit code: {0}".format(code))


def run(method):
    global LOG_PATH
    LOG_PATH = create_log_file(method)
    params = [get_parameter(index) for index in range(0, 40)]
    py3 = find_python_executable(params)
    params_file = write_params_file(method, params)

    if not os.path.isfile(py3):
        fail(u"Python 3 executable was not found: {0}".format(ensure_text(py3)))
    if not os.path.isfile(BACKEND):
        fail(u"Python 3 backend script was not found: {0}".format(ensure_text(BACKEND)))

    command = [
        py3, "-u", BACKEND,
        "--params-file", params_file,
        "--run-log", LOG_PATH,
    ]
    run_process(command)
    log_line("Finished: {0}".format(datetime.datetime.now()))


def main(method):
    try:
        run(method)
    except Exception:
        error_text = traceback.format_exc()
        arcpy.AddError(error_text)
        log_line(error_text)
        raise
