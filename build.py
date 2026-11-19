"""Build a self-contained PSD Prep: no Python or installs needed on the
machines that run it. Run on the OS you ship (no cross-builds):

    python3 -m venv build-venv
    build-venv/bin/pip install pyinstaller pyside6-essentials pillow numpy
    build-venv/bin/python build.py

Linux: build on the OLDEST distro you support (Rocky 8 -> runs on Rocky 8 and 9).
No Linux box with internet? Use GitHub Actions or ./build_docker.sh (see README).
Output: dist/PSDPrep/  (tar/zip that folder and copy it anywhere)
"""
import datetime as _dt
import os
import shutil
import stat
import subprocess as _sp
import sys
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent
NAME = "PSDPrep"
SEP = os.pathsep
WIN, MAC = sys.platform.startswith("win"), sys.platform == "darwin"

# --- build stamp (date + commit) shown in the window title --------------------
sha = os.environ.get("GITHUB_SHA", "")[:7]
if not sha:
    try:
        sha = _sp.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                      text=True).stdout.strip()
    except OSError:
        sha = ""
stamp = _dt.date.today().isoformat() + (f" · {sha}" if sha else "")
(ROOT / "psdprep" / "_build_stamp.py").write_text(f'BUILD = "{stamp}"\n', encoding="utf-8")
print(f"[build] stamp: {stamp}")

args = [
    str(ROOT / "app_main.py"),
    "--name", NAME,
    "--onedir",              # starts fast; onefile unpacks Qt to /tmp on every launch
    "--noconfirm", "--clean",
    "--hidden-import", "psdprep.gui",
    "--hidden-import", "psdprep.selftest",
    "--hidden-import", "psdprep.fixtures",
    "--hidden-import", "psdprep._build_stamp",
    "--hidden-import", "PIL.ImageCms",
    *sum((["--exclude-module", m] for m in (
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick",
        "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtMultimedia",
        "PySide6.QtDataVisualization", "PySide6.QtPdf", "tkinter")), []),
]
if WIN or MAC:
    args.append("--windowed")
if not WIN and not MAC:
    # Qt 6.5+ needs libxcb-cursor, which many Rocky installs lack: ship it inside the app
    for lib in ("/usr/lib64/libxcb-cursor.so.0", "/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0"):
        if Path(lib).is_file():
            args += ["--add-binary", f"{lib}{SEP}."]
            break
    else:
        print("[build] note: libxcb-cursor not found on build machine; targets may need xcb-util-cursor")

PyInstaller.__main__.run(args)

# --- extras next to the executable -------------------------------------------
dist = ROOT / "dist" / NAME
prof = dist / "profiles"
prof.mkdir(exist_ok=True)
for p in (ROOT / "profiles").glob("*"):
    if p.suffix.lower() in (".icc", ".icm", ".txt"):
        shutil.copy2(p, prof / p.name)
shutil.copy2(ROOT / "README.md", dist / "README.md")

if not WIN and not MAC:
    # Double-clickable launcher for file managers (GNOME: right-click > Allow Launching once)
    desktop = dist / "PSD Prep.desktop"
    desktop.write_text(
        "[Desktop Entry]\nType=Application\nName=PSD Prep\n"
        "Comment=Remove layers, crop and convert PSD files\n"
        + r'Exec=sh -c "cd \\"\\$(dirname \\"%k\\")\\" && exec ./' + NAME + '"' + "\n"
        "Icon=image-x-generic\nTerminal=false\nCategories=Graphics;\n")
    desktop.chmod(desktop.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    sh = dist / "Run PSD Prep.sh"
    sh.write_text('#!/bin/sh\ncd "$(dirname "$0")" && exec ./' + NAME + ' "$@"\n')
    sh.chmod(0o755)

print(f"\n[build] done -> {dist}")
