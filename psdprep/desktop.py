"""Linux menu / desktop shortcut with absolute paths (a .desktop file can't
locate a program relative to itself)."""
from __future__ import annotations

import os
import stat
import subprocess
import sys


def _quote(path: str) -> str:
    """Desktop Entry spec quoting for an Exec argument."""
    for ch in ("\\", '"', "`", "$"):
        path = path.replace(ch, "\\" + ch)
    return f'"{path}"'


def entry_text(exe: str) -> str:
    folder = os.path.dirname(exe)
    return ("[Desktop Entry]\nType=Application\nVersion=1.0\nName=PSD Prep\n"
            "Comment=Remove layers, crop and convert PSD files\n"
            f"Exec={_quote(exe)} %f\nPath={folder}\n"
            "Icon=image-x-generic\nTerminal=false\nCategories=Graphics;\n"
            "MimeType=image/vnd.adobe.photoshop;image/x-photoshop;application/x-photoshop;\n")


def install(exe: str = None, desktop_copy: bool = True, quiet: bool = False) -> list:
    """Write ~/.local/share/applications/psdprep.desktop (Activities menu) and,
    optionally, a trusted copy on ~/Desktop. Returns written paths."""
    exe = os.path.abspath(exe or sys.executable)
    text = entry_text(exe)
    written = []
    apps = os.path.join(os.path.expanduser("~"), ".local", "share", "applications")
    targets = [os.path.join(apps, "psdprep.desktop")]
    desk = os.path.join(os.path.expanduser("~"), "Desktop")
    if desktop_copy and os.path.isdir(desk):
        targets.append(os.path.join(desk, "PSD Prep.desktop"))
    for t in targets:
        try:
            os.makedirs(os.path.dirname(t), exist_ok=True)
            old = open(t).read() if os.path.exists(t) else None
            if old != text:
                with open(t, "w") as f:
                    f.write(text)
            os.chmod(t, os.stat(t).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            if t.startswith(desk):
                # GNOME: mark as trusted so double-click launches it
                subprocess.run(["gio", "set", t, "metadata::trusted", "true"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            written.append(t)
        except OSError as e:
            if not quiet:
                print(f"could not write {t}: {e}")
    if not quiet:
        for t in written:
            print(f"wrote {t}")
    return written


def auto_register():
    """Called at GUI start in the packaged Linux app: keep the menu entry
    pointing at wherever the app folder now lives."""
    if sys.platform.startswith("linux") and getattr(sys, "frozen", False):
        try:
            install(desktop_copy=False, quiet=True)
        except Exception:
            pass
