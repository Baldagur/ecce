#!/usr/bin/env bash
# Builds the Linux app inside a Rocky Linux 8 container (oldest glibc we target,
# so the result runs on Rocky 8 and 9). Used by GitHub Actions and build_docker.sh.
set -euxo pipefail
cd "$(dirname "$0")/.."

dnf install -y epel-release
dnf install -y python3.11 python3.11-pip tar findutils git ghostscript \
  mesa-libGL mesa-libEGL libxkbcommon libxkbcommon-x11 fontconfig freetype dbus-libs \
  libX11 libXext libXrender libxcb xcb-util-wm xcb-util-image xcb-util-keysyms \
  xcb-util-renderutil xcb-util-cursor

python3.11 -m venv /tmp/venv
/tmp/venv/bin/pip install --upgrade pip
/tmp/venv/bin/pip install pyinstaller pyside6-essentials pillow numpy pypdfium2

# Bundle Ghostscript's generic CMYK profile as a fallback for untagged CMYK files
# (files with an embedded profile always use their own).
gs_cmyk="$(find /usr/share/ghostscript -name default_cmyk.icc 2>/dev/null | head -1 || true)"
if [ -n "$gs_cmyk" ] && [ ! -f profiles/default_cmyk.icc ]; then
  cp "$gs_cmyk" profiles/default_cmyk.icc
fi

/tmp/venv/bin/python tests/test_roundtrip.py
/tmp/venv/bin/python tests/test_review_fixes.py
/tmp/venv/bin/python build.py
cd dist && tar -czf PSDPrep-linux-x86_64.tar.gz PSDPrep
echo "Built dist/PSDPrep-linux-x86_64.tar.gz"
