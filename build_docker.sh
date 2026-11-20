#!/usr/bin/env bash
# Build the Rocky Linux app on any computer with Docker + internet (Mac, Windows/WSL, Linux).
# Result: dist/PSDPrep-linux-x86_64.tar.gz
set -euo pipefail
cd "$(dirname "$0")"
docker run --rm --platform linux/amd64 -v "$PWD":/src -w /src \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" rockylinux:8 bash -c '
    set -e
    dnf install -y git >/dev/null
    git config --global --add safe.directory /src      # keeps the commit in the build stamp
    bash ci/build_linux.sh
    chown -R "$HOST_UID:$HOST_GID" dist build psdprep/_build_stamp.py 2>/dev/null || true'
