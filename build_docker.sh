#!/usr/bin/env bash
# Build the Rocky Linux app on any computer with Docker + internet (Mac, Windows/WSL, Linux).
# Result: dist/PSDPrep-linux-x86_64.tar.gz
set -euo pipefail
cd "$(dirname "$0")"
docker run --rm --platform linux/amd64 -v "$PWD":/src -w /src rockylinux:8 bash ci/build_linux.sh
