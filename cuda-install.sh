#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_URL="https://tw.download.nvidia.com/Windows/595.79/595.79-desktop-win10-win11-64bit-international-nsd-dch-whql.exe"
DRIVER_EXE="${SCRIPT_DIR}/nvidia-driver-595.79.exe"

if [[ "${OSTYPE:-}" != msys* && "${OSTYPE:-}" != cygwin* ]]; then
  echo "This script is intended for Git Bash / MSYS2 / Cygwin on Windows."
  exit 1
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is already available:"
  nvcc --version
  echo
  echo "If Cargo still cannot find CUDA, run:"
  echo "  source \"${SCRIPT_DIR}/setenv.cuda\""
  exit 0
fi

run_in_powershell() {
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$1"
}

download_driver() {
  echo "Downloading NVIDIA driver from:"
  echo "  ${DRIVER_URL}"
  run_in_powershell "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '${DRIVER_URL}' -OutFile '${DRIVER_EXE}'"
}

install_driver() {
  echo "Starting NVIDIA driver installer..."
  run_in_powershell "Start-Process -FilePath '${DRIVER_EXE}'"
}

if [[ ! -f "${DRIVER_EXE}" ]]; then
  download_driver
else
  echo "Driver installer already exists:"
  echo "  ${DRIVER_EXE}"
fi

install_driver

echo
echo "Installer launched."
echo "Finish the NVIDIA driver upgrade in the GUI, reboot if requested, then run:"
echo "  source \"${SCRIPT_DIR}/setenv.cuda\""
echo "  nvidia-smi"
echo "  nvcc --version"
echo
echo "After that, try:"
echo "  cargo run --features cuda -- --prompt \"用中文介紹 Rust Candle 是什麼\""
