#!/usr/bin/env bash

set -uo pipefail

PROJECT_NAME="qwen3-4b-runner"
BIN_NAME="candle01"
FEATURES="${FEATURES:-}"
BUILD_MODE="${BUILD_MODE:-release}"
DIST_DIR="${DIST_DIR:-dist}"

TARGETS=(
  "x86_64-pc-windows-msvc:windows-x64:.exe"
  "x86_64-unknown-linux-gnu:linux-x64:"
  "aarch64-apple-darwin:macos-arm64:"
)

mkdir -p "${DIST_DIR}"
mkdir -p "${DIST_DIR}/logs"

build_target() {
  local rust_target="$1"
  local label="$2"
  local ext="$3"

  echo "==> building ${label} (${rust_target})"
  local log_file="${DIST_DIR}/logs/${label}.log"

  local cmd=(cargo build --target "${rust_target}")
  if [[ "${BUILD_MODE}" == "release" ]]; then
    cmd+=(--release)
  fi
  if [[ -n "${FEATURES}" ]]; then
    cmd+=(--features "${FEATURES}")
  fi

  if ! "${cmd[@]}" >"${log_file}" 2>&1; then
    echo "FAILED ${label} (${rust_target})"
    echo "  log: ${log_file}"
    return 1
  fi

  local profile_dir="debug"
  if [[ "${BUILD_MODE}" == "release" ]]; then
    profile_dir="release"
  fi

  local src="target/${rust_target}/${profile_dir}/${BIN_NAME}${ext}"
  local dst="${DIST_DIR}/${PROJECT_NAME}-${label}${ext}"

  if [[ ! -f "${src}" ]]; then
    echo "FAILED ${label} (${rust_target}) - binary not found at ${src}"
    return 1
  fi

  cp "${src}" "${dst}"
  echo "OK ${label} -> ${dst}"
  echo "  log: ${log_file}"
  return 0
}

successes=()
failures=()

for entry in "${TARGETS[@]}"; do
  IFS=":" read -r rust_target label ext <<< "${entry}"
  if build_target "${rust_target}" "${label}" "${ext}"; then
    successes+=("${label}")
  else
    failures+=("${label}")
  fi
done

echo
echo "Build summary"
if ((${#successes[@]} > 0)); then
  echo "  succeeded: ${successes[*]}"
fi
if ((${#failures[@]} > 0)); then
  echo "  failed: ${failures[*]}"
  exit 1
fi

echo "  all targets built successfully"
