#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="${HOME}/g1_ws/src/g1_social_common"
RESOURCE_DIR="${PACKAGE_ROOT}/resource"
PYTHON_PACKAGE_DIR="${PACKAGE_ROOT}/g1_social_common"

RESOURCE_FILE="${RESOURCE_DIR}/g1_social_common"
INIT_FILE="${PYTHON_PACKAGE_DIR}/__init__.py"

echo "[INFO] Ensuring g1_social_common package structure exists..."

mkdir -p "${RESOURCE_DIR}"
mkdir -p "${PYTHON_PACKAGE_DIR}"

if [[ ! -e "${RESOURCE_FILE}" ]]; then
    touch "${RESOURCE_FILE}"
    echo "[CREATE] ${RESOURCE_FILE}"
else
    echo "[KEEP]   ${RESOURCE_FILE}"
fi

if [[ ! -e "${INIT_FILE}" ]]; then
    touch "${INIT_FILE}"
    echo "[CREATE] ${INIT_FILE}"
else
    echo "[KEEP]   ${INIT_FILE}"
fi

echo "[DONE] g1_social_common package structure is ready."