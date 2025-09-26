#!/bin/bash
set -ex

# Builds a Numba wheel inside the manylinux container
# Usage: build_wheel_linux.sh <python_executable> <use_tbb> <numpy_version> <llvmlite_wheel_path> <wheels_index_url>

PYTHON_EXECUTABLE=$1 # Use the provided executable path
USE_TBB=${2:-"true"}
NUMPY_VERSION=$3 # Adjusted index
LLVMLITE_WHEEL_PATH=${4:-""} # Adjusted index
WHEELS_INDEX_URL=${5:-"https://pypi.anaconda.org/numba/label/dev/simple"} # Adjusted index

# Set Git safe directory (needed for setuptools-scm in Docker)
git config --global --add safe.directory /io

# Install dependencies
$PYTHON_EXECUTABLE -m pip install build numpy==${NUMPY_VERSION} setuptools wheel

# Install TBB if enabled
if [ "$USE_TBB" = "true" ]; then
    $PYTHON_EXECUTABLE -m pip install tbb==2021.6 tbb-devel==2021.6
fi

# Install llvmlite
if [ -n "$LLVMLITE_WHEEL_PATH" ] && [ -d "$LLVMLITE_WHEEL_PATH" ]; then
    $PYTHON_EXECUTABLE -m pip install --no-cache-dir $LLVMLITE_WHEEL_PATH/*.whl
else
    $PYTHON_EXECUTABLE -m pip install --no-cache-dir -i $WHEELS_INDEX_URL llvmlite
fi

# Change to the mounted workspace directory
cd /io

# Show directory contents for debugging
echo "Contents of /io directory:"
ls -la

# Build wheel from the workspace directory
$PYTHON_EXECUTABLE -m build --wheel --no-isolation

# Create output directory if it doesn't exist
mkdir -p /io/wheelhouse

# Copy wheel to the output directory
cp dist/*.whl /io/wheelhouse/

echo "Wheel build completed successfully"
