#!/bin/bash
set -ex

# Repairs and patches a Numba wheel inside the manylinux container
# Usage: repair_wheel_linux.sh <python_executable> <use_tbb> <wheel_dir>

PYTHON_EXECUTABLE=$1 # Use the provided executable path
USE_TBB=${2:-"true"}
WHEEL_DIR=${3:-"/io/wheelhouse"}

# Debug output
echo "PYTHON_EXECUTABLE: $PYTHON_EXECUTABLE"
echo "USE_TBB: $USE_TBB"
echo "WHEEL_DIR: $WHEEL_DIR"

# Install required tools
$PYTHON_EXECUTABLE -m pip install auditwheel patchelf twine wheel

# Install TBB if enabled
if [ "$USE_TBB" = "true" ]; then
    $PYTHON_EXECUTABLE -m pip install tbb==2021.6 tbb-devel==2021.6
    # Make TBB libraries available to dynamic linker
    find /opt /usr -name "libtbb.so*" -exec cp {} /usr/local/lib/ \; 2>/dev/null
    ldconfig
fi

# Make sure the wheelhouse directory exists
mkdir -p $WHEEL_DIR

# Find the wheel
cd $WHEEL_DIR
echo "Contents of wheel directory ($WHEEL_DIR):"
ls -la

WHEEL_FILE=$(ls -1 numba*.whl 2>/dev/null | head -1)

if [ -z "$WHEEL_FILE" ]; then
    echo "No wheel file found in $WHEEL_DIR"
    exit 1
fi

echo "Repairing wheel: $WHEEL_FILE"

# Create a temporary directory for repair
REPAIR_DIR=$(mktemp -d)
cp "$WHEEL_FILE" "$REPAIR_DIR/"
cd "$REPAIR_DIR"

# Repair with auditwheel
$PYTHON_EXECUTABLE -m auditwheel repair "$WHEEL_FILE" -w "$REPAIR_DIR/wheelhouse"
cd "$REPAIR_DIR/wheelhouse"

# Get the filename of the wheel to be patched
WHEEL_PATCHED=$(ls -1 *.whl | head -1)

# Unpack the wheel for patching
$PYTHON_EXECUTABLE -m wheel unpack $WHEEL_PATCHED
WHEEL_DIR_UNPACKED=$(find . -maxdepth 1 -type d -name "numba-*" | head -1)
cd "$WHEEL_DIR_UNPACKED"

# Patch libraries
if [ -d "numba.libs" ]; then
  cd numba.libs
  LIBTBB=$(ls libtbb* 2>/dev/null || echo "")
  LIBOMP=$(ls libgomp* 2>/dev/null || echo "")
  cd ..
  rm -rf numba.libs

  # Patch TBB libraries if present and TBB is enabled
  if [ "$USE_TBB" = "true" ] && [ -n "$LIBTBB" ]; then
    echo "Patching TBB libraries"
    TBBEXT=$(echo "$LIBTBB" | grep -oP "(\\.so.*)" || echo ".so")
    patchelf numba/np/ufunc/tbbpool*.so --replace-needed $LIBTBB libtbb$TBBEXT
    patchelf numba/np/ufunc/tbbpool*.so --remove-rpath
    ldd numba/np/ufunc/tbbpool*.so
    readelf -d numba/np/ufunc/tbbpool*.so
  fi

  # Patch OpenMP libraries if present
  if [ -n "$LIBOMP" ]; then
    echo "Patching OpenMP libraries"
    OMPEXT=$(echo "$LIBOMP" | grep -oP "(\\.so.*)" || echo ".so")
    patchelf numba/np/ufunc/omppool*.so --replace-needed $LIBOMP libgomp$OMPEXT
    patchelf numba/np/ufunc/omppool*.so --remove-rpath
    ldd numba/np/ufunc/omppool*.so
    readelf -d numba/np/ufunc/omppool*.so
  fi

  # Fix executable bit on scripts
  if [ -d "numba-*.data/scripts" ]; then
    chmod +x numba-*.data/scripts/*
  fi
fi

cd ..

# Repack the wheel
rm -f *.whl
$PYTHON_EXECUTABLE -m wheel pack "$WHEEL_DIR_UNPACKED"

# Get the filename of the final repacked wheel
WHEEL_REPACKED=$(ls -1 *.whl | head -1)

# Remove the original non-manylinux wheel from the target directory
echo "Removing original wheel: $WHEEL_DIR/$WHEEL_FILE"
rm -f "$WHEEL_DIR/$WHEEL_FILE"

# Move final (repaired and repacked) wheel back to output directory
echo "Copying final wheel $WHEEL_REPACKED to $WHEEL_DIR/"
cp "$WHEEL_REPACKED" "$WHEEL_DIR/"

# Verify the final wheel (in the temp dir before cleanup)
$PYTHON_EXECUTABLE -m twine check "$WHEEL_REPACKED"

echo "Wheel repair and patch completed successfully"
