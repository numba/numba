##############################################
# Numba GPU build and test script for CI     #
##############################################
set -e

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Switch to project root; also root of repo checkout
cd "$WORKSPACE"

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

# Test with NVIDIA Bindings on CUDA 11.5
if [ $CUDA_TOOLKIT_VER == "11.5" ]
then
  export NUMBA_CUDA_USE_NVIDIA_BINDING=1;
else
  export NUMBA_CUDA_USE_NVIDIA_BINDING=0;
fi;

# Test with Minor Version Compatibility on CUDA 11.8
if [ $CUDA_TOOLKIT_VER == "11.8" ]
then
  export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1;
else
  export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=0;
fi;

# Test with different NumPy and Python versions with each toolkit (it's not
# worth testing the Cartesian product of versions here, we just need to test
# with different CUDA, NumPy, and Python versions).
declare -A CTK_NUMPY_VMAP=( ["11.2"]="1.22" ["11.3"]="1.23" ["11.5"]="1.25" ["11.8"]="1.26")
NUMPY_VER="${CTK_NUMPY_VMAP[$CUDA_TOOLKIT_VER]}"

declare -A CTK_PYTHON_VMAP=( ["11.2"]="3.9" ["11.3"]="3.10" ["11.5"]="3.11" ["11.8"]="3.12")
PYTHON_VER="${CTK_PYTHON_VMAP[$CUDA_TOOLKIT_VER]}"

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Create testing env"
. /opt/conda/etc/profile.d/conda.sh
gpuci_mamba_retry create -n numba_ci -y \
                  "python=${PYTHON_VER}" \
                  "cudatoolkit=${CUDA_TOOLKIT_VER}" \
                  "rapidsai-nightly::cubinlinker" \
                  "conda-forge::ptxcompiler" \
                  "numba/label/dev::llvmlite" \
                  "numpy=${NUMPY_VER}" \
                  "scipy" \
                  "cffi" \
                  "psutil" \
                  "gcc_linux-64=7" \
                  "gxx_linux-64=7" \
                  "setuptools"

conda activate numba_ci

if [ $NUMBA_CUDA_USE_NVIDIA_BINDING == "1" ]
then
  gpuci_logger "Install NVIDIA CUDA Python bindings";
  gpuci_mamba_retry install cuda-python=11.8 cuda-cudart=11.5 cuda-nvrtc=11.5;
fi;

gpuci_logger "Install numba"
python setup.py develop

gpuci_logger "Check Compiler versions"
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources

gpuci_logger "Dump system information from Numba"
python -m numba -s

gpuci_logger "Run tests in numba.cuda.tests"
python -m numba.runtests numba.cuda.tests -v -m
