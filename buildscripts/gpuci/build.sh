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
                  "numba/label/dev::llvmlite" \
                  "numpy=1.21" \
                  "scipy" \
                  "cffi" \
                  "psutil" \
                  "gcc_linux-64=7" \
                  "gxx_linux-64=7"

conda activate numba_ci

if [ $NUMBA_CUDA_USE_NVIDIA_BINDING == "1" ]
then
  gpuci_logger "Install NVIDIA CUDA Python bindings";
  gpuci_mamba_retry install nvidia::cuda-python=11.7.0;
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
