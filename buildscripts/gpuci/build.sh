##############################################
# Numba GPU build and test script for CI      #
##############################################
set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Switch to project root; also root of repo checkout
cd "$WORKSPACE"

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Install testing dependencies"
gpuci_mamba_retry install -y "cudatoolkit=${CUDA_TOOLKIT_VER}"

gpuci_logger "Install numba"
python setup.py build_ext --inplace
python setup.py develop

gpuci_logger "Check Python version"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

gpuci_logger "Dump system information from Numba"
python -m numba -s

gpuci_logger "Run tests in numba.cuda.tests"
python -m numba.runtests numba.cuda.tests -m
