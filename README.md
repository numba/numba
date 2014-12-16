numbapro
========

Proprietary version of Numba that provides multi-core and GPU support

Building and installing for development
---------------------------------------

Create and activate a new conda environment (where `$PYVER` is your preferred
Python version):

```
conda create -n numbapro_dev python=$PYVER numpy scipy cudatoolkit patchelf
source activate numbapro_dev
```

Install numba master and llvmlite master from the Numba binstar channel:

```
conda install -c numba numba
```

Init and update submodules:

```
git submodule init
git submodule update
```

Build and install the cudalib conda recipe:

```
conda build --no-binstar-upload cudalib/condarecipe
CUDALIB_PKG=`conda build --output cudalib/condarecipe
conda install CUDALIB_PKG
```

Build and install Numbapro:

```
python setup.py build_ext --inplace
python setup.py install # Optional, if you're not going to run in-place
```

Test with:

```
python runtests.py
```

If you make any changes to cudalib, it will be necessary to rebuild and
reinstall the cudalib conda recipe. Otherwise, it will not need rebuilding if
changes are only made to the `numbapro` module.
