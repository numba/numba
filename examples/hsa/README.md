HSA
===

The following instructions assumes Linux 64-bit platform (preferably Ubuntu 14
.04)

Install Base Requirements
--------------------------

Download Linux 64-bit Miniconda distribution

```bash
wget http://repo.continuum.io/miniconda/Miniconda-3.7.0-Linux-x86_64.sh
```

Install Miniconda

```bash
bash Miniconda-3.7.0-Linux-x86_64.sh
```

Use conda to install dependencies

```bash
conda install -c https://conda.binstar.org/numba llvmlite numpy
```

Build Numba
-----------

Checkout Numba source code

```bash
git clone https://github.com/numba/numba.git
cd numba
git checkout hsa
```

In order to build numba with HSA support, you need to set the environment variable
HSA_PATH to point where to find HSA include files/libraries
```bash
export HSA_PATH=/opt/hsa
```

Build inplace
```bash
python setup.py build_ext --inplace
```

Running Numba tests (non HSA)

```bash
python runtests.py -m -vb
```

Note: `-m` for multiprocess testing; `-v` for verbose; `-b` for buffered
stdout/stderr and defer print to the end.

Once the test passes, install numba

```bash
python setup.py install
```

Running HSA
-----------

Running unittests

```bash
LD_LIBRARY_PATH=/opt/hsa_ld_library_path NUMBA_HSA_DRIVER=/opt/hsa_ld_library_path/libhsa-runtime64.so LD_PRELOAD=/lib/x86_64-linux-gnu/libm.so.6 python -m numba.hsa.tests.hsadrv.runtests
```

The `LD_PRELOAD=/lib/x86_64-linux-gnu/libm.so.6` is necessary to workaround a
missing symbol issue with the libm shipped with Miniconda.

The `/opt/hsa_ld_library_path` contains:

- libhsakmt.so.1
- libhsa-runtime64.so
- libhsa-runtime64.so.1

Run HSA vector_copy example

```bash
LD_LIBRARY_PATH=/opt/hsa_ld_library_path NUMBA_HSA_DRIVER=/opt/hsa_ld_library_path/libhsa-runtime64.so LD_PRELOAD=/lib/x86_64-linux-gnu/libm.so.6 python examples/hsa/vector_copy.py
```


