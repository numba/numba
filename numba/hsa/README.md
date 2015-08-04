Setup
-----

`libhsakmt.so.1`, `libhsa-runtime64.so`, `libhsa-runtime-ext64.so` must be in
 the `LD_LIBRARY_PATH`.
 
Known Issue
-----------

Must use the `system-5.8-2` package at the `numba` binstar channel.

```bash
conda install -c numba system
```
    
    
Run Tests
---------

    python -m numba.hsa.tests.hsadrv.runtests
    python -m numba.hsa.tests.hsapy.runtests

