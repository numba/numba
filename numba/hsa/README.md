Setup
-----

`libhsakmt.so.1`, `libhsa-runtime64.so`, `libhsa-runtime-ext64.so` must be in
 the `LD_LIBRARY_PATH`.
 
`libhsa-runtime64.so` and `libhsa-runtime-ext64.so` are in `/opt/hsa/lib`
`libhsakmt.so.1` has no default location and is available from https://github.com/HSAFoundation/HSA-Drivers-Linux-AMD
 
Get and install the deb file from HSAIL-HLC-Stable from https://github
.com/HSAFoundation/HSAIL-HLC-Stable

Optional.  Change if path is not at default location:

    export NUMBA_HSA_DRIVER=/opt/hsa/lib/libhsa-runtime64.so
    export HSAILBIN=/opt/amd/bin
    
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


Notes
-----

* Very verbose: all driver calls are logged
