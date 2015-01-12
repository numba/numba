 Environment variables
----------------------


    export LD_LIBRARY_PATH=/opt/hsa_ld_library_path
    export LD_PRELOAD=/lib/x86_64-linux-gnu/libm.so.6
    export NUMBA_HSA_DRIVER=/opt/hsa_ld_library_path/libhsa-runtime64.so
    export HSA_PATH=/opt/hsa
    
To enable HSAIL compiling:

    export HSAILBIN=/path/to/HSAIL-HLC-Stable/bin
    
Get HSAIL-HLC-Stable from https://github.com/HSAFoundation/HSAIL-HLC-Stable

Run Tests
---------

    python -m numba.hsa.tests.hsadrv.runtests
    python -m numba.hsa.tests.hsapy.runtests


Notes
-----

* Very verbose: all driver calls are logged
