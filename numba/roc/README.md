Setup
-----

`libhsakmt.so.1`, `libhsa-runtime64.so`, `libhsa-runtime-ext64.so` must be in
 the `LD_LIBRARY_PATH`.

The standard location of these libraries are in `/opt/hsa/lib`.  Thus,
user can simply do `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/hsa/lib`


Run Tests
---------

The HSA test suite can be executed from the base of the source tree with:

```bash
python runtests.py numba.hsa.tests -vb
```

The test suite can also be executed inside the python interpreter with:

```python
import numba.hsa
numba.hsa.test("-vb")
```

Or directly from the terminal with:

```bash
python -c 'import numba.hsa; numba.hsa.test("-vb")'
```

Note that the "-vb" flags are optional.  The "-v" flag enables verbose mode
that will print the name of each test.  The "-b" flag enables capturing
the stdout messages printed from within the tests.

