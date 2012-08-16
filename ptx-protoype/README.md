LLVM to PTX Test
----------------

**gen_add.py** -- Generate a simple add kernel to add.ptx;
**testadd.cpp** -- Implement test driver for add.ptx;

Steps:
python gen_add.py
make
./testadd
