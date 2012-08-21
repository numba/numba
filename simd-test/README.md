= Short Description =

This is a sample that tests simd options to implement ufuncs.

Test the performance of ufuncs that perform a gather in order to
use simd instructions and compare it with a simple kernel written
in plain C that performs the same operations. There are also tests
to figure out what is the overhead of splitting the operation and
its memory access. There are extra memory copies that are performed
so there is an overhead there.

The point is figuring out the size of that overhead and at what point
it is worth it. After the memory gathering operations are to be
performed on aligned packed data, so SIMD can be used to full effect.

As the data is split in chunks that fit the L1 cache, there will be
some extra loops and that may imply some overhead as well.


= Build and run instruction =
There is a Makefile that supports the following:

make install - to install the python extension
make clean   - to clean local files
make         - just builds the .so
make test    - runs the test that times the execution

You can take a look inside the make file to figure out how everything
is actually done.