= Short Description =

This is a prototype that bases on
[the ufunc example](http://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html).
The ufunc calls `parallel_ufunc` to compute using multiple threads.
...


Work items are equally splitted among all worker threads, initially.
Once a thread completes, it tries to steal work items from other threads.
The workqueue is locked using atomic compare-and-swap.
...



= Build and run instruction =

python setup.py build_ext -i

# all test has a "test_" prefix
./test_race.sh               # try to discover race condition
python test_*.py

