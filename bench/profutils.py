#
#
#

import time
import numpy as np

_time_func = time.clock

def print_profile_results(profile_results):
    """ print results from a profile_functions run
    """
    for x in profile_results:
        print '%(name)s avg: %(avg)f max: %(max)f min: %(min)f' % x 

def profile_functions(tests, times = 10):
    """ Return timing information on profile runs of a list of functions

    Run functions specified in tests, measuring time. Repeat the run the 
    number specified in the parameter.  Each entry in Tests will be a tuple 
    containing a name for the test, the function to call and the list of 
    arguments.
    """
    def timeit_func(func, args):
        t0 = _time_func()
        func(*args)
        t1 = _time_func()
        return t1-t0
    
    def profile_func(test):
        timings = [timeit_func(test[1], test[2]) for i in range(0, times)]
        return { 'name': test[0],
                 'avg': np.average(timings), 
                 'max': np.max(timings),
                 'min': np.min(timings)}

    return [profile_func(x) for x in tests]

