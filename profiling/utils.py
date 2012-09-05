import time
import numpy as np

REPEAT = 20

def run_timing(cmd, *args, **kwargs):
    timings = []
    for i in range(REPEAT):
        s = time.time()
        cmd(*args, **kwargs)
        e = time.time()
        dt = e - s
        timings.append(dt)
    return timings

def print_timing(name, timings):
    print "%s\t%s\t%s\t%s" % (name, np.min(timings), np.average(timings), np.max(timings))
