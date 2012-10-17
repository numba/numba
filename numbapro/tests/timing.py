from time import time, clock
time_prec = abs(time() -  time())
clock_prec = abs(clock() - clock())
assert time_prec > 0 or clock_prec > 0, (time_prec, clock_prec)
if clock_prec > time_prec:
    time = clock

