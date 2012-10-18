from time import time, clock
def _dummy():
    j = 1
    for i in range(100):
        j += i
    return 0

time_prec = abs(time() - _dummy() - time())
clock_prec = abs(clock() - _dummy() - clock())
assert time_prec > 0 or clock_prec > 0, (time_prec, clock_prec)
if clock_prec > time_prec:
    time = clock


if __name__ == '__main__':
    print 'time()', time_prec
    print 'clock()', clock_prec
