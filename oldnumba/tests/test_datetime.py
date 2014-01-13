import numpy
import numba
import numba.vectorize
from numba.vectorize import vectorize

@numba.autojit(nopython=True)
def datetime_identity(datetime):
    return datetime
    
@numba.autojit(nopython=True)
def timedelta_identity(delta):
    return delta

@numba.autojit(nopython=True)
def create_python_datetime(year, month, day, hour, min, sec):
    return datetime.datetime(year, month, day, hour, min, sec)

@numba.autojit(nopython=True)
def create_numpy_datetime(datetime_str):
    return numpy.datetime64(datetime_str)

@numba.autojit(nopython=True)
def create_numpy_timedelta(delta, units):
    return numpy.timedelta64(delta, units)

@numba.autojit(nopython=True)
def create_python_datetime_from_string(datetime_str):
    year = datetime_str[0:4]
    month = datetime_str[5:7]
    day = datetime_str[8:10]
    hour = datetime_str[11:13]
    min = datetime_str[14:16]
    sec = datetime_str[18:20]
    return datetime.datetime(int(year), int(month), int(day),
        int(hour), int(min), int(sec))

@numba.autojit(nopython=True)
def create_numpy_datetime_from_string(datetime_str):
    year = datetime_str[0:4]
    month = datetime_str[5:7]
    day = datetime_str[8:10]
    hour = datetime_str[11:13]
    min = datetime_str[14:16]
    sec = datetime_str[18:20]
    return numpy.datetime64('{0}-{1}-{2}T{3}:{4}:{5}Z'.format(year, month, day,
        hour, min, sec))

@numba.autojit(nopython=True)
def extract_year(date):
    return date.year

@numba.autojit(nopython=True)
def extract_month(date):
    return date.month

@numba.autojit(nopython=True)
def extract_day(date):
    return date.day

@numba.autojit(nopython=True)
def extract_hour(date):
    return date.hour

@numba.autojit(nopython=True)
def extract_min(date):
    return date.min

@numba.autojit(nopython=True)
def extract_sec(date):
    return date.sec

@numba.autojit(nopython=True)
def datetime_delta(d0, d1):
    return d1 - d0

@numba.autojit(nopython=True)
def datetime_add_timedelta(d, t):
    return d + t

@numba.autojit(nopython=True)
def datetime_subtract_timedelta(d, t):
    return d - t

# JNB: vectorize doesn't work for struct-like types right now
#@vectorize([numba.datetime(units='D')(numba.datetime(units='D'))])
def ufunc_inc_day(a):
    return a + numpy.timedelta64(1, 'D')

@numba.jit(numba.int64(numba.string_), nopython=True)
def cast_datetime_to_int(datetime_str):
    x = numpy.datetime64(datetime_str)
    return x

@numba.autojit(nopython=True)
def datetime_array_index(datetimes, index):
    return datetimes[index]

@numba.jit(numba.datetime(units='M')(numba.datetime(units='M')[:], numba.int_),
    nopython=True)
def datetime_array_index2(datetimes, index):
    return datetimes[index]

@numba.autojit(nopython=True)
def timedelta_array_index(timedeltas, index):
    return timedeltas[index]

@numba.jit(numba.timedelta(units='M')(numba.timedelta(units='M')[:], numba.int_),
    nopython=True)
def timedelta_array_index2(timedeltas, index):
    return timedeltas[index]

def test_datetime():

    datetime = numpy.datetime64('2014-01-01')
    assert datetime_identity(datetime) == datetime

    delta = numpy.timedelta64(1)
    assert timedelta_identity(delta) == delta

    datetime_str = '2014'
    datetime = numpy.datetime64(datetime_str)
    control = numpy.datetime64(datetime_str)
    assert create_numpy_datetime(datetime_str) == control

    datetime_str = '2014-01'
    datetime = numpy.datetime64(datetime_str)
    control = numpy.datetime64(datetime_str)
    assert create_numpy_datetime(datetime_str) == control

    datetime_str = '2014-01-02'
    datetime = numpy.datetime64(datetime_str)
    control = numpy.datetime64(datetime_str)
    assert create_numpy_datetime(datetime_str) == control

    if numpy.version.version[0:3] != '1.6':
        datetime_str = '2014-01-02T03Z'
        datetime = numpy.datetime64(datetime_str)
        control = numpy.datetime64(datetime_str)
        assert create_numpy_datetime(datetime_str) == control

        datetime_str = '2014-01-02T03:04Z'
        datetime = numpy.datetime64(datetime_str)
        control = numpy.datetime64(datetime_str)
        assert create_numpy_datetime(datetime_str) == control

    datetime_str = '2014-01-02T03:04:05Z'
    datetime = numpy.datetime64(datetime_str)
    control = numpy.datetime64(datetime_str)
    assert create_numpy_datetime(datetime_str) == control

    # JNB: string concatenation doesn't work right now
    #assert create_numpy_datetime_from_string(datetime_str) == control

    if numpy.version.version[0:3] != '1.6':
        control = numpy.timedelta64(2014, 'Y')
        assert create_numpy_timedelta(2014, 'Y') == control
        control = numpy.timedelta64(100, 'M')
        assert create_numpy_timedelta(100, 'M') == control
        control = numpy.timedelta64(10000, 'D')
        assert create_numpy_timedelta(10000, 'D') == control
        control = numpy.timedelta64(100, 'h')
        assert create_numpy_timedelta(100, 'h') == control
        control = numpy.timedelta64(100, 'm')
        assert create_numpy_timedelta(100, 'm') == control    
        control = numpy.timedelta64(100, 's')
        assert create_numpy_timedelta(100, 's') == control    

    datetime_str = '2014-01-02T03:04:05Z'
    assert extract_year(numpy.datetime64(datetime_str)) == 2014
    assert extract_month(numpy.datetime64(datetime_str)) == 1
    assert extract_day(numpy.datetime64(datetime_str)) == 2
    assert extract_hour(numpy.datetime64(datetime_str)) == 3
    assert extract_min(numpy.datetime64(datetime_str)) == 4
    assert extract_sec(numpy.datetime64(datetime_str)) == 5

    datetime1 = numpy.datetime64('2014')
    datetime2 = numpy.datetime64('2015')
    control = datetime2 - datetime1
    assert datetime_delta(datetime1, datetime2) == control

    datetime1 = numpy.datetime64('2014-01')
    datetime2 = numpy.datetime64('2015-01')
    control = datetime2 - datetime1
    assert datetime_delta(datetime1, datetime2) == control

    datetime1 = numpy.datetime64('2014-01-01')
    datetime2 = numpy.datetime64('2015-01-02')
    control = datetime2 - datetime1
    assert datetime_delta(datetime1, datetime2) == control

    datetime1 = numpy.datetime64('2014-01-01T01Z')
    datetime2 = numpy.datetime64('2015-01-04T02Z')
    control = datetime2 - datetime1
    assert datetime_delta(datetime1, datetime2) == control

    datetime1 = numpy.datetime64('2014-01-01T01:01Z')
    datetime2 = numpy.datetime64('2015-01-04T02:02Z')
    control = datetime2 - datetime1
    assert datetime_delta(datetime1, datetime2) == control

    datetime1 = numpy.datetime64('2014-01-01T01:01:01Z')
    datetime2 = numpy.datetime64('2015-01-04T02:02:02Z')
    control = datetime2 - datetime1
    assert datetime_delta(datetime1, datetime2) == control

    datetime = numpy.datetime64('2014-01-01')
    if numpy.version.version[0:3] != '1.6':
        timedelta = numpy.timedelta64(1, 'D')
    else:
        timedelta = numpy.timedelta64(1)
    control = datetime + timedelta
    assert datetime_add_timedelta(datetime, timedelta) == control

    datetime = numpy.datetime64('2014-01-01T01:02:03Z')
    if numpy.version.version[0:3] != '1.6':
        timedelta = numpy.timedelta64(-10000, 's')
    else:
        timedelta = numpy.timedelta64(-10000)
    control = datetime + timedelta
    assert datetime_add_timedelta(datetime, timedelta) == control

    datetime = numpy.datetime64('2014')
    if numpy.version.version[0:3] != '1.6':
        timedelta = numpy.timedelta64(10, 'Y')
    else:
        timedelta = numpy.timedelta64(10)
    control = datetime - timedelta
    assert datetime_subtract_timedelta(datetime, timedelta) == control

    datetime = numpy.datetime64('2014-01-01T01:02:03Z')
    if numpy.version.version[0:3] != '1.6':
        timedelta = numpy.timedelta64(-10000, 'm')
    else:
        timedelta = numpy.timedelta64(-10000)
    control = datetime - timedelta
    assert datetime_subtract_timedelta(datetime, timedelta) == control

    datetime_str  ='2014'
    datetime = numpy.datetime64(datetime_str)
    assert cast_datetime_to_int(datetime_str) == \
        int(numpy.array(datetime, numpy.int64))

    # cast datetime to number of days since epoch
    datetime_str  ='2014-01-01'
    datetime = numpy.datetime64(datetime_str)
    assert cast_datetime_to_int(datetime_str) == \
        int(numpy.array(datetime, numpy.int64))

    # cast datetime to number of seconds since epoch
    datetime_str  ='2014-01-02T03:04:05Z'
    datetime = numpy.datetime64(datetime_str)
    assert cast_datetime_to_int(datetime_str) == \
        int(numpy.array(datetime, numpy.int64))

    datetimes = numpy.array(['2014-01', '2014-02', '2014-03'],
        dtype=numpy.datetime64)
    assert datetime_array_index(datetimes, 0) == datetimes[0]
    assert datetime_array_index2(datetimes, 1) == datetimes[1]

    timedeltas = numpy.array([1, 2, 3], dtype='m8[M]')
    assert timedelta_array_index(timedeltas, 0) == timedeltas[0]
    assert timedelta_array_index2(timedeltas, 1) == timedeltas[1]

    # JNB: vectorize doesn't work for struct-like types right now
    #array = numpy.array(['2014-01-01', '2014-01-02', '2014-01-03'],
    #    dtype=numpy.datetime64)
    #assert ufunc_inc_day(array) == numpy.array(
    #    ['2014-01-02', '2014-01-03', '2014-01-04'], dtype=numpy.datetime64)

if __name__ == "__main__":
    test_datetime()
