import datetime
import numpy
import numba

@numba.autojit(nopython=True)
def create_python_datetime(year, month, day, hour, min, sec):
    return datetime.datetime(year, month, day, hour, min, sec)

@numba.autojit(nopython=True)
def create_numpy_datetime(datetime_str):
    return numpy.datetime64(datetime_str)

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

#@numba.autojit(nopython=True)
#def datetime_delta(d0, d1):
#    return d1 - d0

#@numba.jit(numba.int64(numba.datetime), nopython=True)
#def cast_datetime_to_int(x):
#    return x

def test_datetime():

    datetime_components = (2014, 1, 2, 3, 4, 5)
    datetime_str = '2014-01-02T03:04:05Z'

    # JNB: only test numpy datetimes for now
    #assert extract_year(datetime.datetime(*datetime_components)) == 2014
    #assert extract_month(datetime.datetime(*datetime_components)) == 1
    #assert extract_day(datetime.datetime(*datetime_components)) == 2
    #assert extract_hour(datetime.datetime(*datetime_components)) == 3
    #assert extract_min(datetime.datetime(*datetime_components)) == 4
    #assert extract_sec(datetime.datetime(*datetime_components)) == 5

    assert extract_year(numpy.datetime64(datetime_str)) == 2014
    assert extract_month(numpy.datetime64(datetime_str)) == 1
    assert extract_day(numpy.datetime64(datetime_str)) == 2
    assert extract_hour(numpy.datetime64(datetime_str)) == 3
    assert extract_min(numpy.datetime64(datetime_str)) == 4
    assert extract_sec(numpy.datetime64(datetime_str)) == 5

    #control = datetime.datetime(*datetime_components)
    #assert create_python_datetime(*datetime_components) == control
    #assert create_python_datetime_from_string("2014-01-02T03:04:05Z") == control
    
    control = numpy.datetime64(datetime_str)
    assert create_numpy_datetime(datetime_str) == control
    # JNB: string concatenation doesn't work right now
    #assert create_numpy_datetime_from_string(datetime_str) == control

    #datetime1 = numpy.datetime64('2014-01-01')
    #datetime2 = numpy.datetime64('2014-01-02')
    #datetime_delta(datetime1, datetime2)

    #cast_datetime_to_int(numpy.datetime64('2014-01-02'))

if __name__ == "__main__":
    test_datetime()
