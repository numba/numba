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

def test_datetime():

    datetime_components = (2014, 1, 2, 3, 4, 5)
    assert extract_year(datetime.datetime(*datetime_components)) == 2014
    assert extract_month(datetime.datetime(*datetime_components)) == 1
    assert extract_day(datetime.datetime(*datetime_components)) == 2
    assert extract_hour(datetime.datetime(*datetime_components)) == 3
    assert extract_min(datetime.datetime(*datetime_components)) == 4
    assert extract_sec(datetime.datetime(*datetime_components)) == 5

    control = datetime.datetime(2014, 1, 2, 3, 4, 5)
    assert create_python_datetime(2014, 1, 2, 3, 4, 5) == control
    assert create_python_datetime_from_string("2014-01-02T03:04:05Z") == control
    
    datetime_str = '2014-01-02T03:04:05Z'
    control = numpy.datetime64(datetime_str)
    x = create_numpy_datetime(datetime_str)
    print control, x
    assert x == control

if __name__ == "__main__":
    test_datetime()
