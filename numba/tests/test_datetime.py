from datetime import datetime
import numba

@numba.autojit(nopython=True)
def create_datetime(year, month, day):
    return datetime(year, month, day)

@numba.autojit(nopython=True)
def create_datetime_from_string(datetime_str):
    year = datetime_str[0:4]
    month = datetime_str[5:7]
    day = datetime_str[8:10]
    return datetime(int(year), int(month), int(day))

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
def datetime_delta(d0, d1):
    return d1 - d0

def test_datetime():

    assert extract_year(datetime(2014, 1, 2)) == 2014
    assert extract_month(datetime(2014, 1, 2)) == 1
    assert extract_day(datetime(2014, 1, 2)) == 2

    control = datetime(2014, 1, 2)
    assert create_datetime(2014, 1, 2) == control
    assert create_datetime_from_string("2014-01-02") == control

if __name__ == "__main__":
    test_datetime()
