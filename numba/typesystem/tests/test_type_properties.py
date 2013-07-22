from numba.typesystem import *

assert int_.is_int
assert int_.is_numeric
assert long_.is_int
assert long_.is_numeric
assert not long_.is_long

assert float_.is_float
assert float_.is_numeric
assert double.is_float
assert double.is_numeric
assert not double.is_double

assert object_.is_object
assert list_(int_, 2).is_list
assert list_(int_, 2).is_object

assert function(void, [double]).is_function
