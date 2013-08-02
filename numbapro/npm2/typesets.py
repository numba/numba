from .types import (int8, int16, int32, int64,
                    uint8, uint16, uint32, uint64,
                    float32, float64,
                    complex64, complex128)

signed_set = frozenset([int8, int16, int32, int64])
unsigned_set = frozenset([uint8, uint16, uint32, uint64])
integer_set = signed_set | unsigned_set
float_set = frozenset([float32, float64])
complex_set = frozenset([complex64, complex128])


