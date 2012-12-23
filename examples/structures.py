from numba import struct, jit, double
import numpy as np

record_type = struct([('x', double), ('y', double)])
record_dtype = record_type.get_dtype()
a = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=record_dtype)

@jit(argtypes=[record_type[:]])
def hypot(data):
    # return types of numpy functions are inferred
    result = np.empty_like(data, dtype=np.float64) 
    # notice access to structure elements 'x' and 'y' via attribute access
    # You can also index by field name or field index:
    #       data[i].x == data[i]['x'] == data[i][0]
    for i in range(data.shape[0]):
        result[i] = np.sqrt(data[i].x * data[i].x + data[i].y * data[i].y)

    return result

print hypot(a)

# Notice inferred return type
print hypot.signature
# Notice native sqrt calls and for.body direct access to memory...
#print hypot.lfunc
