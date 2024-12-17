# pickle_func.py

import numba
import cloudpickle

VALUE = 1

@numba.jit
def func():
    with numba.objmode(val = "int64"):
        val = VALUE
    return val


pf = cloudpickle.dumps(func)

with open("/tmp/pf.pkl", "wb") as cpf:
    cpf.write(pf)