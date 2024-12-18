# use_pickled_func.py

import cloudpickle

with open("/tmp/pf.pkl", "rb") as cpf:
    pf = cpf.read()

func = cloudpickle.loads(pf)

print("VALUE" in func.py_func.__globals__)

func()
