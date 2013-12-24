from __future__ import print_function
from numba import bytecode, interpreter


def sum_1d(s, e):
    c = 0
    for i in range(s, e):
        c += i
    return c


def sum_2d(s, e):
    c = 0
    for i in range(s, e):
        for j in range(s, e):
            c += i * j
    return c


def while_count(s, e):
    i = s
    c = 0
    while i < e:
        c += 1
        i += 1
    return c


def copy_arrays(a, b):
    for i in range(a.shape[0]):
        b[i] = a[i]


def main():
    bc = bytecode.ByteCode(func=while_count)
    print(bc.dump())

    interp = interpreter.Interpreter(bytecode=bc)
    interp.interpret()
    interp.dump()

    for syn in interp.syntax_info:
        print(syn)

    interp.verify()

if __name__ == '__main__':
    main()
