def sum1d(s, e):
    c = 0
    for i in range(s, e):
        c += i
    return c


def sum2d(s, e):
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


def redefine1():
    x = 0
    for i in range(5):
        x += 1
    x = 0. + x
    for i in range(5):
        x += 1
    return x