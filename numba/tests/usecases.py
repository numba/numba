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
        c += i
        i += 1
    return c


def copy_arrays(a, b):
    for i in range(a.shape[0]):
        b[i] = a[i]


def copy_arrays2d(a, b):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[i, j] = a[i, j]


def redefine1():
    x = 0
    for i in range(5):
        x += 1
    x = 0. + x
    for i in range(5):
        x += 1
    return x


def andor(x, y):
    return (x > 0 and x < 10) or (y > 0 and y < 10)


def ifelse1(x, y):
    if x > y:
        return 1
    elif x == 0 or y == 0:
        return 2
    else:
        return 3


def string1(x, y):
    a = "whatzup"
    return a + str(x + y)
