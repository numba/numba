"""
Usecases with Python 3 syntax in the signatures. This is a separate module
in order to avoid syntax errors with Python 2.
"""


class TestClass1(object):
    def __init__(self, x, y, z=1, *, a=5):
        self.x = x
        self.y = y
        self.z = z
        self.a = a


class TestClass2(object):
    def __init__(self, x, y, z=1, *args, a=5):
        self.x = x
        self.y = y
        self.z = z
        self.args = args
        self.a = a
