from __future__ import print_function, absolute_import, division
from numbapro import vectorize
from numbapro.testsupport import unittest


def discriminant(a, b, c):
    return a + b + c


class TestErrorArgs(unittest.TestCase):
    def test_narg_error(self):
        sig = ['float32(float32,float32)', 'float64(float64,float64)']
        try:
            cu_discriminant = vectorize(sig, target='gpu')(discriminant)
        except TypeError as e:
            self.assertTrue("Mismatch number of argument types" in str(e))
        else:
            raise AssertionError("Excepting an expection")


if __name__ == '__main__':
    unittest.main()
