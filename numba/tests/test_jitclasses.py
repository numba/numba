import unittest
from numba import types, typing, compiler, targets


class Car(object):
    def __init__(self, value):
        self.value = value


def use_car_value(car):
    return car.value


class TestJITClasses(unittest.TestCase):
    def test_jitclass(self):
        carattrs = {
            "value": types.int32,
        }

        tyctx = typing.Context()
        tyctx.insert_class(Car, carattrs)

        cgctx = targets.CPUContext()
        cgctx.insert_class(Car, carattrs)

        car_object = types.Object(Car)
        argtys = (car_object,)

        flags = compiler.Flags()
        func, err = compiler.compile_extra(tyctx, cgctx, use_car_value,
                                           args=argtys, return_type=None,
                                           flags=flags)
        if err:
            raise err

        car = Car(value=123)
        self.assertEqual(use_car_value(car), func(car))




if __name__ == '__main__':
    unittest.main()

