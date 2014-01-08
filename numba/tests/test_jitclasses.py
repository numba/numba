import unittest
from numba import types, typing, compiler, targets, utils


class Car(object):
    def __init__(self, value):
        self.value = value

    def move(self, x):
        self.value += x


def use_car_value(car):
    return car.value


def use_car_move(car, x):
    car.move(x)
    return car.value


class TestJITClasses(unittest.TestCase):

    def setUp(self):
        move_signature = typing.signature(types.none, types.int32)
        carattrs = {
            "value": types.int32,
            "move" : typing.new_method(Car.move, move_signature),
        }

        self.carattrs = carattrs

    def test_use_car_value(self):
        tyctx = typing.Context()
        tyctx.insert_class(Car, self.carattrs)

        cgctx = targets.CPUContext()
        cgctx.insert_class(Car, self.carattrs)

        car_object = types.Object(Car)
        argtys = (car_object,)

        flags = compiler.Flags()
        cr = compiler.compile_extra(tyctx, cgctx, use_car_value, args=argtys,
                                    return_type=None, flags=flags)
        func = cr.entry_point

        if cr.typing_error:
            raise cr.typing_error

        car = Car(value=123)
        self.assertEqual(use_car_value(car), func(car))

        def bm_python():
            use_car_value(car)

        def bm_numba():
            func(car)

        python = utils.benchmark(bm_python, maxct=100)
        numba = utils.benchmark(bm_numba, maxct=100)

        print(python)
        print(numba)

    def test_use_car_move(self):
        tyctx = typing.Context()
        tyctx.insert_class(Car, self.carattrs)

        cgctx = targets.CPUContext()
        cgctx.insert_class(Car, self.carattrs)

        car_object = types.Object(Car)
        argtys = (car_object, types.int32)

        flags = compiler.Flags()
        cr = compiler.compile_extra(tyctx, cgctx, use_car_move, args=argtys,
                                    return_type=None, flags=flags)
        func = cr.entry_point

        if cr.typing_error:
            raise cr.typing_error

        car1 = Car(value=123)
        car2 = Car(value=123)
        self.assertEqual(use_car_move(car1, 321), func(car2, 321))

        def bm_python():
            use_car_move(car1, 321)

        def bm_numba():
            func(car2, 321)

        python = utils.benchmark(bm_python, maxct=100)
        numba = utils.benchmark(bm_numba, maxct=100)

        print(python)
        print(numba)


if __name__ == '__main__':
    unittest.main()

