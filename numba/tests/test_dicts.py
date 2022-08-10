from numba import njit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase, force_pyobj_flags


def build_map():
    return {0: 1, 2: 3}

def build_map_from_local_vars():
    # There used to be a crash due to wrong IR generation for STORE_MAP
    x = TestCase
    return {0: x, x: 1}


class DictTestCase(TestCase):

    def test_build_map(self, flags=force_pyobj_flags):
        self.run_nullary_func(build_map, flags=flags)

    def test_build_map_from_local_vars(self, flags=force_pyobj_flags):
        self.run_nullary_func(build_map_from_local_vars, flags=flags)


class TestCompiledDict(TestCase):
    """Testing `dict()` and `{}` usage that are redirected to
    `numba.typed.Dict`.
    """
    def test_use_dict(self):
        # Test dict()
        @njit
        def foo():
            d = dict()
            d[1] = 2
            return d

        d = foo()
        self.assertEqual(d, {1: 2})

    def test_unsupported_dict_usage(self):
        # Test dict(dict())
        from numba.core.typing.dictdecl import _message_dict_support

        @njit
        def foo():
            d = dict()
            d[1] = 2
            return dict(d)

        with self.assertRaises(TypingError) as raises:
            foo()

        self.assertIn(_message_dict_support, str(raises.exception))

    def test_use_curlybraces(self):
        # Test {} with empty args
        @njit
        def foo():
            d = {}
            d[1] = 2
            return d

        d = foo()
        self.assertEqual(d, {1: 2})


    def test_use_curlybraces_with_init1(self):
        # Test {} with 1 item
        @njit
        def foo():
            return {1: 2}

        d = foo()
        self.assertEqual(d, {1: 2})

    def test_use_curlybraces_with_initmany(self):
        # Test {} with many items
        @njit
        def foo():
            return {1: 2.2, 3: 4.4, 5: 6.6}

        d = foo()
        self.assertEqual(d, {1: 2.2, 3: 4.4, 5: 6.6})

    def test_curlybraces_init_with_coercion(self):
        # Type coercion at dict init is tested
        @njit
        def foo():
            return {1: 2.2, 3: 4, 5: 6}

        self.assertEqual(foo(), foo.py_func())

    def test_use_curlybraces_with_manyvar(self):
        # Test using variable in {}
        @njit
        def foo(x, y):
            return {x: 1, y: x + y}

        x, y = 10, 20
        self.assertEqual(foo(x, y), foo.py_func(x, y))

    def test_mixed_curlybraces_and_dict(self):
        # Test mixed use of {} and dict()
        @njit
        def foo():
            k = dict()
            k[1] = {1: 3}
            k[2] = {4: 2}
            return k

        self.assertEqual(foo(), foo.py_func())

    def test_dict_use_with_list_value(self):
        # Test using reflected list as value
        @njit
        def foo():
            d = {}
            for i in range(10):
                d[i] = list(range(i))
            return d

        self.assertEqual(foo(), foo.py_func())

    def test_dict_use_with_mutated_list_value(self):
        # Test using a mutated reflected list as value
        @njit
        def foo():
            l = list(range(3))
            d = {0: l}
            l.append(4)
            return d

        print(foo())
        self.assertEqual(foo(), foo.py_func())
        self.assertEqual(len(foo()[0]), 4)

    def test_dict_use_with_set_value(self):
        # Test using set as value
        @njit
        def foo():
            d = {}
            for i in range(10):
                d[i] = set(range(i))
            return d

        self.assertEqual(foo(), foo.py_func())

    def test_dict_use_with_mutated_set_value(self):
        # Test using a mutated set as value
        @njit
        def foo():
            s = set(range(3))
            d = {0: s}
            s.add(4)
            return d

        print(foo())
        self.assertEqual(foo(), foo.py_func())
        self.assertEqual(len(foo()[0]), 4)


    def test_dict_use_with_none_value(self):
        # Test that NoneType cannot be used as value for Dict
        @njit
        def foo():
            k = {1: None}
            return k

        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn(
            "Dict.value_type cannot be of type none",
            str(raises.exception),
        )


    def test_dict_use_with_optional_value(self):
        # Test that Optional cannot be used as value for Dict
        @njit
        def foo(choice):
            optional = 2.5 if choice else None
            k = {1: optional}
            return k

        with self.assertRaises(TypingError) as raises:
            foo(True)
        self.assertIn(
            "Dict.value_type cannot be of type OptionalType(float64)",
            str(raises.exception),
        )

    def test_dict_use_with_optional_key(self):
        # Test that Optional cannot be used as a key for Dict
        @njit
        def foo(choice):
            k = {2.5 if choice else None: 1}
            return k

        with self.assertRaises(TypingError) as raises:
            foo(True)
        self.assertIn(
            "Dict.key_type cannot be of type OptionalType(float64)",
            str(raises.exception),
        )

    def test_dict_use_with_none_key(self):
        # Test that NoneType cannot be used as a key for Dict
        @njit
        def foo():
            k = {None: 1}
            return k

        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn(
            "Dict.key_type cannot be of type none",
            str(raises.exception),
        )

if __name__ == '__main__':
    unittest.main()
