import array
import numpy as np

from numba import unittest_support as unittest
from numba import jit, njit, cfunc, cffi_support, types, errors
from numba.core.compiler import compile_isolated, Flags
from numba.tests.support import TestCase, tag

import numba.tests.cffi_usecases as mod
import unittest


enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()


@unittest.skipUnless(cffi_support.SUPPORTED,
                     "CFFI not supported -- please install the cffi module")
class TestCFFI(TestCase):

    # Need to run the tests serially because of race conditions in
    # cffi's OOL mode.
    _numba_parallel_test_ = False

    def setUp(self):
        mod.init()
        mod.init_ool()

    def test_type_map(self):
        signature = cffi_support.map_type(mod.ffi.typeof(mod.cffi_sin))
        self.assertEqual(len(signature.args), 1)
        self.assertEqual(signature.args[0], types.double)

    def _test_function(self, pyfunc, flags=enable_pyobj_flags):
        cres = compile_isolated(pyfunc, [types.double], flags=flags)
        cfunc = cres.entry_point

        for x in [-1.2, -1, 0, 0.1, 3.14]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_sin_function(self):
        self._test_function(mod.use_cffi_sin)

    def test_bool_function_ool(self):
        pyfunc = mod.use_cffi_boolean_true
        cres = compile_isolated(pyfunc, (), flags=no_pyobj_flags)
        cfunc = cres.entry_point
        self.assertEqual(pyfunc(), True)
        self.assertEqual(cfunc(), True)

    def test_sin_function_npm(self):
        self._test_function(mod.use_cffi_sin, flags=no_pyobj_flags)

    def test_sin_function_ool(self, flags=enable_pyobj_flags):
        self._test_function(mod.use_cffi_sin_ool)

    def test_sin_function_npm_ool(self):
        self._test_function(mod.use_cffi_sin_ool, flags=no_pyobj_flags)

    def test_two_funcs(self):
        # Check that two constant functions don't get mixed up.
        self._test_function(mod.use_two_funcs)

    def test_two_funcs_ool(self):
        self._test_function(mod.use_two_funcs_ool)

    def test_function_pointer(self):
        pyfunc = mod.use_func_pointer
        cfunc = jit(nopython=True)(pyfunc)
        for (fa, fb, x) in [
            (mod.cffi_sin, mod.cffi_cos, 1.0),
            (mod.cffi_sin, mod.cffi_cos, -1.0),
            (mod.cffi_cos, mod.cffi_sin, 1.0),
            (mod.cffi_cos, mod.cffi_sin, -1.0),
            (mod.cffi_sin_ool, mod.cffi_cos_ool, 1.0),
            (mod.cffi_sin_ool, mod.cffi_cos_ool, -1.0),
            (mod.cffi_cos_ool, mod.cffi_sin_ool, 1.0),
            (mod.cffi_cos_ool, mod.cffi_sin_ool, -1.0),
            (mod.cffi_sin, mod.cffi_cos_ool, 1.0),
            (mod.cffi_sin, mod.cffi_cos_ool, -1.0),
            (mod.cffi_cos, mod.cffi_sin_ool, 1.0),
            (mod.cffi_cos, mod.cffi_sin_ool, -1.0)]:
            expected = pyfunc(fa, fb, x)
            got = cfunc(fa, fb, x)
            self.assertEqual(got, expected)
        # A single specialization was compiled for all calls
        self.assertEqual(len(cfunc.overloads), 1, cfunc.overloads)

    def test_user_defined_symbols(self):
        pyfunc = mod.use_user_defined_symbols
        cfunc = jit(nopython=True)(pyfunc)
        self.assertEqual(pyfunc(), cfunc())

    def check_vector_sin(self, cfunc, x, y):
        cfunc(x, y)
        np.testing.assert_allclose(y, np.sin(x))

    def _test_from_buffer_numpy_array(self, pyfunc, dtype):
        x = np.arange(10).astype(dtype)
        y = np.zeros_like(x)
        cfunc = jit(nopython=True)(pyfunc)
        self.check_vector_sin(cfunc, x, y)

    def test_from_buffer_float32(self):
        self._test_from_buffer_numpy_array(mod.vector_sin_float32, np.float32)

    def test_from_buffer_float64(self):
        self._test_from_buffer_numpy_array(mod.vector_sin_float64, np.float64)

    def test_from_buffer_struct(self):
        n = 10
        x = np.arange(n) + np.arange(n * 2, n * 3) * 1j
        y = np.zeros(n)
        real_cfunc = jit(nopython=True)(mod.vector_extract_real)
        real_cfunc(x, y)
        np.testing.assert_equal(x.real, y)
        imag_cfunc = jit(nopython=True)(mod.vector_extract_imag)
        imag_cfunc(x, y)
        np.testing.assert_equal(x.imag, y)

    def test_from_buffer_pyarray(self):
        pyfunc = mod.vector_sin_float32
        cfunc = jit(nopython=True)(pyfunc)
        x = array.array("f", range(10))
        y = array.array("f", [0] * len(x))
        self.check_vector_sin(cfunc, x, y)

    def test_from_buffer_error(self):
        pyfunc = mod.vector_sin_float32
        cfunc = jit(nopython=True)(pyfunc)
        # Non-contiguous array
        x = np.arange(10).astype(np.float32)[::2]
        y = np.zeros_like(x)
        with self.assertRaises(errors.TypingError) as raises:
            cfunc(x, y)
        self.assertIn("from_buffer() unsupported on non-contiguous buffers",
                      str(raises.exception))

    def test_from_buffer_numpy_multi_array(self):
        c1 = np.array([1, 2], order='C', dtype=np.float32)
        c1_zeros = np.zeros_like(c1)
        c2 = np.array([[1, 2], [3, 4]], order='C', dtype=np.float32)
        c2_zeros = np.zeros_like(c2)
        f1 = np.array([1, 2], order='F', dtype=np.float32)
        f1_zeros = np.zeros_like(f1)
        f2 = np.array([[1, 2], [3, 4]], order='F', dtype=np.float32)
        f2_zeros = np.zeros_like(f2)
        f2_copy = f2.copy('K')
        pyfunc = mod.vector_sin_float32
        cfunc = jit(nopython=True)(pyfunc)
        # No exception because of C layout and single dimension
        self.check_vector_sin(cfunc, c1, c1_zeros)
        # No exception because of C layout
        cfunc(c2, c2_zeros)
        sin_c2 = np.sin(c2)
        sin_c2[1] = [0, 0]  # Reset to zero, since cfunc only processes one row
        np.testing.assert_allclose(c2_zeros, sin_c2)
        # No exception because of single dimension
        self.check_vector_sin(cfunc, f1, f1_zeros)
        # Exception because multi-dimensional with F layout
        with self.assertRaises(errors.TypingError) as raises:
            cfunc(f2, f2_zeros)
        np.testing.assert_allclose(f2, f2_copy)
        self.assertIn("from_buffer() only supports multidimensional arrays with C layout",
                      str(raises.exception))

    def test_indirect_multiple_use(self):
        """
        Issue #2263

        Linkage error due to multiple definition of global tracking symbol.
        """
        my_sin = mod.cffi_sin

        # Use two jit functions that references `my_sin` to ensure multiple
        # modules
        @jit(nopython=True)
        def inner(x):
            return my_sin(x)

        @jit(nopython=True)
        def foo(x):
            return inner(x) + my_sin(x + 1)

        # Error occurs when foo is being compiled
        x = 1.123
        self.assertEqual(foo(x), my_sin(x) + my_sin(x + 1))


@unittest.skipUnless(cffi_support.SUPPORTED,
                     "CFFI not supported -- please install the cffi module")
class TestCFFILinkedList(TestCase):
    def setUp(self):
        ffi_mod = mod.load_ool_linkedlist()
        self.lib = ffi_mod.lib
        self.ffi = ffi_mod.ffi

    def _create_linked_list(self, n):
        lib = self.lib
        @njit
        def create_linked_list(n):
            l = lib.list_new()
            for i in range(n):
                lib.list_append(l, i)
            return l

        return create_linked_list(n)

    def test_create_linked_list(self):
        n = 100
        ll = self._create_linked_list(n)
        self.assertEqual(self.lib.list_len(ll), n)
        self.assertEqual(self.lib.list_sum(ll), sum(range(n)))

    def test_traverse_list(self):
        n = 100
        ll = self._create_linked_list(n)
        lib = self.lib
        ffi = self.ffi

        @njit
        def list_sum(l):
            n = l.node
            s = 0
            while n != ffi.NULL:
                s += n.value
                n = n.next
            return s

        s = list_sum(ll)

        self.assertEqual(s, lib.list_sum(ll))

    def test_create_list(self):
        n = 100
        ll_ref = self._create_linked_list(n)
        lib = self.lib
        ffi = self.ffi

        @njit
        def create_new_list(n):
            head = ffi.new("Head*")
            nodes = [ffi.new("Node*")]
            head.node = nodes[0]
            head.node.value = 0
            last = head.node
            for i in range(1, n):
                node_ref = ffi.new("Node*")
                nodes.append(node_ref)
                last.next = node_ref
                last = last.next
                last.value = i
            return head, nodes

        ll, nodes = create_new_list(n)
        self.assertEqual(lib.list_len(ll_ref), lib.list_len(ll))
        self.assertEqual(lib.list_sum(ll_ref), lib.list_sum(ll))
        node = ll.node
        for i in range(n):
            self.assertEqual(node.value, i)
            node = node.next
        self.assertEqual(node, ffi.NULL)

    def test_const_lowering(self):
        n = 100
        ll = self._create_linked_list(n)
        lib = self.lib
        ffi = self.ffi

        @njit
        def list_sum():
            # we pass ll as constant
            n = ll.node
            s = 0
            while n != ffi.NULL:
                s += n.value
                n = n.next
            return s

        self.assertEqual(lib.list_sum(ll), list_sum())

    def test_array(self):
        n = 100
        ffi = self.ffi
        nodes = ffi.new('Node[{}]'.format(n))
        for i in range(n):
            nodes[i].value = i

        @njit
        def sum_array(nodes):
            s = 0
            for i in range(len(nodes)):
                s += nodes[i].value
            return s

        self.assertEqual(sum_array(nodes), sum(range(n)))

    def test_allocate_array(self):
        ffi = self.ffi

        @njit
        def create_array():
            nodes = ffi.new("Node[100]")
            for i in range(len(nodes)):
                nodes[i].value = i
            return nodes

        nodes = create_array()
        self.assertEqual(sum(n.value for n in nodes), sum(range(100)))

    def test_iter_arry(self):
        ffi = self.ffi
        lib = self.lib
        nodes = ffi.new("Node[100]")

        @njit
        def set_array(nodes):
            for i, n in enumerate(nodes):
                n.value = i

        set_array(nodes)

        self.assertEqual(sum(n.value for n in nodes), sum(range(100)))

        @njit
        def create_and_set_array():
            nodes = ffi.new('Node[100]')
            for i in range(len(nodes)):
                nodes[i].value = i
            return nodes

        nodes2 = create_and_set_array()
        self.assertEqual(sum(n.value for n in nodes2), sum(range(100)))

    def test_create_destroy(self):
        ffi = self.ffi
        lib = self.lib
        import random

        @njit
        def create_and_destroy():
            nodes = ffi.new('Node[100]')
            nodes[10].value = random.randint(0, 100)
            # for i in range(100):
            #     nodes[i].value = random.randint(0, 100)
            # return nodes[random.randint(0, 99)].value

        create_and_destroy()
        # self.assertTrue(val > 0 and val < 99)


    def test_callback(self):
        n = 100
        ffi = self.ffi
        lib = self.lib
        ll = self._create_linked_list(n)
        node_t = cffi_support.map_type(ffi.typeof("Node*"))

        @cfunc(types.void(node_t))
        def double_node_value(node):
            node.value = node.value * 2

        lib.list_map(ll, double_node_value.cffi)

        self.assertEqual(lib.list_sum(ll), 2 * sum(range(n)))

if __name__ == '__main__':
    unittest.main()
