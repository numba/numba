import numpy as np

from numba.cuda.testing import CUDATestCase

from numba import cuda
from numba.cuda.vector_types import vector_types


def make_kernel(vtype):
    """
    Returns a jit compiled kernel that constructs a vector types of
    the given type, using the exact number of primitive types to
    construct the vector type.
    """
    vobj = vtype.user_facing_object
    base_type = vtype.base_type

    def kernel_1elem(res):
        v = vobj(base_type(0))
        res[0] = v.x

    def kernel_2elem(res):
        v = vobj(base_type(0), base_type(1))
        res[0] = v.x
        res[1] = v.y

    def kernel_3elem(res):
        v = vobj(base_type(0), base_type(1), base_type(2))
        res[0] = v.x
        res[1] = v.y
        res[2] = v.z

    def kernel_4elem(res):
        v = vobj(
            base_type(0),
            base_type(1),
            base_type(2),
            base_type(3)
        )
        res[0] = v.x
        res[1] = v.y
        res[2] = v.z
        res[3] = v.w

    host_function = {
        1: kernel_1elem,
        2: kernel_2elem,
        3: kernel_3elem,
        4: kernel_4elem
    }[vtype.num_elements]
    return cuda.jit(host_function)


def make_fancy_creation_kernel(vtype):
    """
    Returns a jit compiled kernel that constructs a vector type using the
    "fancy" construction, that is, with arbitrary combinations of primitive
    types and vector types, as long as the total element of the construction
    is the same as the number of elements of the vector type.
    """
    base_type = vtype.base_type
    v2 = getattr(cuda, f"{vtype.name[:-1]}2")
    v3 = getattr(cuda, f"{vtype.name[:-1]}3")
    v4 = getattr(cuda, f"{vtype.name[:-1]}4")

    def kernel(res):
        one = base_type(1.0)
        two = base_type(2.0)
        three = base_type(3.0)
        four = base_type(4.0)

        # Construct a 2-component vector type, possible combination includes:
        # v2(p, p), v2(v2), (p is a primitive type)

        # 2 3
        f2 = v2(two, three)
        # 2 3
        f2_2 = v2(f2)

        # Construct a 3-component vector type, possible combination includes:
        # v3(p, p, p), v3(p, v2), v3(v2, p), v3(v3), (p is a primitive type)

        # 2 3 1
        f3 = v3(f2, one)
        # 1 2 3
        f3_2 = v3(one, f2)
        # 1 2 3
        f3_3 = v3(one, two, three)
        # 2 3 1
        f3_4 = v3(f3)

        # Construct a 4-component vector type, possible combination includes:
        # v4(p, p, p, p), v4(p, p, v2), v4(p, v2, p), v4(v2, p, p), v4(v2, v2),
        # v4(p, v3), v4(v3, p), v4(v4), (p is a primitive type)

        # 1 2 3 4
        f4 = v4(one, two, three, four)
        # 2 3 1 4
        f4_2 = v4(f2, one, four)
        # 1 2 3 4
        f4_3 = v4(one, f2, four)
        # 1 4 2 3
        f4_4 = v4(one, four, f2)
        # 2 3 2 3
        f4_5 = v4(f2, f2)
        # 2 3 1 4
        f4_6 = v4(f3, four)
        # 4 2 3 1
        f4_7 = v4(four, f3)
        # 1 2 3 4
        f4_8 = v4(f4)

        # Write all constructed elements out to the result array

        res[0] = f2.x
        res[1] = f2.y
        res[2] = f2_2.x
        res[3] = f2_2.y

        j = 4
        for i, f in enumerate((f3, f3_2, f3_3, f3_4)):
            res[i * 3 + j] = f.x
            res[i * 3 + j + 1] = f.y
            res[i * 3 + j + 2] = f.z

        j += 12
        for i, f in enumerate((f4, f4_2, f4_3, f4_4, f4_5, f4_6, f4_7, f4_8)):
            res[i * 4 + j] = f.x
            res[i * 4 + j + 1] = f.y
            res[i * 4 + j + 2] = f.z
            res[i * 4 + j + 3] = f.w

    return cuda.jit(kernel)


class TestCudaVectorType(CUDATestCase):

    def test_creation_readout(self):
        for vty in vector_types.values():
            with self.subTest(vty=vty):
                arr = np.zeros((vty.num_elements,))
                kernel = make_kernel(vty)
                kernel[1, 1](arr)
                np.testing.assert_almost_equal(
                    arr, np.array(range(vty.num_elements))
                )

    def test_fancy_creation_readout(self):
        for vty in vector_types.values():
            kernel = make_fancy_creation_kernel(vty)

            expected = np.array([
                2, 3,
                2, 3,
                2, 3, 1,
                1, 2, 3,
                1, 2, 3,
                2, 3, 1,
                1, 2, 3, 4,
                2, 3, 1, 4,
                1, 2, 3, 4,
                1, 4, 2, 3,
                2, 3, 2, 3,
                2, 3, 1, 4,
                4, 2, 3, 1,
                1, 2, 3, 4
            ])
            arr = np.zeros(expected.shape)
            kernel[1, 1](arr)
            np.testing.assert_almost_equal(arr, expected)
