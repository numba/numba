import pytest
import numpy as np
from numpy.dtypes import StringDType
from numba import njit
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError
import threading
from queue import Queue


def has_StringDType():
    return numpy_version >= (2, 0)


# Lengths hit inline/arena/heap storage thresholds
LENGTHS = [8, 50, 120]
KINDS = ["ascii", "unicode"]
NA_OBJECTS = [None, "NA"]

# Common descriptor that round-trips Python None as the StringDType NA value.
DEFAULT_DT = StringDType(na_object=None)

ASCII_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789_-+*/"
)
UNICODE_ALPHABET = "世界國語文漢字仮名ΩπßäçЖДФ"


def _cycle_string(alphabet: str, n: int, offset: int = 0) -> str:
    m = len(alphabet)
    return "".join(alphabet[(i + offset) % m] for i in range(n))


def make_str(kind: str, n: int, variant: int = 0, inject=None) -> str:
    alphabet = ASCII_ALPHABET if kind == "ascii" else UNICODE_ALPHABET
    s = _cycle_string(alphabet, max(1, n), offset=(variant * 3) % len(alphabet))
    if n == 0:
        return ""
    if inject is not None and n > 0:
        mid = max(0, (n - 1) // 2)
        marker = str(inject)[0]
        s = s[:mid] + marker + s[mid + 1:]
    return s[:n]


@njit
def j_head(a):
    return a[0]


@njit
def j_tail(a):
    return a[-1]


@njit
def j_step2(a):
    return a[::2]


@njit
def j_step_minus2(a):
    return a[::-2]


@njit
def j_take(a, idx):
    return a[idx]


@njit
def j_mask(a, m):
    return a[m]


@njit
def j_assign(dst, src):
    for i in range(dst.shape[0]):
        dst[i] = src[i]


@njit
def j_copy_like(a):
    return a.copy()


@njit
def j_copy_scalar(arr):
    out = np.empty(1, dtype=arr.dtype)
    out[0] = arr[0]
    return out


@njit
def j_item(a):
    return a.item()


@njit
def j_len_at(a, i):
    return len(a[i])


@njit
def j_eq_at(a, i, j):
    return a[i] == a[j]


@njit
def j_ne_at(a, i, j):
    return a[i] != a[j]


@njit
def j_lt_at(a, i, j):
    return a[i] < a[j]


@njit
def j_le_at(a, i, j):
    return a[i] <= a[j]


@njit
def j_gt_at(a, i, j):
    return a[i] > a[j]


@njit
def j_ge_at(a, i, j):
    return a[i] >= a[j]


@njit
def j_concat_pair(a, b):
    out = np.empty(1, dtype=a.dtype)
    out[0] = a[0] + b[0]
    return out


@njit
def j_pair_add(a, b):
    n = a.shape[0]
    out = np.empty(n, dtype=a.dtype)
    for i in range(n):
        out[i] = a[i] + b[i]
    return out


@njit
def j_eq_ij(a, i, b, j):
    return a[i] == b[j]


@njit
def j_startswith_bounds(a, b):
    s = a[0]
    p = b[0]
    return (
        s.startswith(p, 0),
        s.startswith(p, 1),
        s.startswith(p, 1, max(1, len(s) - 1)),
    )


@njit
def j_endswith_bounds(a, b):
    s = a[0]
    p = b[0]
    return (
        s.endswith(p),
        s.endswith(p, 0, len(s)),
        s.endswith(p, 0, max(1, len(s) - 1)),
    )


@njit
def j_transforms(a):
    s = a[0]
    return (
        s.lower(),
        s.upper(),
        s.casefold(),
        s.title(),
        s.strip(),
        s.lstrip(),
        s.rstrip(),
    )


@njit
def j_replaces(a, b, c):
    return a[0].replace(b[0], c[0])


@njit
def j_preds(a):
    s = a[0]
    return (
        s.isalpha(),
        s.isalnum(),
        s.isspace(),
        s.isnumeric(),
        s.isdecimal(),
        s.isdigit(),
        s.isprintable(),
        s.isascii(),
        s.isidentifier(),
        s.islower(),
        s.isupper(),
        s.istitle(),
    )


@njit
def j_numeric(a, sub):
    s = a[0]
    t = sub[0]
    return (
        s.count(t),
        s.find(t),
        s.rfind(t),
        s.index(t),
        s.rindex(t),
        s.find(t, 1),
        s.rfind(t, 0, 5),
        s.count(t, 0, 6),
    )


@njit
def j_broadcast_row(row, out):
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = row[j]
    return out


@njit
def j_mk_empty(template, size):
    return np.empty(size, dtype=template.dtype)


@njit
def j_mk_zeros(template, size):
    return np.zeros(size, dtype=template.dtype)


@njit
def j_mk_full(size, template):
    out = np.empty(size, dtype=template.dtype)
    for i in range(size):
        out[i] = template[0]
    return out


@njit
def j_reshape(a):
    return a.reshape((2, 3))


@njit
def j_transpose(a):
    return a.T


@njit
def j_ravel(a):
    return a.ravel()


@njit
def j_copy_F(a, out):
    m, n = a.shape
    for j in range(n):
        for i in range(m):
            out[i, j] = a[i, j]
    return out


@njit
def j_col_assign_F(a, out):
    m, n = a.shape
    for j in range(n):
        for i in range(m):
            out[i, j] = a[i, 0]
    return out


@njit
def j_masked_assign(dst, src, mask):
    k = 0
    for i in range(dst.shape[0]):
        if mask[i]:
            dst[i] = src[k]
            k += 1
    return dst


@njit
def j_stride_overwrite(a):
    n = a.shape[0]
    for k in range(n // 2):
        a[1 + 2 * k] = a[2 * k]
    return a


@njit
def j_reverse_copy(a):
    n = a.shape[0]
    out = np.empty(n, dtype=a.dtype)
    for i in range(n):
        out[i] = a[n - 1 - i]
    return out


@njit
def j_bad_write(a):
    a[0] = None


@njit
def j_put_int(a):
    a[0] = 123


@njit
def j_ewise_add(a, b):
    return np.add(a, b)


# Helper to give nice pytest ids
def _ids_len(n):
    return f"n{n}"


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
@pytest.mark.parametrize("n", LENGTHS, ids=_ids_len)
@pytest.mark.parametrize("kind", KINDS, ids=str)
class TestIndexingSlicing:
    def test_head_tail_and_step(self, n, kind):
        s1 = make_str(kind, n, 0)
        s2 = make_str(kind, n, 1)
        a = np.array([s1, None, s2, None], dtype=DEFAULT_DT)
        assert j_head(a) == s1
        assert j_tail(a) is None
        s = j_step2(a)
        assert s.shape[0] == 2
        assert s[0] == s1
        assert s[1] == s2

    def test_negative_step_slice(self, n, kind):
        s1 = make_str(kind, n, 0)
        s2 = make_str(kind, n, 1)
        s3 = make_str(kind, n, 2)
        a = np.array([s1, None, s2, None, s3], dtype=DEFAULT_DT)
        out = j_step_minus2(a)
        assert out.shape == (3,)
        assert out[0] == s3
        assert out[1] == s2
        assert out[2] == s1

    def test_fancy_indexing_and_mask(self, n, kind):
        s1 = make_str(kind, n, 0)
        s2 = make_str(kind, n, 1)
        a = np.array([s1, None, s2, None], dtype=DEFAULT_DT)
        out = j_take(a, np.array([1, 0, 3], dtype=np.int64))
        assert out[0] is None
        assert out[1] == s1
        assert out[2] is None

        m = np.array([True, False, True, False])
        mb = j_mask(a, m)
        assert mb.shape[0] == 2
        assert mb[0] == s1
        assert mb[1] == s2

    def test_item_python_value(self, n, kind):
        s = make_str(kind, n, 0)
        a = np.array([s], dtype=DEFAULT_DT)
        b = np.array([None], dtype=DEFAULT_DT)
        assert j_item(a) == s
        assert j_item(b) is None


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
@pytest.mark.parametrize("n", LENGTHS, ids=_ids_len)
@pytest.mark.parametrize("kind", KINDS, ids=str)
class TestAssignmentCopy:
    def test_cross_array_assignment_preserves_na(self, n, kind):
        s1 = make_str(kind, n, 0)
        s2 = make_str(kind, n, 1)
        src = np.array([s1, None, s2], dtype=DEFAULT_DT)
        dst = np.array(["", "", ""], dtype=DEFAULT_DT)
        j_assign(dst, src)
        assert dst[0] == s1
        assert dst[1] is None
        assert dst[2] == s2

    def test_copy_long_and_short(self, n, kind):
        s = make_str(kind, n, 0)
        out = j_copy_scalar(np.array([s], dtype=DEFAULT_DT))
        assert out[0] == s

    def test_copy_method(self, n, kind):
        s = make_str(kind, max(n, 60), 0)
        src = np.array([s], dtype=DEFAULT_DT)
        out = j_copy_like(src)
        assert out[0] == s


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
@pytest.mark.parametrize("creation_type", ["empty", "zeros", "full"], ids=str)
@pytest.mark.parametrize("n", LENGTHS, ids=_ids_len)
@pytest.mark.parametrize("kind", KINDS, ids=str)
class TestCreation:
    def test_array_creation(self, creation_type, n, kind):
        if creation_type == "empty":
            template = np.array([""], dtype=DEFAULT_DT)
            arr = j_mk_empty(template, 3)
            assert arr.shape == (3,)
            assert arr.dtype == DEFAULT_DT
        elif creation_type == "zeros":
            template = np.array([""], dtype=DEFAULT_DT)
            arr = j_mk_zeros(template, 3)
            assert arr.shape == (3,)
            assert arr.dtype == DEFAULT_DT
            for i in range(3):
                assert arr[i] == ""
        elif creation_type == "full":
            test_str = make_str(kind, n, 0)
            template = np.array([test_str], dtype=DEFAULT_DT)
            arr = j_mk_full(3, template)
            assert arr.shape == (3,)
            for i in range(3):
                assert arr[i] == test_str


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
@pytest.mark.parametrize("with_na", [False, True], ids=["noNA", "withNA"])
@pytest.mark.parametrize("n", LENGTHS, ids=_ids_len)
@pytest.mark.parametrize("kind", KINDS, ids=str)
class TestBroadcasting:
    def test_broadcast_row(self, with_na, n, kind):
        s1 = make_str(kind, n, 0)
        s2 = make_str(kind, n, 1)
        s3 = make_str(kind, n, 2)
        row = np.array([s1, None if with_na else s2, s3], dtype=DEFAULT_DT)
        out = np.empty((2, 3), dtype=DEFAULT_DT)
        res = j_broadcast_row(row, out)
        assert res.shape == (2, 3)
        for i in range(2):
            assert res[i, 0] == s1
            if with_na:
                assert res[i, 1] is None
            else:
                assert res[i, 1] == s2
            assert res[i, 2] == s3


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
@pytest.mark.parametrize(
    "operation",
    ["reshape", "transpose", "ravel"],
    ids=str,
)
@pytest.mark.parametrize("n", LENGTHS, ids=_ids_len)
@pytest.mark.parametrize("kind", KINDS, ids=str)
class TestShapeOps:
    def test_shape_operations(self, operation, n, kind):
        base = make_str(kind, 1, 0)[0]
        if operation == "reshape":
            strings = [base * (n + i) for i in range(6)]
            a = np.array(strings, dtype=DEFAULT_DT)
            res = j_reshape(a)
            assert res.shape == (2, 3)
            assert res[0, 0] == strings[0]
            assert res[0, 1] == strings[1]
            assert res[1, 2] == strings[5]
        elif operation == "transpose":
            s1 = base * n
            s2 = base * (n + 1)
            s3 = base * (n + 2)
            s4 = base * (n + 3)
            a = np.array([[s1, s2], [s3, s4]], dtype=DEFAULT_DT)
            res = j_transpose(a)
            assert res.shape == (2, 2)
            assert res[0, 0] == s1
            assert res[0, 1] == s3
            assert res[1, 0] == s2
            assert res[1, 1] == s4
        elif operation == "ravel":
            strings = [base * (n + i) for i in range(6)]
            a = np.array([strings[:3], strings[3:]], dtype=DEFAULT_DT)
            res = j_ravel(a)
            assert res.shape == (6,)
            assert res[0] == strings[0]
            assert res[2] == strings[2]
            assert res[5] == strings[5]


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
@pytest.mark.parametrize("n", LENGTHS, ids=_ids_len)
@pytest.mark.parametrize("kind", KINDS, ids=str)
class TestCompareConcat:
    def test_eq_ord_concat_basic(self, n, kind):
        s1 = make_str(kind, n, 0)
        s2 = make_str(kind, n, 1)
        arr = np.array([s1, s2], dtype=DEFAULT_DT)
        assert j_eq_at(arr, 0, 0)
        assert not j_eq_at(arr, 0, 1)
        py_ord = s1 < s2
        assert j_lt_at(arr, 0, 1) == py_ord
        c = j_concat_pair(
            np.array([s1], dtype=DEFAULT_DT),
            np.array([s2], dtype=DEFAULT_DT),
        )
        assert c[0] == s1 + s2

    def test_na_semantics(self, n, kind):
        s = make_str(kind, n, 0)
        arr = np.array([s, None], dtype=DEFAULT_DT)
        assert j_eq_at(arr, 1, 1)
        assert not j_eq_at(arr, 0, 1)
        assert j_ne_at(arr, 0, 1)
        # ordering with NA -> False
        assert not j_lt_at(arr, 1, 0)
        assert not j_gt_at(arr, 1, 0)
        assert not j_le_at(arr, 1, 0)
        assert not j_ge_at(arr, 1, 0)

    def test_len_behavior(self, n, kind):
        s = make_str(kind, n, 0)
        arr = np.array([s, None], dtype=DEFAULT_DT)
        assert j_len_at(arr, 0) == len(s)
        with pytest.raises(ValueError):
            j_len_at(arr, 1)


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
@pytest.mark.parametrize("n", LENGTHS, ids=_ids_len)
@pytest.mark.parametrize("kind", KINDS, ids=str)
class TestStringMethods:
    def test_transforms_and_preds(self, n, kind):
        # include edges and mixed case
        base = make_str("ascii", max(3, n), 0)
        s = "  A" + base[2:2 + max(1, n - 2)] + "z  "
        a = np.array([s], dtype=DEFAULT_DT)
        lo, up, cf, ti, st, ls, rs = j_transforms(a)
        assert lo == s.lower()
        assert up == s.upper()
        assert cf == s.casefold()
        assert ti == s.title()
        assert st == s.strip()
        assert ls == s.lstrip()
        assert rs == s.rstrip()

        parts = j_replaces(
            np.array(["banana"], dtype=DEFAULT_DT),
            np.array(["na"], dtype=DEFAULT_DT),
            np.array(["X"], dtype=DEFAULT_DT),
        )
        assert parts == "baXX"

        p = j_preds(np.array(["Abc"], dtype=DEFAULT_DT))
        assert p == (
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            True,
        )

    def test_numeric_methods_and_bounds(self, n, kind):
        s = make_str("ascii", max(6, n), 0)
        sub = s[2:4]
        a = np.array([s], dtype=DEFAULT_DT)
        nums = j_numeric(a, np.array([sub], dtype=DEFAULT_DT))
        py = (
            s.count(sub),
            s.find(sub),
            s.rfind(sub),
            s.index(sub),
            s.rindex(sub),
            s.find(sub, 1),
            s.rfind(sub, 0, 5),
            s.count(sub, 0, 6),
        )
        assert nums == py

    def test_methods_na_behavior(self, n, kind):
        na = np.array([None], dtype=DEFAULT_DT)
        with pytest.raises(ValueError):
            j_transforms(na)
        pr = j_preds(na)
        assert pr == (
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        )
        with pytest.raises(ValueError):
            j_numeric(na, np.array(["x"], dtype=DEFAULT_DT))

    def test_startswith_endswith_bounds(self, n, kind):
        if kind == "ascii":
            s = "abcdef" + make_str(kind, max(0, n - 6), 1)
            p1, p2 = "bc", "de"
        else:
            s = "世界界國" + make_str(kind, max(0, n - 4), 1)
            p1, p2 = "界國", "世界"
        a = np.array([s], dtype=DEFAULT_DT)
        b1 = np.array([p1], dtype=DEFAULT_DT)
        b2 = np.array([p2], dtype=DEFAULT_DT)
        sw = j_startswith_bounds(a, b1)
        ew = j_endswith_bounds(a, b2)
        py_sw = (
            s.startswith(p1, 0),
            s.startswith(p1, 1),
            s.startswith(p1, 1, max(1, len(s) - 1)),
        )
        py_ew = (
            s.endswith(p2),
            s.endswith(p2, 0, len(s)),
            s.endswith(p2, 0, max(1, len(s) - 1)),
        )
        assert sw == py_sw
        assert ew == py_ew


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
@pytest.mark.parametrize(
    "na_lhs,na_rhs",
    [(None, None), (None, "NA"), ("NA", None), ("NA", "NA")],
    ids=[
        "L:None-R:None",
        "L:None-R:NA",
        "L:NA-R:None",
        "L:NA-R:NA",
    ],
)
@pytest.mark.parametrize("n", LENGTHS, ids=_ids_len)
@pytest.mark.parametrize("kind", KINDS, ids=str)
class TestCrossDescriptor:
    def test_pair_add_and_eq(self, na_lhs, na_rhs, n, kind):
        dt_l = StringDType(na_object=na_lhs)
        dt_r = StringDType(na_object=na_rhs)
        s1 = make_str(kind, n, 0)
        s2 = make_str(kind, n, 1)
        lhs_na = None if na_lhs is None else "NA"
        rhs_na = None if na_rhs is None else "NA"
        a = np.array([s1, "", lhs_na], dtype=dt_l)
        b = np.array([s2, s1, rhs_na], dtype=dt_r)
        out = j_pair_add(a, b)
        assert out.dtype == dt_l
        assert out[0] == s1 + s2
        assert out[1] == s1
        if na_lhs is None:
            assert out[2] is None
        else:
            assert out[2] == "NA"

        # Cross-descriptor equality: NA==NA True; NA vs value False
        assert j_eq_ij(a, 2, b, 2)
        assert not j_eq_ij(a, 0, b, 2)
        assert not j_eq_ij(a, 2, b, 0)

    def test_mixed_descriptor_ewise_add_typing_error(
        self, na_lhs, na_rhs, n, kind
    ):
        dt_l = StringDType(na_object=na_lhs)
        dt_r = StringDType(na_object=na_rhs)
        a = np.array([make_str(kind, n, 0), make_str(kind, n, 1)], dtype=dt_l)
        b = np.array([make_str(kind, n, 1), make_str(kind, n, 2)], dtype=dt_r)
        with pytest.raises(TypingError):
            j_ewise_add(a, b)


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
class TestMemoryOrderAndStrides:
    def test_fortran_order_copy_and_column_assign(self):
        aF = np.empty((3, 4), dtype=DEFAULT_DT, order="F")
        aF[0, 0] = "A"
        aF[1, 0] = make_str("ascii", 70, 0)
        aF[2, 0] = None
        aF[0, 1] = make_str("ascii", 2, 1)
        aF[1, 1] = make_str("ascii", 120, 2)
        aF[2, 1] = ""
        aF[0, 2] = make_str("ascii", 7, 0)
        aF[1, 2] = make_str("ascii", 64, 1)
        aF[2, 2] = "世"
        aF[0, 3] = "界"
        aF[1, 3] = "g"
        aF[2, 3] = make_str("ascii", 80, 2)

        cpy = j_copy_F(aF, np.empty((3, 4), dtype=DEFAULT_DT, order="F"))
        for i in range(3):
            for j in range(4):
                assert cpy[i, j] == aF[i, j]

        col = j_col_assign_F(aF, np.empty((3, 4), dtype=DEFAULT_DT, order="F"))
        for i in range(3):
            for j in range(4):
                if i == 2:
                    assert col[i, j] is None
                else:
                    assert col[i, j] == aF[i, 0]

    def test_strided_overlapping_slice_and_reverse(self):
        base = np.array([
            "A",
            make_str("ascii", 90, 0),
            None,
            make_str("ascii", 2, 1),
            make_str("ascii", 120, 2),
            "",
            make_str("ascii", 7, 0),
            make_str("ascii", 64, 1),
        ], dtype=DEFAULT_DT)
        got1 = j_stride_overwrite(base.copy())
        ref1 = base.astype(object).copy()
        for k in range(ref1.shape[0] // 2):
            ref1[1 + 2 * k] = ref1[2 * k]
        for i in range(ref1.shape[0]):
            assert got1[i] == ref1[i]

        got2 = j_reverse_copy(base)
        ref2 = base[::-1]
        for i in range(ref2.shape[0]):
            assert got2[i] == ref2[i]


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
class TestBooleanMaskAndScalarBroadcast:
    def test_boolean_mask_assignment(self):
        dst = np.array([
            "a",
            make_str("ascii", 70, 0),
            make_str("ascii", 5, 1),
            None,
            make_str("ascii", 150, 2),
            "e",
        ], dtype=DEFAULT_DT)
        mask = np.array([True, False, True, True, False, True])
        src = np.array([
            make_str("ascii", 80, 0),
            "Y",
            make_str("ascii", 2, 1),
            make_str("ascii", 120, 2),
        ], dtype=DEFAULT_DT)
        out = j_masked_assign(dst.copy(), src, mask)
        exp = dst.astype(object).copy()
        exp[0] = src[0]
        exp[2] = src[1]
        exp[3] = src[2]
        exp[5] = src[3]
        for i in range(dst.shape[0]):
            assert out[i] == exp[i]

    def test_scalar_broadcast_assignment(self):
        # 1D short
        dst1 = np.empty(5, dtype=DEFAULT_DT)
        val1 = np.array(["a"], dtype=DEFAULT_DT)
        for i in range(dst1.shape[0]):
            dst1[i] = val1[0]
        for i in range(5):
            assert dst1[i] == "a"

        # 1D long
        dst2 = np.empty(4, dtype=DEFAULT_DT)
        longv = make_str("ascii", 120, 1)
        val2 = np.array([longv], dtype=DEFAULT_DT)
        for i in range(dst2.shape[0]):
            dst2[i] = val2[0]
        for i in range(4):
            assert dst2[i] == longv

        # 2D NA
        dst3 = np.empty((2, 3), dtype=DEFAULT_DT)
        na = np.array([None], dtype=DEFAULT_DT)
        for i in range(dst3.shape[0]):
            for j in range(dst3.shape[1]):
                dst3[i, j] = na[0]
        for i in range(2):
            for j in range(3):
                assert dst3[i, j] is None


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
class TestParallelAndAliasing:
    def test_self_assignment_long_string(self):
        @njit
        def self_assign(a):
            a[0] = a[0]
            return a
        s = make_str("ascii", 150, 0)
        a = np.array([s], dtype=DEFAULT_DT)
        out = self_assign(a)
        assert out[0] == s

    def test_long_string_copy_isolated_subprocess(self):
        # Reproduce a memory issue observed when this test ran alone.
        import sys
        import subprocess
        import textwrap
        code = textwrap.dedent(
            r'''
import numpy as np
from numba import njit

@njit
def copy_one(src):
    dst = np.empty(1, dtype=src.dtype)
    dst[0] = src[0]
    return dst

long = 'a' * 120
src = np.array([long], dtype=np.dtypes.StringDType(na_object=None))
dst = copy_one(src)
assert dst[0] == long
''')
        subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
class TestTypingErrors:
    def test_error_path_na_write_does_not_poison_allocator(self):
        a = np.array(["x"], dtype=DEFAULT_DT)
        with pytest.raises(TypingError):
            j_bad_write(a)

    def test_coerce_false_rejects_non_unicode_in_njit(self):
        dt = StringDType(coerce=False, na_object=None)
        arr = np.array(["x"], dtype=dt)
        with pytest.raises(TypingError):
            j_put_int(arr)

    def test_ufunc_add_ewise_cross_descriptor_typing_error(self):
        dt1 = DEFAULT_DT
        dt2 = StringDType(na_object="NA")
        a = np.array(["x", "y"], dtype=dt1)
        b = np.array(["u", "v"], dtype=dt2)
        with pytest.raises(TypingError):
            j_ewise_add(a, b)


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
def test_na_unpack_returns_none_from_thread():
    # Ensure NA values unpack to Python None safely from another thread
    # when using a descriptor with na_object=None (no custom sentinel).
    a = np.array([None], dtype=DEFAULT_DT)
    q = Queue()

    def worker():
        q.put(j_item(a))  # uses helper to convert to Python object

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    res = q.get(timeout=5)
    assert res is None


@pytest.mark.skipif(
    not has_StringDType(),
    reason="requires NumPy >= 2.0 with StringDType",
)
class TestStressMemory:
    def test_allocator_stress_mixed_lengths(self):
        # Choose size/iters to keep total ~5s
        n = 256
        iters = 10000

        # Init a base array with mixed lengths/kinds and periodic NA
        a0 = np.empty(n, dtype=DEFAULT_DT)
        for i in range(n):
            kind = "ascii" if (i % 2 == 0) else "unicode"
            base_len = LENGTHS[i % len(LENGTHS)]
            a0[i] = make_str(kind, base_len + (i % 5), (i // 7) % 3)
        for i in range(0, n, 17):
            a0[i] = None
        mask = np.array([(i % 3) == 0 for i in range(n)], dtype=np.bool_)
        src_na = np.empty_like(a0)
        for i in range(n):
            src_na[i] = None if (i % 4) == 0 else a0[i]

        def stress_churn(a, tmp, mask, src_na, iters):
            n = a.shape[0]
            acc = False
            for t in range(iters):
                # 1) Copy every other element from its predecessor
                for i in range(n // 2):
                    a[2 * i + 1] = a[2 * i]
                # 2) Reverse copy via tmp then back
                for i in range(n):
                    tmp[i] = a[n - 1 - i]
                for i in range(n):
                    a[i] = tmp[i]
                # 3) Masked assign from tmp, wrapping source index
                k = 0
                for i in range(n):
                    if mask[i]:
                        a[i] = tmp[k]
                        k += 1
                        if k == n:
                            k = 0
                # 4) Periodic NA propagation via copy from src_na
                if (t & 7) == 0:
                    step = 1 if n < 8 else (n // 4)
                    for i in range(0, n, step):
                        a[i] = src_na[i]
                # 5) Do ephemeral concatenations to stress the allocator
            for i in range(0, n, 3):
                x = a[i]
                y = a[(i + 1) % n]
                d = None if (x is None or y is None) else (x + y)
                # Use the result to avoid DCE; comparisons force materialization
                acc = acc or (d == d)
                # Touch tmp to keep array-typed writes active
                if i == 0:
                    tmp[0] = d
                return acc
        jit_stress_churn = njit(stress_churn)

        # Run jit churn
        jit_a = a0.copy()
        jit_out = np.empty_like(jit_a)
        jit_stress_churn(jit_a, jit_out, mask, src_na, iters)

        # Run Python reference churn
        py_a = a0.copy()
        py_out = np.empty_like(py_a)
        # Use the original Python implementation of the exact same function body
        stress_churn(py_a, py_out, mask, src_na, iters)

        # Compare content element-wise with NA semantics
        for i in range(n):
            if py_a[i] is None:
                assert jit_a[i] is None
            else:
                assert jit_a[i] == py_a[i]
