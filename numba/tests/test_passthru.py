from contextlib import contextmanager
import gc
from numba import int64, jit, objmode, PassThruContainer, types
from numba import cgutils
from numba.extending import box, lower_builtin, NativeValue, type_callable, typeof_impl
from numba.jitclass import _box
from numba.runtime.nrt import rtsys
import pytest
from numba.jitclass.passthru import create_pass_thru_native, PassThruTypeBase, create_pass_thru_type
from sys import getrefcount
from unittest import expectedFailure,TestCase


@contextmanager
def check_numba_allocations(self, extra_allocations=0, refcount_changes={},**track_refcounts):
    refcounts = {k: getrefcount(v) for k, v in track_refcounts.items()}
    for k, v in refcount_changes.items():
        refcounts[k] += v

    for k, v in track_refcounts.items():
        if isinstance(v, _box.Box):
            self.assertEqual(_box.box_get_meminfoptr(v), 0, "'{}' already has meminfo.".format(k))
    del v

    before = rtsys.get_allocation_stats()

    try:
        yield track_refcounts.values()      # ATTN: this will be in 'random' order on older pythons

        refcounts_after = {k: getrefcount(v) for k, v in track_refcounts.items()}
        for k, v in track_refcounts.items():
            if isinstance(v, _box.Box):
                self.assertEqual(get_nrt_refcount(v), 1, "'{}'s meminfo refcount is {} != 1.".format(k, get_nrt_refcount(v)))
        del v
        del track_refcounts

        after = rtsys.get_allocation_stats()

        self.assertTrue(refcounts == refcounts_after and after.alloc - before.alloc - extra_allocations == after.free - before.free)
    finally:
        # try to trigger any __del__ that might not been called b/c of exceptions,
        # otherwise these show up randomly later
        gc.collect()



class PassThruContainerTest(TestCase):
    def test_identity(self):
        obj = dict(a=1)

        @jit(nopython=True)
        def container_identity(c):
            with objmode(a='int64'):
                c.obj['b'] = 2
                a = c.obj['a']

            return c, a

        with check_numba_allocations(self, c=PassThruContainer(obj)) as (c,):
            r, a = container_identity(c)

            self.assertIs(c, r)
            self.assertEqual(a, 1)
            self.assertEqual(c.obj['a'], 1)
            self.assertEqual(c.obj['b'], 2)
            del c, r


    def test_forget(self):
        obj = dict(a=1)

        @jit(nopython=True)
        def forget_container(c):
            with objmode(a="int64"):
                a = c.obj['a']
                c.obj['b'] = 2

            return a

        with check_numba_allocations(self, c=PassThruContainer(obj)) as (c,):
            a = forget_container(c)

            self.assertEqual(a, 1)
            self.assertEqual(c.obj['a'], 1)
            self.assertEqual(c.obj['b'], 2)
            del c


    def test_eq(self):
        obj1 = dict(a=1)
        obj2 = dict(A=1)

        @jit(nopython=True)
        def container_eq(c1, c2, c3, c4):
            with objmode(a="int64", A="int64"):
                a = c1.obj['a']
                A = c3.obj['A']
                c1.obj['b'] = 2
                c3.obj['b'] = 2

            return a, A, c1 == c1, c1 == c2, c1 == c3, c1 == c4, c2 == c2, c2 == c3, c2 == c4, c3 == c3, c3 == c4, c4 == c4

        with check_numba_allocations(
                self,
                c1=PassThruContainer(obj1),
                c2=PassThruContainer(obj1),
                c3=PassThruContainer(obj2),
                c4=PassThruContainer(obj2)
        ) as (c1, c2, c3, c4):
            a, A, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10 = container_eq(c1, c2, c3, c4)

            self.assertEqual(a, 1)
            self.assertEqual(A, 1)
            self.assertEqual((r1, r2, r3, r4, r5, r6, r7, r8, r9, r10), (c1 == c1, c1 == c2, c1 == c3, c1 == c4, c2 == c2, c2 == c3, c2 == c4, c3 == c3, c3 == c4, c4 == c4))
            self.assertEqual(c1.obj['a'], 1)
            self.assertEqual(c1.obj['b'], 2)
            self.assertEqual(c3.obj['A'], 1)
            self.assertEqual(c3.obj['b'], 2)
            del c1, c2, c3, c4

    @expectedFailure
    def test_hash(self):
        obj = dict(a=1)

        @jit(nopython=True)
        def container_hash(c):
            return hash(c)

        with check_numba_allocations(self, c=PassThruContainer(obj)) as (c,):
            res = container_hash(c)

            self.assertEqual(hash(c), res)
            del c


############################ PassThru ############################
# simple pass through type, no attributes accessible from nopython
##################################################################
class MyPassThru(_box.Box):
    def __init__(self):
        super(MyPassThru, self).__init__()
        self.something_not_boxable = object()


class MyPassThruType(PassThruTypeBase):
    def __init__(self):
        super(MyPassThruType, self).__init__()


@typeof_impl.register(MyPassThru)
def type_object(val, context):
    return MyPassThruType()


create_pass_thru_type(MyPassThruType)


class MyPassThruTest(TestCase):
    def test_forget(self):
        with check_numba_allocations(self, o=MyPassThru()) as (o,):
            _id = forget(o)
            self.assertEqual(_id, id(o.something_not_boxable))
            del o

    def test_identity(self):
        with check_numba_allocations(self, o=MyPassThru()) as (o,):
            o2 = identity(o)
            self.assertIs(o, o2)
            del o, o2

    def test_doubling(self):
        with check_numba_allocations(self, o=MyPassThru()) as (o,):
            o2, o3 = double(o)
            self.assertIs(o, o2)
            self.assertIs(o, o3)
            del o, o2, o3

    def test_list_identity(self):
        with check_numba_allocations(self, x=MyPassThru(), y=MyPassThru(), z=MyPassThru()) as (x, y, z):
            l = [x, y, z]
            l2 = identity(l)
            self.assertIs(l, l2)
            del x, y, z, l, l2


    def test_list_create(self):
        with check_numba_allocations(self, x=MyPassThru(), y=MyPassThru()) as (x, y):
            l = create_passthru_list(x, y)

            self.assertIs(l[0], x)
            self.assertIs(l[1], y)

            del l, x, y

    def test_list_doubling(self):
        with check_numba_allocations(self, x=MyPassThru(), y=MyPassThru(), z=MyPassThru()) as (x, y, z):
            l = [x, y, z]

            l2, l3 = double(l)
            self.assertIs(l, l2)
            self.assertIs(l, l3)
            self.assertIs(l[0], x)
            self.assertIs(l2[0], x)
            self.assertIs(l3[0], x)

            del x, y, z, l, l2, l3


############################# PassThruComplex #############################
# pass through type with several attrs accessible from nopython,
# including another pass through type and a list
###########################################################################

class PassThruComplex(_box.Box):
    def __init__(self, int_attr, passthru_attr, list_attr=[]):
        super(PassThruComplex, self).__init__()
        self.int_attr = int_attr
        self.passthru_attr = passthru_attr
        self.list_attr = list_attr

        self.something_not_boxable = object()


class PassThruComplexType(PassThruTypeBase):
    nopython_attrs = [
        ('int_attr', lambda fe_type: int64),
        ('passthru_attr', lambda fe_type: MyPassThruType()),
         ('list_attr', lambda fe_type: types.List(MyPassThruType()))
    ]

    def __init__(self):
        super(PassThruComplexType, self).__init__()


def unbox_passthru_complex_payload(typ, obj, context):
    pass_thru = cgutils.create_struct_proxy(typ)(context.context, context.builder)

    pass_thru.parent = obj

    int_attr = context.pyapi.object_getattr_string(obj, 'int_attr')
    pass_thru.int_attr = context.unbox(int64, int_attr).value
    context.pyapi.decref(int_attr)

    passthru_attr = context.pyapi.object_getattr_string(obj, 'passthru_attr')
    pass_thru.passthru_attr = context.unbox(MyPassThruType(), passthru_attr).value
    context.pyapi.decref(passthru_attr)

    list_attr_type = types.List(MyPassThruType())

    list_attr = context.pyapi.object_getattr_string(obj, 'list_attr')
    list_attr_unboxed = context.unbox(list_attr_type, list_attr)
    pass_thru.list_attr = list_attr_unboxed.value
    context.pyapi.decref(list_attr)

    # should have done this more often, errors most likely erased by subsequent pyapi call anyway ...
    is_error = cgutils.is_not_null(context.builder, context.pyapi.err_occurred())

    def cleanup():
        list_attr_unboxed.cleanup()

    return NativeValue(pass_thru._getvalue(), is_error=is_error, cleanup=cleanup)


PassThruComplexPayloadType = create_pass_thru_type(PassThruComplexType, unbox_payload=unbox_passthru_complex_payload)


@typeof_impl.register(PassThruComplex)
def type_object(val, context):
    return PassThruComplexType()


@box(PassThruComplexPayloadType)
def box_passthru_complex_payload(typ, val, context):
    pass_thru = cgutils.create_struct_proxy(typ)(context.context, context.builder, value=val)

    int_attr = context.box(int64, pass_thru.int_attr)
    passthru_attr = context.box(MyPassThruType(), pass_thru.passthru_attr)
    list_attr = context.box(types.List(MyPassThruType()), pass_thru.list_attr)

    class_obj = context.pyapi.unserialize(context.pyapi.serialize_object(PassThruComplex))
    res = context.pyapi.call_function_objargs(class_obj, (int_attr, passthru_attr, list_attr))

    context.pyapi.decref(int_attr)
    context.pyapi.decref(passthru_attr)
    context.pyapi.decref(list_attr)

    #context.context.nrt.decref(context.builder, typ, val)

    return res


@type_callable(PassThruComplex)
def type_passthru_complex_constructor(context):
    def passthru_complex_constructor_typer(int_attr, passthru_attr, list_attr):
        return PassThruComplexType()

    return passthru_complex_constructor_typer


@lower_builtin(PassThruComplex, types.Integer, MyPassThruType, types.List)
def passthru_complex_constructor(context, builder, sig, args):
    typ = sig.return_type
    int_attr_ty, passthru_attr_ty, list_attr_ty = sig.args
    int_attr, passthru_attr, list_attr = args

    pass_thru, _ = create_pass_thru_native(context, builder, typ)
    payload = typ.get_payload(context, builder, pass_thru)

    context.nrt.incref(builder, int_attr_ty, int_attr)
    context.nrt.incref(builder, passthru_attr_ty, passthru_attr)
    context.nrt.incref(builder, list_attr_ty, list_attr)

    payload.int_attr = int_attr
    payload.passthru_attr = passthru_attr
    payload.list_attr = list_attr

    return pass_thru._getvalue()


class PassThruComplexTest(TestCase):
    def test_forget(self):
        with check_numba_allocations(self, x=MyPassThru(), y=MyPassThru(), z=MyPassThru()) as (x, y, z):
            o = PassThruComplex(42, x, [y, z])

            _id = forget(o)
            self.assertEqual(_id, id(o.something_not_boxable))
            del o, x, y, z

    def test_identity(self):
        with check_numba_allocations(self, x=MyPassThru(), y=MyPassThru(), z=MyPassThru()) as (x, y, z):
            o = PassThruComplex(42, x, [y, z])

            o2 = identity(o)
            self.assertIs(o, o2)

            del o, o2, x, y, z

    def test_doubling(self):
        with check_numba_allocations(self, x=MyPassThru(), y=MyPassThru(), z=MyPassThru()) as (x, y, z):
            o = PassThruComplex(42, x, [y, z])

            o2, o3 = double(o)
            self.assertIs(o, o2)
            self.assertIs(o, o3)

            del o, o2, o3, x, y, z

    def test_attr_access(self):
        with check_numba_allocations(self, x=MyPassThru(), y=MyPassThru(), z=MyPassThru()) as (x, y, z):
            o = PassThruComplex(42, x, [y, z])

            value, x2, (y2, z2) = attr_access(o, 'int_attr', 'passthru_attr', 'list_attr')

            self.assertEqual(value, 42)
            self.assertIs(x2, x)
            self.assertIs(y2, y)
            self.assertIs(z2, z)

            del o, x, x2, y, y2, z, z2

    def test_list_create(self):
        with check_numba_allocations(self, x=MyPassThru(), y=MyPassThru(), z=MyPassThru()) as (x, y, z):
            o1 = PassThruComplex(42, x, [y, z])
            o2 = PassThruComplex(43, x, [y, z])

            l = create_passthru_list(o1, o2)

            self.assertEqual(o1.int_attr, 42)
            self.assertEqual(o2.int_attr, 43)
            self.assertIs(o1.passthru_attr, x)
            self.assertEqual(o1.list_attr[0], y)
            self.assertEqual(o1.list_attr[1], z)

            del l, o1, o2, x, y, z

    def test_native_create(self):
        with check_numba_allocations(self, x=MyPassThru(), y=MyPassThru(), z=MyPassThru()) as (x, y, z):
            o = create_passthru_complex_native(43, x, y, z)

            self.assertEqual(o.int_attr, 43)
            self.assertIs(o.passthru_attr, x)
            self.assertEqual(o.list_attr[0], y)
            self.assertEqual(o.list_attr[1], z)

            del o, x, y, z

    def test_native_create_and_double(self):
        with check_numba_allocations(self, x=MyPassThru(), y=MyPassThru(), z=MyPassThru()) as (x, y, z):
            o1, o2 = create_passthru_complex_native_and_double(43, x, y, z)

            self.assertIs(o1, o2)
            self.assertEqual(o1.int_attr, 43)
            self.assertIs(o1.passthru_attr, x)
            self.assertEqual(o1.list_attr[0], y)
            self.assertEqual(o1.list_attr[1], z)

            del o1, o2, x, y, z


############################ test functions ############################
# test function for basic allocation tests
########################################################################
@jit(nopython=True)
def forget(o):
    with objmode(x='int64'):
        x = id(o.something_not_boxable)

    return x


@jit(nopython=True)
def identity(x):
    return x


@jit(nopython=True)
def double(x):
    return x, x


def attr_access(o, *names):
    body = ", ".join(f"o.{n}" for n in names)
    f = jit(nopython=True)(eval(f"lambda o: ({body})"))

    return f(o)


@jit(nopython=True)
def create_passthru_list(list_attr1, list_attr2):
    return [list_attr1, list_attr2]


@jit(nopython=True)
def create_passthru_complex_native(int_attr, passthru_attr, list_attr1, list_attr2):
    l = [list_attr1, list_attr2]
    o = PassThruComplex(int_attr, passthru_attr, l)
    return o


@jit(nopython=True)
def create_passthru_complex_native_and_double(int_attr, passthru_attr, list_attr1, list_attr2):
    l = [list_attr1, list_attr2]
    o = PassThruComplex(int_attr, passthru_attr, l)

    return o, o


def get_nrt_refcount(box_obj):
    from numba.config import MACHINE_BITS
    from ctypes import cast, c_int32, c_int64, POINTER

    refcount_ctype = c_int64 if MACHINE_BITS == 64 else c_int32

    ptr = _box.box_get_meminfoptr(box_obj)
    if ptr == 0:
        raise ValueError("'MemInfo' not initialized, yet.")

    return cast(ptr, POINTER(refcount_ctype)).contents.value