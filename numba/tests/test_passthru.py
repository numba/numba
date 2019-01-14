from contextlib import contextmanager
import gc
from numba import jit, objmode, PassThruContainer, types
from numba import cgutils
from numba.config import MACHINE_BITS
from numba.datamodel import models
from numba.extending import box, lower_builtin, NativeValue, register_model, type_callable, typeof_impl, unbox, \
    make_attribute_wrapper
from numba.jitclass import _box
from numba.runtime.nrt import rtsys
from numba.jitclass.passthru import PassThru, PassThruType, pass_thru_type, pass_thru_constructor
from sys import getrefcount
from unittest import TestCase


@contextmanager
def check_numba_allocations(self, create_tracked_objects, extra_allocations=0, refcount_changes={}):
    track_refcounts = create_tracked_objects()
    refcounts = {k: getrefcount(v) for k, v in track_refcounts.items()}
    for k, v in refcount_changes.items():
        refcounts[k] += v

    for k, v in track_refcounts.items():
        if isinstance(v, _box.Box):
            self.assertEqual(_box.box_get_meminfoptr(v), 0, "'{}' already has meminfo.".format(k))

    if track_refcounts:
        # if loop ran at least once clean-up v
        del v

    before = rtsys.get_allocation_stats()

    try:
        tracked = tuple(v for k, v in sorted(track_refcounts.items()))
        yield tracked
        del tracked

        refcounts_after = {k: getrefcount(v) for k, v in track_refcounts.items()}
        for k, v in track_refcounts.items():
            if isinstance(v, _box.Box):
                self.assertEqual(
                    get_nrt_refcount(v), 1, "'{}'s meminfo refcount is {} != 1.".format(k, get_nrt_refcount(v))
                )

        if track_refcounts:
            # if loop ran at least once clean-up v
            del v
        del track_refcounts

        after = rtsys.get_allocation_stats()

        self.assertEqual(
            refcounts,refcounts_after
        )
        self.assertEqual(
            after.alloc - before.alloc - extra_allocations, after.free - before.free
        )
    finally:
        # try to trigger any __del__ that might not been called b/c of exceptions,
        # otherwise these show up randomly later
        gc.collect()


class PassThruContainerTest(TestCase):
    def test_identity(self):
        obj = dict(a=1)

        @jit(nopython=True)
        def container_identity(c):
            with objmode(a='intp'):
                c.obj['b'] = 2
                a = c.obj['a']

            return c, a

        with check_numba_allocations(self, (lambda: dict(c=PassThruContainer(obj)))) as (c,):
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
            with objmode(a="intp"):
                a = c.obj['a']
                c.obj['b'] = 2

            return a

        with check_numba_allocations(self, (lambda: dict(c=PassThruContainer(obj)))) as (c,):
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
            with objmode(a="intp", A="intp"):
                a = c1.obj['a']
                A = c3.obj['A']
                c1.obj['b'] = 2
                c3.obj['b'] = 2

            return (
                a, A, c1 == c1, c1 == c2, c1 == c3, c1 == c4, c2 == c2, c2 == c3, c2 == c4, c3 == c3, c3 == c4, c4 == c4
            )

        with check_numba_allocations(
                self,
                (
                    lambda: dict(
                        c1=PassThruContainer(obj1),
                        c2=PassThruContainer(obj1),
                        c3=PassThruContainer(obj2),
                        c4=PassThruContainer(obj2)
                    )
                )
        ) as (c1, c2, c3, c4):
            a, A, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10 = container_eq(c1, c2, c3, c4)

            self.assertEqual(a, 1)
            self.assertEqual(A, 1)
            self.assertEqual(
                (r1, r2, r3, r4, r5, r6, r7, r8, r9, r10),
                (c1 == c1, c1 == c2, c1 == c3, c1 == c4, c2 == c2, c2 == c3, c2 == c4, c3 == c3, c3 == c4, c4 == c4)
            )
            self.assertEqual(c1.obj['a'], 1)
            self.assertEqual(c1.obj['b'], 2)
            self.assertEqual(c3.obj['A'], 1)
            self.assertEqual(c3.obj['b'], 2)
            del c1, c2, c3, c4

    def test_hash(self):
        obj = dict(a=1)

        @jit(nopython=True)
        def container_hash(c):
            return hash(c)

        with check_numba_allocations(self, (lambda: dict(c=PassThruContainer(obj)))) as (c,):
            res = container_hash(c)

            self.assertEqual(hash(c), res)
            del c


############################ PassThru ############################
# simple pass through type, no attributes accessible from nopython
##################################################################
class MyPassThru(object):
    def __init__(self):
        self.something_not_boxable = object()


@typeof_impl.register(MyPassThru)
def type_my_passthru(val, context):
    return pass_thru_type


class MyPassThruTest(TestCase):
    def test_forget(self):
        with check_numba_allocations(self, (lambda: dict(o=MyPassThru()))) as (o,):
            _id = forget(o)
            self.assertEqual(_id, id(o.something_not_boxable) & (2**MACHINE_BITS - 1))
            del o

    def test_identity(self):
        with check_numba_allocations(self, (lambda: dict(o=MyPassThru()))) as (o,):
            o2 = identity(o)
            self.assertIs(o, o2)
            del o, o2

    def test_doubling(self):
        with check_numba_allocations(self, (lambda: dict(o=MyPassThru()))) as (o,):
            o2, o3 = double(o)
            self.assertIs(o, o2)
            self.assertIs(o, o3)
            del o, o2, o3

    def test_list_forget(self):
        @jit(nopython=True)
        def forget_list(x):
            return 1

        def create_tracked():
            o = MyPassThru()
            l = [o, o, o]

            return dict(l=l, o=o)

        with check_numba_allocations(self, create_tracked) as (l, o):
            one = forget_list(l)
            self.assertIs(one, 1)
            del o, l

    def test_list_identity(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            l = [x, y, z]
            l2 = identity(l)
            self.assertIs(l, l2)
            del x, y, z, l, l2

    def test_list_create(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru()))) as (x, y):
            l = create_passthru_list(x, y)

            self.assertIs(l[0], x)
            self.assertIs(l[1], y)

            del l, x, y

    def test_list_copy(self):
        @jit(nopython=True)
        def copy_list(x):
            the_copy = []
            for ii in range(len(x)):
                v = x[ii]
                the_copy.append(v)

            return the_copy  # y

        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru()))) as (x, y):
            l1 = [x, y]

            l2 = copy_list(l1)

            self.assertIs(l2[0], x)
            self.assertIs(l2[1], y)

            del l1, l2, x, y

    def test_list_doubling(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            l = [x, y, z]

            l2, l3 = double(l)
            self.assertIs(l, l2)
            self.assertIs(l, l3)
            self.assertIs(l[0], x)
            self.assertIs(l2[0], x)
            self.assertIs(l3[0], x)

            del x, y, z, l, l2, l3

    def test_native_create(self):
        @jit(nopython=True)
        def create_pass_thru():
            # this will raise NotImplementedError from boxer
            return PassThru()

        with check_numba_allocations(self, (lambda: {})):
            with self.assertRaises(NotImplementedError) as raises:
                create_pass_thru()

        self.assertIn("Native creation of 'PassThruType' not implemented.", str(raises.exception))


############################# PassThruComplex #############################
# pass through type with several attrs accessible from nopython,
# including another pass through type and a list
###########################################################################

class PassThruComplex(object):
    def __init__(self, int_attr, passthru_attr, list_attr=[]):
        self.int_attr = int_attr
        self.passthru_attr = passthru_attr
        self.list_attr = list_attr

        self.something_not_boxable = object()


class PassThruComplexType(PassThruType):
    def __init__(self):
        super(PassThruComplexType, self).__init__()


@register_model(PassThruComplexType)
class PassThruComplexModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        members = [
            ('parent', pass_thru_type),
            ('int_attr', types.intp),
            ('passthru_attr', pass_thru_type),
            ('list_attr', types.List(pass_thru_type))
        ]
        super(PassThruComplexModel, self).__init__(dmm, fe_typ, members)


make_attribute_wrapper(PassThruComplexType, 'int_attr', 'int_attr')
make_attribute_wrapper(PassThruComplexType, 'passthru_attr', 'passthru_attr')
make_attribute_wrapper(PassThruComplexType, 'list_attr', 'list_attr')


@typeof_impl.register(PassThruComplex)
def type_my_pass_thru_complex(val, context):
    return PassThruComplexType()


@unbox(PassThruComplexType)
def unbox_passthru_complex(typ, obj, context):
    pass_thru = cgutils.create_struct_proxy(typ)(context.context, context.builder)

    pass_thru.parent = context.unbox(pass_thru_type, obj).value

    int_attr = context.pyapi.object_getattr_string(obj, 'int_attr')
    pass_thru.int_attr = context.unbox(types.intp, int_attr).value
    context.pyapi.decref(int_attr)

    passthru_attr = context.pyapi.object_getattr_string(obj, 'passthru_attr')
    pass_thru.passthru_attr = context.unbox(pass_thru_type, passthru_attr).value
    context.pyapi.decref(passthru_attr)

    # We want to have lists of PassThruComplexType. We must copy the list as nopython lists have
    # a cleanup functions that will not get called for list members (ie forwarding like
    # NativeValue(..., cleanup=list_attr_unboxed.cleanup) will not work).
    list_attr_type = types.List(pass_thru_type) # force list reflected=False

    list_attr = context.pyapi.object_getattr_string(obj, 'list_attr')
    py_list_obj = context.pyapi.unserialize(context.pyapi.serialize_object(list))
    list_attr_copy = context.pyapi.call_function_objargs(py_list_obj, [list_attr])
    context.pyapi.decref(list_attr)
    context.pyapi.decref(py_list_obj)

    list_attr_unboxed = context.unbox(list_attr_type, list_attr_copy)
    pass_thru.list_attr = list_attr_unboxed.value
    context.pyapi.decref(list_attr_copy)
    list_attr_unboxed.cleanup()

    # should have done this more often, errors most likely erased by subsequent pyapi call anyway ...
    is_error = cgutils.is_not_null(context.builder, context.pyapi.err_occurred())

    return NativeValue(pass_thru._getvalue(), is_error=is_error)


@box(PassThruComplexType)
def box_passthru_complex(typ, val, context):
    pass_thru = cgutils.create_struct_proxy(typ)(context.context, context.builder, value=val)

    # try to box parent, if this succeeds we are done else create a new Python object
    # we need to incref here as the boxer will decref and we might still need the parent
    # to create a new Python object
    context.context.nrt.incref(context.builder, pass_thru_type, pass_thru.parent)
    obj = context.box(pass_thru_type, pass_thru.parent)

    with context.builder.if_else(cgutils.is_null(context.builder, obj)) as (no_python_created, python_created):
        with python_created:
            context.context.nrt.decref(context.builder, typ, val)
            bb_python_created = context.builder.basic_block

        with no_python_created:
            # clear the NotImplementedError set by context.box(pass_thru_type, val)
            context.pyapi.err_clear()

            int_attr = context.box(types.intp, pass_thru.int_attr)
            passthru_attr = context.box(pass_thru_type, pass_thru.passthru_attr)
            list_attr = context.box(types.List(pass_thru_type), pass_thru.list_attr)

            class_obj = context.pyapi.unserialize(context.pyapi.serialize_object(PassThruComplex))
            new_obj = context.pyapi.call_function_objargs(class_obj, (int_attr, passthru_attr, list_attr))

            # store the newly created object onto the meminfo such that only one instance is created
            # even if the same nopython instance gets returned twice (or more)
            parent = cgutils.create_struct_proxy(pass_thru_type)(
                context.context, context.builder, value=pass_thru.parent
            )
            context.context.nrt.meminfo_set_data(context.builder, parent.meminfo, new_obj)

            context.pyapi.decref(int_attr)
            context.pyapi.decref(passthru_attr)
            context.pyapi.decref(list_attr)

            # only need to decref parent, other members already decref'ed by the boxers
            context.context.nrt.decref(context.builder, pass_thru_type, pass_thru.parent)

            bb_no_python_created = context.builder.basic_block

    res = context.builder.phi(cgutils.voidptr_t)
    res.add_incoming(obj, bb_python_created)
    res.add_incoming(new_obj, bb_no_python_created)

    return res


@type_callable(PassThruComplex)
def type_passthru_complex_constructor(context):
    def passthru_complex_constructor_typer(int_attr, passthru_attr, list_attr):
        return PassThruComplexType()

    return passthru_complex_constructor_typer


@lower_builtin(PassThruComplex, types.Integer, pass_thru_type, types.List)
def passthru_complex_constructor(context, builder, sig, args):
    typ = sig.return_type
    int_attr_ty, passthru_attr_ty, list_attr_ty = sig.args
    int_attr, passthru_attr, list_attr = args

    pass_thru = cgutils.create_struct_proxy(typ)(context, builder)

    pass_thru.parent = pass_thru_constructor(context, builder, pass_thru_type(), [])

    context.nrt.incref(builder, int_attr_ty, int_attr)
    context.nrt.incref(builder, passthru_attr_ty, passthru_attr)
    context.nrt.incref(builder, list_attr_ty, list_attr)

    pass_thru.int_attr = int_attr
    pass_thru.passthru_attr = passthru_attr
    pass_thru.list_attr = list_attr

    return pass_thru._getvalue()


class PassThruComplexTest(TestCase):
    def test_forget(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, [y, z])

            _id = forget(o)
            self.assertEqual(_id, id(o.something_not_boxable) & (2**MACHINE_BITS - 1))
            del o, x, y, z

    def test_identity(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, [y, z])

            o2 = identity(o)
            self.assertIs(o, o2)

            del o, o2, x, y, z

    def test_doubling(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, [y, z])

            o2, o3 = double(o)
            self.assertIs(o, o2)
            self.assertIs(o, o3)

            del o, o2, o3, x, y, z

    def test_attr_access(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, [y, z])

            value, x2, (y2, z2) = attr_access(o, 'int_attr', 'passthru_attr', 'list_attr')

            self.assertEqual(value, 42)
            self.assertIs(x2, x)
            self.assertIs(y2, y)
            self.assertIs(z2, z)

            del o, x, x2, y, y2, z, z2

    def test_list_create(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o1 = PassThruComplex(42, x, [y, z])
            o2 = PassThruComplex(43, x, [y, z])

            l = create_passthru_list(o1, o2)

            self.assertEqual(o1.int_attr, 42)
            self.assertEqual(o2.int_attr, 43)
            self.assertIs(o1.passthru_attr, x)
            self.assertEqual(o1.list_attr[0], y)
            self.assertEqual(o1.list_attr[1], z)

            del l, o1, o2, x, y, z

    def test_list_copy(self):
        @jit(nopython=True)
        def copy_list(x):
            the_copy = []
            for ii in range(len(x.list_attr)):
                v = x.list_attr[ii]
                the_copy.append(v)

            return the_copy  # y

        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, [y, z])

            l2 = copy_list(o)

            self.assertIs(l2[0], y)
            self.assertIs(l2[1], z)

            del o, l2, x, y, z

    def test_list_identity(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, [y, z])

            l = [o, o, o]
            l2 = identity(l)
            self.assertIs(l, l2)
            del o, x, y, z, l, l2

    def test_list_forget(self):
        @jit(nopython=True)
        def forget_list(x):
            return 1

        def create_tracked():
            from numpy import ones
            x = MyPassThru()
            y = MyPassThru()
            z = MyPassThru()
            o = PassThruComplex(42, x, [y, z])
            l = [o, o, o]

            return dict(l=l, o=o, x=x, y=y, z=z)

        with check_numba_allocations(self, create_tracked) as (l, o, x, y, z):
            one = forget_list(l)
            one = forget_list(l)
            # one = forget_list(l)
            self.assertIs(one, 1)
            del x, y, z, l, o

    def test_native_create(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = create_passthru_complex_native(43, x, y, z)

            self.assertEqual(o.int_attr, 43)
            self.assertIs(o.passthru_attr, x)
            self.assertEqual(o.list_attr[0], y)
            self.assertEqual(o.list_attr[1], z)

            del o, x, y, z

    def test_native_create_and_double(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
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
    with objmode(x='uintp'):
        # TODO: cannot get 32bit to work w/o truncation
        x = id(o.something_not_boxable) & (2**MACHINE_BITS - 1)

    return x


@jit(nopython=True)
def identity(x):
    return x


@jit(nopython=True)
def double(x):
    return x, x


def attr_access(o, *names):
    body = ", ".join("o.{}".format(n) for n in names)
    f = jit(nopython=True)(eval("lambda o: ({})".format(body)))

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
    from ctypes import cast, c_int32, c_int64, POINTER

    refcount_ctype = c_int64 if MACHINE_BITS == 64 else c_int32

    ptr = _box.box_get_meminfoptr(box_obj)
    if ptr == 0:
        raise ValueError("'MemInfo' not initialized, yet.")

    return cast(ptr, POINTER(refcount_ctype)).contents.value
