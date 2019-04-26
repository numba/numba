from __future__ import absolute_import, print_function

from collections import OrderedDict
import types as pytypes
import inspect
import operator

from llvmlite import ir as llvmir

from numba import types
from numba.targets.registry import cpu_target
from numba import njit
from numba.typing import templates
from numba.datamodel import default_manager, models
from numba.targets import imputils
from numba import cgutils, utils, errors
from numba.six import exec_, Sequence
from . import _box


##############################################################################
# Data model


class InstanceModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        cls_data_ty = types.ClassDataType(fe_typ)
        # MemInfoPointer uses the `dtype` attribute to traverse for nested
        # NRT MemInfo.  Since we handle nested NRT MemInfo ourselves,
        # we will replace provide MemInfoPointer with an opaque type
        # so that it does not raise exception for nested meminfo.
        dtype = types.Opaque('Opaque.' + str(cls_data_ty))
        members = [
            ('meminfo', types.MemInfoPointer(dtype)),
            ('data', types.CPointer(cls_data_ty)),
        ]
        super(InstanceModel, self).__init__(dmm, fe_typ, members)


class InstanceDataModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        clsty = fe_typ.class_type
        members = [(_mangle_attr(k), v) for k, v in clsty.struct.items()]
        super(InstanceDataModel, self).__init__(dmm, fe_typ, members)


default_manager.register(types.ClassInstanceType, InstanceModel)
default_manager.register(types.ClassDataType, InstanceDataModel)
default_manager.register(types.ClassType, models.OpaqueModel)


def _mangle_attr(name):
    """
    Mangle attributes.
    The resulting name does not startswith an underscore '_'.
    """
    return 'm_' + name


##############################################################################
# Class object

_ctor_template = """
def ctor({args}):
    return __numba_cls_({args})
"""


def _getargs(fn_sig):
    """
    Returns list of positional and keyword argument names in order.
    """
    params = fn_sig.parameters
    args = []
    for k, v in params.items():
        if (v.kind & v.POSITIONAL_OR_KEYWORD) == v.POSITIONAL_OR_KEYWORD:
            args.append(k)
        else:
            msg = "%s argument type unsupported in jitclass" % v.kind
            raise errors.UnsupportedError(msg)
    return args


class JitClassType(type):
    """
    The type of any jitclass.
    """
    def __new__(cls, name, bases, dct):
        if len(bases) != 1:
            raise TypeError("must have exactly one base class")
        [base] = bases
        if isinstance(base, JitClassType):
            raise TypeError("cannot subclass from a jitclass")
        assert 'class_type' in dct, 'missing "class_type" attr'
        outcls = type.__new__(cls, name, bases, dct)
        outcls._set_init()
        return outcls

    def _set_init(cls):
        """
        Generate a wrapper for calling the constructor from pure Python.
        Note the wrapper will only accept positional arguments.
        """
        init = cls.class_type.instance_type.methods['__init__']
        init_sig = utils.pysignature(init)
        # get postitional and keyword arguments
        # offset by one to exclude the `self` arg
        args = _getargs(init_sig)[1:]
        cls._ctor_sig = init_sig
        ctor_source = _ctor_template.format(args=', '.join(args))
        glbls = {"__numba_cls_": cls}
        exec_(ctor_source, glbls)
        ctor = glbls['ctor']
        cls._ctor = njit(ctor)

    def __instancecheck__(cls, instance):
        if isinstance(instance, _box.Box):
            return instance._numba_type_.class_type is cls.class_type
        return False

    def __call__(cls, *args, **kwargs):
        # The first argument of _ctor_sig is `cls`, which here
        # is bound to None and then skipped when invoking the constructor.
        bind = cls._ctor_sig.bind(None, *args, **kwargs)
        bind.apply_defaults()
        return cls._ctor(*bind.args[1:], **bind.kwargs)


##############################################################################
# Registration utils

def _validate_spec(spec):
    for k, v in spec.items():
        if not isinstance(k, str):
            raise TypeError("spec keys should be strings, got %r" % (k,))
        if not isinstance(v, types.Type):
            raise TypeError("spec values should be Numba type instances, got %r"
                            % (v,))


def _fix_up_private_attr(clsname, spec):
    """
    Apply the same changes to dunder names as CPython would.
    """
    out = OrderedDict()
    for k, v in spec.items():
        if k.startswith('__') and not k.endswith('__'):
            k = '_' + clsname + k
        out[k] = v
    return out


def _add_linking_libs(context, call):
    """
    Add the required libs for the callable to allow inlining.
    """
    libs = getattr(call, "libs", ())
    if libs:
        context.add_linking_libs(libs)


def register_class_type(cls, spec, class_ctor, builder):
    """
    Internal function to create a jitclass.

    Args
    ----
    cls: the original class object (used as the prototype)
    spec: the structural specification contains the field types.
    class_ctor: the numba type to represent the jitclass
    builder: the internal jitclass builder
    """
    # Normalize spec
    if isinstance(spec, Sequence):
        spec = OrderedDict(spec)
    _validate_spec(spec)

    # Fix up private attribute names
    spec = _fix_up_private_attr(cls.__name__, spec)

    # Copy methods from base classes
    clsdct = {}
    for basecls in reversed(inspect.getmro(cls)):
        clsdct.update(basecls.__dict__)

    methods = dict((k, v) for k, v in clsdct.items()
                   if isinstance(v, pytypes.FunctionType))
    props = dict((k, v) for k, v in clsdct.items()
                 if isinstance(v, property))

    others = dict((k, v) for k, v in clsdct.items()
                  if k not in methods and k not in props)

    # Check for name shadowing
    shadowed = (set(methods) | set(props)) & set(spec)
    if shadowed:
        raise NameError("name shadowing: {0}".format(', '.join(shadowed)))

    docstring = others.pop('__doc__', "")
    _drop_ignored_attrs(others)
    if others:
        msg = "class members are not yet supported: {0}"
        members = ', '.join(others.keys())
        raise TypeError(msg.format(members))

    for k, v in props.items():
        if v.fdel is not None:
            raise TypeError("deleter is not supported: {0}".format(k))

    jitmethods = {}
    for k, v in methods.items():
        jitmethods[k] = njit(v)

    jitprops = {}
    for k, v in props.items():
        dct = {}
        if v.fget:
            dct['get'] = njit(v.fget)
        if v.fset:
            dct['set'] = njit(v.fset)
        jitprops[k] = dct

    # Instantiate class type
    class_type = class_ctor(cls, ConstructorTemplate, spec, jitmethods,
                            jitprops)

    cls = JitClassType(cls.__name__, (cls,), dict(class_type=class_type,
                                                  __doc__=docstring))

    # Register resolution of the class object
    typingctx = cpu_target.typing_context
    typingctx.insert_global(cls, class_type)

    # Register class
    targetctx = cpu_target.target_context
    builder(class_type, methods, typingctx, targetctx).register()

    return cls


class ConstructorTemplate(templates.AbstractTemplate):
    """
    Base class for jitclass constructor templates.
    """

    def generic(self, args, kws):
        # Redirect resolution to __init__
        instance_type = self.key.instance_type
        ctor = instance_type.jitmethods['__init__']
        boundargs = (instance_type.get_reference_type(),) + args
        disp_type = types.Dispatcher(ctor)
        sig = disp_type.get_call_type(self.context, boundargs, kws)
        # Actual constructor returns an instance value (not None)
        out = templates.signature(instance_type, *sig.args[1:])
        return out


def _drop_ignored_attrs(dct):
    # ignore anything defined by object
    drop = set(['__weakref__',
                '__module__',
                '__dict__'])
    for k, v in dct.items():
        if isinstance(v, (pytypes.BuiltinFunctionType,
                          pytypes.BuiltinMethodType)):
            drop.add(k)
        elif getattr(v, '__objclass__', None) is object:
            drop.add(k)

    for k in drop:
        del dct[k]


class ClassBuilder(object):
    """
    A jitclass builder for a mutable jitclass.  This will register
    typing and implementation hooks to the given typing and target contexts.
    """
    class_impl_registry = imputils.Registry()
    implemented_methods = set()

    def __init__(self, class_type, methods, typingctx, targetctx):
        self.class_type = class_type
        self.methods = methods
        self.typingctx = typingctx
        self.targetctx = targetctx

    def register(self):
        """
        Register to the frontend and backend.
        """
        # Register generic implementations for all jitclasses
        self._register_methods(self.class_impl_registry,
                               self.class_type.instance_type)
        # NOTE other registrations are done at the top-level
        # (see ctor_impl and attr_impl below)
        self.targetctx.install_registry(self.class_impl_registry)

    def _register_methods(self, registry, instance_type):
        """
        Register method implementations for the given instance type.
        """
        for meth in instance_type.jitmethods:

            # There's no way to retrive the particular method name
            # inside the implementation function, so we have to register a
            # specific closure for each different name
            if meth not in self.implemented_methods:
                self._implement_method(registry, meth)
                self.implemented_methods.add(meth)

    def _implement_method(self, registry, attr):
        # create a separate instance of imp method to avoid closure clashing
        def get_imp():
            def imp(context, builder, sig, args):
                instance_type = sig.args[0]
                method = instance_type.jitmethods[attr]
                disp_type = types.Dispatcher(method)
                call = context.get_function(disp_type, sig)
                out = call(builder, args)
                _add_linking_libs(context, call)
                return imputils.impl_ret_new_ref(context, builder,
                                                 sig.return_type, out)
            return imp

        def _getsetitem_gen(getset):
            _dunder_meth = "__%s__" % getset
            op = getattr(operator, getset)

            @templates.infer_global(op)
            class GetSetItem(templates.AbstractTemplate):
                def generic(self, args, kws):
                    instance = args[0]
                    if isinstance(instance, types.ClassInstanceType) and \
                            _dunder_meth in instance.jitmethods:
                        meth = instance.jitmethods[_dunder_meth]
                        disp_type = types.Dispatcher(meth)
                        sig = disp_type.get_call_type(self.context, args, kws)
                        return sig

            # lower both {g,s}etitem and __{g,s}etitem__ to catch the calls
            # from python and numba
            imputils.lower_builtin((types.ClassInstanceType, _dunder_meth),
                                   types.ClassInstanceType,
                                   types.VarArg(types.Any))(get_imp())
            imputils.lower_builtin(op,
                                   types.ClassInstanceType,
                                   types.VarArg(types.Any))(get_imp())

        dunder_stripped = attr.strip('_')
        if dunder_stripped in ("getitem", "setitem"):
            _getsetitem_gen(dunder_stripped)
        else:
            registry.lower((types.ClassInstanceType, attr),
                           types.ClassInstanceType,
                           types.VarArg(types.Any))(get_imp())


@templates.infer_getattr
class ClassAttribute(templates.AttributeTemplate):
    key = types.ClassInstanceType

    def generic_resolve(self, instance, attr):
        if attr in instance.struct:
            # It's a struct field => the type is well-known
            return instance.struct[attr]

        elif attr in instance.jitmethods:
            # It's a jitted method => typeinfer it
            meth = instance.jitmethods[attr]
            disp_type = types.Dispatcher(meth)

            class MethodTemplate(templates.AbstractTemplate):
                key = (self.key, attr)

                def generic(self, args, kws):
                    args = (instance,) + tuple(args)
                    sig = disp_type.get_call_type(self.context, args, kws)
                    return sig.as_method()

            return types.BoundFunction(MethodTemplate, instance)

        elif attr in instance.jitprops:
            # It's a jitted property => typeinfer its getter
            impdct = instance.jitprops[attr]
            getter = impdct['get']
            disp_type = types.Dispatcher(getter)
            sig = disp_type.get_call_type(self.context, (instance,), {})
            return sig.return_type


@ClassBuilder.class_impl_registry.lower_getattr_generic(types.ClassInstanceType)
def get_attr_impl(context, builder, typ, value, attr):
    """
    Generic getattr() for @jitclass instances.
    """
    if attr in typ.struct:
        # It's a struct field
        inst = context.make_helper(builder, typ, value=value)
        data_pointer = inst.data
        data = context.make_data_helper(builder, typ.get_data_type(),
                                        ref=data_pointer)
        return imputils.impl_ret_borrowed(context, builder,
                                          typ.struct[attr],
                                          getattr(data, _mangle_attr(attr)))
    elif attr in typ.jitprops:
        # It's a jitted property
        getter = typ.jitprops[attr]['get']
        sig = templates.signature(None, typ)
        dispatcher = types.Dispatcher(getter)
        sig = dispatcher.get_call_type(context.typing_context, [typ], {})
        call = context.get_function(dispatcher, sig)
        out = call(builder, [value])
        _add_linking_libs(context, call)
        return imputils.impl_ret_new_ref(context, builder, sig.return_type, out)

    raise NotImplementedError('attribute {0!r} not implemented'.format(attr))


@ClassBuilder.class_impl_registry.lower_setattr_generic(types.ClassInstanceType)
def set_attr_impl(context, builder, sig, args, attr):
    """
    Generic setattr() for @jitclass instances.
    """
    typ, valty = sig.args
    target, val = args

    if attr in typ.struct:
        # It's a struct member
        inst = context.make_helper(builder, typ, value=target)
        data_ptr = inst.data
        data = context.make_data_helper(builder, typ.get_data_type(),
                                        ref=data_ptr)

        # Get old value
        attr_type = typ.struct[attr]
        oldvalue = getattr(data, _mangle_attr(attr))

        # Store n
        setattr(data, _mangle_attr(attr), val)
        context.nrt.incref(builder, attr_type, val)

        # Delete old value
        context.nrt.decref(builder, attr_type, oldvalue)

    elif attr in typ.jitprops:
        # It's a jitted property
        setter = typ.jitprops[attr]['set']
        disp_type = types.Dispatcher(setter)
        sig = disp_type.get_call_type(context.typing_context,
                                      (typ, valty), {})
        call = context.get_function(disp_type, sig)
        call(builder, (target, val))
        _add_linking_libs(context, call)
    else:
        raise NotImplementedError(
            'attribute {0!r} not implemented'.format(attr))


def imp_dtor(context, module, instance_type):
    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = llvmir.FunctionType(llvmir.VoidType(),
                                     [llvoidptr, llsize, llvoidptr])

    fname = "_Dtor.{0}".format(instance_type.name)
    dtor_fn = module.get_or_insert_function(dtor_ftype,
                                            name=fname)
    if dtor_fn.is_declaration:
        # Define
        builder = llvmir.IRBuilder(dtor_fn.append_basic_block())

        alloc_fe_type = instance_type.get_data_type()
        alloc_type = context.get_value_type(alloc_fe_type)

        ptr = builder.bitcast(dtor_fn.args[0], alloc_type.as_pointer())
        data = context.make_helper(builder, alloc_fe_type, ref=ptr)

        context.nrt.decref(builder, alloc_fe_type, data._getvalue())

        builder.ret_void()

    return dtor_fn


@ClassBuilder.class_impl_registry.lower(types.ClassType,
                                        types.VarArg(types.Any))
def ctor_impl(context, builder, sig, args):
    """
    Generic constructor (__new__) for jitclasses.
    """
    # Allocate the instance
    inst_typ = sig.return_type
    alloc_type = context.get_data_type(inst_typ.get_data_type())
    alloc_size = context.get_abi_sizeof(alloc_type)

    meminfo = context.nrt.meminfo_alloc_dtor(
        builder,
        context.get_constant(types.uintp, alloc_size),
        imp_dtor(context, builder.module, inst_typ),
    )
    data_pointer = context.nrt.meminfo_data(builder, meminfo)
    data_pointer = builder.bitcast(data_pointer,
                                   alloc_type.as_pointer())

    # Nullify all data
    builder.store(cgutils.get_null_value(alloc_type),
                  data_pointer)

    inst_struct = context.make_helper(builder, inst_typ)
    inst_struct.meminfo = meminfo
    inst_struct.data = data_pointer

    # Call the jitted __init__
    # TODO: extract the following into a common util
    init_sig = (sig.return_type,) + sig.args

    init = inst_typ.jitmethods['__init__']
    disp_type = types.Dispatcher(init)
    call = context.get_function(disp_type, types.void(*init_sig))
    _add_linking_libs(context, call)
    realargs = [inst_struct._getvalue()] + list(args)
    call(builder, realargs)

    # Prepare return value
    ret = inst_struct._getvalue()

    return imputils.impl_ret_new_ref(context, builder, inst_typ, ret)
