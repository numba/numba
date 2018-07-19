"""
Implementation of compiled C callbacks (@cfunc).
"""

from __future__ import print_function, division, absolute_import

import ctypes

from copy import deepcopy

from llvmlite import ir

import numba.types as types

from . import config, sigutils, utils, compiler
from .caching import NullCache, FunctionCache
from .dispatcher import _FunctionCompiler
from .targets import registry
from .typing import signature
from .typing.ctypes_utils import to_ctypes


MINIMAL_ABI_TUPLE_SIZE = 16  # bytes
MAX_ABI_COERCION_SIZE = 8  # bytes

class _CFuncCompiler(_FunctionCompiler):

    def _customize_flags(self, flags):
        flags.set('no_cpython_wrapper', True)
        # Disable compilation of the IR module, because we first want to
        # add the cfunc wrapper.
        flags.set('no_compile', True)
        # Object mode is not currently supported in C callbacks
        # (no reliable way to get the environment)
        flags.set('enable_pyobject', False)
        if flags.force_pyobject:
            raise NotImplementedError("object mode not allowed in C callbacks")
        return flags


class CFunc(object):
    """
    A compiled C callback, as created by the @cfunc decorator.

    Produced LLVM IR function signature adheres amd64 ABI and corresponds to
    a LLVM IR which would be produced by clang compiler for given C signature

    amd64 ABI:
    1) return values larger than MINIMAL_ABI_TUPLE_SIZE
    must be written into memory location supplied by the caller
    in ret_ptr*
    2) Structure arguments larger than MINIMAL_TYPE_SIZE must be passed by
    pointer
    3.1) Structure arguments smaller than MINIMAL_TYPE_SIZE and which can be
    aligned by the largrest field without padding, must be passed flat
    3.2) Structure arguments smaller than MINIMAL_TYPE_SIZE which cannot be aligned
    without padding, must be passed by pointer
    4) Small integers are coerced and merged into larger integers
    5) Two floats are merged in a vector.
    5) Struct fields are aligned to the largest field size
    6) Boolean fields are always at least 1 byte inside a struct
    """
    _targetdescr = registry.cpu_target

    def __init__(self, pyfunc, sig, locals, options,
                 pipeline_class=compiler.Pipeline):
        args, return_type = sig
        if return_type is None:
            raise TypeError("C callback needs an explicit return type")
        self.__name__ = pyfunc.__name__
        self.__qualname__ = getattr(pyfunc, '__qualname__', self.__name__)
        self.__wrapped__ = pyfunc

        self._pyfunc = pyfunc
        self._sig = signature(return_type, *args)
        self._compiler = _CFuncCompiler(pyfunc, self._targetdescr,
                                        options, locals,
                                        pipeline_class=pipeline_class)

        self._wrapper_name = None
        self._wrapper_address = None
        self._cache = NullCache()
        self._cache_hits = 0

        self.return_by_ptr = False

        self._sig.args = tuple(map(_replace_boolean, self._sig.args))
        self._sig.return_type = _replace_boolean(self._sig.return_type)

        self.wrapper_sig = deepcopy(self._sig)
        # compute wrapper signature
        self._compute_wrapper_signature()

    def _compute_wrapper_signature(self):
        self._compute_arguments()
        self._compute_return_type()
        self._adjust_signature()

    def _compute_arguments(self):
        new_args = []
        args = self._sig.args
        for arg in args:
            if isinstance(arg, types.Tuple):
                if _get_llvm_type_size(arg) > MINIMAL_ABI_TUPLE_SIZE:
                    # Replace big structs by pointers
                    new_arg = types.CPointer(arg)
                    new_args.append(new_arg)
                else:
                    # Unnest small structs
                    args_ = _flatten(arg)
                    new_args2 = _merge_tuple_args(args_)
                    if len(new_args2) > 2:
                        # should not unnest into more than 2 args
                        new_arg = types.CPointer(arg)
                        new_args.append(new_arg)
                    else:
                        new_args += new_args2
            else:
                new_args.append(arg)
        self.wrapper_sig.args = new_args

    def _compute_return_type(self):
        retty = self.wrapper_sig.return_type
        new_retty = retty
        if isinstance(retty, types.Tuple):
            if _get_llvm_type_size(retty) > MINIMAL_ABI_TUPLE_SIZE:
                # Replace big struct by pointer
                new_retty = types.CPointer(retty)
                self.return_by_ptr = True
            else:
                # Unnest small structs
                args_ = _flatten(retty)
                new_args2 = _merge_tuple_args(args_)
                # return one argument flat
                if len(new_args2) == 1:
                    new_retty = new_args2[0]
                # if the coercion was successful, make a new tuple  with
                # coerced fields and pass by value
                elif list(args_) != list(new_args2) or len(args_) == 2:
                    new_retty = make_tuple(new_args2)
                else:
                    # pass the original tuple by ptr
                    new_retty = types.CPointer(retty)
                    self.return_by_ptr = True

        self.wrapper_sig.return_type = new_retty

    def _adjust_signature(self):
        if self.return_by_ptr:
            # add retty as one of the arguments
            # return void
            retty = self.wrapper_sig.return_type
            assert isinstance(retty, types.CPointer)
            self.wrapper_sig.args.insert(0, retty)
            self.wrapper_sig.return_type = types.CVoid()

    def enable_caching(self):
        self._cache = FunctionCache(self._pyfunc)

    def compile(self):
        # Use cache and compiler in a critical section
        with compiler.lock_compiler:
            # Try to load from cache
            cres = self._cache.load_overload(self._sig, self._targetdescr.target_context)
            if cres is None:
                cres = self._compile_uncached()
                self._cache.save_overload(self._sig, cres)
            else:
                self._cache_hits += 1

            self._library = cres.library
            self._wrapper_name = cres.fndesc.llvm_cfunc_wrapper_name
            self._wrapper_address = self._library.get_pointer_to_function(self._wrapper_name)

    def _compile_uncached(self):
        sig = self._sig

        # Compile native function
        cres = self._compiler.compile(sig.args, sig.return_type)
        assert not cres.objectmode  # disabled by compiler above
        fndesc = cres.fndesc

        # Compile C wrapper
        # Note we reuse the same library to allow inlining the Numba
        # function inside the wrapper.
        library = cres.library
        module = library.create_ir_module(fndesc.unique_name)
        context = cres.target_context
        wrap_sig = self.wrapper_sig
        ll_argtypes = [context.get_argument_type(ty) for ty in wrap_sig.args]
        ll_return_type = context.get_value_type(wrap_sig.return_type)

        wrapty = ir.FunctionType(ll_return_type, ll_argtypes)
        wrapfn = module.add_function(wrapty, fndesc.llvm_cfunc_wrapper_name)
        builder = ir.IRBuilder(wrapfn.append_basic_block('entry'))

        self._build_c_wrapper(context, builder, cres, wrapfn.args)

        library.add_ir_module(module)
        library.finalize()

        return cres

    def _build_c_wrapper(self, context, builder, cres, c_args):
        sig = self._sig
        pyapi = context.get_python_api(builder)
        if self.return_by_ptr:
            # omit the first argument which is the res ptr
            retptr = c_args[0]
            retptr.add_attribute("noalias")
            retptr.add_attribute("sret")
            c_args = c_args[1:]
        else:
            retptr = None
        context.get_python_api(builder)

        fnty = context.call_conv.get_function_type(sig.return_type, sig.args)
        function_ir = builder.module.add_function(
            fnty, cres.fndesc.llvm_func_name)

        status, out = self._generate_func_call(context, builder, function_ir, c_args)

        with builder.if_then(status.is_error, likely=False):
            # If (and only if) an error occurred, acquire the GIL
            # and use the interpreter to write out the exception.
            gil_state = pyapi.gil_ensure()
            context.call_conv.raise_error(builder, pyapi, status)
            cstr = context.insert_const_string(builder.module, repr(self))
            strobj = pyapi.string_from_string(cstr)
            pyapi.err_write_unraisable(strobj)
            pyapi.decref(strobj)
            pyapi.gil_release(gil_state)

        self._generate_return(builder, out, retptr, context)

    def _generate_func_call(self, context, builder, function_ir, c_args):
        sig = self._sig

        def _replace_with_pointer(c_arg):
            c_arg.add_attribute("byval")
            return builder.load(c_arg)

        new_cargs = []
        c_arg_ind = 0
        for arg_type in sig.args:
            if isinstance(arg_type, types.Tuple):
                if _get_llvm_type_size(arg_type) > MINIMAL_ABI_TUPLE_SIZE:
                    c_arg = c_args[c_arg_ind]
                    new_cargs.append(_replace_with_pointer(c_arg))
                    c_arg_ind += 1
                else:
                    args_ = _flatten(arg_type)
                    new_args2 = _merge_tuple_args(args_)
                    # should not unnest into more than 2 args
                    if len(new_args2) > 2:
                        c_arg = c_args[c_arg_ind]
                        new_cargs.append(_replace_with_pointer(c_arg))
                        c_arg_ind += 1
                    else:
                        # tuple has been unnested
                        # allocate a new tuple
                        # copy the arguments
                        allocated_tuple_ptr = builder.alloca(
                            context.get_value_type(arg_type))

                        shift_by = 0
                        for i in range(len(new_args2)):
                            c_arg = c_args[c_arg_ind]
                            shifted_ptr = builder.gep(allocated_tuple_ptr,
                                                      [ir.Constant(
                                                          ir.IntType(32),
                                                          shift_by)])
                            cast_ptr = builder.bitcast(shifted_ptr,
                                                       ir.PointerType(
                                                           c_arg.type))
                            builder.store(c_arg, cast_ptr)
                            size_sum = 0
                            target_field_size = _get_llvm_type_size(
                                new_args2[i])
                            for arg in args_[shift_by:]:
                                size_sum += _get_llvm_type_size(arg)
                                shift_by += 1
                                if size_sum >= target_field_size:
                                    break

                        c_arg_ind += 1
                        allocated_tuple = builder.load(allocated_tuple_ptr)
                        new_cargs.append(allocated_tuple)
            else:
                c_arg = c_args[c_arg_ind]
                new_cargs.append(c_arg)
                c_arg_ind += 1

        return context.call_conv.call_function(builder, function_ir,
                                               sig.return_type, sig.args,
                                               new_cargs)

    def _generate_return(self, builder, out_val, retptr, context):
        if retptr:
            builder.store(out_val, retptr)
            builder.ret_void()
        else:
            # cast out val to the function return type
            numba_retty = context.get_value_type(self._sig.return_type)
            c_retty = context.get_value_type(
                self.wrapper_sig.return_type)
            allocated_ret_ptr = builder.alloca(numba_retty)
            builder.store(out_val, allocated_ret_ptr)
            res_ptr = builder.bitcast(allocated_ret_ptr,
                                      ir.PointerType(c_retty))
            ret = builder.load(res_ptr)
            builder.ret(ret)



    @property
    def native_name(self):
        """
        The process-wide symbol the C callback is exposed as.
        """
        # Note from our point of view, the C callback is the wrapper around
        # the native function.
        return self._wrapper_name

    @property
    def address(self):
        """
        The address of the C callback.
        """
        return self._wrapper_address

    @utils.cached_property
    def cffi(self):
        """
        A cffi function pointer representing the C callback.
        """
        import cffi
        ffi = cffi.FFI()
        # cffi compares types by name, so using precise types would risk
        # spurious mismatches (such as "int32_t" vs. "int").
        return ffi.cast("void *", self.address)

    @utils.cached_property
    def ctypes(self):
        """
        A ctypes function object representing the C callback.
        """
        ctypes_args = [to_ctypes(ty) for ty in self._sig.args]
        ctypes_restype = to_ctypes(self._sig.return_type)
        functype = ctypes.CFUNCTYPE(ctypes_restype, *ctypes_args)
        return functype(self.address)

    def inspect_llvm(self):
        """
        Return the LLVM IR of the C callback definition.
        """
        return self._library.get_llvm_str()

    @property
    def cache_hits(self):
        return self._cache_hits

    def __repr__(self):
        return "<Numba C callback %r>" % (self.__qualname__,)


def _get_llvm_type_size(type_):
    # compute size of a type inside a tuple
    try:
        return int(type_.bitwidth / 8)
    except AttributeError:
        pass
    if isinstance(type_, types.Boolean):
        return 1
    if isinstance(type_, types.Tuple):
        return _get_llvm_type_size(type_.types)
    elif isinstance(type_, types.UniTuple):
        return _get_llvm_type_size(type_.dtype) * type_.count
    elif isinstance(type_, types.Record):
        return type_.size
    elif isinstance(type_, types.CPointer):
        return 8
    elif isinstance(type_, types.NoneType):
        return 0
    try:
        return sum([_get_llvm_type_size(ty) for ty in type_])
    except TypeError:
        pass
    assert False, "Cannot compute size of " + type_.name



def _merge_tuple_args(args_):
    new_args = []
    to_merge = []
    # Merge small scalar arguments
    for arg_ in args_:
        arg_size = _get_llvm_type_size(arg_)
        if arg_size < MAX_ABI_COERCION_SIZE:
            if _get_llvm_type_size(to_merge) + \
                    arg_size <= MAX_ABI_COERCION_SIZE:
                to_merge.append(arg_)
            else:
                _append_not_none(new_args, _coerce_types(to_merge, True))
                to_merge = [arg_]
        else:
            _append_not_none(new_args, _coerce_types(to_merge, True))
            to_merge = []
            new_args.append(arg_)

    align_64 = False
    if any(map(lambda a: _get_llvm_type_size(a) >= MAX_ABI_COERCION_SIZE,
               args_)):
        align_64 = True
    if to_merge:
        _append_not_none(new_args, _coerce_types(to_merge, align_64))
    return new_args


def _coerce_types(to_merge, align=False):
    size = _get_llvm_type_size(to_merge)
    if size == 0:
        return
    if len(to_merge) == 1:
        return to_merge[0]
    if len(to_merge) == 2:
        if all(map(lambda ty: isinstance(ty, types.Float), to_merge)):
            # merge float32, float32 to <2 x float>
            return types.Vector(types.float32, 2)
    if align:
        # align by the biggest primitive byte size
        return types.int64

    # int32 forces 32 bit alignment
    if types.int32 in to_merge:
        return types.int64
    # int16 forces 16 bit alignment
    if types.int16 in to_merge:
        # here we know that all types are <= 16 bits
        size = len(to_merge) * 2
    type_str = "int" + str(size * 8)

    return types.Integer(type_str)


def _append_not_none(list_, el):
    if el is not None:
        list_.append(el)


def _replace_boolean(typ):
    # structs should be 8 bit aligned
    # replace i1 with i8
    def _rec(typ_):
        assert isinstance(typ_, types.Type)
        if isinstance(typ_, types.Tuple):
            child_types = [_rec(t) for t in typ_.types]
            return make_tuple(child_types)
        if isinstance(typ_, types.Boolean):
            return types.int8
        return typ_

    if not isinstance(typ, types.Tuple):
        return typ
    return _rec(typ)


def _flatten(iterable_):
    """
    Flatten nested iterable of (tuple, list).
    """

    def rec(iterable__):
        for i in iterable__:
            if isinstance(i, (tuple, list)):
                for j in rec(i):
                    yield j
            elif isinstance(i, (types.UniTuple, types.Tuple)):
                for j in rec(i.types):
                    yield j
            else:
                yield i

    if not isinstance(iterable_,
                      (tuple, list, types.UniTuple, types.Tuple)):
        return iterable_,
    return tuple(rec(iterable_))


def make_tuple(child_types):
    out = types.Tuple([])
    out.types = tuple(child_types)
    out.name = "(%s)" % ', '.join(str(i) for i in child_types)
    out.count = len(child_types)
    return out
