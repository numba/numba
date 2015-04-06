from __future__ import absolute_import, print_function, division
import numpy

from .. import jit, typeof, utils, types, numpy_support
from ..typing import npydecl
from ..typing.templates import AbstractTemplate, signature
from ..targets import npyimpl
from . import _internal, ufuncbuilder

class DUFuncKernel(npyimpl._Kernel):
    dufunc = None

    def __init__(self, context, builder, outer_sig):
        super(DUFuncKernel, self).__init__(context, builder, outer_sig)
        self.inner_sig, self.cres = self.dufunc.find_ewise_function(
            outer_sig.args)

    def generate(self, *args):
        isig = self.inner_sig
        osig = self.outer_sig
        cast_args = [self.cast(val, inty, outty)
                     for val, inty, outty in zip(args, osig.args, isig.args)]
        if self.cres.objectmode:
            func_type = self.context.call_conv.get_function_type(
                types.pyobject, [types.pyobject] * len(isig.args))
        else:
            func_type = self.context.call_conv.get_function_type(
                isig.return_type, isig.args)
        module = self.builder.block.function.module
        entry_point = module.get_or_insert_function(
            func_type, name=self.cres.fndesc.llvm_func_name)
        entry_point.attributes.add("alwaysinline")
        _, res = self.context.call_conv.call_function(
            self.builder, entry_point, isig.return_type, isig.args, cast_args)
        return self.cast(res, isig.return_type, osig.return_type)

class DUFunc(_internal._DUFunc):
    # NOTE: __base_kwargs must be kept in synch with the kwlist in
    # _internal.c:dufunc_init()
    __base_kwargs = set(('identity', '_keepalive', 'nin', 'nout'))

    def __init__(self, py_func, **kws):
        dispatcher = jit(target='npyufunc')(py_func)
        self.targetoptions = {}
        # Loop over a copy of the keys instead of the keys themselves,
        # since we're changing the dictionary while looping.
        kws_keys = tuple(kws.keys())
        for key in kws_keys:
            if key not in self.__base_kwargs:
                self.targetoptions[key] = kws.pop(key)
        kws['identity'] = ufuncbuilder._BaseUFuncBuilder.parse_identity(
            kws.pop('identity', None))
        super(DUFunc, self).__init__(dispatcher, **kws)
        self._install_type()
        self._install_cg()

    def _compile_for_args(self, *args, **kws):
        nin = self.ufunc.nin
        args_len = len(args)
        assert (args_len == nin) or (args_len == nin + self.ufunc.nout)
        assert not kws
        argtys = []
        # To avoid a mismatch in how Numba types values as opposed to
        # Numpy, we need to first check for scalars.  For example, on
        # 64-bit systems, numba.typeof(3) => int32, but
        # numpy.array(3).dtype => int64.
        for arg in args[:nin]:
            if numpy_support.is_arrayscalar(arg):
                argtys.append(numpy_support.map_arrayscalar_type(arg))
            else:
                argty = typeof(arg)
                if isinstance(argty, types.Array):
                    argty = argty.dtype
                argtys.append(argty)
        return self._compile_for_argtys(tuple(argtys))

    def _compile_for_argtys(self, argtys):
        """Given a tuple of argument types (these should be the array
        dtypes, and not the array types themselves), compile the
        element-wise function for those inputs, generate a UFunc loop
        wrapper, and register the loop with the Numpy ufunc object for
        this DUFunc.
        """
        cres, argtys, return_type = ufuncbuilder._compile_element_wise_function(
            self._dispatcher, self.targetoptions, argtys)
        actual_sig = ufuncbuilder._finalize_ufunc_signature(
            cres, argtys, return_type)
        dtypenums, ptr, env = ufuncbuilder._build_element_wise_ufunc_wrapper(
            cres, actual_sig)
        self._add_loop(utils.longint(ptr), dtypenums)
        self._keepalive.append((ptr, cres.library, env))

    def _install_type(self, typingctx=None):
        """Constructs and installs a typing class for a DUFunc object in the
        input typing context.  If no typing context is given, then
        _install_type() installs into the typing context of the
        dispatcher object (should be same default context used by
        jit() and njit()).
        """
        if typingctx is None:
            typingctx = self.dispatcher.targetdescr.typing_context
        _ty_cls = type('DUFuncTyping_' + self.ufunc.__name__,
                       (AbstractTemplate,),
                       dict(key=self, generic=self._type_me))
        typingctx.insert_user_function(self, _ty_cls)

    def find_ewise_function(self, ewise_types):
        """Given a tuple of element-wise argument types, find a matching
        signature in the dispatcher.

        Returns a 2-tuple containing the matching signature, and
        compilation result.  Will return two None's if no matching
        signature was found.
        """
        for sig, cres in self.dispatcher.overloads.items():
            if sig.args == ewise_types:
                return sig, cres
        return None, None

    def _type_me(self, argtys, kwtys):
        """Overloads (defines) AbstractTemplate.generic() for the typing class
        built by DUFunc._install_type().

        Returns the call-site signature after either validating the
        element-wise signature or compiling for it.
        """
        assert not kwtys
        ufunc = self.ufunc
        _handle_inputs_result = npydecl.Numpy_rules_ufunc._handle_inputs(
            ufunc, argtys, kwtys)
        base_types, explicit_outputs, ndims = _handle_inputs_result
        ewise_types = tuple(base_types[:-len(explicit_outputs)])
        sig, cres = self.find_ewise_function(ewise_types)
        if sig is None:
            # Matching element-wise signature was not found; must
            # compile.
            self._compile_for_argtys(ewise_types)
            sig, cres = self.find_ewise_function(ewise_types)
            assert sig is not None
        if explicit_outputs:
            outtys = list(explicit_outputs)
        elif ufunc.nout == 1:
            # XXX What does Numpy do about memory layout when not
            # given an explicit output array?
            outtys = [types.Array(sig.return_type, ndims, 'A')]
        else:
            raise NotImplementedError("typing gufuncs (nout > 1)")
        outtys.extend(argtys)
        return signature(*outtys)

    def _install_cg(self, targetctx=None):
        """Constructs and installs a typing class for a DUFunc object in the
        input typing context.  If no typing context is given, then
        _install_type() installs into the typing context of the
        dispatcher object (should be same default context used by
        jit() and njit()).
        """
        if targetctx is None:
            targetctx = self.dispatcher.targetdescr.target_context
        _any = types.Any
        _arr = types.Kind(types.Array)
        sig0 = _any(*((_any,) * self.ufunc.nin + (_arr,) * self.ufunc.nout))
        sig1 = _any(*((_any,) * self.ufunc.nin))
        targetctx.insert_func_defn([(self._lower_me, [
            (self, sig0),
            (self, sig1),
        ])])

    def _lower_me(self, context, builder, sig, args):
        kernel = type('DUFuncKernel_' + self.ufunc.__name__,
                      (DUFuncKernel,),
                      dict(dufunc=self))
        return npyimpl.numpy_ufunc_kernel(context, builder, sig, args, kernel)
