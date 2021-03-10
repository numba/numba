import types as pytypes
import textwrap


def stub_generator(nargs, glbls, kwargs=None):
    def stub(tyctx):
        # body is supplied when the function is magic'd into life via glbls
        return body(tyctx)  # noqa: F821
    if kwargs is None:
        kwargs = {}
    # create new code parts
    stub_code = stub.__code__
    co_args = [stub_code.co_argcount + nargs + len(kwargs)]

    new_varnames = [*stub_code.co_varnames]
    new_varnames.extend([f'tmp{x}' for x in range(nargs)])
    new_varnames.extend([x for x, _ in kwargs.items()])
    from numba.core import utils
    if utils.PYVERSION >= (3, 8):
        co_args.append(stub_code.co_posonlyargcount)
    co_args.append(stub_code.co_kwonlyargcount)
    co_args.extend([stub_code.co_nlocals + nargs + len(kwargs),
                    stub_code.co_stacksize,
                    stub_code.co_flags,
                    stub_code.co_code,
                    stub_code.co_consts,
                    stub_code.co_names,
                    tuple(new_varnames),
                    stub_code.co_filename,
                    stub_code.co_name,
                    stub_code.co_firstlineno,
                    stub_code.co_lnotab,
                    stub_code.co_freevars,
                    stub_code.co_cellvars
                    ])

    new_code = pytypes.CodeType(*co_args)

    # get function
    new_func = pytypes.FunctionType(new_code, glbls)
    return new_func


class OverloadWrapper(object):

    def __init__(self, function=None):
        assert function is not None
        self._function = function
        self._BIND_TYPES = dict()
        self._selector = None
        self._TYPER = None
        # run to register overload, the intrinsic sorts out the binding to the
        # registered impls at the point the overload is evaluated, i.e. this
        # is all lazy.
        self._build()

    def wrap_typing(self):
        """
        Use this to replace @infer_global, it records the decorated function
        as a typer for the argument `concrete_function`.
        """
        def inner(typing_class):
            # arg is the typing class
            self._TYPER = typing_class
            # HACK: This is a hack, infer_global maybe?
            self._TYPER.key = self._function
            return typing_class
        return inner

    def wrap_impl(self, *args):
        """
        Use this to replace @lower*, it records the decorated function as the
        lowering implementation
        """
        def inner(lowerer):
            self._BIND_TYPES[args] = lowerer
            return lowerer
        return inner

    def _assemble(self):
        """ Assembles the OverloadSelector definitions from the registered
        typing to lowering map.
        """
        from numba.core.base import OverloadSelector
        self._selector = OverloadSelector()
        msg = f"No entries in the typing->lowering map for {self._function}"
        assert self._BIND_TYPES, msg
        for sig, impl in self._BIND_TYPES.items():
            self._selector.append(impl, sig)

    def _build(self):
        from numba.core.extending import overload, intrinsic

        @overload(self._function, strict=False)
        def ol_generated(*ol_args, **ol_kwargs):

            def body(tyctx):
                msg = f"No typer registered for {self._function}"
                assert self._TYPER is not None, msg
                typing = self._TYPER(tyctx)
                sig = typing.apply(ol_args, ol_kwargs)
                if sig is None:
                    from numba.core import errors
                    # TODO: Something about this, it might be possible to
                    # fish out the actual arguments. Needs kwargs plugging
                    # in to the error message too.
                    err = ("No match. No implementation of %s found for "
                           "argument type(s) %s" % (self._function,
                                                    str(ol_args)))
                    raise errors.TypingError(err)
                if self._selector is None:
                    self._assemble()
                lowering = self._selector.find(sig.args)
                msg = (f"Could not find implementation to lower {sig} for ",
                       f"{self._function}")
                assert lowering is not None, msg
                return sig, lowering

            stub = stub_generator(len(ol_args), {'body': body}, ol_kwargs)
            intrin = intrinsic(stub)

            # This is horrible, need to generate a jit wrapper function that
            # walks the ol_kwargs into the intrin with a signature that
            # matches the lowering sig. The actual kwarg var names matter,
            # they have to match exactly.
            arg_str = ','.join([f'tmp{x}' for x in range(len(ol_args))])
            kws_str = ','.join(ol_kwargs.keys())
            call_str = ','.join([x for x in (arg_str, kws_str) if x])
            # NOTE: The jit_wrapper functions cannot take `*args`
            # albeit this an obvious choice for accepting an unknown number
            # of arguments. If this is done, `*args` ends up as a cascade of
            # Tuple assembling in the IR which ends up with literal
            # information being lost. As a result the _exact_ argument list
            # is generated to match the number of arguments and kwargs.
            name = str(self._function)
            # This is to name the function with something vaguely identifiable
            name = ''.join([x if x not in {'>','<',' ','-'} else '_'
                            for x in name])
            gen = textwrap.dedent(("""
            def jit_wrapper_{}({}):
                return intrin({})
            """)).format(name, call_str, call_str)
            l = {}
            g = {'intrin': intrin}
            exec(gen, g, l)
            return l['jit_wrapper_{}'.format(name)]


class Gluer():
    def __init__(self):
        self._registered = dict()

    def __call__(self, func):
        if func in self._registered:
            return self._registered[func]
        else:
            wrapper = OverloadWrapper(func)
            self._registered[func] = wrapper
            return wrapper


overload_glue = Gluer()
del Gluer


def glue_typing(*args):
    return overload_glue(args[0]).wrap_typing()


def glue_lowering(*args):
    return overload_glue(args[0]).wrap_impl(*args[1:])
