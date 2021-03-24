"""
Target Options
"""
import operator

from numba.core import config, utils
from numba.core.targetconfig import TargetConfig, Option


class TargetOptions:
    class Mapping:
        def __init__(self, flag_name, apply=lambda x: x):
            self.flag_name = flag_name
            self.apply = apply

    def finalize(self, settings, flags):
        pass

    @classmethod
    def parse_as_flags(cls, flags, options):
        opt = cls()
        # try:
        opt.apply(flags, options)
        opt.finalize(flags, options)
        # except:
        #     raise TypeError(f"failed in {opt}")
        # finalized Flags(enable_looplift=True, enable_pyobject=True, enable_pyobject_looplift=True, boundscheck=None, nrt=True)
        # print('finalized', flags.summary())
        return flags

    def apply(self, flags, options):
        # Find all Mapping instances in the class
        mappings = {}
        cls = type(self)
        for k in dir(cls):
            v = getattr(cls, k)
            if isinstance(v, cls.Mapping):
                mappings[k] = v

        used = set()
        for k, mapping in mappings.items():
            if k in options:
                v = mapping.apply(options[k])
                setattr(flags, mapping.flag_name, v)
                used.add(k)

        unused = set(options) - used
        if unused:
            # Unread options?
            print(mappings)
            print(self.__dict__)
            raise NameError(f"Unrecognized options: {unused}. Known options are {mappings.keys()}")


_mapping = TargetOptions.Mapping


class DefaultOptions:
    nopython = _mapping("enable_pyobject", operator.not_)
    forceobj = _mapping("force_pyobject")
    looplift = _mapping("enable_looplift")
    _nrt = _mapping("nrt")
    debug = _mapping("debuginfo")
    boundscheck = _mapping("boundscheck")
    nogil = _mapping("release_gil")

    no_rewrites = _mapping("no_rewrites")
    no_cpython_wrapper = _mapping("no_cpython_wrapper")
    no_cfunc_wrapper = _mapping("no_cfunc_wrapper")

    parallel = _mapping("parallel")
    fastmath = _mapping("fastmath")
    error_model = _mapping("error_model")
    inline = _mapping("inline")


def include_default_options(*args):
    glbs = {k: getattr(DefaultOptions, k) for k in args}

    return type("OptionMixins", (), glbs)

