"""
Target Options
"""

from numba.core import config, utils


class TargetOptions:
    OPTIONS = {}

    def __init__(self):
        self.values = {}

    def from_dict(self, dic):
        for k, v in dic.items():
            try:
                ctor = self.OPTIONS[k]
            except KeyError:
                fmt = "%r does not support option: '%s'"
                raise KeyError(fmt % (self.__class__, k))
            else:
                self.values[k] = ctor(v)

    @classmethod
    def parse_as_flags(cls, flags, options):
        opt = cls()
        opt.from_dict(options)
        opt.set_flags(flags)
        return flags

    def set_flags(self, flags):
        """
        Provide default flags setting logic.
        Subclass can override.
        """
        from numba.core.targetconfig import TargetConfig
        if isinstance(flags, TargetConfig):
            return self._set_flags_new(flags)
        else:
            raise NotImplementedError(type(flags))

    def _set_flags_new(self, flags):
        kws = self.values.copy()

        if kws.pop('nopython', False) == False:
            flags.enable_pyobject = True

        if kws.pop("forceobj", False):
            flags.force_pyobject = True

        if kws.pop('looplift', True):
            flags.enable_looplift = True

        cstk = utils.ConfigStack()
        if "_nrt" in kws:
            flags.nrt = kws.pop("_nrt")
        elif cstk:
            top = cstk.top()
            if top.is_set("nrt"):
                flags.nrt = top.nrt
        else:
            flags.nrt = True


        debug_mode = kws.pop('debug', config.DEBUGINFO_DEFAULT)

        # boundscheck is supplied
        if 'boundscheck' in kws:
            boundscheck = kws.pop("boundscheck")
            if boundscheck is None and debug_mode:
                # if it's None and debug is on then set it
                flags.boundscheck = True
            else:
                # irrespective of debug set it to the requested value
                flags.boundscheck = boundscheck
        else:
            # no boundscheck given, if debug mode, set it
            if debug_mode:
                flags.boundscheck = True

        if debug_mode:
            flags.debuginfo = True


        if kws.pop('nogil', False):
            flags.release_gil = True

        if kws.pop('no_rewrites', False):
            flags.no_rewrites = True

        if kws.pop('no_cpython_wrapper', False):
            flags.no_cpython_wrapper = True

        if kws.pop('no_cfunc_wrapper', False):
            flags.no_cfunc_wrapper = True

        if 'parallel' in kws:
            flags.auto_parallel = kws.pop('parallel')

        if 'fastmath' in kws:
            flags.fastmath = kws.pop('fastmath')

        if 'error_model' in kws:
            flags.error_model = kws.pop('error_model')

        if 'inline' in kws:
            flags.inline = kws.pop('inline')

        flags.enable_pyobject_looplift = True

        if kws:
            # Unread options?
            raise NameError("Unrecognized options: %s" % kws.keys())
