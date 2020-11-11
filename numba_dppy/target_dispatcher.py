from numba.core import registry, serialize, dispatcher
from numba import types
from numba.core.errors import UnsupportedError
import dpctl
from numba.core.compiler_lock import global_compiler_lock


class TargetDispatcher(serialize.ReduceMixin, metaclass=dispatcher.DispatcherMeta):
    __numba__ = 'py_func'

    target_offload_gpu = '__dppl_offload_gpu__'
    target_offload_cpu = '__dppl_offload_cpu__'
    target_dppl = 'dppl'

    def __init__(self, py_func, wrapper, target, parallel_options, compiled=None):

        self.__py_func = py_func
        self.__target = target
        self.__wrapper = wrapper
        self.__compiled = compiled if compiled is not None else {}
        self.__parallel = parallel_options
        self.__doc__ = py_func.__doc__
        self.__name__ = py_func.__name__
        self.__module__ = py_func.__module__

    def __call__(self, *args, **kwargs):
        return self.get_compiled()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.get_compiled(), name)

    def __get__(self, obj, objtype=None):
        return self.get_compiled().__get__(obj, objtype)

    def __repr__(self):
        return self.get_compiled().__repr__()

    @classmethod
    def _rebuild(cls, py_func, wrapper, target, parallel, compiled):
        self = cls(py_func, wrapper, target, parallel, compiled)
        return self

    def get_compiled(self, target=None):
        if target is None:
            target = self.__target

        disp = self.get_current_disp()
        if not disp in self.__compiled.keys():
            with global_compiler_lock:
                if not disp in self.__compiled.keys():
                    self.__compiled[disp] = self.__wrapper(self.__py_func, disp)

        return self.__compiled[disp]

    def __is_with_context_target(self, target):
        return target is None or target == TargetDispatcher.target_dppl

    def get_current_disp(self):
        target = self.__target
        parallel = self.__parallel
        offload = isinstance(parallel, dict) and parallel.get('offload') is True

        if (dpctl.is_in_device_context() or offload):
            if not self.__is_with_context_target(target):
                raise UnsupportedError(f"Can't use 'with' context with explicitly specified target '{target}'")
            if parallel is False or (isinstance(parallel, dict) and parallel.get('offload') is False):
                raise UnsupportedError(f"Can't use 'with' context with parallel option '{parallel}'")

            from numba.dppl import dppl_offload_dispatcher

            if target is None:
                if dpctl.get_current_device_type() == dpctl.device_type.gpu:
                    return registry.dispatcher_registry[TargetDispatcher.target_offload_gpu]
                elif dpctl.get_current_device_type() == dpctl.device_type.cpu:
                    return registry.dispatcher_registry[TargetDispatcher.target_offload_cpu]
                else:
                    if dpctl.is_in_device_context():
                        raise UnsupportedError('Unknown dppl device type')
                    if offload:
                        if dpctl.has_gpu_queues():
                            return registry.dispatcher_registry[TargetDispatcher.target_offload_gpu]
                        elif dpctl.has_cpu_queues():
                            return registry.dispatcher_registry[TargetDispatcher.target_offload_cpu]

        if target is None:
            target = 'cpu'

        return registry.dispatcher_registry[target]

    def _reduce_states(self):
        return dict(
            py_func=self.__py_func,
            wrapper=self.__wrapper,
            target=self.__target,
            parallel=self.__parallel,
            compiled=self.__compiled
        )
