from numba.core import dispatcher, compiler
from numba.core.registry import cpu_target, dispatcher_registry


class CPUDispatcher(dispatcher.Dispatcher):
    targetdescr = cpu_target

    def __init__(self, py_func, locals={}, targetoptions={}, impl_kind='direct', pipeline_class=compiler.Compiler):
        dispatcher.Dispatcher.__init__(self, py_func, locals=locals,
            targetoptions=targetoptions, impl_kind=impl_kind, pipeline_class=pipeline_class)


dispatcher_registry['cpu'] = CPUDispatcher
