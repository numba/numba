from numba.core import dispatcher, compiler
from numba.core.registry import cpu_target, dispatcher_registry
from numba.dppy.compiler import DPPyCompiler


class CPUDispatcher(dispatcher.Dispatcher):
    targetdescr = cpu_target

    def __init__(self, py_func, locals={}, targetoptions={}, impl_kind='direct', pipeline_class=compiler.Compiler):
        if ('parallel' in targetoptions and isinstance(targetoptions['parallel'], dict) and
                'spirv' in targetoptions['parallel'] and  targetoptions['parallel']['spirv'] == True):
            dispatcher.Dispatcher.__init__(self, py_func, locals=locals,
                    targetoptions=targetoptions, impl_kind=impl_kind, pipeline_class=DPPyCompiler)
        else:
            dispatcher.Dispatcher.__init__(self, py_func, locals=locals,
                targetoptions=targetoptions, impl_kind=impl_kind, pipeline_class=pipeline_class)


dispatcher_registry['cpu'] = CPUDispatcher
