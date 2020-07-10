from numba.core import dispatcher, compiler
from numba.core.registry import cpu_target, dispatcher_registry


class CPUDispatcher(dispatcher.Dispatcher):
    targetdescr = cpu_target

    def __init__(self, py_func, locals={}, targetoptions={}, impl_kind='direct', pipeline_class=compiler.Compiler):
        if ('parallel' in targetoptions and isinstance(targetoptions['parallel'], dict) and
                'offload' in targetoptions['parallel'] and  targetoptions['parallel']['offload'] == True):
            import numba.dppl_config as dppl_config
            if dppl_config.dppl_present:
                from numba.dppl.compiler import DPPLCompiler
                dispatcher.Dispatcher.__init__(self, py_func, locals=locals,
                        targetoptions=targetoptions, impl_kind=impl_kind, pipeline_class=DPPLCompiler)
            else:
                print("---------------------------------------------------------------------------")
                print("WARNING : offload=True option ignored. Ensure OpenCL drivers are installed.")
                print("---------------------------------------------------------------------------")
                dispatcher.Dispatcher.__init__(self, py_func, locals=locals,
                    targetoptions=targetoptions, impl_kind=impl_kind, pipeline_class=pipeline_class)
        else:
            dispatcher.Dispatcher.__init__(self, py_func, locals=locals,
                targetoptions=targetoptions, impl_kind=impl_kind, pipeline_class=pipeline_class)


dispatcher_registry['cpu'] = CPUDispatcher
