from numba import environment as _env
from numba import pipeline as _nb_pipeline
from numba.utils import WriteOnceTypedProperty
from . import pipeline as _cuda_pipeline

import logging
logger = logging.getLogger(__name__)

class PTXUtils(object):
    def __init__(self, env):
        self.__env = env

    @property
    def env(self):
        "The parent environment"
        return self.__env

    @property
    def device_functions(self):
        return self.env.device_functions

    def generate_ptx(self, module, kernels):
        from numbapro.cudapipeline import nvvm, driver
        context = driver.get_or_create_context()
        cc_major = context.device.COMPUTE_CAPABILITY[0]

        _hack_to_implement_pymodulo(module)

        for kernel in kernels:
            nvvm.set_cuda_kernel(kernel)
        _link_llvm_math_intrinsics(module, cc_major)

        arch = 'compute_%d0' % cc_major

        nvvm.fix_data_layout(module)

        # NOTE: It seems to be invalid to run passes on the LLVM for PTX.
        #       LLVM assumes it is CPU code and does the wrong kind of optimization.
        #    pmb = _lp.PassManagerBuilder.new()
        #    pmb.opt_level = 2 # O3 causes bar.sync to be duplicated in unrolled loop
        #    pm = _lp.PassManager.new()
        #    pmb.populate(pm)
        #    pm.run(module)

        ptx = nvvm.llvm_to_ptx(str(module), arch=arch)
        return ptx

    def link_device_function(self, lfunc):
        toinline = []

        for func in lfunc.module.functions:
            if func.uses:
                uselist = list(func.uses)
                if func.is_declaration: # declared and is used
                    bag = self.device_functions.get(func.name)
                    if bag is not None:
                        pyfunc, linkee, inline =bag
                        lfunc.module.link_in(linkee.module, preserve=True)
                        if inline:
                            toinline.extend(uselist)
        for call in toinline:
            callee = call.called_function
            _lc.inline_function(call)


class CudaEnvironment(_env.NumbaEnvironment):

    ptxutils = WriteOnceTypedProperty(PTXUtils,
                                      "PTX codegen utilities")

    device_functions = WriteOnceTypedProperty(dict,
                                          "A dictinoary of device functions")

    def __init__(self, name, *args, **kws):
        super(CudaEnvironment, self).__init__(name, *args, **kws)

        # add cbuilder_library
        assert not hasattr(self.context, 'cbuilder_library')
        numba_env = _env.NumbaEnvironment.get_environment()
        self.context.cbuilder_library = numba_env.context.cbuilder_library
        # setup pipeline
        orders = _cuda_pipeline.get_orders()
        composer = _nb_pipeline.ComposedPipelineStage
        self.pipelines.update({
            self.default_pipeline: composer(orders.default),
            'type_infer': composer(orders.type_infer)
        })

        self.ptxutils = PTXUtils(self)
        self.device_functions = {}

    def add_device_function(self, pyfunc, lfunc, inline):
        self.device_functions[lfunc.name] = pyfunc, lfunc, inline



#
# Internals
#
import re
from llvm import core as _lc
from llvm import ee as _le
from llvm import passes as _lp


regex_py_modulo = re.compile('__numba_specialized_\d+___py_modulo')

def _hack_to_implement_pymodulo(module):
    '''XXX: I should fix the linkage instead.
        '''
    for func in module.functions:
        if regex_py_modulo.match(func.name):
            assert func.is_declaration
            func.add_attribute(_lc.ATTR_ALWAYS_INLINE)
            func.linkage = _lc.LINKAGE_LINKONCE_ODR
            bb = func.append_basic_block('entry')
            b = _lc.Builder.new(bb)
            if func.type.pointee.return_type.kind == _lc.TYPE_INTEGER:
                rem = b.srem
            else:
                raise Exception("Does not support modulo of float-point number.")
            b.ret(rem(*func.args))
            del b
            del bb

def _generate_ptx(module, kernels):
    from numbapro.cudapipeline import nvvm, driver
    context = driver.get_or_create_context()
    cc_major = context.device.COMPUTE_CAPABILITY[0]

    _hack_to_implement_pymodulo(module)

    for kernel in kernels:
        _link_device_function(kernel)
        nvvm.set_cuda_kernel(kernel)
    _link_llvm_math_intrinsics(module, cc_major)

    arch = 'compute_%d0' % cc_major

    nvvm.fix_data_layout(module)

    # NOTE: It seems to be invalid to run passes on the LLVM for PTX.
    #       LLVM assumes it is CPU code and does the wrong kind of optimization.
    #    pmb = _lp.PassManagerBuilder.new()
    #    pmb.opt_level = 2 # O3 causes bar.sync to be duplicated in unrolled loop
    #    pm = _lp.PassManager.new()
    #    pmb.populate(pm)
    #    pm.run(module)

    ptx = nvvm.llvm_to_ptx(str(module), arch=arch)
    return ptx


from . import ptx

CUDA_MATH_INTRINSICS_2 = {
    'llvm.exp.f32': ptx.exp_f32,
    'llvm.exp.f64': ptx.exp_f64,
    'fabsf'       : ptx.fabs_f32, # libm
    'fabs'        : ptx.fabs_f64, # libm
    'llvm.fabs.f32': ptx.fabs_f32,
    'llvm.fabs.f64': ptx.fabs_f64,
    'llvm.log.f32': ptx.log_f32,
    'llvm.log.f64': ptx.log_f64,
    'llvm.pow.f32': ptx.pow_f32,
    'llvm.pow.f64': ptx.pow_f64,
}

CUDA_MATH_INTRINSICS_3 = CUDA_MATH_INTRINSICS_2.copy()
CUDA_MATH_INTRINSICS_3.update({
                              # intentionally empty
                              })


def _link_llvm_math_intrinsics(module, cc):
    '''Discover and implement llvm math intrinsics that are not supported
        by NVVM.  NVVM only supports llvm.sqrt at this point (11/1/2012).
        '''
    to_be_implemented = {}   # new-function object -> inline ptx object
    to_be_removed = set()    # functions to be deleted
    inlinelist = []          # functions to be inlined


    library_map = {
        2 : CUDA_MATH_INTRINSICS_2,
        3 : CUDA_MATH_INTRINSICS_3,
    }
    if cc not in library_map:
        raise Exception("NumbaPro does not support CC%d.x" % cc)
    library = library_map[cc]

    # find all known math intrinsics and implement them.
    for lfunc in module.functions:
        for instr in lfunc.uses:
            fn = instr.called_function
            if fn is not None: # maybe a inline asm
                fname = fn.name
                if fname in library:
                    inlinelist.append(instr)
                    to_be_removed.add(fn)
                    ftype = fn.type.pointee
                    newfn = module.get_or_insert_function(ftype, "numbapro.%s" % fname)
                    ptxcode = library[fname]
                    to_be_implemented[newfn] = ptxcode
                    instr.called_function = newfn  # replace the function
                else:
                    logger.debug("Unknown LLVM intrinsic %s", fname)

    # implement all the math functions with inline ptx
    for fn, ptx in to_be_implemented.items():
        entry = fn.append_basic_block('entry')
        builder = _lc.Builder.new(entry)
        value = builder.call(ptx, fn.args)
        builder.ret(value)
        to_be_removed.add(fn)

    # inline all the functions
    for callinstr in inlinelist:
        ok = _lc.inline_function(callinstr)
        assert ok
    
    for fn in to_be_removed:
        fn.delete()

