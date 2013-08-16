from numbapro.npm import types, cgutils
from . import ptx

def SRegImplFactory(func):
    class SRegImpl(object):
        function = func, (), types.uint32

        def generic_implement(self, context, args, argtys, retty):
            builder = context.builder
            sreg = ptx.SREG_MAPPING[self.function[0]]
            retty = self.function[-1].llvm_as_value()
            func = cgutils.get_function(builder, sreg, retty, ())
            return builder.call(func, ())
    return SRegImpl

TidX = SRegImplFactory(ptx._ptx_sreg_tidx)
TidY = SRegImplFactory(ptx._ptx_sreg_tidy)
TidZ = SRegImplFactory(ptx._ptx_sreg_tidz)

NTidX = SRegImplFactory(ptx._ptx_sreg_ntidx)
NTidY = SRegImplFactory(ptx._ptx_sreg_ntidy)
NTidZ = SRegImplFactory(ptx._ptx_sreg_ntidz)

CTAidX = SRegImplFactory(ptx._ptx_sreg_ctaidx)
CTAidY = SRegImplFactory(ptx._ptx_sreg_ctaidy)

NCTAidX = SRegImplFactory(ptx._ptx_sreg_nctaidx)
NCTAidY = SRegImplFactory(ptx._ptx_sreg_nctaidy)

extensions = [
    TidX, TidY, TidZ,
    NTidX, NTidY, NTidZ,
    CTAidX, CTAidY,
    NCTAidX, NCTAidY,
]

