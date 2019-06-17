from __future__ import print_function, absolute_import

from numba import roc
from numba.npyufunc import deviceufunc

from numba.roc import dispatch

vectorizer_stager_source = '''
def __vectorized_{name}({args}, __out__):

    __tid__ = __hsa__.get_local_id(0)
    __blksz__ = __hsa__.get_local_size(0)
    __blkid__ = __hsa__.get_group_id(0)

    __tid0__ = __tid__ + __blksz__ * (4 * __blkid__)
    __tid1__ = __tid__ + __blksz__ * (4 * __blkid__ + 1)
    __tid2__ = __tid__ + __blksz__ * (4 * __blkid__ + 2)
    __tid3__ = __tid__ + __blksz__ * (4 * __blkid__ + 3)

    __ilp0__ = __tid0__ < __out__.shape[0]
    if not __ilp0__:
        # Early escape
        return
    __ilp1__ = __tid1__ < __out__.shape[0]
    __ilp2__ = __tid2__ < __out__.shape[0]
    __ilp3__ = __tid3__ < __out__.shape[0]

    if __ilp3__:
        __args0__ = {argitems_0}
        __args1__ = {argitems_1}
        __args2__ = {argitems_2}
        __args3__ = {argitems_3}

        __r0__ = __core__(*__args0__)
        __r1__ = __core__(*__args1__)
        __r2__ = __core__(*__args2__)
        __r3__ = __core__(*__args3__)

        __out__[__tid0__] = __r0__
        __out__[__tid1__] = __r1__
        __out__[__tid2__] = __r2__
        __out__[__tid3__] = __r3__

    elif __ilp2__:
        __args0__ = {argitems_0}
        __args1__ = {argitems_1}
        __args2__ = {argitems_2}

        __r0__ = __core__(*__args0__)
        __r1__ = __core__(*__args1__)
        __r2__ = __core__(*__args2__)

        __out__[__tid0__] = __r0__
        __out__[__tid1__] = __r1__
        __out__[__tid2__] = __r2__

    elif __ilp1__:
        __args0__ = {argitems_0}
        __args1__ = {argitems_1}

        __r0__ = __core__(*__args0__)
        __r1__ = __core__(*__args1__)

        __out__[__tid0__] = __r0__
        __out__[__tid1__] = __r1__

    else:
        __args0__ = {argitems_0}
        __r0__ = __core__(*__args0__)
        __out__[__tid0__] = __r0__

'''


class HsaVectorize(deviceufunc.DeviceVectorize):
    def _compile_core(self, sig):
        hsadevfn = roc.jit(sig, device=True)(self.pyfunc)
        return hsadevfn, hsadevfn.cres.signature.return_type

    def _get_globals(self, corefn):
        glbl = self.pyfunc.__globals__
        glbl.update({'__hsa__': roc,
                     '__core__': corefn})
        return glbl

    def _compile_kernel(self, fnobj, sig):
        return roc.jit(sig)(fnobj)

    def _get_kernel_source(self, template, sig, funcname):
        args = ['a%d' % i for i in range(len(sig.args))]

        def make_argitems(n):
            out = ', '.join('%s[__tid%d__]' % (i, n) for i in args)
            if len(args) < 2:
                # Less than two arguments.
                # We need to wrap the argument in a tuple because
                # we use stararg later.
                return "({0},)".format(out)
            else:
                return out

        fmts = dict(name=funcname,
                    args=', '.join(args),
                    argitems_0=make_argitems(n=0),
                    argitems_1=make_argitems(n=1),
                    argitems_2=make_argitems(n=2),
                    argitems_3=make_argitems(n=3))
        src = template.format(**fmts)
        return src

    def build_ufunc(self):
        return dispatch.HsaUFuncDispatcher(self.kernelmap)

    @property
    def _kernel_template(self):
        return vectorizer_stager_source


# ------------------------------------------------------------------------------
# Generalized HSA ufuncs

_gufunc_stager_source = '''
def __gufunc_{name}({args}):
    __tid__ = __hsa__.get_global_id(0)
    if __tid__ < {checkedarg}:
        __core__({argitems})
'''


class HsaGUFuncVectorize(deviceufunc.DeviceGUFuncVectorize):
    def build_ufunc(self):
        engine = deviceufunc.GUFuncEngine(self.inputsig, self.outputsig)
        return dispatch.HSAGenerializedUFunc(kernelmap=self.kernelmap,
                                             engine=engine)

    def _compile_kernel(self, fnobj, sig):
        return roc.jit(sig)(fnobj)

    @property
    def _kernel_template(self):
        return _gufunc_stager_source

    def _get_globals(self, sig):
        corefn = roc.jit(sig, device=True)(self.pyfunc)
        glbls = self.py_func.__globals__.copy()
        glbls.update({'__hsa__': roc,
                      '__core__': corefn})
        return glbls

