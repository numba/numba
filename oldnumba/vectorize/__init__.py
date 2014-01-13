# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
__all__ = [
           'vectorize',
           'Vectorize',
           'BasicVectorize',
           'BasicASTVectorize',
           'GUVectorize',
           ]


from .basic import BasicVectorize, BasicASTVectorize
from .gufunc import GUFuncVectorize as GUVectorize

from numba.utils import process_sig
import warnings

#_bytecode_vectorizers = {
#    'cpu': BasicVectorize,
#}

_ast_vectorizers = {
    'cpu': BasicASTVectorize,
}

_vectorizers = {
#    'bytecode': _bytecode_vectorizers,
    'ast'     : _ast_vectorizers,
}

def _prepare_sig(sig):
    if isinstance(sig, str):
        _name, restype, argtypes = process_sig(str(sig), None)
    else:
        argtypes = sig.args
        restype = sig.return_type

    kws = {}
    if restype is not None:
        kws['restype'] = restype
    if argtypes is not None:
        kws['argtypes'] = argtypes
    return kws

def install_vectorizer(backend, target, vectorizer):
    lib = _vectorizers.get(backend, {})
    if not lib:
        _vectorizers[backend] = lib
    if target in lib:
        warnings.warn("overriding vectorizer %s:%s:%s" % (backend,
                                                          target,
                                                          vectorizer),
                      UserWarning)
    lib[target] = vectorizer

def Vectorize(func, backend='ast', target='cpu'):
    """
        Instantiate a vectorizer given the backend and target.

        func: the function to vectorize
        backend: 'ast'
        Default: 'ast'
        target: 'basic'
        Default: 'basic'
        """
    assert backend in _vectorizers, tuple(_vectorizers)
    targets = _vectorizers[backend]
    assert target in targets, tuple(targets)
    if target in targets:
        return targets[target](func)
    else: # fall back
        raise NotImplementedError

def vectorize(signatures, backend='ast', target='cpu'):
    '''vectorize(type_signatures[, target='cpu'])

    A decorator to create numpy ufunc object from Numba compiled code.
    
    :param type_signatures: an iterable of type signatures, which are either 
                            function type object or a string describing the 
                            function type.
    
    :param target: a string for hardware target; e.g. "cpu", "parallel", "gpu".
                   For support for "parallel" and "gpu", use NumbaPro.

    :returns: a vectorizers object.
    
    Example::

        @vectorize(['float32(float32, float32)',
                    'float64(float64, float64)'])
        def sum(a, b):
            return a + b
    '''
    def _vectorize(fn):
        vect = Vectorize(fn, backend=backend, target=target)
        for sig in signatures:
            kws = _prepare_sig(sig)
            vect.add(**kws)
        ufunc = vect.build_ufunc()
        return ufunc
    return _vectorize

def get_include():
    from os.path import dirname
    return dirname(__file__)
