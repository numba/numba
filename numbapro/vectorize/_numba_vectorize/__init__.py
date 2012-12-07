__all__ = [
           'vectorize',
           'Vectorize',
           'BasicVectorize',
           'BasicASTVectorize',
           ]


from .basic import BasicVectorize, BasicASTVectorize
from numba.decorators import _process_sig
import warnings

_bytecode_vectorizers = {
    'cpu': BasicVectorize,
}

_ast_vectorizers = {
    'cpu': BasicASTVectorize,
}

_vectorizers = {
    'bytecode': _bytecode_vectorizers,
    'ast'     : _ast_vectorizers,
}

def _prepare_sig(sig):
    if isinstance(sig, str):
        _name, restype, argtypes = _process_sig(str(sig), None)
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
        backend: 'bytecode', 'ast' or 'mini'.
        Default: 'bytecode'
        target: 'basic', 'parallel', 'stream' or 'gpu'
        Default: 'basic'
        """
    assert backend in _vectorizers, tuple(backends)
    targets = _vectorizers[backend]
    assert target in targets, tuple(targets)
    if target in targets:
        return targets[target](func)
    else: # fall back
        warning.warn("fallback to bytecode vectorizer")
        # Use the default bytecode backend
        return _bytecode_vectorizers[target](func)

def vectorize(signatures, backend='ast', target='cpu'):
    def _vectorize(fn):
        vect = Vectorize(fn, backend=backend, target=target)
        for sig in signatures:
            kws = _prepare_sig(sig)
            vect.add(**kws)
        ufunc = vect.build_ufunc()
        return ufunc
    return _vectorize
