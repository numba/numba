"""
Decorator to assign the right jit for different targets.
"""
from numba import cuda, jit
from Lib import inspect
# from Lib.textwrap import dedent


# HACK
def get_myjit(target="cuda",
              inline="never",
              device=True,
              parallel=False,
              debug=False,
              opt=True
              ):
    """
    Get a decorator that assigns the right jit for different targets.

    Parameters
    ----------
    target : str, optional
        Whether to compute on "cuda" or on "cpu". The default is "cuda".
    self.inline : str, optional
        Whether to inline functions. The default is "never".

    Returns
    -------
    myjit : function
        The decorator with the specified keyword arguments.

    """
    if debug and opt:
        raise ValueError("debug and opt should not both be True")

    def myjit(f):
        """
        Decorator to assign the right jit for different targets.

        In case of non-cuda targets, all instances of `cuda.local.array`
        are replaced by `np.empty`. This is a dirty fix, hopefully in the
        near future numba will support numpy array allocation and this will
        not be necessary anymore

        Modified from: https://github.com/numba/numba/issues/2571

        Parameters
        ----------
        f : function
            function for which the decorator is applied to.

        Returns
        -------
        newfun : function
            cuda.jit or jit version of f.

        """
        source = inspect.getsource(f).splitlines()
        assert '@myjit' in source[0] or '@mycudajit' in source[0]
        indent_spaces = len(source[0]) - len(source[0].lstrip())
        source = [s[indent_spaces:] for s in source]
        source = '\n'.join(source[1:]) + '\n'

        # source = inspect.dedent(source)

        if target == 'cuda':
            source = source.replace('prange', 'range')
            exec(source)
            fun = eval(f.__name__)
            newfun = cuda.jit(f, device=device, inline=inline,
                              debug=debug, opt=opt)
            # needs to be exported to globals
            globals()[f.__name__] = newfun
            return newfun

        elif target == 'cpu':
            source = source.replace('cuda.local.array', 'np.empty')
            if not parallel:
                source = source.replace('prange', 'range')
            exec(source)
            fun = eval(f.__name__)
            newfun = jit(fun, nopython=True, inline=inline)
            # needs to be exported to globals
            globals()[f.__name__] = newfun
            return newfun

    return myjit
