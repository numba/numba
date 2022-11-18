import types

from numba.core import utils


def copy_code_type(code: types.CodeType, **kwargs) -> types.CodeType:
    """Copy CodeType with mutations.

    Parameters
    ----------
    code: CodeType
    **kwargs:
        Attributes to mutate. Only support "argcount", "nlocals", "codestring",
        "constants", "names", and "varnames". These names matches argument in
        CodeType. See CodeType docstring for details.
    """
    co_args = [kwargs.get('argcount', code.co_argcount)]
    if utils.PYVERSION >= (3, 8):
        co_args.append(code.co_posonlyargcount)
    co_args.append(code.co_kwonlyargcount)
    co_args.extend([
        kwargs.get('nlocals', code.co_nlocals),
        code.co_stacksize,
        code.co_flags,
        kwargs.get('codestring', code.co_code),
        kwargs.get('constants', code.co_consts),
        kwargs.get('names', code.co_names),
        kwargs.get('varnames', code.co_varnames),
        code.co_filename,
        code.co_name,
    ])
    if utils.PYVERSION >= (3, 11):
        co_args.append(code.co_qualname)
    co_args.extend([code.co_firstlineno,
                    code.co_lnotab])
    if utils.PYVERSION >= (3, 11):
        co_args.append(code.co_exceptiontable)
    co_args.extend([
        code.co_freevars,
        code.co_cellvars,
    ])
    return types.CodeType(*co_args)
