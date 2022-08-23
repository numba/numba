import ast
import hashlib
import inspect
import textwrap
import typing as pt
import warnings
from typing import TYPE_CHECKING

from numba.core.serialize import dumps


if TYPE_CHECKING:
    from numba.core import dispatcher, ccallback

DispatcherOrFunctionType = pt.TypeVar(
    'DispatcherOrFunctionType', type["ccallback.CFunc"], type["dispatcher.Dispatcher"]
)


def get_index_key(py_func, caller_class: "DispatcherOrFunctionType"):
    """
    Compute index key for the given signature and codegen.
    It includes a description of the OS, target architecture and hashes of
    the bytecode for the function and, if the function has a __closure__,
    a hash of the cell_contents.
    """
    # retrieve codebytes and cvarbytes of this function
    codebytes = py_func.__code__.co_code
    if py_func.__closure__ is not None:
        cvars = tuple([x.cell_contents for x in py_func.__closure__])
        # Note: cloudpickle serializes a function differently depending
        #       on how the process is launched; e.g. multiprocessing.Process
        cvarbytes = dumps(cvars)
    else:
        cvarbytes = b''
    # retrieve codebytes and cvarbytes of each dispatcher used in py_func
    for dep in get_function_dependencies(py_func, (caller_class,)):
        index_key = dep.cache_index_key
        codebytes += index_key[0].encode()
        cvarbytes += index_key[1].encode()

    hasher = lambda x: hashlib.sha256(x).hexdigest()

    return hasher(codebytes), hasher(cvarbytes)


def get_function_dependencies(
        py_func, fc_classes: pt.Tuple[DispatcherOrFunctionType]
) -> pt.List[DispatcherOrFunctionType]:
    """ given a py func, will return all dispatchers on which this depends


    this function supports the cache mechanism, so it ignores dependencies
    that are not callable objects. The classes used to identify what to include
    in the hash and what to ignore, are given by parameter ``fc_classes``

    The following cases are planned, but not all of them implemented
    - the called function is a global: implemented. The dependency is returned.
    - the called function is a builtin: implemented. The dependency is ignored.
    - the called function is a local: not implemented

    :param py_func: python function object to analyze
    :param fc_classes: classes to be considered callable objects that should
     be included in the hashes
    :return:
    :return:
    """
    deps = []
    unknown = []
    # Retrieve all function calls
    source = textwrap.dedent(inspect.getsource(py_func))
    ast_parsed = ast.parse(source)
    ast_walker = ast.walk(ast_parsed)
    # we create a list of all function calls, which is filtered in several
    # steps so only dispatchers remain
    disp_calls = (op.func.id
                  for op in ast_walker
                  if isinstance(op, ast.Call) and hasattr(op.func, 'id')
                  )
    disp_calls = set(disp_calls)
    # filter out builtins
    if not hasattr(py_func, "__builtins__"):
        builtin_names = __builtins__
        disp_calls = (name for name in disp_calls if name not in builtin_names)
    else:
        builtin_names = py_func.__builtins__
        disp_calls = (name for name in disp_calls if name not in builtin_names)
    # filter out input parameters
    sig_params = list(inspect.signature(py_func).parameters.keys())
    disp_calls = (name for name in disp_calls if name not in sig_params)
    # filter out globals if not a Dispatcher
    # retrieve object from globals if a Dispatcher
    fc_globals = py_func.__globals__
    for var_name in disp_calls:
        if var_name in fc_globals:
            var_obj = fc_globals[var_name]
            if isinstance(var_obj, fc_classes):
                deps.append(var_obj)
            # else ignore, not a Dispatcher
        else:
            unknown.append(var_name)

    if unknown:
        warnings.warn(f"Functions {unknown} will be cached but changes made to "
                      f"them might not be detected, resulting in possible use"
                      f"of stale cached versions")
    if not unknown:
        return deps
    # case: function is a getitem or attribute of a global
    # not yet implemented
    return deps
