"""
Generate documentation for all registered implementation for lowering
using reStructured text.
"""

from __future__ import print_function

import os.path
from io import StringIO
from collections import defaultdict
import inspect
from functools import partial
import textwrap


import numba
from numba.targets.registry import cpu_target


def gather_function_info(backend):
    fninfos = defaultdict(list)
    basepath = os.path.dirname(os.path.dirname(numba.__file__))
    for fn, osel in backend._defns.items():
        for sig, impl in osel.versions:
            info = {}
            fninfos[fn].append(info)
            info['fn'] = fn
            info['sig'] = sig
            code, firstlineno = inspect.getsourcelines(impl)
            path = inspect.getsourcefile(impl)
            info['impl'] = {
                'name': impl.__qualname__,
                'filename': os.path.relpath(path, start=basepath),
                'lines': (firstlineno, firstlineno + len(code) - 1),
                'source': textwrap.dedent(''.join(code)),
                'docstring': impl.__doc__
            }

    return fninfos


def bind_file_to_print(fobj):
    return partial(print, file=fobj)


def format_signature(sig):
    def fmt(c):
        try:
            return c.__name__
        except AttributeError:
            return repr(c).strip('\'"')
    out = tuple(map(fmt, sig))
    return '`({0})`'.format(', '.join(out))


def format_function_infos(fninfos):
    with StringIO() as buf:
        print = bind_file_to_print(buf)

        title_line = "Lowering Listing"
        print(title_line)
        print('=' * len(title_line))

        def format_fname(fn):
            try:
                fname = "{0}.{1}".format(fn.__module__, fn.__qualname__)
            except AttributeError:
                fname = repr(fn)
            return fn, fname

        for fn, fname in sorted(map(format_fname, fninfos), key=lambda x: x[1]):
            impinfos = fninfos[fn]
            header_line = "Lower Impl ``{0}``".format(fname)
            print(header_line)
            print('-' * len(header_line))
            print()

            formatted_sigs = map(lambda x: format_signature(x['sig']), impinfos)
            sorted_impinfos = zip(formatted_sigs, impinfos)

            for fmtsig, info in sorted(sorted_impinfos, key=lambda x: x[0]):
                sig_line = 'signature {0}'.format(fmtsig)
                print(sig_line)
                print('~' * len(sig_line))
                print()

                impl = info['impl']
                print('by', '``{0}``'.format(impl['name']),
                      '`{0}`'.format(impl['filename']),
                      'lines {0}-{1}'.format(*impl['lines']))
                print()
                print(".. code-block:: python")
                print()
                print(textwrap.indent(impl['source'], prefix=' ' * 4))
                print()

        return buf.getvalue()


# Main routine for this module:

def gen_lower_listing(path=None):
    """
    Generate lowering listing to ``path`` or (if None) to stdout.
    """
    cpu_backend = cpu_target.target_context
    cpu_backend.refresh()

    fninfos = gather_function_info(cpu_backend)
    out = format_function_infos(fninfos)

    if path is None:
        print(out)
    else:
        with open(path, 'w') as fobj:
            print(out, file=fobj)


if __name__ == '__main__':
    gen_lower_listing()
