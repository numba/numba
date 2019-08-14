"""
This file contains `__main__` so that it can be run as a commandline tool.

This file contains function to inspect Numba's support for a given Python
module or a Python package.
"""


from __future__ import print_function

import sys
from functools import partial
import pkgutil
import types as pytypes
import textwrap
from collections import defaultdict
import html

from numba.targets.registry import cpu_target
from numba.tests.support import captured_stdout


def inspect_module(module, target=None):
    target = target or cpu_target
    tyct = target.typing_context
    # Make sure we have loaded all extensions
    tyct.refresh()
    target.target_context.refresh()
    # Walk the module
    for name in dir(module):
        if name.startswith('_'):
            # Skip
            continue
        obj = getattr(module, name)
        info = dict(module=module, name=name, obj=obj)
        supported_types = (pytypes.FunctionType, pytypes.BuiltinFunctionType)
        if isinstance(obj, supported_types):
            # Try getting the function type
            source_infos = {}
            try:
                nbty = tyct.resolve_value_type(obj)
            except ValueError:
                nbty = None
                explained = 'not supported'
            else:
                # Make a longer explanation of the type
                explained = tyct.explain_function_type(nbty)
                for temp in nbty.templates:
                    try:
                        source_infos[temp] = temp.get_source_info()
                    except AttributeError:
                        source_infos[temp] = None

            info['numba_type'] = nbty
            info['explained'] = explained
            info['source_infos'] = source_infos
            yield info


class _Stat(object):
    def __init__(self):
        self.supported = 0
        self.unsupported = 0

    @property
    def total(self):
        total = self.supported + self.unsupported
        return total

    @property
    def ratio(self):
        ratio = self.supported / self.total * 100
        return ratio

    def describe(self):
        return "supported = {supported} / {total} = {ratio:.2f}%".format(
            supported=self.supported,
            total=self.total,
            ratio=self.ratio,
        )

    def __repr__(self):
        return "{clsname}({describe})".format(
            clsname=self.__class__.__name__,
            describe=self.describe(),
        )


def list_modules_in_package(package):
    onerror_ignore = lambda _: None

    prefix = package.__name__ + "."
    package_walker = pkgutil.walk_packages(
        package.__path__,
        prefix,
        onerror=onerror_ignore,
    )
    for pkginfo in package_walker:
        modname = pkginfo.name

        # Suppress private module '_'?
        if any(x.startswith('_') for x in modname.split('.')):
            continue

        # In case importing of the module print to stdout
        with captured_stdout():
            try:
                # Import the module
                mod = __import__(modname)
            except Exception:
                continue

            # Extract the module
            for part in modname.split('.')[1:]:
                try:
                    mod = getattr(mod, part)
                except AttributeError:
                    # Suppress error in getting the attribute
                    mod = None
                    break

        # Ignore if mod is not a module
        if not isinstance(mod, pytypes.ModuleType):
            # Skip non-module
            continue

        yield mod


def print_module_info(mod_sequence, target=None, print=print):
    stats = defaultdict(_Stat)
    for mod in mod_sequence:
        modname = mod.__name__
        print(modname.center(80, '='))
        for info in inspect_module(mod, target=target):
            print('-', info['module'].__name__, info['obj'].__name__)
            nbtype = info['numba_type']
            indent = '    '
            if nbtype is not None:
                print('{}{}'.format(indent, nbtype))
                stats[modname].supported += 1
            else:
                stats[modname].unsupported += 1
            explained = info['explained']
            print(textwrap.indent(explained, prefix=indent))
    print('=' * 80)
    print('Stats:')
    for k, v in stats.items():
        print(' - {} : {}'.format(k, v.describe()))


def print_module_info_html(mod_sequence, target=None, print=print):
    print('<h1>', 'Numba Support Inspector', '</h1>')
    stats = defaultdict(_Stat)
    quote = html.escape
    for mod in mod_sequence:
        modname = mod.__name__
        print('<h2>', quote(modname), '</h2>')
        print('<ul>')
        for info in inspect_module(mod, target=target):
            print('<li>')
            print('{}.<b>{}</b>'.format(
                quote(info['module'].__name__),
                quote(info['obj'].__name__),
            ))
            nbtype = info['numba_type']
            if nbtype is not None:
                print(' : <b>{}</b>'.format(quote(str(nbtype))))
                stats[modname].supported += 1
            else:
                stats[modname].unsupported += 1
            explained = info['explained']
            print('<div><pre>', quote(explained), '</pre></div>')
            for tempcls, source in info['source_infos'].items():
                if source:
                    impl = source['name']
                    filename = source['filename']
                    lines = source['lines']
                    print(
                        "<p>defined by <b>{}</b> at {}:{}-{}</p>".format(
                            impl, filename, *lines,
                        ),
                    )
                    print('<p>{}</p>'.format(source['docstring']) or '')

            print('</li>')
        print('</ul>')

    print('<h2>', 'Stats', '</h2>')
    print('<ul>')
    for k, v in stats.items():
        print('<li>{} : {}</li>'.format(quote(k), quote(v.describe())))
    print('</ul>')


def print_help(programname):
    print("""
Inspect Numba support for a given top-level package.

Usage:
    {programname} <package>

    `package`: is a top level package name.
    """.format(programname=programname))


def main():
    try:
        [_, package_name] = sys.argv
    except ValueError:
        print_help(sys.argv[0])
        sys.exit(1)
    else:
        package = __import__(package_name)
        if hasattr(package, '__path__'):
            mods = list_modules_in_package(package)
        else:
            mods = [package]
        filename = '{}.numba_support.html'.format(package_name)
        with open(filename, 'w') as fout:
            print_module_info_html(mods, print=partial(print, file=fout))


if __name__ == '__main__':
    main()
