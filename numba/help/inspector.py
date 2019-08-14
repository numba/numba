"""
This file contains `__main__` so that it can be run as a commandline tool.

This file contains function to inspect Numba's support for a given Python
module or a Python package.
"""


from __future__ import print_function

import argparse
import pkgutil
import types as pytypes
import html


from numba._version import get_versions
from numba.targets.registry import cpu_target
from numba.tests.support import captured_stdout


commit = get_versions()['full'].split('.')[0]
github_url = 'https://github.com/numba/numba/blob/{commit}/{path}#L{firstline}-L{lastline}'


def inspect_function(function, target=None):
    target = target or cpu_target
    tyct = target.typing_context
    # Make sure we have loaded all extensions
    tyct.refresh()
    target.target_context.refresh()

    info = {}
    # Try getting the function type
    source_infos = {}
    try:
        nbty = tyct.resolve_value_type(function)
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
    return info


def inspect_module(module, target=None):
    # Walk the module
    for name in dir(module):
        if name.startswith('_'):
            # Skip
            continue
        obj = getattr(module, name)
        info = dict(module=module, name=name, obj=obj)
        supported_types = (pytypes.FunctionType, pytypes.BuiltinFunctionType)
        if isinstance(obj, supported_types):
            info.update(inspect_function(obj, target=target))
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
        if self.total == 0:
            return "empty"
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
        modname = pkginfo[1]

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


class Formatter(object):
    def __init__(self, fileobj):
        self._fileobj = fileobj

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self._fileobj)


class HTMLFormatter(Formatter):

    def escape(self, text):
        return html.escape(text)

    def title(self, text):
        self.print('<h1>', text, '</h2>')

    def begin_module_section(self, modname):
        self.print('<h2>', modname, '</h2>')
        self.print('<ul>')

    def end_module_section(self):
        self.print('</ul>')

    def write_supported_item(self, modname, itemname, typename, explained, sources):
        self.print('<li>')
        self.print('{}.<b>{}</b>'.format(
            modname,
            itemname,
        ))
        self.print(': <b>{}</b>'.format(typename))
        self.print('<div><pre>', explained, '</pre></div>')

        self.print("<ul>")
        for tcls, source in sources.items():
            if source:
                self.print("<li>")
                impl = source['name']
                filename = source['filename']
                lines = source['lines']
                self.print(
                    "<p>defined by <b>{}</b> at {}:{}-{}</p>".format(
                        self.escape(impl), self.escape(filename), *lines,
                    ),
                )
                self.print('<p>{}</p>'.format(
                    self.escape(source['docstring'] or '')
                ))
            else:
                self.print("<li>{}".format(self.escape(str(tcls))))
            self.print("</li>")
        self.print("</ul>")
        self.print('</li>')

    def write_unsupported_item(self, modname, itemname):
        self.print('<li>')
        self.print('{}.<b>{}</b>: UNSUPPORTED'.format(
            modname,
            itemname,
        ))
        self.print('</li>')

    def write_statistic(self, stats):
        self.print('<p>{}</p>'.format(stats.describe()))


class ReSTFormatter(Formatter):

    def escape(self, text):
        return text

    def title(self, text):
        self.print(text)
        self.print('=' * len(text))
        self.print()

    def begin_module_section(self, modname):
        self.print(modname)
        self.print('-' * len(modname))
        self.print()

    def end_module_section(self):
        self.print()

    def write_supported_item(self, modname, itemname, typename, explained, sources):
        self.print('.. function:: {}.{}'.format(modname, itemname))
        self.print()
        # if explained:
        #     self.print('   .. code-block:: text')
        #     self.print()
        #     self.print('      {}'.format('\n      '.join(explained.splitlines())))
        #     self.print()
        for tcls, source in sources.items():
            if source:
                impl = source['name']
                filename = source['filename']
                lines = source['lines']
                source_link = github_url.format(
                    commit=commit,
                    path=filename,
                    firstline=lines[0],
                    lastline=lines[1],
                )
                self.print(
                    "   - defined by ``{}`` at `{}:{}-{} <{}>`_".format(
                        impl, filename, lines[0], lines[1], source_link,
                    ),
                )

            else:
                self.print("   - defined by ``{}``".format(str(tcls)))
        self.print()

    def write_unsupported_item(self, modname, itemname):
        pass
        # self.print('.. function:: {}.{}'.format(modname, itemname))
        # self.print()
        # self.print('   unsupported')
        # self.print()

    def write_statistic(self, stat):
        if stat.supported == 0:
            self.print("this module is not supported")
        else:
            msg = "Not showing {} unsupported functions."
            self.print(msg.format(stat.unsupported))
            self.print()
            self.print(stat.describe())
        self.print()


def print_module_info(formatter, package_name, mod_sequence, target=None):
    formatter.title('Listings for {}'.format(package_name))
    for mod in mod_sequence:
        stat = _Stat()
        modname = mod.__name__
        formatter.begin_module_section(formatter.escape(modname))
        for info in inspect_module(mod, target=target):
            nbtype = info['numba_type']
            if nbtype is not None:
                stat.supported += 1
                formatter.write_supported_item(
                    modname=formatter.escape(info['module'].__name__),
                    itemname=formatter.escape(info['obj'].__name__),
                    typename=formatter.escape(str(nbtype)),
                    explained=formatter.escape(info['explained']),
                    sources=info['source_infos'],
                )

            else:
                stat.unsupported += 1
                formatter.write_unsupported_item(
                    modname=formatter.escape(info['module'].__name__),
                    itemname=formatter.escape(info['obj'].__name__),
                )

        formatter.write_statistic(stat)
        formatter.end_module_section()


def print_help(programname):
    print("""
Inspect Numba support for a given top-level package.

Usage:
    {programname} <package>

    `package`: is a top level package name.
    """.format(programname=programname))


program_description = """
Inspect Numba support for a given top-level package.
""".strip()


def main():
    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument(
        'package', metavar='package', type=str,
        help='Package to inspect',
    )
    parser.add_argument(
        '--format', dest='format', default='html',
        help='Output format; i.e. "html", "rst"',
    )
    parser.add_argument(
        '--file', dest='file', default='inspector_output',
        help='Output filename. Defaults to "inspector_output"',
    )

    args = parser.parse_args()
    package_name = args.package
    output_format = args.format
    filename = args.file
    run_inspector(package_name, filename, output_format)


def run_inspector(package_name, filename, output_format):
    package = __import__(package_name)
    if hasattr(package, '__path__'):
        mods = list_modules_in_package(package)
    else:
        mods = [package]

    if output_format == 'html':
        with open(filename + '.html', 'w') as fout:
            fmtr = HTMLFormatter(fileobj=fout)
            print_module_info(fmtr, package_name, mods)
    elif output_format == 'rst':
        with open(filename + '.rst', 'w') as fout:
            fmtr = ReSTFormatter(fileobj=fout)
            print_module_info(fmtr, package_name, mods)
    else:
        raise ValueError("{} is not supported".format(output_format))


if __name__ == '__main__':
    main()
