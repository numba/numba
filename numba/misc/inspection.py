"""Miscellaneous inspection tools
"""
import ast
import inspect
import linecache
import os
import warnings
import types as pt
import typing
from tempfile import NamedTemporaryFile, TemporaryDirectory

from numba.core.errors import NumbaWarning


def disassemble_elf_to_cfg(elf, mangled_symbol):
    """
    Gets the CFG of the disassembly of an ELF object, elf, at mangled name,
    mangled_symbol, and renders it appropriately depending on the execution
    environment (terminal/notebook).
    """
    try:
        import r2pipe
    except ImportError:
        raise RuntimeError("r2pipe package needed for disasm CFG")

    def get_rendering(cmd=None):
        from numba.pycc.platform import Toolchain # import local, circular ref
        if cmd is None:
            raise ValueError("No command given")

        with TemporaryDirectory() as tmpdir:
            # Write ELF as a temporary file in the temporary dir, do not delete!
            with NamedTemporaryFile(delete=False, dir=tmpdir) as f:
                f.write(elf)
                f.flush()  # force write, radare2 needs a binary blob on disk

            # Now try and link the ELF, this helps radare2 _a lot_
            linked = False
            try:
                raw_dso_name = f'{os.path.basename(f.name)}.so'
                linked_dso = os.path.join(tmpdir, raw_dso_name)
                tc = Toolchain()
                tc.link_shared(linked_dso, (f.name,))
                obj_to_analyse = linked_dso
                linked = True
            except Exception as e:
                # link failed, mention it to user, radare2 will still be able to
                # analyse the object, but things like dwarf won't appear in the
                # asm as comments.
                msg = ('Linking the ELF object with the distutils toolchain '
                       f'failed with: {e}. Disassembly will still work but '
                       'might be less accurate and will not use DWARF '
                       'information.')
                warnings.warn(NumbaWarning(msg))
                obj_to_analyse = f.name

            # catch if r2pipe can actually talk to radare2
            try:
                flags = ['-2', # close stderr to hide warnings
                         '-e io.cache=true', # fix relocations in disassembly
                         '-e scr.color=1',  # 16 bit ANSI colour terminal
                         '-e asm.dwarf=true', # DWARF decode
                         '-e scr.utf8=true', # UTF8 output looks better
                         ]
                r = r2pipe.open(obj_to_analyse, flags=flags)
                r.cmd('aaaaaa') # analyse as much as possible
                # If the elf is linked then it's necessary to seek as the
                # DSO ctor/dtor is at the default position
                if linked:
                    # r2 only matches up to 61 chars?! found this by experiment!
                    mangled_symbol_61char = mangled_symbol[:61]
                    # switch off demangle, the seek is on a mangled symbol
                    r.cmd('e bin.demangle=false')
                    # seek to the mangled symbol address
                    r.cmd(f's `is~ {mangled_symbol_61char}[1]`')
                    # switch demangling back on for output purposes
                    r.cmd('e bin.demangle=true')
                data = r.cmd('%s' % cmd) # print graph
                r.quit()
            except Exception as e:
                if "radare2 in PATH" in str(e):
                    msg = ("This feature requires 'radare2' to be "
                           "installed and available on the system see: "
                           "https://github.com/radareorg/radare2. "
                           "Cannot find 'radare2' in $PATH.")
                    raise RuntimeError(msg)
                else:
                    raise e
        return data

    class DisasmCFG(object):

        def _repr_svg_(self):
            try:
                import graphviz
            except ImportError:
                raise RuntimeError("graphviz package needed for disasm CFG")
            jupyter_rendering = get_rendering(cmd='agfd')
            # this just makes it read slightly better in jupyter notebooks
            jupyter_rendering.replace('fontname="Courier",',
                                      'fontname="Courier",fontsize=6,')
            src = graphviz.Source(jupyter_rendering)
            return src.pipe('svg').decode('UTF-8')

        def __repr__(self):
            return get_rendering(cmd='agf')

    return DisasmCFG()


class _ClassFinder(ast.NodeVisitor):

    def __init__(self, cls, tree, lines, qualname):
        self.stack = []
        self.cls = cls
        self.tree = tree
        self.lines = lines
        self.qualname = qualname
        self.lineno_found = []

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.stack.append('<locals>')
        self.generic_visit(node)
        self.stack.pop()
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        if self.qualname == '.'.join(self.stack):
            # Return the decorator for the class if present
            if node.decorator_list:
                line_number = node.decorator_list[0].lineno
            else:
                line_number = node.lineno

            # decrement by one since lines starts with indexing by zero
            self.lineno_found.append((line_number - 1, node.end_lineno))
        self.generic_visit(node)
        self.stack.pop()

    def get_lineno(self):
        self.visit(self.tree)
        lineno_found_number = len(self.lineno_found)
        if lineno_found_number == 0:
            raise OSError('could not find class definition')
        elif lineno_found_number == 1:
            return self.lineno_found[0][0]
        else:
            # We have multiple candidates for the class definition.
            # Now we have to guess.

            # First, let's see if there are any method definitions
            for member in self.cls.__dict__.values():
                if isinstance(member, pt.FunctionType):
                    for lineno, end_lineno in self.lineno_found:
                        if (
                            lineno <= member.__code__.co_firstlineno
                            <= end_lineno
                        ):
                            return lineno

            class_strings = [
                (''.join(self.lines[lineno: end_lineno]), lineno)
                for lineno, end_lineno in self.lineno_found
            ]

            # Maybe the class has a docstring and it's unique?
            if self.cls.__doc__:
                ret = None
                for candidate, lineno in class_strings:
                    if self.cls.__doc__.strip() in candidate:
                        if ret is None:
                            ret = lineno
                        else:
                            break
                else:
                    if ret is not None:
                        return ret

            # We are out of ideas, just return the last one found, which is
            # slightly better than previous ones
            return self.lineno_found[-1][0]


def find_class_source(obj: typing.Type) -> str:
    """
    Find the source code for the given class definition.

    The builtin ``inspect.getsource`` doesn't work when classes in the same
    file share a name, it just returns the first one it finds. Thankfully this
    issue has already been addressed in
    https://github.com/python/cpython/pull/106815 so we can lift the code from
    there to roll our own inspection function.

    At the time of writing, the latest Python version released 3.12.4. Can't
    figure out when this patch to ``inspect`` actually lands.
    """
    file = inspect.getsourcefile(obj)
    if file:
        # Invalidate cache if needed.
        linecache.checkcache(file)
    else:
        file = inspect.getfile(obj)
        # Allow filenames in form of "<something>" to pass through.
        # `doctest` monkeypatches `linecache` module to enable
        # inspection, so let `linecache.getlines` to be called.
        if not (file.startswith('<') and file.endswith('>')):
            raise OSError('source code not available')
    module = inspect.getmodule(obj, file)
    if module:
        lines = linecache.getlines(file, module.__dict__)
    else:
        lines = linecache.getlines(file)
    if not lines:
        raise OSError('could not get source code')
    tree = ast.parse("".join(lines))
    class_finder = _ClassFinder(obj, tree, lines, obj.__qualname__)
    lnum = class_finder.get_lineno()
    return "".join(inspect.getblock(lines[lnum:]))
