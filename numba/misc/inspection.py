"""Miscellaneous inspection tools
"""
from tempfile import NamedTemporaryFile


def disassemble_elf_to_cfg(elf):
    """
    Gets the CFG of the disassembly of an ELF object, elf, and renders it
    appropriately depending on the execution environment (terminal/notebook).
    """
    try:
        import r2pipe
    except ImportError:
        raise RuntimeError("r2pipe package needed for disasm CFG")

    def get_rendering(cmd=None):
        if cmd is None:
            raise ValueError("No command given")

        with NamedTemporaryFile(delete=False) as f:
            f.write(elf)
            f.flush()  # force write, radare2 needs a binary blob on disk

            # catch if r2pipe can actually talk to radare2
            try:
                flags = ['-e io.cache=true',  # fix relocations in disassembly
                         '-e scr.color=1',  # 16 bit ANSI colour terminal
                         ]
                r = r2pipe.open(f.name, flags=flags)
                data = r.cmd('af;%s' % cmd)
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
