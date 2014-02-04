from __future__ import print_function, absolute_import
import inspect
import re
from collections import Mapping, defaultdict
import textwrap
from contextlib import closing
from numba.io_support import StringIO
from numba import ir


class SourceLines(Mapping):
    def __init__(self, func):

        try:
            lines, startno = inspect.getsourcelines(func)
        except IOError:
            self.lines = ()
            self.startno = 0
        else:
            self.lines = textwrap.dedent(''.join(lines)).splitlines()
            self.startno = startno

    def __getitem__(self, lineno):
        try:
            return self.lines[lineno - self.startno].rstrip()
        except IndexError:
            return ''

    def __iter__(self):
        return iter((self.startno + i) for i in range(len(self.lines)))

    def __len__(self):
        return len(self.lines)

    @property
    def avail(self):
        return bool(self.lines)


class TypeAnnotation(object):
    def __init__(self, interp, typemap, calltypes):
        self.text = self.annotate(interp, typemap, calltypes)

    def annotate(self, interp, typemap, calltypes):
        source = SourceLines(interp.bytecode.func)
        # if not source.avail:
        #     return "Source code unavailable"

        # Prepare annotations
        groupedinst = defaultdict(list)
        for blkid, blk in interp.blocks.items():
            groupedinst[blk.loc.line].append("label %d" % blkid)
            for inst in blk.body:
                lineno = inst.loc.line

                if isinstance(inst, ir.Assign):
                    if (isinstance(inst.value, ir.Expr) and
                            inst.value.op ==  'call'):
                        atype = calltypes[inst.value]
                    else:
                        atype = typemap[inst.target.name]

                    aline = "%s = %s  :: %s" % (inst.target, inst.value, atype)
                elif isinstance(inst, ir.SetItem):
                    atype = calltypes[inst]
                    aline = "%s  :: %s" % (inst, atype)
                else:
                    aline = "%s" % inst
                groupedinst[lineno].append("  %s" % aline)

        # Format annotations
        io = StringIO()
        with closing(io):
            if source.avail:
                for num in source:
                    srcline = source[num]
                    ind = _getindent(srcline)
                    print("%s# --- LINE %d --- " % (ind, num), file=io)
                    for inst in groupedinst[num]:
                        print('%s# %s' % (ind, inst), file=io)
                    print(file=io)
                    print(srcline, file=io)
                    print(file=io)
            else:
                print("# Source code unavailable", file=io)
                for num in groupedinst:
                    for inst in groupedinst[num]:
                        print('%s' % (inst,), file=io)
                    print(file=io)

            return io.getvalue()

    def __str__(self):
        return self.text


re_longest_white_prefix = re.compile('^\s*')


def _getindent(text):
    m = re_longest_white_prefix.match(text)
    if not m:
        return ''
    else:
        return ' ' * len(m.group(0))
