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
    def __init__(self, interp, typemap, calltypes, lifted):
        self.filename = interp.bytecode.filename
        self.func = interp.bytecode.func
        self.blocks = interp.blocks
        self.typemap = typemap
        self.calltypes = calltypes
        self.lifted = lifted

    def annotate(self):
        source = SourceLines(self.func)
        # if not source.avail:
        #     return "Source code unavailable"

        # Prepare annotations
        groupedinst = defaultdict(list)
        for blkid, blk in self.blocks.items():
            groupedinst[blk.loc.line].append("label %s" % blkid)
            for inst in blk.body:
                lineno = inst.loc.line

                if isinstance(inst, ir.Assign):
                    if (isinstance(inst.value, ir.Expr) and
                            inst.value.op ==  'call'):
                        atype = self.calltypes[inst.value]
                    else:
                        atype = self.typemap[inst.target.name]

                    aline = "%s = %s  :: %s" % (inst.target, inst.value, atype)
                elif isinstance(inst, ir.SetItem):
                    atype = self.calltypes[inst]
                    aline = "%s  :: %s" % (inst, atype)
                else:
                    aline = "%s" % inst
                groupedinst[lineno].append("  %s" % aline)

        # Format annotations
        io = StringIO()
        with closing(io):
            if source.avail:
                print("# File: %s" % self.filename, file=io)
                for num in source:
                    srcline = source[num]
                    ind = _getindent(srcline)
                    print("%s# --- LINE %d --- " % (ind, num), file=io)
                    for inst in groupedinst[num]:
                        print('%s# %s' % (ind, inst), file=io)
                    print(file=io)
                    print(srcline, file=io)
                    print(file=io)
                if self.lifted:
                    print("# The function contains lifted loops", file=io)
                    for loop in self.lifted:
                        print("# Loop at line %d" % loop.bytecode.firstlineno,
                              file=io)
                        print("# Has %d overloads" % len(loop.overloads),
                              file=io)
                        for cres in loop._compileinfos.values():
                            print(cres.type_annotation, file=io)
            else:
                print("# Source code unavailable", file=io)
                for num in groupedinst:
                    for inst in groupedinst[num]:
                        print('%s' % (inst,), file=io)
                    print(file=io)

            return io.getvalue()

    def __str__(self):
        return self.annotate()


re_longest_white_prefix = re.compile('^\s*')


def _getindent(text):
    m = re_longest_white_prefix.match(text)
    if not m:
        return ''
    else:
        return ' ' * len(m.group(0))
