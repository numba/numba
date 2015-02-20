from __future__ import print_function, absolute_import
import inspect
import re
from collections import Mapping, defaultdict, OrderedDict
import textwrap
from contextlib import closing
from numba.io_support import StringIO
from numba import ir
from . import step
import os
import sys


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
    
    data = {'func_data': OrderedDict()}

    def __init__(self, interp, typemap, calltypes, lifted, args, return_type,
                 func_attr, fancy=None):
        self.filename = interp.bytecode.filename
        self.func = interp.bytecode.func
        self.blocks = interp.blocks
        self.typemap = typemap
        self.calltypes = calltypes
        self.lifted = lifted
        if fancy is None:
            self.fancy = None
        else:
            self.fancy = os.path.join(os.getcwd(), fancy)
        self.filename = interp.loc.filename
        self.linenum = str(interp.loc.line)
        self.signature = str(args) + ' -> ' + str(return_type)
        self.func_attr = func_attr

    def prepare_annotations(self):
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
        return groupedinst

    def annotate(self):
        source = SourceLines(self.func)
        # if not source.avail:
        #     return "Source code unavailable"

        groupedinst = self.prepare_annotations()

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

    def html_annotate(self):
        root = os.path.join(os.path.dirname(__file__))
        template_filename = os.path.join(root, 'template.html')
        with open(template_filename, 'r') as template:
            html = template.read()

        python_source = SourceLines(self.func)
        llvm_instructions = self.prepare_annotations()
        line_nums = [num for num in python_source]

        lifted_lines = [l.bytecode.firstlineno for l in self.lifted]

        func_key = (self.filename + ':' + self.linenum, self.signature)
        if func_key not in TypeAnnotation.data['func_data']:

            TypeAnnotation.data['func_data'][func_key] = {}
            func_data = TypeAnnotation.data['func_data'][func_key]

            func_data['filename'] = self.filename
            func_data['funcname'] = self.func_attr.name

            func_data['python_indent'] = {}
            for num in line_nums:
                indent_len = len(_getindent(python_source[num]))
                func_data['python_indent'][num] = '&nbsp;' * indent_len
        
            func_data['llvm_indent'] = {}
            for num in line_nums:
                func_data['llvm_indent'][num] = []
                for inst in llvm_instructions[num]:
                    indent_len = len(_getindent(inst))
                    func_data['llvm_indent'][num].append('&nbsp;' * indent_len)

            func_data['python_object'] = {}
            func_data['llvm_object'] = {}
            for num in line_nums:
                func_data['python_object'][num] = ''
                func_data['llvm_object'][num] = []
                for inst in llvm_instructions[num]:
                    if num in lifted_lines:
                        func_data['python_object'][num] = 'lifted_tag'
                    elif inst.endswith('pyobject'):
                        func_data['llvm_object'][num].append('object_tag')
                        func_data['python_object'][num] = 'object_tag'
                    else:
                        func_data['llvm_object'][num].append('')

            func_data['python_lines'] = {}
            for num in line_nums:
                func_data['python_lines'][num] = python_source[num].strip()
        
            func_data['llvm_lines'] = {}
            for num in line_nums:
                func_data['llvm_lines'][num] = []
                for inst in llvm_instructions[num]:
                    func_data['llvm_lines'][num].append(inst.strip())

        with open(self.fancy, 'w') as output:
            step.Template(html).stream(output, TypeAnnotation.data.copy())

    def __str__(self):
        if self.fancy is not None:
            self.html_annotate()
            return self.annotate()
        else:
            return self.annotate()


re_longest_white_prefix = re.compile('^\s*')


def _getindent(text):
    m = re_longest_white_prefix.match(text)
    if not m:
        return ''
    else:
        return ' ' * len(m.group(0))
