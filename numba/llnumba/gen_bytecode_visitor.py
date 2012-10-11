#! /usr/bin/env python
# ______________________________________________________________________

import opcode_util

# ______________________________________________________________________

def generate_bytecode_visitor (classname = 'BytecodeVisitor',
                               baseclass = 'object'):
    opnames = list(set((opname.split('+')[0]
                        for opname in opcode_util.OPCODE_MAP.keys())))
    opnames.sort()
    return 'class %s (%s):\n%s\n' % (
        classname, baseclass,
        '\n\n'.join(('    def op_%s (self, i, op, arg):\n'
                     '        raise NotImplementedError("%s.op_%s")' %
                     (opname, classname, opname)
                     for opname in opnames)))

# ______________________________________________________________________

if __name__ == "__main__":
    import sys
    print(generate_bytecode_visitor(*sys.argv[1:]))

# ______________________________________________________________________
# End of gen_bytecode_visitor.py
