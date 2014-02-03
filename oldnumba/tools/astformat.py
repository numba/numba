# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ast, sys

def prettyprint(node, stream=sys.stdout):
    text = ast.dump(node)
    depth = 0
    last = ''
    for i in text:
        if i == ' ':
            continue # ignore space

        indent = ' ' * 4 * depth
        if last in ['(', '[', ','] and i not in [')', ']']:
            stream.write('\n' + indent)

        if i in ['(', '[']:
            depth += 1
        elif i in [')', ']']:
#            if last not in ['(', '[']:
#                stream.write('\n' + indent)
            depth -= 1

        stream.write(i)

        last = i
    stream.write('\n')
