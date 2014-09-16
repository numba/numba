from __future__ import print_function, absolute_import
from .pyalge import datatype

ArrayNode = datatype('ArrayNode', ['data', 'owners', 'depth'])
ArrayDataNode = datatype('ArrayDataNode', ['array_data', 'depth'])
ScalarNode = datatype('ScalarNode', ['value', 'depth'])
# JNB: rename to VariableArrayNode
VariableDataNode = datatype('VariableDataNode', ['name', 'depth'])
UFuncNode = datatype('UFuncNode', ['ufunc', 'args', 'depth'])

UnaryOperation = datatype('UnaryOperation', ['operand', 'op_str', 'depth'])
BinaryOperation = datatype('BinaryOperation', ['lhs', 'rhs', 'op_str', 'depth'])
ArrayAssignOperation = datatype('ArrayAssignOperation', ['operand', 'key', 'value', 'depth'])
WhereOperation = datatype('WhereOperation', ['cond', 'left', 'right', 'depth'])

