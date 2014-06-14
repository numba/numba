from pyalge import datatype

ArrayNode = datatype('ArrayNode', ['data', 'owners'])
ArrayDataNode = datatype('ArrayDataNode', ['array_data'])
ScalarNode = datatype('ScalarNode', ['value'])
# JNB: rename to VariableArrayNode
VariableDataNode = datatype('VariableDataNode', ['name'])
UFuncNode = datatype('UFuncNode', ['ufunc', 'args'])

UnaryOperation = datatype('UnaryOperation', ['operand', 'op_str'])
BinaryOperation = datatype('BinaryOperation', ['lhs', 'rhs', 'op_str'])
ArrayAssignOperation = datatype('ArrayAssignOperation', ['operand', 'key', 'value'])
WhereOperation = datatype('WhereOperation', ['cond', 'left', 'right'])

