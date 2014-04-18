from pyalge import datatype

ArrayNode = datatype('ArrayNode', ['data', 'owners'])
ArrayDataNode = datatype('ArrayDataNode', ['array_data'])
ScalarConstantNode = datatype('ScalarConstantNodeNode', ['value'])

UnaryOperation = datatype('UnaryOperation', ['operand', 'op', 'op_str'])
BinaryOperation = datatype('BinaryOperation', ['lhs', 'rhs', 'op', 'op_str'])
ArrayAssignOperation = datatype('ArrayAssignOperation', ['operand', 'key', 'value'])

