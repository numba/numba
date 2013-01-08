from numba.nodes import *

class TempNode(Node): #, ast.Name):
    """
    Create a temporary to store values in. Does not perform reference counting.
    """

    temp_counter = 0

    def __init__(self, type, name=None, dst_variable=None):
        self.type = type
        self.name = name
        self.variable = Variable(type, name='___numba_%d' % self.temp_counter,
                                 is_local=True)
        TempNode.temp_counter += 1
        self.llvm_temp = None

        self.dst_variable = dst_variable

    def load(self):
        return TempLoadNode(temp=self)

    def store(self):
        return TempStoreNode(temp=self)

    def __repr__(self):
        if self.name:
            name = ", %s" % self.name
        else:
            name = ""
        return "temp(%s%s)" % (self.type, name)

class TempLoadNode(Node):
    _fields = ['temp']

    def __init__(self, temp):
        self.temp = temp
        self.type = temp.type
        self.variable = Variable(self.type)

    def __repr__(self):
        return "load(%s)" % self.temp

class TempStoreNode(TempLoadNode):
    _fields = ['temp']

    def __repr__(self):
        return "store(%s)" % self.temp
