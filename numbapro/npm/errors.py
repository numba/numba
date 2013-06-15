
class CompileError(Exception):
    def __init__(self, lineno, msg):
        msg = "At line %d: %s" % (lineno, msg)
        super(CompileError, self).__init__(msg)

