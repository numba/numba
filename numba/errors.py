# ----------------------------------------------------------------------
# Typing Error

class TypingError(Exception):
    """
    Errors during typing phase
    """

    def __init__(self, msg, loc=None):
        self.msg = msg
        self.loc = loc
        if loc:
            super(TypingError, self).__init__("%s\n%s" % (msg, loc.strformat()))
        else:
            super(TypingError, self).__init__("%s" % (msg,))


# ----------------------------------------------------------------------
# Lowering Error

class LoweringError(Exception):
    """
    Errors during lowering phase
    """

    def __init__(self, msg, loc):
        self.msg = msg
        self.loc = loc
        super(LoweringError, self).__init__("%s\n%s" % (msg, loc.strformat()))


class ForbiddenConstruct(LoweringError):
    pass
