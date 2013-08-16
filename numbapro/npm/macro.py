class Macro(object):
    '''A macro object is expanded to a function call
    '''
    __slots__ = 'name', 'func'

    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __repr__(self):
        return '<macro %s -> %s>' % (self.name, self.func)
