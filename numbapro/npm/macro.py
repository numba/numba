class Macro(object):
    '''A macro object is expanded to a function call
    '''
    __slots__ = 'name', 'func', 'callable'

    def __init__(self, name, func, callable=False):
        self.name = name
        self.func = func
        self.callable = callable

    def __repr__(self):
        return '<macro %s -> %s>' % (self.name, self.func)

