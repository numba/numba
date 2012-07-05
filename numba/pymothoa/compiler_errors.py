# Copyright (c) 2012, Siu Kwan Lam
# All rights reserved.

class CompilerError(Exception):
    message = 'Unspecified compiler error.'
    def __init__(self, node_or_error, additional=''):
        if isinstance(node_or_error, CompilerError):
            assert additional
            error = node_or_error
            super(CompilerError, self).__init__('\n'.join([additional, str(error)]))
            self.line = error.line
            self.col = error.col
            self.inner = error
        else:
            node = node_or_error
            super(CompilerError, self).__init__('\n'.join([self.message, additional]))
            try:
                self.line = node.lineno
                self.col = node.col_offset
            except AttributeError:
                self.line = 1
                self.col = 0

    def is_due_to(self, errcls):
        if hasattr(self, 'inner'):
            return self.inner.is_due_to(errcls)
        else:
            return isinstance(self, errcls)

class VariableRedeclarationError(CompilerError):
    message = 'Variable redeclared.'

class InvalidCall(CompilerError):
    message = 'Invalid call to non-function object.'

class FunctionDeclarationError(CompilerError):
    message = 'Function declaration error.'

class UndefinedSymbolError(CompilerError):
    message = '''Symbol has not been defined.
Hint: All variables must be defined using var ( Name = Type, ... ) construct prior to use.'''

class InvalidUseOfConstruct(CompilerError):
    message = 'Invalid use of construct.'

class MissingReturnError(CompilerError):
    message = 'Function missing return statement.'

class InvalidReturnError(CompilerError):
    message = 'Invalid use of return statement.'

class InvalidSubscriptError(CompilerError):
    message = 'Type does not support subscripting.'

class InternalError(CompilerError):
    message = 'Internal Error.'

def wrap_by_function(e, func):
    '''Add source information to the exception.
    '''
    import inspect
    from pymothoa.util import terminal_helpers as term
    # Handle error and print useful information
    modname = func.func_globals['__name__']
    filename = func.func_globals['__file__']
    corrsource = inspect.getsourcelines(func)
    lineno = e.line-1
    corrline = corrsource[0][lineno].rstrip()

    prevlineno = lineno-1
    if prevlineno >= 0:
        prevline = corrsource[0][prevlineno].rstrip()
    else:
        prevline = ''

    nextlineno = lineno+1
    if nextlineno < len(corrsource[0]):
        nextline = corrsource[0][nextlineno].rstrip()
    else:
        nextline = ''

    loc = '(in %s:%d:%d)' % (
                    filename,
                    corrsource[1]+e.line-1,
                    e.col+1
                )

    corrptr = '-'*(e.col) + '^'

    template = '\n'.join([
                        term.header('\nWhen compiling function "%s.%s %s":'),
                        '%s',
                        term.fail('%s'),
                        '%s',
                        '%s\n'
                    ])

    msg = template %(
            modname,
            func.func_name,
            loc,
            prevline,
            corrline,
            corrptr,
            nextline,
        )
    raise CompilerError(e, msg)
