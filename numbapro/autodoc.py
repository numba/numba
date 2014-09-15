import sys
import traceback
from sphinx.ext.autodoc import *


def import_object(self):
    """Import the object given by *self.modname* and *self.objpath* and set
    it as *self.object*.

    Returns True if successful, False if an error occurred.
    """
    dbg = self.env.app.debug
    if self.objpath:
        dbg('[autodoc] from %s import %s',
            self.modname, '.'.join(self.objpath))
    try:
        dbg('[autodoc] import %s', self.modname)

        parts = []
        for part in self.modname.split('.'):
            parts.append(part)
            try:
                __import__('.'.join(parts))
            except ImportError:
                if not parts:
                    raise

                parts.pop()
                break

        self.objpath = self.modname.split('.')[len(parts):] + self.objpath
        self.modname = '.'.join(parts)

        parent = None
        obj = self.module = sys.modules[self.modname]
        dbg('[autodoc] => %r', obj)
        for part in self.objpath:
            parent = obj
            dbg('[autodoc] getattr(_, %r)', part)
            obj = self.get_attr(obj, part)
            dbg('[autodoc] => %r', obj)
            self.object_name = part
        self.parent = parent
        self.object = obj
        return True
    # this used to only catch SyntaxError, ImportError and AttributeError,
    # but importing modules with side effects can raise all kinds of errors
    except (Exception, SystemExit) as e:
        if self.objpath:
            errmsg = 'autodoc: failed to import %s %r from module %r' % \
                     (self.objtype, '.'.join(self.objpath), self.modname)
        else:
            errmsg = 'autodoc: failed to import %s %r' % \
                     (self.objtype, self.fullname)
        if isinstance(e, SystemExit):
            errmsg += ('; the module executes module level statement ' +
                       'and it might call sys.exit().')
        else:
            errmsg += '; the following exception was raised:\n%s' % \
                      traceback.format_exc()
        dbg(errmsg)
        self.directive.warn(errmsg)
        self.env.note_reread()
        return False


Documenter.import_object = import_object
