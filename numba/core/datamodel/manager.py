import weakref

from numba.core import types


class DataModelManager(object):
    """Manages mapping of FE types to their corresponding data model
    """

    def __init__(self):
        # { numba type class -> model factory }
        self._handlers = {}
        # { numba type instance -> model instance }
        self._cache = weakref.WeakKeyDictionary()

    def register(self, fetypecls, handler):
        """Register the datamodel factory corresponding to a frontend-type class
        """
        assert issubclass(fetypecls, types.Type)
        self._handlers[fetypecls] = handler

    def lookup(self, fetype):
        """Returns the corresponding datamodel given the frontend-type instance
        """
        try:
            return self._cache[fetype]
        except KeyError:
            pass
        handler = self._handlers[type(fetype)]
        model = self._cache[fetype] = handler(self, fetype)
        return model

    def __getitem__(self, fetype):
        """Shorthand for lookup()
        """
        return self.lookup(fetype)

    def copy(self):
        """
        Make a copy of the manager.
        Use this to inherit from the default data model and specialize it
        for custom target.
        """
        dmm = DataModelManager()
        dmm._handlers = self._handlers.copy()
        return dmm

