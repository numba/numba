from __future__ import print_function, absolute_import

from numba import types


class DataModelManager(object):
    """Manages mapping of FE types to their corresponding data model
    """

    def __init__(self):
        # handler map
        # key: numba.types.Type subclass
        # value: function
        self._handlers = {}

    def register(self, fetypecls, handler):
        """Register the datamodel factory corresponding to a frontend-type class
        """
        assert issubclass(fetypecls, types.Type)
        self._handlers[fetypecls] = handler

    def lookup(self, fetype):
        """Returns the corresponding datamodel given the frontend-type instance
        """
        handler = self._handlers[type(fetype)]
        return handler(self, fetype)

    def __getitem__(self, fetype):
        """Shorthand for lookup()
        """
        return self.lookup(fetype)
