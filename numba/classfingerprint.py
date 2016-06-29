from __future__ import print_function, division, absolute_import

import types
import hashlib
import marshal
import pickle
import collections


class ClassFingerPrint(object):
    _member_kinds = [types.FunctionType, types.GetSetDescriptorType,
                     property, classmethod, staticmethod]

    def __init__(self, cls):
        if cls is object:
            self.name = repr(object)
            self.bases = ()
            self.dct = {}
        else:
            self.name = cls.__name__
            self.bases = cls.__bases__
            self.dct = cls.__dict__
        self._hasher = self._make_hash()

    def digest(self):
        return self._hasher.digest()

    def hexdigest(self):
        return self._hasher.hexdigest()

    def considered_class_members(self):
        return dict(self._gen_body_info())

    #
    # Internal API
    #

    def _make_hash(self):
        hasher = hashlib.md5()
        hasher.update(pickle.dumps(self._base_infos()))
        for base in self.bases:
            base_digest = ClassFingerPrint(base).digest()
            hasher.update(base_digest)
        for item in self._gen_body_info():
            hasher.update(pickle.dumps(item))
        return hasher

    def _base_infos(self):
        return self.name

    def _gen_body_info(self):
        for k, v in sorted(self.dct.items()):
            yield self._process_body(k, v)

    def _process_body(self, key, value):
        if key in ['__doc__', '__module__']:
            return key, value
        kinds = self._member_kinds
        for kind in kinds:
            if isinstance(value, kind):
                return key, self._handle(kind, value)
        else:
            # unknown class attributes
            fmt = "fail to handle {0} of {1}"
            raise TypeError(fmt.format(key, type(value)))

    def _handle(self, kind, value):
        fn = getattr(self, '_handle_' + kind.__name__)
        return kind.__name__, fn(value)

    def _handle_function(self, value):
        return _dump_function(value)

    def _handle_getset_descriptor(self, value):
        return   # ignored

    def _handle_property(self, value):
        data = []
        for attr in ['fget', 'fset', 'fdel']:
            fn = getattr(value, attr, None)
            if fn:
                data.append(_dump_function(fn))
        return data

    def _handle_staticmethod(self, value):
        return _dump_function(value.__func__)

    def _handle_classmethod(self, value):
        return _dump_function(value.__func__)


def _dump_function(fn):
    return marshal.dumps(fn.__code__)

