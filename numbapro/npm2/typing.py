class Type(object):
    def __new__(cls, obj):
        if isinstance(obj, Type):
            return obj
        else:
            cls.check_protocol(obj)
            instance = object.__new__(cls)
            instance.obj = obj
            return instance

    @classmethod
    def check_protocol(cls, obj):
        assert hasattr(obj, 'coerce'), 'coerce method must be defined'

    def coerce(self, other, noraise=False):
        ret = self.obj.coerce(Type(other).obj)
        if ret is None and not noraise:
            raise TypeError('can not coerce %s -> %s' % (other, self))
        return ret


class Infer(object):
    def __init__(self, intp, args, return_type=None):
        print args
        print return_type
        self.args = args
        self.return_type = return_type

        self.typemap = {}

    def infer(self):
        return self.typemap

