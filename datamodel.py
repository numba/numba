class DataModel(object):
    def __init__(self, localty, dataty, retty, argty):
        """
        Args
        ----
        localty:
            Type representation within a function (e.g. on the stack)
        dataty:
            Type representation as storage (e.g. in an array)
        resty: type or sequence types
            Type representation as return value
            Can be a sequence of types.
        argty: type or sequence types
            Type representation as argument.
            Can be a sequence of types.
        """
        self._dataty = dataty
        self._localty = localty
        self._retty = retty
        self._argty = argty

    @property
    def argument_type(self):
        return self._argty

    @property
    def return_type(self):
        return self._retty

    @property
    def local_type(self):
        return self._localty

    @property
    def data_type(self):
        return self._dataty

    def as_data(self, builder, value):
        return value

    def as_argument(self, builder, value):
        return value

    def as_return(self, builder, value):
        return value

    def inverse_as_data(self, builder, value):
        return value

    def inverse_as_argument(self, builder, value):
        return value

    def inverse_as_return(self, builder, value):
        return value


class DataModelTest(object):
    def test_as_data(self, builder, value):
        tmp = self.as_data(builder, value)
        asvalue = self.inverse_as_data(builder, tmp)
        return value.type == asvalue

    def test_as_argument(self, builder, value):
        tmp = self.as_argument(builder, value)
        asvalue = self.inverse_as_argument(builder, tmp)
        return value.type == asvalue

    def test_as_return_value(self, builder, value):
        tmp = self.as_return(builder, value)
        asvalue = self.inverse_as_return(builder, tmp)
        return value.type == asvalue


class PrimitiveDataModel(DataModel):
    def __init__(self, dataty):
        super(PrimitiveDataModel, self).__init__(dataty, dataty, dataty,
                                                 dataty)


class FunctionModel(object):
    def __init__(self, retmodel, argmodels):
        pass
