
Extending the Numba backend
===========================

.. todo:: write this

Helper Lib
----------

``numba/_helperlib.c`` addition of struct and adapter function:

.. code-block:: c

    typedef struct {
        double lo;
        double hi;
    } intervalstruct_t;

    static
    int Numba_adapt_interval(PyObject *obj, intervalstruct_t* ivstruct) {
        PyObject* lodata = PyObject_GetAttrString(obj, "lo");
        ivstruct->lo = PyFloat_AsDouble(lodata);
        PyObject* hidata = PyObject_GetAttrString(obj, "hi");
        ivstruct->hi = PyFloat_AsDouble(hidata);

        return 0;
    }

Building C Helpers dict:

.. code-block:: c

    static PyObject *
    build_c_helpers_dict(void)
    {
        PyObject *dct = PyDict_New();
        if (dct == NULL)
            goto error;

    #define declmethod(func) do {                          \
        PyObject *val = PyLong_FromVoidPtr(&Numba_##func); \
        if (val == NULL) goto error;                       \
        if (PyDict_SetItemString(dct, #func, val)) {       \
            Py_DECREF(val);                                \
            goto error;                                    \
        }                                                  \
        Py_DECREF(val);                                    \
    } while (0)

        declmethod(sdiv);
        declmethod(srem);
        declmethod(udiv);
        declmethod(urem);
        declmethod(cpow);
        declmethod(complex_adaptor);
        declmethod(extract_record_data);
        declmethod(release_record_buffer);
        declmethod(adapt_ndarray);
        declmethod(ndarray_new);
        declmethod(extract_np_datetime);
        declmethod(create_np_datetime);
        declmethod(extract_np_timedelta);
        declmethod(create_np_timedelta);
        declmethod(recreate_record);
        declmethod(round_even);
        declmethod(roundf_even);
        declmethod(fptoui);
        declmethod(fptouif);
        declmethod(gil_ensure);
        declmethod(gil_release);
        declmethod(adapt_interval);
    #define MATH_UNARY(F, R, A) declmethod(F);
    #define MATH_BINARY(F, R, A, B) declmethod(F);
        #include "mathnames.inc"
    #undef MATH_UNARY
    #undef MATH_BINARY

    #undef declmethod
        return dct;
    error:
        Py_XDECREF(dct);
        return NULL;
    }

Python API
----------

In ``numba.pythonapi``. Add to ``to_native_value``::

    elif isinstance(typ, types.IntervalType):
        return self.to_native_interval(obj)

Add methods::

    def to_native_interval(self, interval):
        voidptr = Type.pointer(Type.int(8))
        nativeivcls = self.context.make_interval()
        nativeiv = nativeivcls(self.context, self.builder)
        ivptr = nativeiv._getpointer()
        ptr = self.builder.bitcast(ivptr, voidptr)
        errcode = self.interval_adaptor(interval, ptr)
        failed = cgutils.is_not_null(self.builder, errcode)
        with cgutils.if_unlikely(self.builder, failed):
            # TODO
            self.builder.unreachable()
        return self.builder.load(ivptr)

    def interval_adaptor(self, interval, ptr):
        voidptr = Type.pointer(Type.int(8))
        fnty = Type.function(Type.int(), [self.pyobj, voidptr])
        fn = self._get_function(fnty, name="numba_adapt_interval")
        fn.args[0].add_attribute(lc.ATTR_NO_CAPTURE)
        fn.args[1].add_attribute(lc.ATTR_NO_CAPTURE)
        return self.builder.call(fn, (interval, ptr))

Target Interval Objects
-----------------------

``numba.targets.intervalobj.py``::

    from numba import cgutils, types
    from numba.targets.imputils import builtin_attr, impl_attribute

    def make_interval():
        """
        Return the Structure representation of an interval
        """

        # This structure should be kept in sync with Numba_adapt_interval()
        # in _helperlib.c.
        class IntervalTemplate(cgutils.Structure):
            _fields = [('lo', types.float64),
                       ('hi', types.float64),
                      ]

        return IntervalTemplate

    @builtin_attr
    @impl_attribute(types.Kind(types.IntervalType), 'lo', types.float64)
    def interval_lo(context, builder, typ, value):
        ivty = make_interval()
        iv = ivty(context, builder, value)
        return iv.lo

    @builtin_attr
    @impl_attribute(types.Kind(types.IntervalType), 'hi', types.float64)
    def interval_hi(context, builder, typ, value):
        ivty = make_interval()
        iv = ivty(context, builder, value)
        return iv.hi

Base Target
-----------

Add ``get_data_type`` handling for interval type and ``make_interval`` method.
