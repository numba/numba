/*
* Adapted from NumPy.
*/

#ifndef _NPY_PRIVATE__DATETIME_H_
#define _NPY_PRIVATE__DATETIME_H_


typedef enum {
        NUMBA_FR_Y = 0, /* Years */
        NUMBA_FR_M = 1, /* Months */
        NUMBA_FR_W = 2, /* Weeks */
        /* Gap where NUMBA_FR_B was */
        NUMBA_FR_D = 4, /* Days */
        NUMBA_FR_h = 5, /* hours */
        NUMBA_FR_m = 6, /* minutes */
        NUMBA_FR_s = 7, /* seconds */
        NUMBA_FR_ms = 8,/* milliseconds */
        NUMBA_FR_us = 9,/* microseconds */
        NUMBA_FR_ns = 10,/* nanoseconds */
        NUMBA_FR_ps = 11,/* picoseconds */
        NUMBA_FR_fs = 12,/* femtoseconds */
        NUMBA_FR_as = 13,/* attoseconds */
        NUMBA_FR_GENERIC = 14 /* Generic, unbound units, can convert to anything */
} NUMBA_DATETIMEUNIT;

#define NUMBA_DATETIME_NUMUNITS NUMBA_FR_GENERIC+1

#define NUMBA_DATETIME_MAX_ISO8601_STRLEN (21+3*5+1+3*6+6+1)

#define NUMBA_DATETIME_NAT NPY_MIN_INT64

typedef struct {
        npy_int64 year;
        npy_int32 month, day, hour, min, sec, us, ps, as;
} numba_datetimestruct;

typedef struct {
    NUMBA_DATETIMEUNIT base;
    int num;
} numba_datetime_metadata;


extern char *_datetime_strings[NUMBA_DATETIME_NUMUNITS];
extern int _days_per_month_table[2][12];

void
numpy_pydatetime_import(void);

/*
 * Returns 1 if the given year is a leap year, 0 otherwise.
 */
int
is_leapyear(npy_int64 year);

/*
 * Calculates the days offset from the 1970 epoch.
 */
npy_int64
get_datetimestruct_days(const numba_datetimestruct *dts);

/*
 * Creates a datetime or timedelta dtype using a copy of the provided metadata.
 */
PyArray_Descr *
create_datetime_dtype(int type_num, PyArray_DatetimeMetaData *meta);

/*
 * Creates a datetime or timedelta dtype using the given unit.
 */
PyArray_Descr *
create_datetime_dtype_with_unit(int type_num, NUMBA_DATETIMEUNIT unit);

/*
 * This function returns a pointer to the DateTimeMetaData
 * contained within the provided datetime dtype.
 */
PyArray_DatetimeMetaData *
get_datetime_metadata_from_dtype(PyArray_Descr *dtype);

/*
 * Both type1 and type2 must be either NPY_DATETIME or NPY_TIMEDELTA.
 * Applies the type promotion rules between the two types, returning
 * the promoted type.
 */
PyArray_Descr *
datetime_type_promotion(PyArray_Descr *type1, PyArray_Descr *type2);

/*
 * Converts a datetime from a datetimestruct to a datetime based
 * on some metadata.
 */
int
convert_datetimestruct_to_datetime(PyArray_DatetimeMetaData *meta,
                                    const numba_datetimestruct *dts,
                                    npy_datetime *out);

/*
 * Extracts the month number, within the current year,
 * from a 'datetime64[D]' value. January is 1, etc.
 */
int
days_to_month_number(npy_datetime days);

/*
 * Parses the metadata string into the metadata C structure.
 *
 * Returns 0 on success, -1 on failure.
 */
int
parse_datetime_metadata_from_metastr(char *metastr, Py_ssize_t len,
                                    PyArray_DatetimeMetaData *out_meta);


/*
 * Converts a datetype dtype string into a dtype descr object.
 * The "type" string should be NULL-terminated, and len should
 * contain its string length.
 */
PyArray_Descr *
parse_dtype_from_datetime_typestr(char *typestr, Py_ssize_t len);

/*
 * Converts a substring given by 'str' and 'len' into
 * a date time unit enum value. The 'metastr' parameter
 * is used for error messages, and may be NULL.
 *
 * Returns 0 on success, -1 on failure.
 */
NUMBA_DATETIMEUNIT
parse_datetime_unit_from_string(char *str, Py_ssize_t len, char *metastr);

/*
 * Translate divisors into multiples of smaller units.
 * 'metastr' is used for the error message if the divisor doesn't work,
 * and can be NULL if the metadata didn't come from a string.
 *
 * Returns 0 on success, -1 on failure.
 */
int
convert_datetime_divisor_to_multiple(PyArray_DatetimeMetaData *meta,
                                    int den, char *metastr);

/*
 * Determines whether the 'divisor' metadata divides evenly into
 * the 'dividend' metadata.
 */
npy_bool
datetime_metadata_divides(
                        PyArray_DatetimeMetaData *dividend,
                        PyArray_DatetimeMetaData *divisor,
                        int strict_with_nonlinear_units);

/*
 * This provides the casting rules for the DATETIME data type units.
 *
 * Notably, there is a barrier between 'date units' and 'time units'
 * for all but 'unsafe' casting.
 */
npy_bool
can_cast_datetime64_units(NUMBA_DATETIMEUNIT src_unit,
                          NUMBA_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting);

/*
 * This provides the casting rules for the DATETIME data type metadata.
 */
npy_bool
can_cast_datetime64_metadata(PyArray_DatetimeMetaData *src_meta,
                             PyArray_DatetimeMetaData *dst_meta,
                             NPY_CASTING casting);

/*
 * This provides the casting rules for the TIMEDELTA data type units.
 *
 * Notably, there is a barrier between the nonlinear years and
 * months units, and all the other units.
 */
npy_bool
can_cast_timedelta64_units(NUMBA_DATETIMEUNIT src_unit,
                          NUMBA_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting);

/*
 * This provides the casting rules for the TIMEDELTA data type metadata.
 */
npy_bool
can_cast_timedelta64_metadata(PyArray_DatetimeMetaData *src_meta,
                             PyArray_DatetimeMetaData *dst_meta,
                             NPY_CASTING casting);

/*
 * Computes the conversion factor to convert data with 'src_meta' metadata
 * into data with 'dst_meta' metadata.
 *
 * If overflow occurs, both out_num and out_denom are set to 0, but
 * no error is set.
 */
void
get_datetime_conversion_factor(PyArray_DatetimeMetaData *src_meta,
                                PyArray_DatetimeMetaData *dst_meta,
                                npy_int64 *out_num, npy_int64 *out_denom);

/*
 * Given a pointer to datetime metadata,
 * returns a tuple for pickling and other purposes.
 */
PyObject *
convert_datetime_metadata_to_tuple(PyArray_DatetimeMetaData *meta);

/*
 * Converts a metadata tuple into a datetime metadata C struct.
 *
 * Returns 0 on success, -1 on failure.
 */
int
convert_datetime_metadata_tuple_to_datetime_metadata(PyObject *tuple,
                                        PyArray_DatetimeMetaData *out_meta);

/*
 * Gets a tzoffset in minutes by calling the fromutc() function on
 * the Python datetime.tzinfo object.
 */
int
get_tzoffset_from_pytzinfo(PyObject *timezone, numba_datetimestruct *dts);

/*
 * Converts an input object into datetime metadata. The input
 * may be either a string or a tuple.
 *
 * Returns 0 on success, -1 on failure.
 */
int
convert_pyobject_to_datetime_metadata(PyObject *obj,
                                        PyArray_DatetimeMetaData *out_meta);

/*
 * 'ret' is a PyUString containing the datetime string, and this
 * function appends the metadata string to it.
 *
 * If 'skip_brackets' is true, skips the '[]'.
 *
 * This function steals the reference 'ret'
 */
PyObject *
append_metastr_to_string(PyArray_DatetimeMetaData *meta,
                                    int skip_brackets,
                                    PyObject *ret);

/*
 * Tests for and converts a Python datetime.datetime or datetime.date
 * object into a NumPy numba_datetimestruct.
 *
 * 'out_bestunit' gives a suggested unit based on whether the object
 *      was a datetime.date or datetime.datetime object.
 *
 * If 'apply_tzinfo' is 1, this function uses the tzinfo to convert
 * to UTC time, otherwise it returns the struct with the local time.
 *
 * Returns -1 on error, 0 on success, and 1 (with no error set)
 * if obj doesn't have the neeeded date or datetime attributes.
 */
int
convert_pydatetime_to_datetimestruct(PyObject *obj, numba_datetimestruct *out,
                                     NUMBA_DATETIMEUNIT *out_bestunit,
                                     int apply_tzinfo);

/*
 * Converts a PyObject * into a timedelta, in any of the forms supported
 *
 * If the units metadata isn't known ahead of time, set meta->base
 * to -1, and this function will populate meta with either default
 * values or values from the input object.
 *
 * The 'casting' parameter is used to control what kinds of inputs
 * are accepted, and what happens. For example, with 'unsafe' casting,
 * unrecognized inputs are converted to 'NaT' instead of throwing an error,
 * while with 'safe' casting an error will be thrown if any precision
 * from the input will be thrown away.
 *
 * Returns -1 on error, 0 on success.
 */
int
convert_pyobject_to_timedelta(PyArray_DatetimeMetaData *meta, PyObject *obj,
                                NPY_CASTING casting, npy_timedelta *out);

/*
 * Converts a datetime into a PyObject *.
 *
 * For days or coarser, returns a datetime.date.
 * For microseconds or coarser, returns a datetime.datetime.
 * For units finer than microseconds, returns an integer.
 */
PyObject *
convert_datetime_to_pyobject(npy_datetime dt, PyArray_DatetimeMetaData *meta);

/*
 * Converts a timedelta into a PyObject *.
 *
 * Not-a-time is returned as the string "NaT".
 * For microseconds or coarser, returns a datetime.timedelta.
 * For units finer than microseconds, returns an integer.
 */
PyObject *
convert_timedelta_to_pyobject(npy_timedelta td, PyArray_DatetimeMetaData *meta);

/*
 * Converts a datetime based on the given metadata into a datetimestruct
 */
int
convert_datetime_to_datetimestruct(PyArray_DatetimeMetaData *meta,
                                    npy_datetime dt,
                                    numba_datetimestruct *out);

/*
 * Converts a datetime from a datetimestruct to a datetime based
 * on some metadata. The date is assumed to be valid.
 *
 * TODO: If meta->num is really big, there could be overflow
 *
 * Returns 0 on success, -1 on failure.
 */
int
convert_datetimestruct_to_datetime(PyArray_DatetimeMetaData *meta,
                                    const numba_datetimestruct *dts,
                                    npy_datetime *out);

/*
 * Adjusts a datetimestruct based on a seconds offset. Assumes
 * the current values are valid.
 */
void
add_seconds_to_datetimestruct(numba_datetimestruct *dts, int seconds);

/*
 * Adjusts a datetimestruct based on a minutes offset. Assumes
 * the current values are valid.
 */
void
add_minutes_to_datetimestruct(numba_datetimestruct *dts, int minutes);

/*
 * Returns true if the datetime metadata matches
 */
npy_bool
has_equivalent_datetime_metadata(PyArray_Descr *type1, PyArray_Descr *type2);

/*
 * Casts a single datetime from having src_meta metadata into
 * dst_meta metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
int
cast_datetime_to_datetime(PyArray_DatetimeMetaData *src_meta,
                          PyArray_DatetimeMetaData *dst_meta,
                          npy_datetime src_dt,
                          npy_datetime *dst_dt);

/*
 * Casts a single timedelta from having src_meta metadata into
 * dst_meta metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
int
cast_timedelta_to_timedelta(PyArray_DatetimeMetaData *src_meta,
                          PyArray_DatetimeMetaData *dst_meta,
                          npy_timedelta src_dt,
                          npy_timedelta *dst_dt);

/*
 * Returns true if the object is something that is best considered
 * a Datetime or Timedelta, false otherwise.
 */
npy_bool
is_any_numpy_datetime_or_timedelta(PyObject *obj);

/*
 * Implements a datetime-specific arange
 */
PyArrayObject *
datetime_arange(PyObject *start, PyObject *stop, PyObject *step,
                PyArray_Descr *dtype);

/*
 * Examines all the objects in the given Python object by
 * recursively descending the sequence structure. Returns a
 * datetime or timedelta type with metadata based on the data.
 */
PyArray_Descr *
find_object_datetime_type(PyObject *obj, int type_num);

const char *
npy_casting_to_string(NPY_CASTING casting);

#endif
