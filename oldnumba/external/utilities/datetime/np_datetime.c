/*
 * Adapted from NumPy.
 * This file implements core functionality for NumPy datetime.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <datetime.h>

#include <time.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/arrayobject.h>

#include "npy_config.h"
#include "npy_pycompat.h"

#include "numpy/arrayscalars.h"
#include "_datetime.h"
#include "np_datetime_strings.h"

/*
 * Imports the PyDateTime functions so we can create these objects.
 * This is called during module initialization
 */
void
numpy_pydatetime_import(void)
{
    PyDateTime_IMPORT;
}

/* Exported as DATETIMEUNITS in multiarraymodule.c */
char *_datetime_strings[NUMBA_DATETIME_NUMUNITS] = {
    "Y",
    "M",
    "W",
    "<invalid>",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us",
    "ns",
    "ps",
    "fs",
    "as",
    "generic"
};

/* Days per month, regular year and leap year */
int _days_per_month_table[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

/*
 * Returns 1 if the given year is a leap year, 0 otherwise.
 */
int
is_leapyear(npy_int64 year)
{
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 ||
            (year % 400) == 0);
}

/*
 * Calculates the days offset from the 1970 epoch.
 */
npy_int64
get_datetimestruct_days(const numba_datetimestruct *dts)
{
    int i, month;
    npy_int64 year, days = 0;
    int *month_lengths;

    year = dts->year - 1970;
    days = year * 365;

    /* Adjust for leap years */
    if (days >= 0) {
        /*
         * 1968 is the closest leap year before 1970.
         * Exclude the current year, so add 1.
         */
        year += 1;
        /* Add one day for each 4 years */
        days += year / 4;
        /* 1900 is the closest previous year divisible by 100 */
        year += 68;
        /* Subtract one day for each 100 years */
        days -= year / 100;
        /* 1600 is the closest previous year divisible by 400 */
        year += 300;
        /* Add one day for each 400 years */
        days += year / 400;
    }
    else {
        /*
         * 1972 is the closest later year after 1970.
         * Include the current year, so subtract 2.
         */
        year -= 2;
        /* Subtract one day for each 4 years */
        days += year / 4;
        /* 2000 is the closest later year divisible by 100 */
        year -= 28;
        /* Add one day for each 100 years */
        days -= year / 100;
        /* 2000 is also the closest later year divisible by 400 */
        /* Subtract one day for each 400 years */
        days += year / 400;
    }

    month_lengths = _days_per_month_table[is_leapyear(dts->year)];
    month = dts->month - 1;

    /* Add the months */
    for (i = 0; i < month; ++i) {
        days += month_lengths[i];
    }

    /* Add the days */
    days += dts->day - 1;

    return days;
}

/*
 * Calculates the minutes offset from the 1970 epoch.
 */
npy_int64
get_datetimestruct_minutes(const numba_datetimestruct *dts)
{
    npy_int64 days = get_datetimestruct_days(dts) * 24 * 60;
    days += dts->hour * 60;
    days += dts->min;

    return days;
}

/*
 * Modifies '*days_' to be the day offset within the year,
 * and returns the year.
 */
static npy_int64
days_to_yearsdays(npy_int64 *days_)
{
    const npy_int64 days_per_400years = (400*365 + 100 - 4 + 1);
    /* Adjust so it's relative to the year 2000 (divisible by 400) */
    npy_int64 days = (*days_) - (365*30 + 7);
    npy_int64 year;

    /* Break down the 400 year cycle to get the year and day within the year */
    if (days >= 0) {
        year = 400 * (days / days_per_400years);
        days = days % days_per_400years;
    }
    else {
        year = 400 * ((days - (days_per_400years - 1)) / days_per_400years);
        days = days % days_per_400years;
        if (days < 0) {
            days += days_per_400years;
        }
    }

    /* Work out the year/day within the 400 year cycle */
    if (days >= 366) {
        year += 100 * ((days-1) / (100*365 + 25 - 1));
        days = (days-1) % (100*365 + 25 - 1);
        if (days >= 365) {
            year += 4 * ((days+1) / (4*365 + 1));
            days = (days+1) % (4*365 + 1);
            if (days >= 366) {
                year += (days-1) / 365;
                days = (days-1) % 365;
            }
        }
    }

    *days_ = days;
    return year + 2000;
}

/* Extracts the month number from a 'datetime64[D]' value */
int
days_to_month_number(npy_datetime days)
{
    npy_int64 year;
    int *month_lengths, i;

    year = days_to_yearsdays(&days);
    month_lengths = _days_per_month_table[is_leapyear(year)];

    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            return i + 1;
        }
        else {
            days -= month_lengths[i];
        }
    }

    /* Should never get here */
    return 1;
}

/*
 * Fills in the year, month, day in 'dts' based on the days
 * offset from 1970.
 */
static void
set_datetimestruct_days(npy_int64 days, numba_datetimestruct *dts)
{
    int *month_lengths, i;

    dts->year = days_to_yearsdays(&days);
    month_lengths = _days_per_month_table[is_leapyear(dts->year)];

    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            dts->month = i + 1;
            dts->day = (int)days + 1;
            return;
        }
        else {
            days -= month_lengths[i];
        }
    }
}

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
                                    npy_datetime *out)
{
    npy_datetime ret;
    NUMBA_DATETIMEUNIT base = meta->base;

    /* If the datetimestruct is NaT, return NaT */
    if (dts->year == NUMBA_DATETIME_NAT) {
        *out = NUMBA_DATETIME_NAT;
        return 0;
    }

    /* Cannot instantiate a datetime with generic units */
    if (meta->base == NUMBA_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot create a NumPy datetime other than NaT "
                    "with generic units");
        return -1;
    }

    if (base == NUMBA_FR_Y) {
        /* Truncate to the year */
        ret = dts->year - 1970;
    }
    else if (base == NUMBA_FR_M) {
        /* Truncate to the month */
        ret = 12 * (dts->year - 1970) + (dts->month - 1);
    }
    else {
        /* Otherwise calculate the number of days to start */
        npy_int64 days = get_datetimestruct_days(dts);

        switch (base) {
            case NUMBA_FR_W:
                /* Truncate to weeks */
                if (days >= 0) {
                    ret = days / 7;
                }
                else {
                    ret = (days - 6) / 7;
                }
                break;
            case NUMBA_FR_D:
                ret = days;
                break;
            case NUMBA_FR_h:
                ret = days * 24 +
                      dts->hour;
                break;
            case NUMBA_FR_m:
                ret = (days * 24 +
                      dts->hour) * 60 +
                      dts->min;
                break;
            case NUMBA_FR_s:
                ret = ((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec;
                break;
            case NUMBA_FR_ms:
                ret = (((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000 +
                      dts->us / 1000;
                break;
            case NUMBA_FR_us:
                ret = (((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us;
                break;
            case NUMBA_FR_ns:
                ret = ((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000 +
                      dts->ps / 1000;
                break;
            case NUMBA_FR_ps:
                ret = ((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps;
                break;
            case NUMBA_FR_fs:
                /* only 2.6 hours */
                ret = (((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps) * 1000 +
                      dts->as / 1000;
                break;
            case NUMBA_FR_as:
                /* only 9.2 secs */
                ret = (((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps) * 1000000 +
                      dts->as;
                break;
            default:
                /* Something got corrupted */
                PyErr_SetString(PyExc_ValueError,
                        "NumPy datetime metadata with corrupt unit value");
                return -1;
        }
    }

    /* Divide by the multiplier */
    if (meta->num > 1) {
        if (ret >= 0) {
            ret /= meta->num;
        }
        else {
            ret = (ret - meta->num + 1) / meta->num;
        }
    }

    *out = ret;

    return 0;
}

/*
 * Converts a datetime based on the given metadata into a datetimestruct
 */
int
convert_datetime_to_datetimestruct(PyArray_DatetimeMetaData *meta,
                                    npy_datetime dt,
                                    numba_datetimestruct *out)
{
    npy_int64 perday;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(numba_datetimestruct));
    out->year = 1970;
    out->month = 1;
    out->day = 1;

    /* NaT is signaled in the year */
    if (dt == NUMBA_DATETIME_NAT) {
        out->year = NUMBA_DATETIME_NAT;
        return 0;
    }

    /* Datetimes can't be in generic units */
    if (meta->base == NUMBA_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot convert a NumPy datetime value other than NaT "
                    "with generic units");
        return -1;
    }

    /* TODO: Change to a mechanism that avoids the potential overflow */
    dt *= meta->num;

    /*
     * Note that care must be taken with the / and % operators
     * for negative values.
     */
    switch (meta->base) {
        case NUMBA_FR_Y:
            out->year = 1970 + dt;
            break;

        case NUMBA_FR_M:
            if (dt >= 0) {
                out->year  = 1970 + dt / 12;
                out->month = dt % 12 + 1;
            }
            else {
                out->year  = 1969 + (dt + 1) / 12;
                out->month = 12 + (dt + 1)% 12;
            }
            break;

        case NUMBA_FR_W:
            /* A week is 7 days */
            set_datetimestruct_days(dt * 7, out);
            break;

        case NUMBA_FR_D:
            set_datetimestruct_days(dt, out);
            break;

        case NUMBA_FR_h:
            perday = 24LL;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = (int)dt;
            break;

        case NUMBA_FR_m:
            perday = 24LL * 60;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = (int)(dt / 60);
            out->min = (int)(dt % 60);
            break;

        case NUMBA_FR_s:
            perday = 24LL * 60 * 60;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = (int)(dt / (60*60));
            out->min = (int)((dt / 60) % 60);
            out->sec = (int)(dt % 60);
            break;

        case NUMBA_FR_ms:
            perday = 24LL * 60 * 60 * 1000;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = (int)(dt / (60*60*1000LL));
            out->min = (int)((dt / (60*1000LL)) % 60);
            out->sec = (int)((dt / 1000LL) % 60);
            out->us = (int)((dt % 1000LL) * 1000);
            break;

        case NUMBA_FR_us:
            perday = 24LL * 60LL * 60LL * 1000LL * 1000LL;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = (int)(dt / (60*60*1000000LL));
            out->min = (int)((dt / (60*1000000LL)) % 60);
            out->sec = (int)((dt / 1000000LL) % 60);
            out->us = (int)(dt % 1000000LL);
            break;

        case NUMBA_FR_ns:
            perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = (int)(dt / (60*60*1000000000LL));
            out->min = (int)((dt / (60*1000000000LL)) % 60);
            out->sec = (int)((dt / 1000000000LL) % 60);
            out->us = (int)((dt / 1000LL) % 1000000LL);
            out->ps = (int)((dt % 1000LL) * 1000);
            break;

        case NUMBA_FR_ps:
            perday = 24LL * 60 * 60 * 1000 * 1000 * 1000 * 1000;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = (int)(dt / (60*60*1000000000000LL));
            out->min = (int)((dt / (60*1000000000000LL)) % 60);
            out->sec = (int)((dt / 1000000000000LL) % 60);
            out->us = (int)((dt / 1000000LL) % 1000000LL);
            out->ps = (int)(dt % 1000000LL);
            break;

        case NUMBA_FR_fs:
            /* entire range is only +- 2.6 hours */
            if (dt >= 0) {
                out->hour = (int)(dt / (60*60*1000000000000000LL));
                out->min = (int)((dt / (60*1000000000000000LL)) % 60);
                out->sec = (int)((dt / 1000000000000000LL) % 60);
                out->us = (int)((dt / 1000000000LL) % 1000000LL);
                out->ps = (int)((dt / 1000LL) % 1000000LL);
                out->as = (int)((dt % 1000LL) * 1000);
            }
            else {
                npy_datetime minutes;

                minutes = dt / (60*1000000000000000LL);
                dt = dt % (60*1000000000000000LL);
                if (dt < 0) {
                    dt += (60*1000000000000000LL);
                    --minutes;
                }
                /* Offset the negative minutes */
                add_minutes_to_datetimestruct(out, minutes);
                out->sec = (int)((dt / 1000000000000000LL) % 60);
                out->us = (int)((dt / 1000000000LL) % 1000000LL);
                out->ps = (int)((dt / 1000LL) % 1000000LL);
                out->as = (int)((dt % 1000LL) * 1000);
            }
            break;

        case NUMBA_FR_as:
            /* entire range is only +- 9.2 seconds */
            if (dt >= 0) {
                out->sec = (int)((dt / 1000000000000000000LL) % 60);
                out->us = (int)((dt / 1000000000000LL) % 1000000LL);
                out->ps = (int)((dt / 1000000LL) % 1000000LL);
                out->as = (int)(dt % 1000000LL);
            }
            else {
                npy_datetime seconds;

                seconds = dt / 1000000000000000000LL;
                dt = dt % 1000000000000000000LL;
                if (dt < 0) {
                    dt += 1000000000000000000LL;
                    --seconds;
                }
                /* Offset the negative seconds */
                add_seconds_to_datetimestruct(out, seconds);
                out->us = (int)((dt / 1000000000000LL) % 1000000LL);
                out->ps = (int)((dt / 1000000LL) % 1000000LL);
                out->as = (int)(dt % 1000000LL);
            }
            break;

        default:
            PyErr_SetString(PyExc_RuntimeError,
                        "NumPy datetime metadata is corrupted with invalid "
                        "base unit");
            return -1;
    }

    return 0;
}


/*
 * Converts a substring given by 'str' and 'len' into
 * a date time unit multiplier + enum value, which are populated
 * into out_meta. Other metadata is left along.
 *
 * 'metastr' is only used in the error message, and may be NULL.
 *
 * Returns 0 on success, -1 on failure.
 */
int
parse_datetime_extended_unit_from_string(char *str, Py_ssize_t len,
                                    char *metastr,
                                    PyArray_DatetimeMetaData *out_meta)
{
    char *substr = str, *substrend = NULL;
    int den = 1;

    /* First comes an optional integer multiplier */
    out_meta->num = (int)strtol(substr, &substrend, 10);
    if (substr == substrend) {
        out_meta->num = 1;
    }
    substr = substrend;

    /* Next comes the unit itself, followed by either '/' or the string end */
    substrend = substr;
    while (substrend-str < len && *substrend != '/') {
        ++substrend;
    }
    if (substr == substrend) {
        goto bad_input;
    }
    out_meta->base = parse_datetime_unit_from_string(substr,
                                        substrend-substr, metastr);
    if (out_meta->base == -1) {
        return -1;
    }
    substr = substrend;

    /* Next comes an optional integer denominator */
    if (substr-str < len && *substr == '/') {
        substr++;
        den = (int)strtol(substr, &substrend, 10);
        /* If the '/' exists, there must be a number followed by ']' */
        if (substr == substrend || *substrend != ']') {
            goto bad_input;
        }
        substr = substrend + 1;
    }
    else if (substr-str != len) {
        goto bad_input;
    }

    if (den != 1) {
        if (convert_datetime_divisor_to_multiple(
                                out_meta, den, metastr) < 0) {
            return -1;
        }
    }

    return 0;

bad_input:
    if (metastr != NULL) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\" at position %d",
                metastr, (int)(substr-metastr));
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\"",
                str);
    }

    return -1;
}

/*
 * Parses the metadata string into the metadata C structure.
 *
 * Returns 0 on success, -1 on failure.
 */
int
parse_datetime_metadata_from_metastr(char *metastr, Py_ssize_t len,
                                    PyArray_DatetimeMetaData *out_meta)
{
    char *substr = metastr, *substrend = NULL;

    /* Treat the empty string as generic units */
    if (len == 0) {
        out_meta->base = NUMBA_FR_GENERIC;
        out_meta->num = 1;

        return 0;
    }

    /* The metadata string must start with a '[' */
    if (len < 3 || *substr++ != '[') {
        goto bad_input;
    }

    substrend = substr;
    while (substrend - metastr < len && *substrend != ']') {
        ++substrend;
    }
    if (substrend - metastr == len || substr == substrend) {
        substr = substrend;
        goto bad_input;
    }

    /* Parse the extended unit inside the [] */
    if (parse_datetime_extended_unit_from_string(substr, substrend-substr,
                                                    metastr, out_meta) < 0) {
        return -1;
    }

    substr = substrend+1;

    if (substr - metastr != len) {
        goto bad_input;
    }

    return 0;

bad_input:
    if (substr != metastr) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\" at position %d",
                metastr, (int)(substr-metastr));
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\"",
                metastr);
    }

    return -1;
}

static NUMBA_DATETIMEUNIT _multiples_table[16][4] = {
    {12, 52, 365},                            /* NUMBA_FR_Y */
    {NUMBA_FR_M, NUMBA_FR_W, NUMBA_FR_D},
    {4,  30, 720},                            /* NUMBA_FR_M */
    {NUMBA_FR_W, NUMBA_FR_D, NUMBA_FR_h},
    {7,  168, 10080},                         /* NUMBA_FR_W */
    {NUMBA_FR_D, NUMBA_FR_h, NUMBA_FR_m},
    {0},                                      /* Gap for removed NUMBA_FR_B */
    {0},
    {24, 1440, 86400},                        /* NUMBA_FR_D */
    {NUMBA_FR_h, NUMBA_FR_m, NUMBA_FR_s},
    {60, 3600},                               /* NUMBA_FR_h */
    {NUMBA_FR_m, NUMBA_FR_s},
    {60, 60000},                              /* NUMBA_FR_m */
    {NUMBA_FR_s, NUMBA_FR_ms},
    {1000, 1000000},                          /* >=NUMBA_FR_s */
    {0, 0}
};



/*
 * Translate divisors into multiples of smaller units.
 * 'metastr' is used for the error message if the divisor doesn't work,
 * and can be NULL if the metadata didn't come from a string.
 *
 * This function only affects the 'base' and 'num' values in the metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
int
convert_datetime_divisor_to_multiple(PyArray_DatetimeMetaData *meta,
                                    int den, char *metastr)
{
    int i, num, ind;
    NUMBA_DATETIMEUNIT *totry;
    NUMBA_DATETIMEUNIT *baseunit;
    int q, r;

    if (meta->base == NUMBA_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
            "Can't use 'den' divisor with generic units");
        return -1;
    }

    ind = ((int)meta->base - (int)NUMBA_FR_Y)*2;
    totry = _multiples_table[ind];
    baseunit = _multiples_table[ind + 1];

    num = 3;
    if (meta->base == NUMBA_FR_W) {
        num = 4;
    }
    else if (meta->base > NUMBA_FR_D) {
        num = 2;
    }
    if (meta->base >= NUMBA_FR_s) {
        ind = ((int)NUMBA_FR_s - (int)NUMBA_FR_Y)*2;
        totry = _multiples_table[ind];
        baseunit = _multiples_table[ind + 1];
        baseunit[0] = meta->base + 1;
        baseunit[1] = meta->base + 2;
        if (meta->base == NUMBA_FR_as - 1) {
            num = 1;
        }
        if (meta->base == NUMBA_FR_as) {
            num = 0;
        }
    }

    for (i = 0; i < num; i++) {
        q = totry[i] / den;
        r = totry[i] % den;
        if (r == 0) {
            break;
        }
    }
    if (i == num) {
        if (metastr == NULL) {
            PyErr_Format(PyExc_ValueError,
                    "divisor (%d) is not a multiple of a lower-unit "
                    "in datetime metadata", den);
        }
        else {
            PyErr_Format(PyExc_ValueError,
                    "divisor (%d) is not a multiple of a lower-unit "
                    "in datetime metadata \"%s\"", den, metastr);
        }
        return -1;
    }
    meta->base = baseunit[i];
    meta->num *= q;

    return 0;
}

/*
 * Lookup table for factors between datetime units, except
 * for years and months.
 */
static npy_uint32
_datetime_factors[] = {
    1,  /* Years - not used */
    1,  /* Months - not used */
    7,  /* Weeks -> Days */
    1,  /* Business Days - was removed but a gap still exists in the enum */
    24, /* Days -> Hours */
    60, /* Hours -> Minutes */
    60, /* Minutes -> Seconds */
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1,   /* Attoseconds are the smallest base unit */
    0    /* Generic units don't have a conversion */
};

/*
 * Returns the scale factor between the units. Does not validate
 * that bigbase represents larger units than littlebase, or that
 * the units are not generic.
 *
 * Returns 0 if there is an overflow.
 */
static npy_uint64
get_datetime_units_factor(NUMBA_DATETIMEUNIT bigbase, NUMBA_DATETIMEUNIT littlebase)
{
    npy_uint64 factor = 1;
    int unit = (int)bigbase;
    while (littlebase > unit) {
        factor *= _datetime_factors[unit];
        /*
         * Detect overflow by disallowing the top 16 bits to be 1.
         * That alows a margin of error much bigger than any of
         * the datetime factors.
         */
        if (factor&0xff00000000000000ULL) {
            return 0;
        }
        ++unit;
    }
    return factor;
}

/* Euclidean algorithm on two positive numbers */
static npy_uint64
_uint64_euclidean_gcd(npy_uint64 x, npy_uint64 y)
{
    npy_uint64 tmp;

    if (x > y) {
        tmp = x;
        x = y;
        y = tmp;
    }
    while (x != y && y != 0) {
        tmp = x % y;
        x = y;
        y = tmp;
    }

    return x;
}

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
                                npy_int64 *out_num, npy_int64 *out_denom)
{
    int src_base, dst_base, swapped;
    npy_uint64 num = 1, denom = 1, tmp, gcd;

    /* Generic units change to the destination with no conversion factor */
    if (src_meta->base == NUMBA_FR_GENERIC) {
        *out_num = 1;
        *out_denom = 1;
        return;
    }
    /*
     * Converting to a generic unit from something other than a generic
     * unit is an error.
     */
    else if (dst_meta->base == NUMBA_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot convert from specific units to generic "
                    "units in NumPy datetimes or timedeltas");
        *out_num = 0;
        *out_denom = 0;
        return;
    }

    if (src_meta->base <= dst_meta->base) {
        src_base = src_meta->base;
        dst_base = dst_meta->base;
        swapped = 0;
    }
    else {
        src_base = dst_meta->base;
        dst_base = src_meta->base;
        swapped = 1;
    }

    if (src_base != dst_base) {
        /*
         * Conversions between years/months and other units use
         * the factor averaged over the 400 year leap year cycle.
         */
        if (src_base == NUMBA_FR_Y) {
            if (dst_base == NUMBA_FR_M) {
                num *= 12;
            }
            else if (dst_base == NUMBA_FR_W) {
                num *= (97 + 400*365);
                denom *= 400*7;
            }
            else {
                /* Year -> Day */
                num *= (97 + 400*365);
                denom *= 400;
                /* Day -> dst_base */
                num *= get_datetime_units_factor(NUMBA_FR_D, dst_base);
            }
        }
        else if (src_base == NUMBA_FR_M) {
            if (dst_base == NUMBA_FR_W) {
                num *= (97 + 400*365);
                denom *= 400*12*7;
            }
            else {
                /* Month -> Day */
                num *= (97 + 400*365);
                denom *= 400*12;
                /* Day -> dst_base */
                num *= get_datetime_units_factor(NUMBA_FR_D, dst_base);
            }
        }
        else {
            num *= get_datetime_units_factor(src_base, dst_base);
        }
    }

    /* If something overflowed, make both num and denom 0 */
    if (denom == 0) {
        PyErr_Format(PyExc_OverflowError,
                    "Integer overflow while computing the conversion "
                    "factor between NumPy datetime units %s and %s",
                    _datetime_strings[src_base],
                    _datetime_strings[dst_base]);
        *out_num = 0;
        *out_denom = 0;
        return;
    }

    /* Swap the numerator and denominator if necessary */
    if (swapped) {
        tmp = num;
        num = denom;
        denom = tmp;
    }

    num *= src_meta->num;
    denom *= dst_meta->num;

    /* Return as a fraction in reduced form */
    gcd = _uint64_euclidean_gcd(num, denom);
    *out_num = (npy_int64)(num / gcd);
    *out_denom = (npy_int64)(denom / gcd);
}

/*
 * Determines whether the 'divisor' metadata divides evenly into
 * the 'dividend' metadata.
 */
npy_bool
datetime_metadata_divides(
                        PyArray_DatetimeMetaData *dividend,
                        PyArray_DatetimeMetaData *divisor,
                        int strict_with_nonlinear_units)
{
    npy_uint64 num1, num2;

    /* Generic units divide into anything */
    if (divisor->base == NUMBA_FR_GENERIC) {
        return 1;
    }
    /* Non-generic units never divide into generic units */
    else if (dividend->base == NUMBA_FR_GENERIC) {
        return 0;
    }

    num1 = (npy_uint64)dividend->num;
    num2 = (npy_uint64)divisor->num;

    /* If the bases are different, factor in a conversion */
    if (dividend->base != divisor->base) {
        /*
         * Years and Months are incompatible with
         * all other units (except years and months are compatible
         * with each other).
         */
        if (dividend->base == NUMBA_FR_Y) {
            if (divisor->base == NUMBA_FR_M) {
                num1 *= 12;
            }
            else if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }
        else if (divisor->base == NUMBA_FR_Y) {
            if (dividend->base == NUMBA_FR_M) {
                num2 *= 12;
            }
            else if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }
        else if (dividend->base == NUMBA_FR_M || divisor->base == NUMBA_FR_M) {
            if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }

        /* Take the greater base (unit sizes are decreasing in enum) */
        if (dividend->base > divisor->base) {
            num2 *= get_datetime_units_factor(divisor->base, dividend->base);
            if (num2 == 0) {
                return 0;
            }
        }
        else {
            num1 *= get_datetime_units_factor(dividend->base, divisor->base);
            if (num1 == 0) {
                return 0;
            }
        }
    }

    /* Crude, incomplete check for overflow */
    if (num1&0xff00000000000000LL || num2&0xff00000000000000LL ) {
        return 0;
    }

    return (num1 % num2) == 0;
}

/*
 * This provides the casting rules for the DATETIME data type units.
 *
 * Notably, there is a barrier between 'date units' and 'time units'
 * for all but 'unsafe' casting.
 */
npy_bool
can_cast_datetime64_units(NUMBA_DATETIMEUNIT src_unit,
                          NUMBA_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting)
{
    switch (casting) {
        /* Allow anything with unsafe casting */
        case NPY_UNSAFE_CASTING:
            return 1;

        /*
         * Only enforce the 'date units' vs 'time units' barrier with
         * 'same_kind' casting.
         */
        case NPY_SAME_KIND_CASTING:
            if (src_unit == NUMBA_FR_GENERIC || dst_unit == NUMBA_FR_GENERIC) {
                return src_unit == dst_unit;
            }
            else {
                return (src_unit <= NUMBA_FR_D && dst_unit <= NUMBA_FR_D) ||
                       (src_unit > NUMBA_FR_D && dst_unit > NUMBA_FR_D);
            }

        /*
         * Enforce the 'date units' vs 'time units' barrier and that
         * casting is only allowed towards more precise units with
         * 'safe' casting.
         */
        case NPY_SAFE_CASTING:
            if (src_unit == NUMBA_FR_GENERIC || dst_unit == NUMBA_FR_GENERIC) {
                return src_unit == dst_unit;
            }
            else {
                return (src_unit <= dst_unit) &&
                       ((src_unit <= NUMBA_FR_D && dst_unit <= NUMBA_FR_D) ||
                        (src_unit > NUMBA_FR_D && dst_unit > NUMBA_FR_D));
            }

        /* Enforce equality with 'no' or 'equiv' casting */
        default:
            return src_unit == dst_unit;
    }
}

/*
 * This provides the casting rules for the TIMEDELTA data type units.
 *
 * Notably, there is a barrier between the nonlinear years and
 * months units, and all the other units.
 */
npy_bool
can_cast_timedelta64_units(NUMBA_DATETIMEUNIT src_unit,
                          NUMBA_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting)
{
    switch (casting) {
        /* Allow anything with unsafe casting */
        case NPY_UNSAFE_CASTING:
            return 1;

        /*
         * Only enforce the 'date units' vs 'time units' barrier with
         * 'same_kind' casting.
         */
        case NPY_SAME_KIND_CASTING:
            if (src_unit == NUMBA_FR_GENERIC || dst_unit == NUMBA_FR_GENERIC) {
                return src_unit == dst_unit;
            }
            else {
                return (src_unit <= NUMBA_FR_M && dst_unit <= NUMBA_FR_M) ||
                       (src_unit > NUMBA_FR_M && dst_unit > NUMBA_FR_M);
            }

        /*
         * Enforce the 'date units' vs 'time units' barrier and that
         * casting is only allowed towards more precise units with
         * 'safe' casting.
         */
        case NPY_SAFE_CASTING:
            if (src_unit == NUMBA_FR_GENERIC || dst_unit == NUMBA_FR_GENERIC) {
                return src_unit == dst_unit;
            }
            else {
                return (src_unit <= dst_unit) &&
                       ((src_unit <= NUMBA_FR_M && dst_unit <= NUMBA_FR_M) ||
                        (src_unit > NUMBA_FR_M && dst_unit > NUMBA_FR_M));
            }

        /* Enforce equality with 'no' or 'equiv' casting */
        default:
            return src_unit == dst_unit;
    }
}

/*
 * This provides the casting rules for the DATETIME data type metadata.
 */
npy_bool
can_cast_datetime64_metadata(PyArray_DatetimeMetaData *src_meta,
                             PyArray_DatetimeMetaData *dst_meta,
                             NPY_CASTING casting)
{
    switch (casting) {
        case NPY_UNSAFE_CASTING:
            return 1;

        case NPY_SAME_KIND_CASTING:
            return can_cast_datetime64_units(src_meta->base, dst_meta->base,
                                             casting);

        case NPY_SAFE_CASTING:
            return can_cast_datetime64_units(src_meta->base, dst_meta->base,
                                                             casting) &&
                   datetime_metadata_divides(src_meta, dst_meta, 0);

        default:
            return src_meta->base == dst_meta->base &&
                   src_meta->num == dst_meta->num;
    }
}

/*
 * This provides the casting rules for the TIMEDELTA data type metadata.
 */
npy_bool
can_cast_timedelta64_metadata(PyArray_DatetimeMetaData *src_meta,
                             PyArray_DatetimeMetaData *dst_meta,
                             NPY_CASTING casting)
{
    switch (casting) {
        case NPY_UNSAFE_CASTING:
            return 1;

        case NPY_SAME_KIND_CASTING:
            return can_cast_timedelta64_units(src_meta->base, dst_meta->base,
                                             casting);

        case NPY_SAFE_CASTING:
            return can_cast_timedelta64_units(src_meta->base, dst_meta->base,
                                                             casting) &&
                   datetime_metadata_divides(src_meta, dst_meta, 1);

        default:
            return src_meta->base == dst_meta->base &&
                   src_meta->num == dst_meta->num;
    }
}

const char *
npy_casting_to_string(NPY_CASTING casting)
{
    switch (casting) {
        case NPY_NO_CASTING:
            return "'no'";
        case NPY_EQUIV_CASTING:
            return "'equiv'";
        case NPY_SAFE_CASTING:
            return "'safe'";
        case NPY_SAME_KIND_CASTING:
            return "'same_kind'";
        case NPY_UNSAFE_CASTING:
            return "'unsafe'";
        default:
            return "<unknown>";
    }
}

/*
 * Tests whether a datetime64 can be cast from the source metadata
 * to the destination metadata according to the specified casting rule.
 *
 * Returns -1 if an exception was raised, 0 otherwise.
 */
int
raise_if_datetime64_metadata_cast_error(char *object_type,
                            PyArray_DatetimeMetaData *src_meta,
                            PyArray_DatetimeMetaData *dst_meta,
                            NPY_CASTING casting)
{
    if (can_cast_datetime64_metadata(src_meta, dst_meta, casting)) {
        return 0;
    }
    else {
        PyObject *errmsg;
        errmsg = PyUString_FromFormat("Cannot cast %s "
                    "from metadata ", object_type);
        errmsg = append_metastr_to_string(src_meta, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        errmsg = append_metastr_to_string(dst_meta, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        Py_DECREF(errmsg);
        return -1;
    }
}

/*
 * Tests whether a timedelta64 can be cast from the source metadata
 * to the destination metadata according to the specified casting rule.
 *
 * Returns -1 if an exception was raised, 0 otherwise.
 */
int
raise_if_timedelta64_metadata_cast_error(char *object_type,
                            PyArray_DatetimeMetaData *src_meta,
                            PyArray_DatetimeMetaData *dst_meta,
                            NPY_CASTING casting)
{
    if (can_cast_timedelta64_metadata(src_meta, dst_meta, casting)) {
        return 0;
    }
    else {
        PyObject *errmsg;
        errmsg = PyUString_FromFormat("Cannot cast %s "
                    "from metadata ", object_type);
        errmsg = append_metastr_to_string(src_meta, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        errmsg = append_metastr_to_string(dst_meta, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        Py_DECREF(errmsg);
        return -1;
    }
}

/*
 * Computes the GCD of the two date-time metadata values. Raises
 * an exception if there is no reasonable GCD, such as with
 * years and days.
 *
 * The result is placed in 'out_meta'.
 *
 * Returns 0 on success, -1 on failure.
 */
int
compute_datetime_metadata_greatest_common_divisor(
                        PyArray_DatetimeMetaData *meta1,
                        PyArray_DatetimeMetaData *meta2,
                        PyArray_DatetimeMetaData *out_meta,
                        int strict_with_nonlinear_units1,
                        int strict_with_nonlinear_units2)
{
    NUMBA_DATETIMEUNIT base;
    npy_uint64 num1, num2, num;

    /* If either unit is generic, adopt the metadata from the other one */
    if (meta1->base == NUMBA_FR_GENERIC) {
        *out_meta = *meta2;
        return 0;
    }
    else if (meta2->base == NUMBA_FR_GENERIC) {
        *out_meta = *meta1;
        return 0;
    }

    num1 = (npy_uint64)meta1->num;
    num2 = (npy_uint64)meta2->num;

    /* First validate that the units have a reasonable GCD */
    if (meta1->base == meta2->base) {
        base = meta1->base;
    }
    else {
        /*
         * Years and Months are incompatible with
         * all other units (except years and months are compatible
         * with each other).
         */
        if (meta1->base == NUMBA_FR_Y) {
            if (meta2->base == NUMBA_FR_M) {
                base = NUMBA_FR_M;
                num1 *= 12;
            }
            else if (strict_with_nonlinear_units1) {
                goto incompatible_units;
            }
            else {
                base = meta2->base;
                /* Don't multiply num1 since there is no even factor */
            }
        }
        else if (meta2->base == NUMBA_FR_Y) {
            if (meta1->base == NUMBA_FR_M) {
                base = NUMBA_FR_M;
                num2 *= 12;
            }
            else if (strict_with_nonlinear_units2) {
                goto incompatible_units;
            }
            else {
                base = meta1->base;
                /* Don't multiply num2 since there is no even factor */
            }
        }
        else if (meta1->base == NUMBA_FR_M) {
            if (strict_with_nonlinear_units1) {
                goto incompatible_units;
            }
            else {
                base = meta2->base;
                /* Don't multiply num1 since there is no even factor */
            }
        }
        else if (meta2->base == NUMBA_FR_M) {
            if (strict_with_nonlinear_units2) {
                goto incompatible_units;
            }
            else {
                base = meta1->base;
                /* Don't multiply num2 since there is no even factor */
            }
        }

        /* Take the greater base (unit sizes are decreasing in enum) */
        if (meta1->base > meta2->base) {
            base = meta1->base;
            num2 *= get_datetime_units_factor(meta2->base, meta1->base);
            if (num2 == 0) {
                goto units_overflow;
            }
        }
        else {
            base = meta2->base;
            num1 *= get_datetime_units_factor(meta1->base, meta2->base);
            if (num1 == 0) {
                goto units_overflow;
            }
        }
    }

    /* Compute the GCD of the resulting multipliers */
    num = _uint64_euclidean_gcd(num1, num2);

    /* Fill the 'out_meta' values */
    out_meta->base = base;
    out_meta->num = (int)num;
    if (out_meta->num <= 0 || num != (npy_uint64)out_meta->num) {
        goto units_overflow;
    }

    return 0;

incompatible_units: {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Cannot get "
                    "a common metadata divisor for "
                    "NumPy datetime metadata ");
        errmsg = append_metastr_to_string(meta1, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" and "));
        errmsg = append_metastr_to_string(meta2, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" because they have "
                    "incompatible nonlinear base time units"));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        Py_DECREF(errmsg);
        return -1;
    }
units_overflow: {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Integer overflow "
                    "getting a common metadata divisor for "
                    "NumPy datetime metadata ");
        errmsg = append_metastr_to_string(meta1, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" and "));
        errmsg = append_metastr_to_string(meta2, 0, errmsg);
        PyErr_SetObject(PyExc_OverflowError, errmsg);
        Py_DECREF(errmsg);
        return -1;
    }
}

/*
 * Converts a substring given by 'str' and 'len' into
 * a date time unit enum value. The 'metastr' parameter
 * is used for error messages, and may be NULL.
 *
 * Generic units have no representation as a string in this form.
 *
 * Returns 0 on success, -1 on failure.
 */
NUMBA_DATETIMEUNIT
parse_datetime_unit_from_string(char *str, Py_ssize_t len, char *metastr)
{
    /* Use switch statements so the compiler can make it fast */
    if (len == 1) {
        switch (str[0]) {
            case 'Y':
                return NUMBA_FR_Y;
            case 'M':
                return NUMBA_FR_M;
            case 'W':
                return NUMBA_FR_W;
            case 'D':
                return NUMBA_FR_D;
            case 'h':
                return NUMBA_FR_h;
            case 'm':
                return NUMBA_FR_m;
            case 's':
                return NUMBA_FR_s;
        }
    }
    /* All the two-letter units are variants of seconds */
    else if (len == 2 && str[1] == 's') {
        switch (str[0]) {
            case 'm':
                return NUMBA_FR_ms;
            case 'u':
                return NUMBA_FR_us;
            case 'n':
                return NUMBA_FR_ns;
            case 'p':
                return NUMBA_FR_ps;
            case 'f':
                return NUMBA_FR_fs;
            case 'a':
                return NUMBA_FR_as;
        }
    }

    /* If nothing matched, it's an error */
    if (metastr == NULL) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime unit \"%s\" in metadata",
                str);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime unit in metadata string \"%s\"",
                metastr);
    }
    return -1;
}


PyObject *
convert_datetime_metadata_to_tuple(PyArray_DatetimeMetaData *meta)
{
    PyObject *dt_tuple;

    dt_tuple = PyTuple_New(2);
    if (dt_tuple == NULL) {
        return NULL;
    }

    PyTuple_SET_ITEM(dt_tuple, 0,
            PyUString_FromString(_datetime_strings[meta->base]));
    PyTuple_SET_ITEM(dt_tuple, 1,
            PyInt_FromLong(meta->num));

    return dt_tuple;
}

/*
 * Converts a metadata tuple into a datetime metadata C struct.
 *
 * Returns 0 on success, -1 on failure.
 */
int
convert_datetime_metadata_tuple_to_datetime_metadata(PyObject *tuple,
                                        PyArray_DatetimeMetaData *out_meta)
{
    char *basestr = NULL;
    Py_ssize_t len = 0, tuple_size;
    int den = 1;
    PyObject *unit_str = NULL;

    if (!PyTuple_Check(tuple)) {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Require tuple for tuple to NumPy "
                                      "datetime metadata conversion, not ");
        PyUString_ConcatAndDel(&errmsg, PyObject_Repr(tuple));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        Py_DECREF(errmsg);
        return -1;
    }

    tuple_size = PyTuple_GET_SIZE(tuple);
    if (tuple_size < 2 || tuple_size > 4) {
        PyErr_SetString(PyExc_TypeError,
                        "Require tuple of size 2 to 4 for "
                        "tuple to NumPy datetime metadata conversion");
        return -1;
    }

    unit_str = PyTuple_GET_ITEM(tuple, 0);
    Py_INCREF(unit_str);
    if (PyUnicode_Check(unit_str)) {
        /* Allow unicode format strings: convert to bytes */
        PyObject *tmp = PyUnicode_AsASCIIString(unit_str);
        Py_DECREF(unit_str);
        if (tmp == NULL) {
            return -1;
        }
        unit_str = tmp;
    }
    if (PyBytes_AsStringAndSize(unit_str, &basestr, &len) < 0) {
        Py_DECREF(unit_str);
        return -1;
    }

    out_meta->base = parse_datetime_unit_from_string(basestr, len, NULL);
    if (out_meta->base == -1) {
        Py_DECREF(unit_str);
        return -1;
    }

    Py_DECREF(unit_str);

    /* Convert the values to longs */
    out_meta->num = PyInt_AsLong(PyTuple_GET_ITEM(tuple, 1));
    if (out_meta->num == -1 && PyErr_Occurred()) {
        return -1;
    }

    if (tuple_size == 4) {
        den = PyInt_AsLong(PyTuple_GET_ITEM(tuple, 2));
        if (den == -1 && PyErr_Occurred()) {
            return -1;
        }
    }

    if (out_meta->num <= 0 || den <= 0) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid tuple values for "
                        "tuple to NumPy datetime metadata conversion");
        return -1;
    }

    if (den != 1) {
        if (convert_datetime_divisor_to_multiple(out_meta, den, NULL) < 0) {
            return -1;
        }
    }

    return 0;
}

/*
 * Converts an input object into datetime metadata. The input
 * may be either a string or a tuple.
 *
 * Returns 0 on success, -1 on failure.
 */
int
convert_pyobject_to_datetime_metadata(PyObject *obj,
                                      PyArray_DatetimeMetaData *out_meta)
{
    PyObject *ascii = NULL;
    char *str = NULL;
    Py_ssize_t len = 0;

    if (PyTuple_Check(obj)) {
        return convert_datetime_metadata_tuple_to_datetime_metadata(obj,
                                                                out_meta);
    }

    /* Get an ASCII string */
    if (PyUnicode_Check(obj)) {
        /* Allow unicode format strings: convert to bytes */
        ascii = PyUnicode_AsASCIIString(obj);
        if (ascii == NULL) {
            return -1;
        }
    }
    else if (PyBytes_Check(obj)) {
        ascii = obj;
        Py_INCREF(ascii);
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "Invalid object for specifying NumPy datetime metadata");
        return -1;
    }

    if (PyBytes_AsStringAndSize(ascii, &str, &len) < 0) {
        return -1;
    }

    if (len > 0 && str[0] == '[') {
        return parse_datetime_metadata_from_metastr(str, len, out_meta);
    }
    else {
        if (parse_datetime_extended_unit_from_string(str, len,
                                                NULL, out_meta) < 0) {
            return -1;
        }

        return 0;
    }

}

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
                                    PyObject *ret)
{
    PyObject *res;
    int num;
    char *basestr;

    if (ret == NULL) {
        return NULL;
    }

    if (meta->base == NUMBA_FR_GENERIC) {
        /* Without brackets, give a string "generic" */
        if (skip_brackets) {
            PyUString_ConcatAndDel(&ret, PyUString_FromString("generic"));
            return ret;
        }
        /* But with brackets, append nothing */
        else {
            return ret;
        }
    }

    num = meta->num;
    if (meta->base >= 0 && meta->base < NUMBA_DATETIME_NUMUNITS) {
        basestr = _datetime_strings[meta->base];
    }
    else {
        PyErr_SetString(PyExc_RuntimeError,
                "NumPy datetime metadata is corrupted");
        return NULL;
    }

    if (num == 1) {
        if (skip_brackets) {
            res = PyUString_FromFormat("%s", basestr);
        }
        else {
            res = PyUString_FromFormat("[%s]", basestr);
        }
    }
    else {
        if (skip_brackets) {
            res = PyUString_FromFormat("%d%s", num, basestr);
        }
        else {
            res = PyUString_FromFormat("[%d%s]", num, basestr);
        }
    }

    PyUString_ConcatAndDel(&ret, res);
    return ret;
}

/*
 * Adjusts a datetimestruct based on a seconds offset. Assumes
 * the current values are valid.
 */
void
add_seconds_to_datetimestruct(numba_datetimestruct *dts, int seconds)
{
    int minutes;

    dts->sec += seconds;
    if (dts->sec < 0) {
        minutes = dts->sec / 60;
        dts->sec = dts->sec % 60;
        if (dts->sec < 0) {
            --minutes;
            dts->sec += 60;
        }
        add_minutes_to_datetimestruct(dts, minutes);
    }
    else if (dts->sec >= 60) {
        minutes = dts->sec / 60;
        dts->sec = dts->sec % 60;
        add_minutes_to_datetimestruct(dts, minutes);
    }
}

/*
 * Adjusts a datetimestruct based on a minutes offset. Assumes
 * the current values are valid.
 */
void
add_minutes_to_datetimestruct(numba_datetimestruct *dts, int minutes)
{
    int isleap;

    /* MINUTES */
    dts->min += minutes;
    while (dts->min < 0) {
        dts->min += 60;
        dts->hour--;
    }
    while (dts->min >= 60) {
        dts->min -= 60;
        dts->hour++;
    }

    /* HOURS */
    while (dts->hour < 0) {
        dts->hour += 24;
        dts->day--;
    }
    while (dts->hour >= 24) {
        dts->hour -= 24;
        dts->day++;
    }

    /* DAYS */
    if (dts->day < 1) {
        dts->month--;
        if (dts->month < 1) {
            dts->year--;
            dts->month = 12;
        }
        isleap = is_leapyear(dts->year);
        dts->day += _days_per_month_table[isleap][dts->month-1];
    }
    else if (dts->day > 28) {
        isleap = is_leapyear(dts->year);
        if (dts->day > _days_per_month_table[isleap][dts->month-1]) {
            dts->day -= _days_per_month_table[isleap][dts->month-1];
            dts->month++;
            if (dts->month > 12) {
                dts->year++;
                dts->month = 1;
            }
        }
    }
}

/*
 * Tests for and converts a Python datetime.datetime or datetime.date
 * object into a NumPy numba_datetimestruct.
 *
 * While the C API has PyDate_* and PyDateTime_* functions, the following
 * implementation just asks for attributes, and thus supports
 * datetime duck typing. The tzinfo time zone conversion would require
 * this style of access anyway.
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
                                     int apply_tzinfo)
{
    PyObject *tmp;
    int isleap;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(numba_datetimestruct));
    out->month = 1;
    out->day = 1;

    /* Need at least year/month/day attributes */
    if (!PyObject_HasAttrString(obj, "year") ||
            !PyObject_HasAttrString(obj, "month") ||
            !PyObject_HasAttrString(obj, "day")) {
        return 1;
    }

    /* Get the year */
    tmp = PyObject_GetAttrString(obj, "year");
    if (tmp == NULL) {
        return -1;
    }
    out->year = PyInt_AsLong(tmp);
    if (out->year == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the month */
    tmp = PyObject_GetAttrString(obj, "month");
    if (tmp == NULL) {
        return -1;
    }
    out->month = PyInt_AsLong(tmp);
    if (out->month == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the day */
    tmp = PyObject_GetAttrString(obj, "day");
    if (tmp == NULL) {
        return -1;
    }
    out->day = PyInt_AsLong(tmp);
    if (out->day == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Validate that the month and day are valid for the year */
    if (out->month < 1 || out->month > 12) {
        goto invalid_date;
    }
    isleap = is_leapyear(out->year);
    if (out->day < 1 ||
                out->day > _days_per_month_table[isleap][out->month-1]) {
        goto invalid_date;
    }

    /* Check for time attributes (if not there, return success as a date) */
    if (!PyObject_HasAttrString(obj, "hour") ||
            !PyObject_HasAttrString(obj, "minute") ||
            !PyObject_HasAttrString(obj, "second") ||
            !PyObject_HasAttrString(obj, "microsecond")) {
        /* The best unit for date is 'D' */
        if (out_bestunit != NULL) {
            *out_bestunit = NUMBA_FR_D;
        }
        return 0;
    }

    /* Get the hour */
    tmp = PyObject_GetAttrString(obj, "hour");
    if (tmp == NULL) {
        return -1;
    }
    out->hour = PyInt_AsLong(tmp);
    if (out->hour == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the minute */
    tmp = PyObject_GetAttrString(obj, "minute");
    if (tmp == NULL) {
        return -1;
    }
    out->min = PyInt_AsLong(tmp);
    if (out->min == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the second */
    tmp = PyObject_GetAttrString(obj, "second");
    if (tmp == NULL) {
        return -1;
    }
    out->sec = PyInt_AsLong(tmp);
    if (out->sec == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the microsecond */
    tmp = PyObject_GetAttrString(obj, "microsecond");
    if (tmp == NULL) {
        return -1;
    }
    out->us = PyInt_AsLong(tmp);
    if (out->us == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    if (out->hour < 0 || out->hour >= 24 ||
            out->min < 0 || out->min >= 60 ||
            out->sec < 0 || out->sec >= 60 ||
            out->us < 0 || out->us >= 1000000) {
        goto invalid_time;
    }

    /* Apply the time zone offset if it exists */
    if (apply_tzinfo && PyObject_HasAttrString(obj, "tzinfo")) {
        tmp = PyObject_GetAttrString(obj, "tzinfo");
        if (tmp == NULL) {
            return -1;
        }
        if (tmp == Py_None) {
            Py_DECREF(tmp);
        }
        else {
            PyObject *offset;
            int seconds_offset, minutes_offset;

            /* The utcoffset function should return a timedelta */
            offset = PyObject_CallMethod(tmp, "utcoffset", "O", obj);
            if (offset == NULL) {
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp);

            /*
             * The timedelta should have a function "total_seconds"
             * which contains the value we want.
             */
            tmp = PyObject_CallMethod(offset, "total_seconds", "");
            if (tmp == NULL) {
                return -1;
            }
            seconds_offset = PyInt_AsLong(tmp);
            if (seconds_offset == -1 && PyErr_Occurred()) {
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp);

            /* Convert to a minutes offset and apply it */
            minutes_offset = seconds_offset / 60;

            add_minutes_to_datetimestruct(out, -minutes_offset);
        }
    }

    /* The resolution of Python's datetime is 'us' */
    if (out_bestunit != NULL) {
        *out_bestunit = NUMBA_FR_us;
    }

    return 0;

invalid_date:
    PyErr_Format(PyExc_ValueError,
            "Invalid date (%d,%d,%d) when converting to NumPy datetime",
            (int)out->year, (int)out->month, (int)out->day);
    return -1;

invalid_time:
    PyErr_Format(PyExc_ValueError,
            "Invalid time (%d,%d,%d,%d) when converting "
            "to NumPy datetime",
            (int)out->hour, (int)out->min, (int)out->sec, (int)out->us);
    return -1;
}

/*
 * Gets a tzoffset in minutes by calling the fromutc() function on
 * the Python datetime.tzinfo object.
 */
int
get_tzoffset_from_pytzinfo(PyObject *timezone_obj, numba_datetimestruct *dts)
{
    PyObject *dt, *loc_dt;
    numba_datetimestruct loc_dts;

    /* Create a Python datetime to give to the timezone object */
    dt = PyDateTime_FromDateAndTime((int)dts->year, dts->month, dts->day,
                            dts->hour, dts->min, 0, 0);
    if (dt == NULL) {
        return -1;
    }

    /* Convert the datetime from UTC to local time */
    loc_dt = PyObject_CallMethod(timezone_obj, "fromutc", "O", dt);
    Py_DECREF(dt);
    if (loc_dt == NULL) {
        return -1;
    }

    /* Convert the local datetime into a datetimestruct */
    if (convert_pydatetime_to_datetimestruct(loc_dt, &loc_dts, NULL, 0) < 0) {
        Py_DECREF(loc_dt);
        return -1;
    }

    Py_DECREF(loc_dt);

    /* Calculate the tzoffset as the difference between the datetimes */
    return (int)(get_datetimestruct_minutes(&loc_dts) -
                 get_datetimestruct_minutes(dts));
}


/*
 * Converts a datetime into a PyObject *.
 *
 * Not-a-time is returned as the string "NaT".
 * For days or coarser, returns a datetime.date.
 * For microseconds or coarser, returns a datetime.datetime.
 * For units finer than microseconds, returns an integer.
 */
PyObject *
convert_datetime_to_pyobject(npy_datetime dt, PyArray_DatetimeMetaData *meta)
{
    PyObject *ret = NULL;
    numba_datetimestruct dts;

    /*
     * Convert NaT (not-a-time) and any value with generic units
     * into None.
     */
    if (dt == NUMBA_DATETIME_NAT || meta->base == NUMBA_FR_GENERIC) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    /* If the type's precision is greater than microseconds, return an int */
    if (meta->base > NUMBA_FR_us) {
        return PyLong_FromLongLong(dt);
    }

    /* Convert to a datetimestruct */
    if (convert_datetime_to_datetimestruct(meta, dt, &dts) < 0) {
        return NULL;
    }

    /*
     * If the year is outside the range of years supported by Python's
     * datetime, or the datetime64 falls on a leap second,
     * return a raw int.
     */
    if (dts.year < 1 || dts.year > 9999 || dts.sec == 60) {
        return PyLong_FromLongLong(dt);
    }

    /* If the type's precision is greater than days, return a datetime */
    if (meta->base > NUMBA_FR_D) {
        ret = PyDateTime_FromDateAndTime(dts.year, dts.month, dts.day,
                                dts.hour, dts.min, dts.sec, dts.us);
    }
    /* Otherwise return a date */
    else {
        ret = PyDate_FromDate(dts.year, dts.month, dts.day);
    }

    return ret;
}

/*
 * Converts a timedelta into a PyObject *.
 *
 * Not-a-time is returned as the string "NaT".
 * For microseconds or coarser, returns a datetime.timedelta.
 * For units finer than microseconds, returns an integer.
 */
PyObject *
convert_timedelta_to_pyobject(npy_timedelta td, PyArray_DatetimeMetaData *meta)
{
    PyObject *ret = NULL;
    npy_timedelta value;
    int days = 0, seconds = 0, useconds = 0;

    /*
     * Convert NaT (not-a-time) into None.
     */
    if (td == NUMBA_DATETIME_NAT) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    /*
     * If the type's precision is greater than microseconds, is
     * Y/M/B (nonlinear units), or is generic units, return an int
     */
    if (meta->base > NUMBA_FR_us ||
                    meta->base == NUMBA_FR_Y ||
                    meta->base == NUMBA_FR_M ||
                    meta->base == NUMBA_FR_GENERIC) {
        return PyLong_FromLongLong(td);
    }

    value = td;

    /* Apply the unit multiplier (TODO: overflow treatment...) */
    value *= meta->num;

    /* Convert to days/seconds/useconds */
    switch (meta->base) {
        case NUMBA_FR_W:
            value *= 7;
            break;
        case NUMBA_FR_D:
            break;
        case NUMBA_FR_h:
            seconds = (int)((value % 24) * (60*60));
            value = value / 24;
            break;
        case NUMBA_FR_m:
            seconds = (int)(value % (24*60)) * 60;
            value = value / (24*60);
            break;
        case NUMBA_FR_s:
            seconds = (int)(value % (24*60*60));
            value = value / (24*60*60);
            break;
        case NUMBA_FR_ms:
            useconds = (int)(value % 1000) * 1000;
            value = value / 1000;
            seconds = (int)(value % (24*60*60));
            value = value / (24*60*60);
            break;
        case NUMBA_FR_us:
            useconds = (int)(value % (1000*1000));
            value = value / (1000*1000);
            seconds = (int)(value % (24*60*60));
            value = value / (24*60*60);
            break;
        default:
            break;
    }
    /*
     * 'value' represents days, and seconds/useconds are filled.
     *
     * If it would overflow the datetime.timedelta days, return a raw int
     */
    if (value < -999999999 || value > 999999999) {
        return PyLong_FromLongLong(td);
    }
    else {
        days = (int)value;
        ret = PyDelta_FromDSU(days, seconds, useconds);
        if (ret == NULL) {
            return NULL;
        }
    }

    return ret;
}


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
                          npy_datetime *dst_dt)
{
    numba_datetimestruct dts;

    /* If the metadata is the same, short-circuit the conversion */
    if (src_meta->base == dst_meta->base &&
            src_meta->num == dst_meta->num) {
        *dst_dt = src_dt;
        return 0;
    }

    /* Otherwise convert through a datetimestruct */
    if (convert_datetime_to_datetimestruct(src_meta, src_dt, &dts) < 0) {
            *dst_dt = NUMBA_DATETIME_NAT;
            return -1;
    }
    if (convert_datetimestruct_to_datetime(dst_meta, &dts, dst_dt) < 0) {
        *dst_dt = NUMBA_DATETIME_NAT;
        return -1;
    }

    return 0;
}

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
                          npy_timedelta *dst_dt)
{
    npy_int64 num = 0, denom = 0;

    /* If the metadata is the same, short-circuit the conversion */
    if (src_meta->base == dst_meta->base &&
            src_meta->num == dst_meta->num) {
        *dst_dt = src_dt;
        return 0;
    }

    /* Get the conversion factor */
    get_datetime_conversion_factor(src_meta, dst_meta, &num, &denom);

    if (num == 0) {
        return -1;
    }

    /* Apply the scaling */
    if (src_dt < 0) {
        *dst_dt = (src_dt * num - (denom - 1)) / denom;
    }
    else {
        *dst_dt = src_dt * num / denom;
    }

    return 0;
}

/*
 * Returns true if the object is something that is best considered
 * a Datetime, false otherwise.
 */
/*static npy_bool
is_any_numpy_datetime(PyObject *obj)
{
    return (PyArray_IsScalar(obj, Datetime) ||
            (PyArray_Check(obj) && (
                PyArray_DESCR((PyArrayObject *)obj)->type_num ==
                                                        NPY_DATETIME)) ||
            PyDate_Check(obj) ||
            PyDateTime_Check(obj));
}*/

/*
 * Returns true if the object is something that is best considered
 * a Timedelta, false otherwise.
 */
/*static npy_bool
is_any_numpy_timedelta(PyObject *obj)
{
    return (PyArray_IsScalar(obj, Timedelta) ||
        (PyArray_Check(obj) && (
            PyArray_DESCR((PyArrayObject *)obj)->type_num == NPY_TIMEDELTA)) ||
        PyDelta_Check(obj));
}*/

/*
 * Returns true if the object is something that is best considered
 * a Datetime or Timedelta, false otherwise.
 */
/*npy_bool
is_any_numpy_datetime_or_timedelta(PyObject *obj)
{
    return obj != NULL &&
           (is_any_numpy_datetime(obj) ||
            is_any_numpy_timedelta(obj));
}*/


