# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm.core

from numba import *

class DateTimeSupportMixin(object):
    "Support for datetimes"

    def _generate_datetime_op(self, op, arg1, arg2):
        (year1, month1, day1, hour1, min1, sec1), \
            (year2, month2, day2, hour2, min2, sec2) = \
            self._extract_datetime(arg1), self._extract_datetime(arg2)
        year, month, day, hour, min, sec = \
            op(year1, month1, day1, hour1, min1, sec1,
                year2, month2, day2, hour2, min2, sec2)
        return self._create_datetime(year, month, day, hour, min, sec)

    def _extract_datetime(self, value):
        "Extract the parts of the datetime"
        return (self.builder.extract_value(value, 0),
                self.builder.extract_value(value, 1),
                self.builder.extract_value(value, 2),
                self.builder.extract_value(value, 3),
                self.builder.extract_value(value, 4),
                self.builder.extract_value(value, 5))

    def _promote_datetime(self, src_type, dst_type, value):
        "Promote a datetime value to value with a larger or smaller datetime type"
        import ipdb;ipdb.set_trace()
        year, month, day, hour, min, sec = self._extract_datetime(value)

        dst_ltype = dst_type.to_llvm(self.context)

        year = self.caster.cast(year, dst_type.subtypes[0])
        month = self.caster.cast(month, dst_type.subtypes[1])
        day = self.caster.cast(day, dst_type.subtypes[2])
        hour = self.caster.cast(hour, dst_type.subtypes[3])
        min = self.caster.cast(min, dst_type.subtypes[4])
        sec = self.caster.cast(sec, dst_type.subtypes[5])
        return self._create_datetime(year, month, day, hour, min, sec)

    def _create_datetime(self, year, month, day, hour, min, sec):
        datetime = llvm.core.Constant.undef(llvm.core.Type.struct([year.type,
                                                                  month.type,
                                                                  day.type,
                                                                  hour.type,
                                                                  min.type,
                                                                  sec.type]))
        datetime = self.builder.insert_value(datetime, year, 0)
        datetime = self.builder.insert_value(datetime, month, 1)
        datetime = self.builder.insert_value(datetime, day, 2)
        datetime = self.builder.insert_value(datetime, hour, 3)
        datetime = self.builder.insert_value(datetime, min, 4)
        datetime = self.builder.insert_value(datetime, sec, 5)
        return datetime


