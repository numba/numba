# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm.core

from numba import *

class DateTimeSupportMixin(object):
    "Support for datetimes"

    def _generate_datetime_op(self, op, arg1, arg2):
        (year1, month1, day1, hour1, min1, sec1), \
            (year2, month2, day2, hour2, min2, sec2) = \
            self._extract(arg1), self._extract(arg2)
        year, month, day, hour, min, sec = \
            op(year1, month1, day1, hour1, min1, sec1,
                year2, month2, day2, hour2, min2, sec2)
        return self._create_datetime(year, month, day, hour, min, sec)

    def _extract(self, value):
        "Extract the parts of the datetime"
        return (self.builder.extract_value(value, 0),
                self.builder.extract_value(value, 1),
                self.builder.extract_value(value, 2),
                self.builder.extract_value(value, 3),
                self.builder.extract_value(value, 4),
                self.builder.extract_value(value, 5))

    def _promote_datetime(self, src_type, dst_type, value):
        "Promote a datetime value to value with a larger or smaller datetime type"
        year, month, day = self._extract(value)

        if dst_type.is_datetime:
            dst_type = dst_type.base_type
        dst_ltype = dst_type.to_llvm(self.context)

        year = self.caster.cast(year, dst_ltype)
        month = self.caster.cast(month, dst_ltype)
        day = self.caster.cast(day, dst_ltype)
        hour = self.caster.cast(hour, dst_ltype)
        min = self.caster.cast(min, dst_ltype)
        sec = self.caster.cast(sec, dst_ltype)
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


