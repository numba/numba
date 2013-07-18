# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm.core

from numba import *

class DateTimeSupportMixin(object):
    "Support for datetimes"

    def _generate_datetime_op(self, op, arg1, arg2):
        (year1, month1, day1), (year2, month2, day2) = self._extract(arg1), self._extract(arg2)
        year, month, day = op(year1, month1, day1, year2, month2, day2)
        return self._create_datetime(year, month, day)

    def _extract(self, value):
        "Extract the parts of the datetime"
        return (self.builder.extract_value(value, 0),
                self.builder.extract_value(value, 1),
                self.builder.extract_value(value, 2))

    def _promote_datetime(self, src_type, dst_type, value):
        "Promote a datetime value to value with a larger or smaller datetime type"
        year, month, day = self._extract(value)

        if dst_type.is_datetime:
            dst_type = dst_type.base_type
        dst_ltype = dst_type.to_llvm(self.context)

        year = self.caster.cast(year, dst_ltype)
        month = self.caster.cast(month, dst_ltype)
        day = self.caster.cast(day, dst_ltype)
        return self._create_datetime(year, month, day)

    def _create_datetime(self, year, month, day):
        datetime = llvm.core.Constant.undef(llvm.core.Type.struct([year.type,
                                                                  month.type,
                                                                  day.type]))
        datetime = self.builder.insert_value(datetime, year, 0)
        datetime = self.builder.insert_value(datetime, month, 1)
        datetime = self.builder.insert_value(datetime, day, 2)
        return datetime


