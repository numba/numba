# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm.core

from numba import *

class DateTimeSupportMixin(object):
    "Support for datetimes"

    def _generate_datetime_op(self, op, arg1, arg2):
        (timestamp1, units1), (timestamp2, units2) = \
            self._extract_datetime(arg1), self._extract_datetime(arg2)
        timestamp, units = op(timestamp1, units1, timestamp2, units2)
        return self._create_datetime(timestamp, units)

    def _extract_datetime(self, value):
        "Extract the parts of the datetime"
        return (self.builder.extract_value(value, 0),
                self.builder.extract_value(value, 1))

    def _extract_timedelta(self, value):
        return (self.builder.extract_value(value, 0),
                self.builder.extract_value(value, 1))

    def _promote_datetime(self, src_type, dst_type, value):
        "Promote a datetime value to value with a larger or smaller datetime type"
        timestamp, units = self._extract_datetime(value)

        dst_ltype = dst_type.to_llvm(self.context)

        timestamp = self.caster.cast(timestamp, dst_type.subtypes[0])
        units = self.caster.cast(units, dst_type.subtypes[1])
        return self._create_datetime(timestamp, units)

    def _promote_timedelta(self, src_type, dst_type, value):
        diff, units = self._extract_timedelta(value)
        dst_ltype = dst_type.to_llvm(self.context)
        diff = self.caster.cast(diff, dst_type.subtypes[0])
        units = self.caster.cast(units, dst_type.subtypes[1])
        return self._create_timedelta(diff, units)

    def _create_datetime(self, timestamp, units):
        datetime = llvm.core.Constant.undef(llvm.core.Type.struct([timestamp.type,
                                                                  units.type]))
        datetime = self.builder.insert_value(datetime, timestamp, 0)
        datetime = self.builder.insert_value(datetime, units, 1)
        return datetime

    def _create_timedelta(self, diff, units):
        timedelta_struct = llvm.core.Constant.undef(llvm.core.Type.struct([diff.type,
                                                                  units.type]))
        timedelta_struct = self.builder.insert_value(timedelta_struct, diff, 0)
        timedelta_struct = self.builder.insert_value(timedelta_struct, units, 1)
        return timedelta_struct

