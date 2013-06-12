# -*- coding: UTF-8 -*-
from __future__ import print_function, division, absolute_import
import sys
from itertools import chain
from collections import namedtuple
from .annotate import format_annotations

def render(program, emit=sys.stdout.write,
           intermediate_names=(), inline=True):
    """
    Render a Program as html.
    """

