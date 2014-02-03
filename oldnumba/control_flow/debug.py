# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import os
import logging

debug = False
#debug = True

logger = logging.getLogger(__name__)
if debug:
    logger.setLevel(logging.DEBUG)

debug_cfg = False
#debug_cfg = True

if debug_cfg:
    dot_output_graph = os.path.expanduser("~/cfg.dot")
else:
    dot_output_graph = False
