from __future__ import print_function, division, absolute_import

import sys
import math
import atexit
import timeit
import functools
import itertools
import tempfile
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import numpy as np
from numba import utils


class Profiler(object):
    """
    Timestamp based profiler
    """

    def __init__(self, timer=None):
        self._timer = timeit.default_timer if timer is None else timer
        self._buf = tempfile.NamedTemporaryFile(mode='w', suffix='.profile',
                                                prefix='numba.', delete=False)
        self._first_timestamp = self._timer()
        # Register atexit handling
        atexit.register(self.exit)

    def _timestamp(self):
        return self._timer() - self._first_timestamp

    def exit(self):
        print("\n\nprofile data written to", self._buf.name)
        self._buf.close()

    def start(self, evt):
        print('S', self._timestamp(), evt, file=self._buf)

    def end(self, evt):
        print('E', self._timestamp(), evt, file=self._buf)

    @contextmanager
    def record(self, evt):
        self.start(evt)
        yield
        self.end(evt)

    def mark(self, evt):
        def decor(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with self.record(evt):
                    return func(*args, **kwargs)

            return wrapped

        return decor


class DummyProfiler(object):
    def __init__(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass

    @contextmanager
    def record(self, *args, **kargs):
        yield

    def mark(self, evt):
        def decor(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped

        return decor


class Plotter(object):
    def __init__(self, fileobj, output):
        self._fileobj = fileobj
        self._output = output

    def render(self):
        from bokeh import plotting as bp
        from bokeh.charts import Bar
        from bokeh.models import HoverTool

        evt_list = []
        dur_list = []
        begin_list = []
        start_list = []
        stop_list = []
        color_list = []

        evt_color = {}
        evt_total = defaultdict(float)

        colors = [
            '#B45C5C',
            '#B4845C',
            '#376C6C',
            '#4A904A',
        ]

        infinte_colors = iter(itertools.cycle(colors))

        evt_range = []
        # Preprocess data
        for evt, ts, te in self._parse():
            evt_list.append(evt)
            dur = te - ts
            begin_list.append(ts + dur / 2)
            start_list.append(ts)
            stop_list.append(te)
            dur_list.append(dur)
            if evt not in evt_color:
                evt_range.append(evt)
                evt_color[evt] = utils.next(infinte_colors)

            color_list.append(evt_color[evt])
            evt_total[evt] += dur

        def format_time(seconds):
            nano = 10 ** -9
            micro = 10 ** -6
            mini = 10 ** -3
            if seconds < micro:
                return "{0:.2f}ns".format(seconds / nano)
            if seconds < mini:
                return "{0:.2f}us".format(seconds / micro)
            if seconds < 1:
                return "{0:.2f}ms".format(seconds / mini)
            return "{0:.2f}s".format(seconds)

        source = bp.ColumnDataSource(
            data=dict(event=evt_list,
                      begin=begin_list,
                      start=start_list,
                      stop=stop_list,
                      duration=dur_list,
                      duration_formated=[format_time(t) for t in dur_list],
                      percent=["{0:.1%}".format(d/stop_list[-1])
                                  for d in dur_list],
                      color=color_list)
        )

        # Plot
        minheight = 150
        bp.output_file(self._output)

        total_seconds = stop_list[-1]

        tools = "resize,hover,save,pan,box_zoom,reset"
        fig_timeline = bp.figure(title='',
                                 width=1200,
                                 height=max(minheight, 50 * len(evt_color)),
                                 y_range=list(reversed(evt_range)),
                                 x_range=(0, total_seconds + 1),
                                 x_axis_location="above",
                                 tools=tools)

        fig_timeline.rect(x="begin", y="event", width="duration", height=1,
                          fill_color="color", source=source, fill_alpha=0.8,
                          line_color="color")
        fig_timeline.ygrid[0].grid_line_color=None
        hover = fig_timeline.select(dict(type=HoverTool))
        hover.snap_to_data = False
        hover.tooltips = OrderedDict([
            ('event', '@event'),
            ('duration', '@duration_formated'),
            ("percent", '@percent'),
            ('start time', '@start'),
            ('stop time', '@stop'),
        ])

        totals = [evt_total[evt] for evt in evt_range]
        totaltimes = dict(total=totals)
        fig_bar = Bar(totaltimes, evt_range, width=1200, height=400,
                      xlabel='events', ylabel='total seconds',
                      tools = "resize,save,pan,box_zoom,reset")

        bp.show(bp.VBox(fig_timeline, fig_bar))

    def _parse(self):
        eventmap = defaultdict(list)
        for line in self._fileobj:
            state, stamp, event = line.rstrip().split(' ', 2)
            stamp = float(stamp)
            if state == 'S':
                eventmap[event].append(stamp)
            elif state == 'E':
                start = eventmap[event].pop()
                end = stamp
                yield event, start, end
            else:
                raise ValueError("invalid format")


def main():
    filename = sys.argv[1]
    with open(filename) as fin:
        plt = Plotter(fin, output=filename + '.html')
        plt.render()


if __name__ == '__main__':
    main()
