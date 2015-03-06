from __future__ import print_function, division, absolute_import

import sys
import atexit
import timeit
import functools
import tempfile
from collections import defaultdict, OrderedDict
from contextlib import contextmanager


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
        from bokeh.models import HoverTool

        evt_list = []
        dur_list = []
        begin_list = []
        end_list = []

        evt_set = set()
        evt_range = []
        # Preprocess data
        for evt, ts, te in self._parse():
            evt_list.append(evt)
            begin_list.append(ts)
            end_list.append(te)
            dur_list.append(te - ts)
            if evt not in evt_set:
                evt_set.add(evt)
                evt_range.append(evt)

        source = bp.ColumnDataSource(
            data=dict(event=evt_list,
                      begin=begin_list,
                      end=end_list,
                      duration=dur_list)
        )

        # Plot
        minheight = 150
        bp.output_file(self._output)

        tools = "resize,hover,save,pan,box_zoom,wheel_zoom,reset"
        fig = bp.figure(title='Execution Timeline',
                        width=1000,
                        height=max(minheight, 50 * len(evt_set)),
                        y_range=evt_range,
                        x_range=(0, end_list[-1] + 1),
                        x_axis_location="above",
                        toolbar_location="left",
                        tools=tools)
        fig.rect(x="begin", y="event", width="duration", height=1,
                 source=source, fill_alpha=0.3,
                 line_color='black')

        hover = fig.select(dict(type=HoverTool))
        hover.snap_to_data = False
        hover.tooltips = OrderedDict([
            ('duration', '@duration'),
            ('start time', '@begin'),
            ('stop time', '@end'),
        ])
        bp.show(fig)

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
