import abc
import enum
import time
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from timeit import default_timer as timer


class EventStatus(enum.Enum):
    START = "START"
    STOP = "STOP"


class EventKind(enum.Enum):
    COMPILER_LOCK = "COMPILER_LOCK"
    COMPILE = "COMPILE"


class Event:
    def __init__(self, kind, status, data=None, exc_details=None):
        self._kind = EventKind(kind)
        self._status = status
        self._data = data
        self._exc_details = exc_details

    @property
    def kind(self):
        return self._kind

    @property
    def status(self):
        return self._status

    @property
    def data(self):
        return self._data

    @property
    def is_start(self):
        return self._status == EventStatus.START

    @property
    def is_stop(self):
        return self._status == EventStatus.STOP

    @property
    def is_failed(self):
        return self._exc_details[0] is None

    def __str__(self):
        return f"Event({self._kind}, {self._status})"


_registered = defaultdict(list)


def register(kind, listener):
    kind = EventKind(kind)
    _registered[kind].append(listener)


def unregister(kind, listener):
    kind = EventKind(kind)
    lst = _registered[kind]
    lst.remove(listener)


def broadcast(event):
    for listener in _registered[event.kind]:
        listener.notify(event)


class Listener(abc.ABC):
    @abc.abstractmethod
    def on_start(self, event):
        pass

    @abc.abstractmethod
    def on_stop(self, event):
        pass

    def notify(self, event):
        if event.is_start:
            self.on_start(event)
        elif event.is_stop:
            self.on_stop(event)
        else:
            raise AssertionError("unreachable")


class TimedListener(Listener):
    def __init__(self):
        self._ts = None
        self._depth = 0

    def on_start(self, event):
        if self._ts is None:
            self._ts = timer()
        self._depth += 1

    def on_stop(self, event):
        self._depth -= 1
        if self._depth == 0:
            self._duration = timer() - self._ts

    @property
    def duration(self):
        return self._duration


class RecordingListener(Listener):
    def __init__(self):
        self.buffer = []

    def on_start(self, event):
        self.buffer.append((time.time(), event))

    def on_stop(self, event):
        self.buffer.append((time.time(), event))


@contextmanager
def install_timer(kind, callback):
    """Install a TimedListener temporarily to measure the duration for
    an event.

    If the context completes successfully, the *callback* function is executed.
    The *callback* function is expected to take a float argument for the
    duration in seconds.
    """
    listener = TimedListener()
    register(kind, listener)
    try:
        yield listener
    finally:
        unregister(kind, listener)
    callback(listener.duration)


@contextmanager
def install_recorder(kind):
    rl = RecordingListener()
    register(kind, rl)
    try:
        yield rl
    finally:
        unregister(kind, rl)


def start_event(kind, data=None):
    evt = Event(kind=kind, status=EventStatus.START, data=data)
    broadcast(evt)


def stop_event(kind, data=None, exc_details=None):
    evt = Event(
        kind=kind, status=EventStatus.STOP, data=data, exc_details=exc_details,
    )
    broadcast(evt)


@contextmanager
def mark_event(kind, data=None):
    with ExitStack() as scope:

        @scope.push
        def on_exit(*exc_details):
            stop_event(kind, data=data, exc_details=exc_details)

        start_event(kind, data=data)
        yield
