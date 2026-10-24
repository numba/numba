"""
The ``numba.core.event`` module provides a simple event system for applications
to register callbacks to listen to specific compiler events.

The following events are built in:

- ``"numba:compile"`` is broadcast when a dispatcher is compiling. Events of
  this kind have ``data`` defined to be a ``dict`` with the following
  key-values:

  - ``"dispatcher"``: the dispatcher object that is compiling.
  - ``"args"``: the argument types.
  - ``"return_type"``: the return type.

- ``"numba:compiler_lock"`` is broadcast when the internal compiler-lock is
  acquired. This is mostly used internally to measure time spent with the lock
  acquired.

- ``"numba:llvm_lock"`` is broadcast when the internal LLVM-lock is acquired.
  This is used internally to measure time spent with the lock acquired.

- ``"numba:run_pass"`` is broadcast when a compiler pass is running.

    - ``"name"``: pass name.
    - ``"qualname"``: qualified name of the function being compiled.
    - ``"module"``: module name of the function being compiled.
    - ``"flags"``: compilation flags.
    - ``"args"``: argument types.
    - ``"return_type"`` return type.

Applications can register callbacks that are listening for specific events using
``register(kind: str, listener: Listener)``, where ``listener`` is an instance
of ``Listener`` that defines custom actions on occurrence of the specific event.

Thread semantics
----------------

There are two ways to attach a listener, with different thread semantics:

- ``register()``/``unregister()`` attach a listener **process-wide**: it
  receives events broadcast on **any** thread. Such a listener must therefore
  be thread-safe itself, because ``notify()`` can be invoked concurrently
  from multiple threads.
- ``install_listener()``/``install_timer()``/``install_recorder()`` attach a
  listener for the **calling thread only**: it receives just the events
  broadcast on the thread that entered the context manager. This makes scoped
  listeners safe to use during concurrent compilation on multiple threads.
"""

import os
import json
import atexit
import abc
import enum
import time
import threading
from timeit import default_timer as timer
from contextlib import contextmanager, ExitStack
from collections import defaultdict

from numba.core import config, utils


class EventStatus(enum.Enum):
    """Status of an event.
    """
    START = enum.auto()
    END = enum.auto()


# Builtin event kinds.
_builtin_kinds = frozenset([
    "numba:compiler_lock",
    "numba:compile",
    "numba:llvm_lock",
    "numba:run_pass",
])


def _guard_kind(kind):
    """Guard to ensure that an event kind is valid.

    All event kinds with a "numba:" prefix must be defined in the pre-defined
    ``numba.core.event._builtin_kinds``.
    Custom event kinds are allowed by not using the above prefix.

    Parameters
    ----------
    kind : str

    Return
    ------
    res : str
    """
    if kind.startswith("numba:") and kind not in _builtin_kinds:
        msg = (f"{kind} is not a valid event kind, "
               "it starts with the reserved prefix 'numba:'")
        raise ValueError(msg)
    return kind


class Event:
    """An event.

    Parameters
    ----------
    kind : str
    status : EventStatus
    data : any; optional
        Additional data for the event.
    exc_details : 3-tuple; optional
        Same 3-tuple for ``__exit__``.
    """
    def __init__(self, kind, status, data=None, exc_details=None):
        self._kind = _guard_kind(kind)
        self._status = status
        self._data = data
        self._exc_details = (None
                             if exc_details is None or exc_details[0] is None
                             else exc_details)

    @property
    def kind(self):
        """Event kind

        Returns
        -------
        res : str
        """
        return self._kind

    @property
    def status(self):
        """Event status

        Returns
        -------
        res : EventStatus
        """
        return self._status

    @property
    def data(self):
        """Event data

        Returns
        -------
        res : object
        """
        return self._data

    @property
    def is_start(self):
        """Is it a *START* event?

        Returns
        -------
        res : bool
        """
        return self._status == EventStatus.START

    @property
    def is_end(self):
        """Is it an *END* event?

        Returns
        -------
        res : bool
        """
        return self._status == EventStatus.END

    @property
    def is_failed(self):
        """Is the event carrying an exception?

        This is used for *END* event. This method will never return ``True``
        in a *START* event.

        Returns
        -------
        res : bool
        """
        return self._exc_details is None

    def __str__(self):
        data = (f"{type(self.data).__qualname__}"
                if self.data is not None else "None")
        return f"Event({self._kind}, {self._status}, data: {data})"

    __repr__ = __str__


_registered = defaultdict(list)
# Lock guarding mutation of ``_registered``.
_registered_lock = threading.Lock()

# Thread-local registry for listeners installed via the ``install_*``
# context managers. Such listeners only receive events broadcast on the
# thread that installed them.
_thread_local = threading.local()


def _get_thread_registered():
    """Return the calling thread's registry of scoped listeners.

    The registry is a ``defaultdict(list)`` mapping event kind to the list
    of listeners installed (via ``install_listener``) on the calling thread.
    It is created lazily on first use.
    """
    registered = getattr(_thread_local, "registered", None)
    if registered is None:
        registered = _thread_local.registered = defaultdict(list)
    return registered


def register(kind, listener):
    """Register a listener for a given event kind.

    The listener is registered process-wide and will receive events
    broadcast on any thread. It must therefore be thread-safe.

    Parameters
    ----------
    kind : str
    listener : Listener
    """
    assert isinstance(listener, Listener)
    kind = _guard_kind(kind)
    with _registered_lock:
        _registered[kind].append(listener)


def unregister(kind, listener):
    """Unregister a listener for a given event kind.

    Parameters
    ----------
    kind : str
    listener : Listener
    """
    assert isinstance(listener, Listener)
    kind = _guard_kind(kind)
    with _registered_lock:
        lst = _registered[kind]
        lst.remove(listener)
        if not lst:
            # Drop the empty entry so a balanced register/unregister
            # pair leaves no trace in the registry.
            del _registered[kind]


def broadcast(event):
    """Broadcast an event to all registered listeners.

    The event is delivered to all process-wide listeners (see
    ``register()``) followed by the listeners installed on the calling
    thread for the event's kind (see ``install_listener()``).

    Parameters
    ----------
    event : Event
    """
    with _registered_lock:
        # Snapshot so concurrent register/unregister cannot skip or
        # double-deliver. Use .get() to avoid autovivifying keys.
        listeners = list(_registered.get(event.kind, ()))
    listeners += list(_get_thread_registered().get(event.kind, ()))
    for listener in listeners:
        listener.notify(event)


class Listener(abc.ABC):
    """Base class for all event listeners.
    """
    @abc.abstractmethod
    def on_start(self, event):
        """Called when there is a *START* event.

        Parameters
        ----------
        event : Event
        """
        pass

    @abc.abstractmethod
    def on_end(self, event):
        """Called when there is a *END* event.

        Parameters
        ----------
        event : Event
        """
        pass

    def notify(self, event):
        """Notify this Listener with the given Event.

        Parameters
        ----------
        event : Event
        """
        if event.is_start:
            self.on_start(event)
        elif event.is_end:
            self.on_end(event)
        else:
            raise AssertionError("unreachable")


class TimingListener(Listener):
    """A listener that measures the total time spent between *START* and
    *END* events during the time this listener is active.
    """
    def __init__(self):
        self._lock = threading.Lock()
        with self._lock:
            self._depth = 0

    def on_start(self, event):
        with self._lock:
            if self._depth == 0:
                self._ts = timer()
            self._depth += 1

    def on_end(self, event):
        with self._lock:
            self._depth -= 1
            if self._depth == 0:
                last = getattr(self, "_duration", 0)
                self._duration = (timer() - self._ts) + last

    @property
    def done(self):
        """Returns a ``bool`` indicating whether a measurement has been made.

        When this returns ``False``, the matching event has never fired.
        If and only if this returns ``True``, ``.duration`` can be read without
        error.
        """
        with self._lock:
            return hasattr(self, "_duration")

    @property
    def duration(self):
        """Returns the measured duration.

        This may raise ``AttributeError``. Users can use ``.done`` to check
        that a measurement has been made.
        """
        with self._lock:
            return self._duration


class RecordingListener(Listener):
    """A listener that records all events and stores them in the ``.buffer``
    attribute as a list of 2-tuple ``(float, Event)``, where the first element
    is the time the event occurred as returned by ``time.time()`` and the second
    element is the event.
    """
    def __init__(self):
        self.buffer = []

    def on_start(self, event):
        self.buffer.append((time.time(), event))

    def on_end(self, event):
        self.buffer.append((time.time(), event))


@contextmanager
def install_listener(kind, listener):
    """Install a listener for event "kind" temporarily within the duration of
    the context.

    The listener is installed on the calling thread only: it receives just
    the events broadcast on the thread that entered this context manager.
    Use ``register()`` instead for process-wide, all-threads delivery.

    Returns
    -------
    res : Listener
        The *listener* provided.

    Examples
    --------

    >>> with install_listener("numba:compile", listener):
    >>>     some_code()  # listener will be active here.
    >>> other_code()     # listener will be unregistered by this point.

    """
    # Capture the per-thread list once so that removal at exit targets the
    # same registry that was appended to at entry.
    lst = _get_thread_registered()[_guard_kind(kind)]
    lst.append(listener)
    try:
        yield listener
    finally:
        lst.remove(listener)


@contextmanager
def install_timer(kind, callback):
    """Install a TimingListener temporarily to measure the duration of
    an event.

    If the context completes successfully, the *callback* function is executed.
    The *callback* function is expected to take a float argument for the
    duration in seconds.

    Returns
    -------
    res : TimingListener

    Examples
    --------

    This is equivalent to:

    >>> with install_listener(kind, TimingListener()) as res:
    >>>    ...
    """
    tl = TimingListener()
    with install_listener(kind, tl):
        yield tl

    if tl.done:
        callback(tl.duration)


@contextmanager
def install_recorder(kind):
    """Install a RecordingListener temporarily to record all events.

    Once the context is closed, users can use ``RecordingListener.buffer``
    to access the recorded events.

    Returns
    -------
    res : RecordingListener

    Examples
    --------

    This is equivalent to:

    >>> with install_listener(kind, RecordingListener()) as res:
    >>>    ...
    """
    rl = RecordingListener()
    with install_listener(kind, rl):
        yield rl


def start_event(kind, data=None):
    """Trigger the start of an event of *kind* with *data*.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    """
    evt = Event(kind=kind, status=EventStatus.START, data=data)
    broadcast(evt)


def end_event(kind, data=None, exc_details=None):
    """Trigger the end of an event of *kind*, *exc_details*.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    exc_details : 3-tuple; optional
        Same 3-tuple for ``__exit__``. Or, ``None`` if no error.
    """
    evt = Event(
        kind=kind, status=EventStatus.END, data=data, exc_details=exc_details,
    )
    broadcast(evt)


@contextmanager
def trigger_event(kind, data=None):
    """A context manager to trigger the start and end events of *kind* with
    *data*. The start event is triggered when entering the context.
    The end event is triggered when exiting the context.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    """
    with ExitStack() as scope:
        @scope.push
        def on_exit(*exc_details):
            end_event(kind, data=data, exc_details=exc_details)

        start_event(kind, data=data)
        yield


def _prepare_chrome_trace_data(listener: RecordingListener):
    """Prepare events in `listener` for serializing as chrome trace data.
    """
    # The spec for the trace event format can be found at:
    # https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit   # noqa
    # This code only uses the JSON Array Format for simplicity.
    pid = os.getpid()
    tid = threading.get_native_id()
    evs = []
    for ts, rec in listener.buffer:
        data = rec.data
        cat = str(rec.kind)
        ts_scaled = ts * 1_000_000   # scale to microseconds
        ph = 'B' if rec.is_start else 'E'
        name = data['name']
        args = data
        ev = dict(
            cat=cat, pid=pid, tid=tid, ts=ts_scaled, ph=ph, name=name,
            args=args,
        )
        evs.append(ev)
    return evs


def _setup_chrome_trace_exit_handler():
    """Setup a RecordingListener and an exit handler to write the captured
    events to file.
    """
    listener = RecordingListener()
    register("numba:run_pass", listener)
    filename = config.CHROME_TRACE

    @atexit.register
    def _write_chrome_trace():
        # The following output file is not multi-process safe.
        evs = _prepare_chrome_trace_data(listener)
        with open(filename, "w") as out:
            json.dump(evs, out, cls=utils._LazyJSONEncoder)


if config.CHROME_TRACE:
    _setup_chrome_trace_exit_handler()
