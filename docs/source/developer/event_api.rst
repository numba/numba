Event API
=========

.. automodule:: numba.core.event
    :members:

Thread safety
-------------

There are two ways to attach a listener, with different thread semantics:

* :func:`~numba.core.event.register` / :func:`~numba.core.event.unregister`
  attach a listener **process-wide**: it receives events broadcast on **any**
  thread. A listener registered this way must be thread-safe itself, because
  its ``notify()`` may be invoked concurrently from multiple threads.
  :class:`~numba.core.event.TimingListener` instances are internally locked,
  so a shared, globally-registered ``TimingListener`` cannot crash or corrupt
  its state, but durations measured from events arriving from multiple
  threads are meaningless.

* :func:`~numba.core.event.install_listener`,
  :func:`~numba.core.event.install_timer` and
  :func:`~numba.core.event.install_recorder` attach a listener for the
  **calling thread only**: it receives just the events broadcast on the
  thread that entered the context manager. This makes scoped listeners safe
  to use during concurrent compilation on multiple threads; each compilation
  measures only its own events.
