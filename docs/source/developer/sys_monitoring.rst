
===========================
Notes on ``sys.monitoring``
===========================

.. note:: This documentation was written at the advent of Python 3.12. Future
          versions of Python may behave differently. It is however hoped that
          most of the concepts herein will remain relevant.

Python 3.12 introduced a new monitoring system under ``sys.monitoring``. This
system lets users monitor a selection of events that may be interesting for e.g.
performance profiling or debugging purposes. Event monitoring is set "per tool",
so that multiple tools can be running at the same time. For each tool the events
can be monitored globally per thread or locally per code object (or a mixture of
both). For each tool-event combination a callback can be registered that will be
called on the occurrence of the event. The callbacks are just regular
functions and can do most of the things supported by Python, they also have the
ability to return a special value to tell the monitoring system to disable
triggering future events for the current code location.

What does this mean for Numba?
------------------------------

When the interpreter "encounters" a monitoring event (it actually issues them)
it triggers any callbacks that are associated with that event across all tools
that have registered monitoring for said event. In the case of Numba there are
problems...

Numba has made it so that there's no Python interpreter involved in the
execution of a function, the function is compiled and its execution path
exists only in machine code. To get to the machine code from the interpreter
the Numba dispatcher is invoked, this is the last place in the stack where
(in ``nopython`` mode) the Python interpreter is readily available. The
dispatcher is also in some way part of the execution of the function, without
the dispatcher the call to the machine code cannot easily happen from user
space. As a result of this, the monitoring types and event types that Numba
can support are somewhat limited as there's such limited interpreter
involvement in execution!

Looking at monitoring types in turn. Local monitoring is requested by setting
monitoring on a code object. In practice this instructs the interpreter to
augment the bytecode at runtime by switching certain opcodes for
"instrumented" opcodes. These instrumented opcodes go via a special path in the
interpreter loop whereby they will issue an "event" in association with a
particular instruction at a particular offset. For example, a ``RETURN`` opcode
might be replaced by an ``INSTRUMENTED_RETURN`` and a ``PY_RETURN`` event
would by issued when the instrumented instruction is interpreted. This event and
the offset at which it occurred being forwarded to the monitoring system.
Unfortunately this presents an issue for Numba, there is no interpreter involved
with execution and so events will not be emitted. It does seem like it would be
possible to handle a few types of event, such as ``PY_START`` and ``PY_RETURN``
by analysing the code object at dispatch time. However, it's possible for a user
to de-instrument the code object and/or dynamically disable monitoring at a
particular code location whilst executing, and as a result emulating the
semantics of this would be prohibitively challenging and would likely require
constant interaction with the interpreter. As a result, Numba does not support
local event monitoring, the compiled function will still execute correctly if it
has been set, it just has no effect on ``sys.monitoring``.

Considering per-thread global monitoring, this manifests as the user setting
some global state on the interpreter for a given thread. This state can be
accessed via the ``sys.monitoring`` Python API, it's also accessible via
CPython internals. This kind of monitoring is a little more amenable to working
with Numba as there's no code object involved and state mutation during
execution can only occur via object mode calls.

What does Numba do in practice?
-------------------------------
As there's no Python or C API to issue events (the concept is heavily linked to
the interpreter itself), Numba has to look for tool-event combinations at
appropriate locations in the dispatch sequence and then manually call the
associated callbacks (essentially doing what the interpreter does when it issues
an event). In the case of the Numba dispatcher, only a few events are relevant
and only four are supported, namely

* ``sys.monitoring.events.PY_START`` (Python function starting).
* ``sys.monitoring.events.PY_RETURN`` (Python function returning).
* ``sys.monitoring.events.RAISE`` (Python function raised an exception).
* ``sys.monitoring.events.PY_UNWIND`` (Python function exiting during exception
  unwinding).

These events don't really exist in the machine code, but would exist had the
interpreter interpreted the equivalent bytecode. The dispatcher therefore checks
for monitoring of ``PY_START`` just before control is transferred to the machine
code and calls any associated callbacks. The same is done for ``PY_RETURN`` just
after control is transferred back to the dispatcher from the machine code. This
behaviour essentially emulates the interpreter executing bytecode and lets
tools such as ``cProfile`` be able to "see" the Numba compiled function as part
of the standard interpreted execution. In the case of an exception being raised
in the machine code, the associated error state is handled just after control is
transferred back to the dispatcher, at this point ``RAISE`` and ``PY_UNWIND``
event monitoring is checked and registered callbacks are invoked.

A note on offsets. The callback functions often take an "offset" argument which
is the bytecode offset at which the event triggering the callback was
encountered. In the case of ``PY_START`` this seems to be associated with the
offset of the ``RESUME`` bytecode. In the case of ``PY_RETURN`` this is
associated with the offset of one of the ``RETURN`` bytecodes, most generally
this would only be known at runtime as there could be multiple return paths. As
a result, Numba elects to just set all offsets to zero. It may eventually be
possible to do some analysis and transfer the appropriate runtime information to
the dispatcher from the machine code, however, at the present time the effort to
do this vastly outweighs the gain.
