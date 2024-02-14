.. _developer-testsuite:

==============================
Debugging the Numba Test Suite
==============================

.. note:: Features described here are only implemented for the parallel 
          test runner (e.g. requires the ``-m`` flag).

The Numba test suite provides some useful features for debugging test failures 
or crashes.

Generating JUnit XML Output
---------------------------

You can generate a JUnit-compatible XML report by running the tests with the 
``--junit`` flag:

.. code-block:: bash

    $ python -m numba.runtests <args> --junit

This will produce a ``junit_numba_<timestamp>.xml`` file containing the test results.

The XML contains some extra metadata as ``testcase`` properties:

- ``pid``: The ID of the worker process running each test case
- ``start_time``: The start time for each test case relative to when the worker 
  process started


Example of the XML file:

.. code-block:: xml

    <?xml version='1.0' encoding='utf-8'?>
    <testsuites time="20.745398832">
        <testsuite name="numba_testsuite">
            <testcase name="TestUsecases.test_andor" classname="numba.tests.test_usecases"
                time="1.7033392500000002">
                <properties>
                    <property name="pid" value="73274" />
                    <property name="start_time" value="1.337771667" />
                </properties>
            </testcase>
            <testcase name="TestUsecases.test_blackscholes_cnd" classname="numba.tests.test_usecases"
                time="1.723800292">
                <properties>
                    <property name="pid" value="73275" />
                    <property name="start_time" value="1.3350285" />
                </properties>
            </testcase>
            ...
        </testsuite>
    </testsuites>


Debugging Timeouts caused by Crashes
------------------------------------

If the test suite times out, it may indicate a crashed worker process. 
In this case, the timeout error will show:

- The active tests that didn't finish before the timeout
- For each affected process ID, the list of tests that process was running

For example:

.. code-block:: none

  TimeoutError: Active tests didn't finish before timeout:
  - test_foo [PID 1234]

  Tests ran by each affected process: 
  - [PID 1234]: test_bar test_baz test_foo

This information can help reproduce crashes that depend on the test execution 
order or state within a process. You can retry just the tests that were running 
in the crashed process.

