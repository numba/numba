"""
Implement background services for the application.
This is implemented as a cooperative concurrent task.
"""

import functools


class Service(object):
    def __init__(self, name="unnamed", arg=None):
        self.name = name
        self.enabled = True
        self.arg = arg
        self._task = self.process(self.arg)
        next(self._task)

    def service(self):
        """
        Request for the service task.
        Servicing is disabled if it is disabled thourght the "enabled"
        attribute.  When the task is executing, the service is disabled to
        avoid recursion.
        """
        if self.enabled:
            enable = self.enabled
            try:
                # Prevent recursion
                self.enabled = False
                next(self._task)
            finally:
                self.enabled = enable

    def process(self, arg):
        """
        Overrided to implement the service task.
        This must be a generator.
        Use `yield` to return control.
        """
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.service()

    def after(self, fn):
        """
        A decorator for a function. Service is triggered on return.
        """
        @functools.wraps(fn)
        def wrap(*args, **kws):
            with self:
                return fn(*args, **kws)
        return wrap

# -----------------------------------------------------------------------------
# The rest are for testing


class HelloService(Service):
    def process(self, arg):
        count = 0
        yield
        while True:
            print("Hello", count)
            count += 1
            yield

def test():
    serv = HelloService("my.hello")
    print("1")
    serv.service()
    print("2")
    serv.service()

    with serv:
        print("3")

    @serv.after
    def nested():
        print("4")

    nested()


if __name__ == '__main__':
    test()
