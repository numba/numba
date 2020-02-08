"""
Implements:
- Threadlocal stack
"""
import threading


class TLStack(object):
    def __init__(self):
        self.local = threading.local()

    @property
    def stack(self):
        try:
            # Retrieve thread local stack
            return self.local.stack
        except AttributeError:
            # Initialize stack for the thread
            self.local.stack = []
            return self.local.stack

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        return self.stack.pop()

    @property
    def top(self):
        return self.stack[-1]

    @property
    def is_empty(self):
        return not self.stack

    def __bool__(self):
        return not self.is_empty

    def __nonzero__(self):
        return self.__bool__()

    def __len__(self):
        return len(self.stack)

    def clear(self):
        self.__init__()
