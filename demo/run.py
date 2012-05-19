print 'hi'
import time
import consumer
print 'bar'
T = consumer.f()
print 'foo'
#time.sleep(2)
class A(object):
    pass
    __metaclass__ = T#consumer.somemeta

print 'baz'
print type(A)
#print type(consumer.bar)
