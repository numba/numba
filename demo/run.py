import provider

a = provider.Provider()
print type(a)
print type(type(a))

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
class B(A):
    pass

print 'baz'
print type(B)

print type(type(a)) is type(B)
#print type(consumer.bar)
