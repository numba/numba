from numba import jit, void, int_

# All methods must be given signatures

@jit
class Shrubbery(object):
    @void(int_, int_)
    def __init__(self, w, h):
        self.width = w
        self.height = h

    @int_()
    def area(self):
        return self.width * self.height

    @void()
    def describe(self):
        print("This shrubbery is ", self.width,
              "by", self.height, "cubits.")
 
shrub = Shrubbery(10, 20)
print(shrub.area())
shrub.describe()
print(shrub._numba_attrs)
print(shrub._numba_attrs._fields_)

class MyClass(Shrubbery):
    def newmethod(self):
        print("This is a new method.")

shrub2 = MyClass(30,40)
shrub2.describe()
shrub2.newmethod()
print(shrub._numba_attrs._fields_)


