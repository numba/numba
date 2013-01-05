% Feeding the Fire 
% Blaze Development ( 3 Months In )
% Continuum Analytics 

# State of the Blaze

![](barnraising.jpg)

# Backend

The Backend Journey

- C!
- Numba!
- Stack VM!
- Register VM!
- Forth!
- Code Generation!
- Numba again!
- llvmpy!
- MLVM!
- Stack VM again!
- Register VM again!
- Interpreter!
- Code Generation again!
- **Runtime**
    - Dispatch to all the things! ( but mostly Numba )

# Vision

* Blaze is 
    * a generalization of NumPy
    * a datashape description language
    * a generalized notion of data access
    * a way to view data stores as arrays and tables

![](numpy_plus.png)

# 

```python
from blaze import Array, dshape
ds = dshape('2, 2, int')

a = Array([1,2,3,4], ds)
```

```python
>>> a
Array
  datashape := 2 2 int64
  values    := [CArray(ptr=36111200)]
  metadata  := [manifest, array_like, chunked]
  layout    := Chunked(dim=0)

[[1, 2],
 [3, 4]]
```

# End-User Perspectives 

Blaze is built around separation of concerns.

* Domain experts
    - Richer structure to express high level ideas.
* Algorithm writers 
    - More information ( type, shape, layout ) to do clever optimizations.
* Researchers
    - A platform in which to explore data and task
      parallelism.


Zen of Blaze
------------

* Express more logic at a high level. ( More Python, Less C )
* Better knowledge informs better code generation and execution.
* Don't copy data when we can push code to data.

# Blaze Source Tree

```
git clone git@github.com:ContinuumIO/blaze.git
```

```
blaze/
    carray/
    datashape/
    dist/
    engine/
    expr/
    include/
    layouts/
    persistence/
    rosetta/
    rts/
    samples/
    sources/
    stratego/

    byteproto.py
    byteprovider.py
    datadescriptor.py
    idx.py
    lib.py
    plan.py
    printer.py
    slicealgebra.py
    table.py
```

# Chunked Arrays

* CArray is the beating heart of Blaze. It is the canonical storage
backend when the user has no preferences on local storage.

* No distinction between storage:
    * Storage on disk.
    * Storage on memory

#

```python
>>> from blaze.carray import carray
>>> a = carray(xrange(10000))
carray((10000,), int64)
    nbytes: 256; cbytes: 8.00 KB; ratio: 0.03
    cparams := _cparams(clevel=5, shuffle=True)
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 ... ]
```

# Modifications

Modifications ( i.e. hacks ) to CArray.

```python
>>> print a.nchunks
2

# Lifting in-memory chunks into Pthon
>>> for c in a.chunks:
...     print c, '\n'
... 
[   0    1    2 ..., 4093 4094 4095] 

[4096 4097 4098 ..., 8189 8190 8191] 

# Pointers to chunks in memory
>>> a.chunks[0].pointer
44653344L

>>> a.leftovers
44653510L
```

Lifting some of CArray internals up into Blaze. Keep the core the
same.

# Datashape

Small DSL, not Python subset!

- Inline 

```python
>>> from blaze import dshape
>>> dshape('3, 3, int32')
```

- IPython Magic

```python
In [0]: %load_ext blaze.datashape.magic
```

- Modules

```haskell
type Pixel A = 800, 600, A, A
type Foo x y = Either( (x,int32),(y, int32) )

type Stock = {
  name   : string,
  min    : int64,
  max    : int64,
  mid    : int64,
  volume : float,
  close  : float,
  open   : float,
}
```

# Types?

- Not your grandfather's type system.
- Leverage the last 20 years of research.
    - Unlike some other languages ( \*cough\* R, Matlab, Julia \*cough\* )
- Datashape is very simple as type systems go. Inference is
  a solved problem. ( sans Records )

# Constructors

A **type constructor** is used to construct new types from given ones.

Datashape types with free parameters in their constructor are called
**parameterized types**.

```
SquareMatrix T = N, N, T
```

Datashape types without free parameters in their constructor are called
**alias types**.

```
SquareIntMatrix = N, N, int32
```

# Product Types

The product operator (,) is used to construct product types. It is a type constructor of two arguments.

```haskell
a, b
```

It is also left associative:

```haskell
((a, b), c) = a, b, c
```

# Sum Types

A sum type is a type representing a collection of heterogeneously typed
values. There are four instances of sum types in Blaze’s type system:

- Variants
- Unions
- Options
- Ranges

# Types, Trees and ADTs, oh my!

Sum types represent branching possibilities of types.

Product types represent sequential combinations of types joined with
external structure.

#

![](tree.png)

#

Type            Notation
-------         ------
Unit Types      ```1```
Type Variable   ```X```
Option Types    ```1 + X```
Sum Types       ```X + Y```
Product Types   ```X • Y```
Disjoint Unions ```X + Y + ... Z```


- ```python
3, 3, int32
```

- ```
1 • 1 • 1
```

- ```
Either(X, Y), int32
```

- ```
(X + Y) • 1
```

- ```
(X • 1) + (Y • 1) 
```

- ```
Either(na, X)
```

- ```
1 + X
```

# Expressions

```
a = NDArray([1,2,3])
e = a+a
e.eval()
```

ATerm IR

```
Add(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
```

Numba Code Generation

```
def ufunc0(op0, op1):
    return (op0 + op1)
```

# ATerm

Human readable representation of ASTs.

It's just Python. Can often just ``eval`` into Python.

```
x
f(x,y)
[1,2,3]
```

Annotations

```python
x{p}
y{p,q}
[1,2,3]{p}
```

```python
f(
    x{dshape("5, int"), contigious},
    y{dshape("5, float"), contigious}
)
```

Pattern Matching

```python
matches("f(x,y)", "f(<term>, <term>)") # True
matches("f(1,g(2,3))", "f(1,g(<int>, <int>))") # True
matches("f(1,g(2,3))", "f(<str>,g(2,3))") # False
```

Term Rewriting

```haskell
E: DNF(Not(Or(A,B))) -> And(Not(A),Not(B))
```

# Inference

Just from the dshape annotations we can infer *a lot* just by
applying simple algorithms before we even hit execution.

Not just type information. Metadata, metacompute. Anything we can
annotate on and write down signatures for!

```
g = (A*B+C*D)**2
```

Operator Constraints
--------------------

```
(+) :: (a,b) -> (a,b) -> (a,b)
(*) :: (a,b) -> (b,c) -> (a,c)
```

Term Constraints
----------------


```
           A : (s, t)
           B : (u, v)
           C : (w, x)
           D : (y, z)

          AB : (a, b)
          CD : (c, d)
     AB + CD : (e, f)
(AB + CD)**2 : (g, h)
```

Constraint Generation
---------------------

```
t = u, a = s, b = v   in AB
x = y, c = w, d = z   in CD
a = c = e, b = d = f  in AB + CD
e = f = g = h         in (AB + CD)**2
```

#

Substitution
------------

```
a = b = c = d = e = f = g = h = s = v = w = z
t = u
x = y
```


Solution
--------

```
A : (a,t)
B : (t,a)
C : (a,x)
D : (x,a)
```


```
g = (A*B+C*D)**2
```

We now have much more knowledge about the expression.

# Layouts

![](layout.png)

#

Example of a order 4 (H_4) curve over a 2D array

```
 0    3    4    5
 1    2    7    6
14   13    8    9
15   12   11   10

 +    +---------+
 |    |         |
 +----+    +----+
           |
 +----+    +----+
 |    |         |
 +    +----+----+
```

#

Coordinate translations for different layouts.

# Hilbert Curve

```python
def rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n-1 - x
            y = n-1 - y
        x, y = y, x
    return x,y

def xy2d(order, x, y):
    n = order / 2
    d = 0
    while n > 0:
        rx = 1 if (x & n) else 0
        ry = 1 if (y & n) else 0
        d += n * n * ((3 * rx) ^ ry)
        x,y = rot(n, x, y, rx, ry)
        n /= 2
    return d

```

# Z Order

```python
def mask(n):
    n &= mask1
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n

def unmask(n):
    n &= mask2
    n = (n ^ (n >> 1)) & 0x33333333
    n = (n ^ (n >> 2)) & 0x0F0F0F0F
    n = (n ^ (n >> 4)) & 0x00FF00FF
    n = (n ^ (n >> 8)) & 0x0000FFFF
    return n

def decode(n):
    return unmask(n), unmask(n >> 1)

def encode(x, y):
    return mask(x) | (mask(y) << 1)

```

# RTS

0. Generate Graph
1. Type Inference
2. Traverse expressions in ATerm expressions.
3. Try to find a function specialized for type, layout and metadata.
       - Use numba to generate kernel ( ``ATerm -> Python AST`` )
       - Link against fast 3rd party library for specialized
         problems.
4. Load Data & Manage Memory
       - Shuffle blocks onto heap
       - Allocate temporaries as needed by data descriptor
       - Unallocate blocks
       - Query data from SQL
       - Copy data to socket ( ZeroMQ, MPI, ... ) if needed
5. Dispatch

- Future: parallel execution
- Future: heterogeneous computation backends ( GPU, cluster )

# What does this give us?

- Blazing fast math
- ... with minimal temporaries
- ... over arbitrary large data
- ... with arbitrary layout
- ![](awesome.gif)

# Long Road Ahead

![](sunrise.jpg)
