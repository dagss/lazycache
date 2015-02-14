# lazycache

Python tool to cache computations by constructing lazy computation trees,
then use them as keys into a cache.

## Example

```
>>> x = lazy(np.zeros(1000), own=True)
>>> x
<lazy 23r5fs ndarray(shape=(1000,), dtype=float64)>



## computed

A `computed` is a "value tagged with its sha256 hash". It serves two
purposes:

* Quickly compare equality of complex/large objects (like NumPy arrays)
* Operations performed on a `computed` will result in building a
  `lazy` program/expression tree

Native immutable objects are hashed inline, a sha256 hash is not computed
for them.

```
>>> computed(3)
computed(3)
>>> computed('hello')
computed('hello')
>>> computed(np.random.normal(size=10**6), own=True)
computed(aaef23:ndarray(shape=1000000,dtype=float64))
```

The `own=True` means "trust that this array will not be mutated", and
should only be used when you have a guarantee for this. Types that
are registered as immutable do not need it. Otherwise,
a `?` is displayed instead of the hash and hashing delayed until
the point of computation.

```
>>> computed(np.random.normal(size=10**6))
computed(?:ndarray(shape=1000000,dtype=float64))
```


## lazy

What we want to do cache our computations is to

 a) Start out with one or more `computed`, those are our hashed inputs

 b) Construct a `lazy` expression tree/program; this gives us
    a hash for "inputs + what to do with them"

 c) Use the `lazy` as a key to do caching.
    
To construct a `lazy`, start out with one or more `computed` and
operate on them. Once a `computed` is in the tree, everything else
is coerced to a computed; this is done so that as much code as possible
should be able to work both for constructing `lazy` and for
carrying out real computations when passed actual values.

```
>>> x = computed(np.zeros(100000), own=True)
>>> e = x + 4 - np.ones(100000) * 4
>>> e
lazy(3f2a23:

    v1 + 4 - (v2 * 4)

with

    v1: 43ta24 ndarray(shape=(100000,), dtype=float64))
    v2: 349t82 ndarray(shape=(100000,), dtype=float64))
)
```

To perform the computation we do

```
>>> y = compute(e)
>>> y
array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
```

At this point it is very simple do build our own cache. Lookups
are fast because the expressions used as keys are simply `lazy`
instances, no actual computation happen.

```
>>> mycache = {}
>>> e in mycache
False
>>> mycache[e] = compute(100000)
>>> (computed(np.zeros(100000)) + 4 - np.ones(100000) * 4) in mycache
True
>>> (np.zeros(100000) + 4 - computed(np.ones(100000)) * 4) in mycache
True
```


## cached

lazycache ships with a builtin cache though so that you don't have to roll
your own and with some extra bells and whistles. You use it on an expression
like you would have used `compute`:

```
>>> y = cache(e)
>>> y
array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
```

However, the result is write-protected (for types where that is possible;
or a copy otherwise):

```
>>> y[0,0] = 3
Traceback
    ...
ValueError: assignment destination is read-only
>>> y is cache(e)
True
```

## pure functions, cached IO

Realistically, all you want to do cannot be expressed as part of the
expression tree. The `@pure` decorator helps create native building
blocks.

Basic example:
```python

def f(x, y):
    return x + y

@pure(version=1)
def g(x, y):
    return x + h(y)
```

A normal function just helps assembling the tree:
```
>>> arr = computed(np.random.normal(size=1000), own=True)
>>> f(arr, arr)
lazy(23fasd:

    v1 + v1

with

    v1: 4322we ndarray(shape=(1000,), dtype=float64))
)
```

In contrast, the pure function is used opaquely:

```
>>> g(arr, arr)
lazy(34fw2c:

    __main__.g(v1)

with

    v1: 4322we ndarray(shape=(1000,), dtype=float64))
)
```

It is now the responsibility of the programmer that `g` returns the
same results every time, including its use of `h` and everything else
that goes on. The name `__main__.g` as well as the indicated version
`1` is made part of the hash; so if the version is bumped and the code
re-run, the `lazy` will get another hash.

A big use-case for `pure` is to cache reading large files from disk:

```python
from lazycache import pure, hash_by_file_date

@pure(hashers={'filename': hash_by_file_date})
def parse_my_file(filename):
    return bazify(file(filename))
```

What will happen here is that the filename argument will go through
`os.path.realpath(os.path.abspath(filename))` prior to being passed
to the function, and the filename and its modification date will be
part of hash:

```
>>> parse_my_file("foo.dat")
lazy(34fw2c:

    __main__.parse_my_file(v1)

with

    v1: er2c23 <file "/home/dagss/data/foo.dat" modified 2015-02-13 19:55:01>
)
```

## Power user API

(registering your own types)

## Licensing

lazycache is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2015, lazycache developpers
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, 
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of Dag Sverre Seljebotn nor the names of other lazycache 
      contributors may be used to endorse or promote products derived from 
      this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    owner or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such
    damage.**
