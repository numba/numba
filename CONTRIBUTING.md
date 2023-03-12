
We welcome people who want to make contributions to Numba, big or small!
Even simple documentation improvements are encouraged.

# Asking questions

Numba has a [discourse forum](https://numba.discourse.group/) for longer/more
involved questions and an IRC channel on
[gitter.im](https://gitter.im/numba/numba) for quick questions and interactive
help.

# Ways to help:

There's lots of ways to help improve Numba, some of these require creating code
changes, see **contributing patches** below.

## Quick things:

* Answer a question asked on [discourse](https://numba.discourse.group/) or
  [gitter.im](https://gitter.im/numba/numba).
* Review a page of documentation, check it makes sense, that it's clear and
  still relevant, that the examples are present, good and working. Fix anything
  that needs updating in a pull request.

## More involved things:

* Review a pull request, you don't need to be a compiler engineer to do an
  initial review of a pull request. It's incredibly helpful to have pull
  requests go through a review to just make sure the code change is well formed,
  documented, efficient and clear. Further, if the code is fixing a bug, making
  sure that tests are present demonstrating it is fixed! Look out for PRs with
  the [`needs initial review`](https://github.com/numba/numba/labels/needs%20initial%20review)
  label. There are also time boxed tasks available on the
  [contributor self-service board](https://github.com/orgs/numba/projects/7).
* Work on fixing or implementing something in the code base, there are a lot of
  [`good first issue's`](https://github.com/numba/numba/labels/good%20first%20issue)
  and [`good second issue's`](https://github.com/numba/numba/labels/good%20first%20issue).
  For implementing new features/functionality, the extension API is the best
  thing to use and a guide to using `@overload` in particular is
  [here](https://numba.readthedocs.io/en/latest/extending/overloading-guide.html)
  and the API documentation is [here](https://numba.readthedocs.io/en/latest/extending/high-level.html#implementing-functions).

## Contributing patches

Please fork the Numba repository on Github, and create a new branch
containing your work.  When you are done, open a pull request.

# Further reading

Please read the [contributing guide](
https://numba.readthedocs.io/en/latest/developer/contributing.html).
