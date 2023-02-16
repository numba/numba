.. Adapted from the dask-contrib policy document https://github.com/dask/dask/blob/89ecb076bbbbd767eeef7e8cd1040cb838e33a5c/docs/source/contrib.rst

Incubator Projects
==================

In addition to the primary `Numba repository <https://github.com/numba/numba>`_, there are other incubator projects that explore new compilation technologies. Since these projects are in the explorative phase of their development, their long-term support and stability are not guaranteed. 

Some of these projects can be owned and maintained by folks from the community and therefore these projects may operate in a different workflow to that of other projects in the Numba organization. However these projects are expected to conform to the `Numba Code of Conduct <https://github.com/numba/numba-governance/blob/accepted/code-of-conduct.md>`_.


Incubator project acceptance criteria
-------------------------------------

If you are working on a project which you think could be a great addition to the Numba incubator projects then please feel free to raise an issue at TBD requesting a transfer.

For a project to be accepted as an Incubator Project it must meet the following criteria:

- The project must aim to complement or enhance the Numba compilation pipeline in some way, maybe it explores an alternative compiler back-end, a different type-inference strategy, different intermediate representations, etc.
- The project must have at least one active maintainer.


Graduation to the Numba repository
----------------------------------

In some cases it may be appropriate to graduate a project to the main Numba project.

This will be assessed on a case-by-case basis by the Numba maintenance team.

For some projects it may make sense to graduate their content into other repositories.

Generally for a project to make it into the Numba project it should meet the following criteria:

- The project much have an active community.
- The project must be maintained by one or more people who are employed to do so and will dedicate a minimum of 1 day per week to Numba.
- Maintainers must engage in providing support for the code.
- The project must have tests which are run via CI and clear acceptance criteria for PRs to enable other maintainers assist.