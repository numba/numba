
Changelog
=========

This directory contains "news fragments" which are short files that contain a
small **ReST**-formatted text that will be added to the next what's new page.

Make sure to use full sentences with correct case and punctuation, and please
try to use Sphinx intersphinx using backticks. The fragment should have a
header line and an underline using ``======`` followed by description of
your user-facing changes as they should appear in the release notes.

Each file should be named like ``<PULL REQUEST>.<TYPE>.rst``, where
``<PULL REQUEST>`` is a pull request number, and ``<TYPE>`` is one of:

* ``highlight``: Adds a highlighted bullet point to use as a possible highlight
  of the release.
* ``np_support``: Addition of new NumPy functionality.
* ``deprecation``: Changes to existing code that will now emit a deprecation warning.
* ``expired``: Removal of a deprecated part of the API.
* ``compatibility``: A change which requires users to change code and is not
  backwards compatible. (Not to be used for removal of deprecated features.)
* ``cuda``: Changes in the CUDA target implementation.
* ``new_feature``: New user facing features like ``kwargs``.
* ``improvement``: General improvements and edge-case changes which are
  not new features or compatibility related.
* ``performance``: Performance changes that should not affect other behaviour.
* ``change``: Other changes
* ``doc``: Documentation related changes.
* ``infrastructure``: Infrastructure/CI related changes. 
* ``bug_fix``: Bug Fixes for exiting features/functionality.

If you are unsure what pull request type to use, don't hesitate to ask in your
PR.

Once you've generated your fragment, to validate it, run 
``python maint/towncrier_rst_validator.py --pull_request_id {PR Number}``
while on your branch and in numba base directory, where PR Number is the
assigned ID for your respective pull request on Github.

You can install ``towncrier`` and run ``towncrier build --draft``
if you want to get a preview of how your change will look in the final release
notes.

.. note::
    This README was adapted from the NumPy changelog readme under the terms of
    the `BSD-3 licence <https://github.com/numpy/numpy/blob/c1ffdbc0c29d48ece717acb5bfbf811c935b41f6/LICENSE.txt>`_.
