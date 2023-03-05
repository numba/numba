
Changelog
=========

This directory contains "news fragments" which are short files that contain a
small **ReST**-formatted text that will be added to the next what's new page.

Make sure to use full sentences with correct case and punctuation, and please
try to use Sphinx intersphinx using backticks. The fragment should have a
header line and an underline using ``========`` followed by description of
your user-facing changes as they should appear in the relase notes.

Each file should be named like ``<PULL REQUEST>.<TYPE>.rst``, where
``<PULL REQUEST>`` is a pull request number, and ``<TYPE>`` is one of:

* ``feature``: Signifying a new feature.

* ``bugfix``: Signifying a bug fix.

* ``doc``: Signifying a documentation improvement.

* ``removal``: Signifying a deprecation or removal of public API.

* ``misc``: A ticket has been closed, but it is not of interest to users.

If you are unsure what pull request type to use, don't hesitate to ask in your
PR.

You can install ``towncrier`` and run ``towncrier build --draft``
if you want to get a preview of how your change will look in the final release
notes.

.. note::
    This README was adapted from the NumPy changelog readme under the terms of
    the MIT licence.