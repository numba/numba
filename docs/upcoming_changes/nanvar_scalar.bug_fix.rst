Fix scalar handling in ``np.nanvar``
------------------------------------

Fixed scalar handling in the ``np.nanvar`` function. Previously, this
function would fail when called with scalar inputs. Now it properly handles
both scalar and array inputs, returning ``0.0`` for numeric scalars and
``nan`` for NaN inputs, matching NumPy behavior.
