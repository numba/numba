Fix scalar handling in ``np.nanmean``
-------------------------------------

Fixed scalar handling in ``np.nanmean``. Previously this function raised a
TypingError when called with scalar inputs. It now supports scalar float,
integer, and boolean inputs, matching NumPy behavior.