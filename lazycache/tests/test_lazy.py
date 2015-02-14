from textwrap import dedent
import numpy as np
from ..lazy import lazy, compute


x = lazy(np.ones(3), own=True)
y = lazy(np.ones(3), own=True)


def test_hash_and_repr():
    assert repr(x) == "<lazy 31f889 ndarray(shape=(3,), dtype=float64)>"
    assert x.__secure_hash__() == '31f88943ab92191a951a50ab89631ffdf47e55a663671b30414b154636fa7e64'
    assert repr((x + y) * 4) == dedent("""\
        <lazy a72c40

          ((v0 + v1) * 4)

        with:

          v0: 31f889 ndarray(shape=(3,), dtype=float64)
          v1: 31f889 ndarray(shape=(3,), dtype=float64)
        )""")
    assert ((x + y) * 4).__secure_hash__() == 'a72c40a2a6bc8584547fe1d5b69a5bd82b632b0264eda07baa5a5a468bd6db4b'


def test_compute():
    assert compute('foo') == 'foo'  # do nothing for ordinary variables
    e = x - x + (4 / (x + x)) * x
    v = compute(e)
    assert np.all(v == 2) and v.shape == (3,)
