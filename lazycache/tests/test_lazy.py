from binascii import hexlify
from textwrap import dedent
import numpy as np
from ..lazy import lazy, compute


x = lazy(np.ones(3), own=True)
y = lazy(np.ones(3), own=True)


def test_hash_and_repr():
    assert repr(x) == "<lazy 31f889 ndarray(shape=(3,), dtype=float64)>"
    assert hexlify(x._secure_hash()) == '31f88943ab92191a951a50ab89631ffdf47e55a663671b30414b154636fa7e64'
    assert repr((x + y) * 4) == dedent("""\
        <lazy 165062

          ((v0 + v1) * 4)

        with:

          v0: 31f889 ndarray(shape=(3,), dtype=float64)
          v1: 31f889 ndarray(shape=(3,), dtype=float64)
        )""")
    assert hexlify(((x + y) * 4)._secure_hash()) == '165062018b6da74c73beb9ef15bda205ddfbf75d8263516e26d5d38328885af5'


def test_compute():
    assert compute('foo') == 'foo'  # do nothing for ordinary variables
    e = x - x + (4 / (x + x)) * x
    v = compute(e)
    assert np.all(v == 2) and v.shape == (3,)
