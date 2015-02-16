from binascii import hexlify
from textwrap import dedent
import numpy as np
from ..lazy import lazy, compute, get_program


x = lazy(np.ones(3), own=True)
y = lazy(np.ones(3), own=True)



def test_compute():
    assert compute('foo') == 'foo'  # do nothing for ordinary variables
    e = x - x + (4 / (x + x)) * x
    v = compute(e)
    assert np.all(v == 2) and v.shape == (3,)


def test_program_tracks_node_creation():
    # Make sure that the way things are computed, and printed in repr, directly
    # reflects exactly how the tree was constructed. So "e = 5 * x; e + e" has
    # the same hash, but not the same program, as (5 * x) + (5 * x).
    e = 5 * x
    assert repr(e + e) == dedent("""\
        <lazy 535687
          input:
            v0: 31f889 ndarray(shape=(3,), dtype=float64)
          program:
            e0: ddc9e1 (5 * v0)
            e1: 535687 (e0 + e0)
        >""")
    v1 = compute(e + e)

    assert repr((5 * x) + (5 * x)) == dedent("""\
        <lazy 535687
          input:
            v0: 31f889 ndarray(shape=(3,), dtype=float64)
          program:
            e0: ddc9e1 (5 * v0)
            e1: ddc9e1 (5 * v0)
            e2: 535687 (e0 + e1)
        >""")
    v2 = compute((5 * x) + (5 * x))
    
    assert np.all(v1 == v2) and np.all(v2 == 10) and v1.shape == v2.shape == (3,)




# TODO:
# Lazy's hash and eq. e + e != (5 * x) + (5 * x)
