import operator
import numpy as np
import struct
import hashlib
from binascii import hexlify
from . import hashing


HASH_FUNCTION = 'sha256'


def secure_hash_digest(x):
    return hashing.hash(x, hash_name=HASH_FUNCTION, coerce_mmap=True).digest()


def make_hasher():
    return hashlib.new(HASH_FUNCTION)


immutable_scalars = {
    int, long, str, unicode, bytes, frozenset, float
}


def is_immutable(x):
    if type(x) in immutable_scalars:
        return True
    elif type(x) is tuple:
        return all(is_immutable(x) for x in x)
    else:
        return False


def should_inline_repr(x):
    return (type(x) in immutable_scalars
            and not (isinstance(x, basestring) and len(x) > 10))


short_reprs = {
    np.ndarray: lambda x: 'ndarray(shape=%r, dtype=%s)' % (x.shape, str(x.dtype))
}


def short_repr(x):
    formatter = short_reprs.get(type(x), repr)
    return formatter(x)


class BaseExpr(object):
    pass


class Expr(BaseExpr):
    def __init__(self, func_name, func_hash, func, args):
        self.func_name = func_name
        self.func_hash = func_hash
        self.func = func
        self.args = args
        h = make_hasher()
        h.update(func_hash)
        for arg in args:
            h.update(arg.hash)
        self.hash = h.digest()

    def format(self, varnames):
        arg_strs = [arg.format(varnames) for arg in self.args]
        return '%s(%s)' % (self.func_name, ', '.join(arg_strs))

    def compute(self):
        computed_args = [arg.compute() for arg in self.args]
        return self.func(*computed_args)


class InfixExpr(Expr):
    def format(self, varnames):
        arg_strs = [arg.format(varnames) for arg in self.args]
        return '(%s)' % (' %s ' % self.func_name).join(arg_strs)


class Leaf(BaseExpr):
    def __init__(self, value, own):
        self.value = value
        self.immutable = is_immutable(value)
        self.own = own or self.immutable
        self.hash = secure_hash_digest(value)

    def compute(self):
        return self.value

    def format(self, varnames):
        name = varnames.get(self, None)
        if name is None:
            if should_inline_repr(self.value):
                name = repr(self.value)
            else:
                name = varnames[self] = 'v%d' % len(varnames)
        return name

    def serialize_for_hash(self, values):
        return values[self].hash


def binop_impl(name, func):
    def method(self, other):
        return Lazy(InfixExpr(name, name, func, (self.expr, lazy(other).expr)))
    return method


def rbinop_impl(name, func):
    def method(self, other):
        return Lazy(InfixExpr(name, name, func, (lazy(other).expr, self.expr)))
    return method


class Lazy(object):

    def __init__(self, expr):
        self.expr = expr

    def _secure_hash(self):
        return self.expr.hash

    def __compute__(self):
        return self.expr.compute()

    def __repr__(self):
        if isinstance(self.expr, Leaf):
            return '<lazy %s %s>' % (hexlify(self.expr.hash[:3]), short_repr(self.expr.value))
        else:
            hashval = self._secure_hash()
            varnames = {}
            expr_repr = self.expr.format(varnames)
            vars_reprs = []
            leaf_and_name_list = varnames.items()
            leaf_and_name_list.sort(key=lambda tup: tup[1])  # sort by varname
            for leaf, name in leaf_and_name_list:
                vars_reprs.append('  %s: %s %s' % (name, hexlify(leaf.hash[:3]), short_repr(leaf.value)))
            return '<lazy {hash}\n\n  {expr}\n\nwith:\n\n{vars}\n)'.format(
                hash=hexlify(hashval[:3]), expr=expr_repr, vars='\n'.join(vars_reprs))

    __add__ = binop_impl('+', operator.add)
    __sub__ = binop_impl('-', operator.sub)
    __mul__ = binop_impl('*', operator.mul)
    __div__ = binop_impl('/', operator.div)

    __radd__ = rbinop_impl('+', operator.add)
    __rsub__ = rbinop_impl('-', operator.sub)
    __rmul__ = rbinop_impl('*', operator.mul)
    __rdiv__ = rbinop_impl('/', operator.div)


def lazy(value, own=False):
    if isinstance(value, Lazy):
        return value
    else:
        return Lazy(Leaf(value, own))


def compute(x):
    if hasattr(x, '__compute__'):
        return x.__compute__()
    else:
        return x
