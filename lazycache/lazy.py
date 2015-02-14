import operator
import numpy as np
import joblib.hashing
import struct


HASH_FUNCTION = 'sha256'


def shash(x):
    return joblib.hashing.hash(x, hash_name=HASH_FUNCTION, coerce_mmap=True)


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


class ValueRef(object):
    def __init__(self, value, own):
        self.value = value
        self.immutable = is_immutable(value)
        self.own = own or self.immutable
        self.hash = shash(value)


class BaseExpr(object):
    pass


class Expr(BaseExpr):
    def __init__(self, func_name, func_hash, func, args):
        self.func_name = func_name
        self.func_hash = func_hash
        self.func = func
        self.args = args

    def format(self, vars, varnames):
        arg_strs = [arg.format(vars, varnames) for arg in self.args]
        return '%s(%s)' % (self.func_name, ', '.join(arg_strs))

    def serialize_for_hash(self, values):
        return (self.func_hash,) + tuple(arg.serialize_for_hash(values) for arg in self.args)

    def compute(self, vars):
        computed_args = [arg.compute(vars) for arg in self.args]
        return self.func(*computed_args)


class InfixExpr(Expr):
    def format(self, vars, varnames):
        arg_strs = [arg.format(vars, varnames) for arg in self.args] 
        return '(%s)' % (' %s ' % self.func_name).join(arg_strs)


class Leaf(BaseExpr):
    def __init__(self):
        pass

    def compute(self, vars):
        return vars[self].value

    def format(self, vars, varnames):
        name = varnames.get(self, None)
        if name is None:
            value = vars[self]
            if should_inline_repr(value.value):
                name = repr(value.value)
            else:
                name = varnames[self] = 'v%d' % len(varnames)
        return name

    def serialize_for_hash(self, values):
        return values[self].hash


def binop_impl(name, func):
    def method(self, other):
        other = lazy(other)
        vars = dict(self.vars)
        vars.update(other.vars)
        return Lazy(InfixExpr(name, name, func, (self.expr, other.expr)), vars)
    return method


def rbinop_impl(name, func):
    def method(self, other):
        other = lazy(other)
        vars = dict(self.vars)
        vars.update(other.vars)
        return Lazy(InfixExpr(name, name, func, (other.expr, self.expr)), vars)
    return method

    
class Lazy(object):

    def __init__(self, expr, vars):
        self.expr = expr
        self.vars = vars

    def __secure_hash__(self):
        if isinstance(self.expr, Leaf):
            return self.vars[self.expr].hash
        else:
            return shash(self.expr.serialize_for_hash(self.vars))

    def __compute__(self):
        return self.expr.compute(self.vars)

    def __repr__(self):
        if isinstance(self.expr, Leaf):
            val = self.vars[self.expr]
            return '<lazy %s %s>' % (val.hash[:6], short_repr(val.value))
        else:
            hashval = self.__secure_hash__()
            varnames = {}
            expr_repr = self.expr.format(self.vars, varnames)
            vars_reprs = []
            node_and_name_list = varnames.items()
            node_and_name_list.sort(key=lambda tup: tup[1])  # sort by varname
            for node, name in node_and_name_list:
                val = self.vars[node]
                vars_reprs.append('  %s: %s %s' % (name, val.hash[:6], short_repr(val.value)))
            return '<lazy {hash}\n\n  {expr}\n\nwith:\n\n{vars}\n)'.format(
                hash=hashval[:6], expr=expr_repr, vars='\n'.join(vars_reprs))

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
        expr = Leaf()
        vars = {expr: ValueRef(value, own)}
        return Lazy(expr, vars)


def compute(x):
    if hasattr(x, '__compute__'):
        return x.__compute__()
    else:
        return x
