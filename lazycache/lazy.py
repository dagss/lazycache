"""


Choices made:

* The lazy program built should, when evaluated, match very closely what
  would have happened without caching. While user code should be 'pure',
  other factors contribute too:

  a) Code doing printout or other debug or non-debug side-effects can still
     be considered "pure" for the purposes of computation. And it's good if
     this follows the order it says in the tree-building code.

  b) If one doesn't have enough memory for all temporary results, it's not
     arbitrary which order things are computed in, and we have no model
     or data to do a good job here and must rely on information implicitly
     provided by the order of building the tree.

  To ensure this we incrementally stamp each node on construction, and
  always evaluate in order of that stamp. Obviously more options could be
  added later..


"""


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



_global_expr_count = 0


class BaseExpr(object):
    def __init__(self):
        global _global_expr_count
        self.stamp = _global_expr_count
        _global_expr_count += 1


class Expr(BaseExpr):
    def __init__(self, func_name, func_hash, func, args):
        super(Expr, self).__init__()
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

    def format_single(self, arg_reprs):
        return '%s(%s)' % (self.func_name, ', '.join(arg_reprs))

    def compute(self):
        computed_args = [arg.compute() for arg in self.args]
        return self.func(*computed_args)

    def gather_program(self, exprs, leafs):
        for arg in self.args:
            arg.gather_program(exprs, leafs)
        exprs.append(self)


class InfixExpr(Expr):
    def format(self, varnames):
        arg_strs = [arg.format(varnames) for arg in self.args]
        return '(%s)' % (' %s ' % self.func_name).join(arg_strs)

    def format_single(self, arg_reprs):
        return '(%s)' % (' %s ' % self.func_name).join(arg_reprs)

class Leaf(BaseExpr):
    def __init__(self, value, own):
        super(Leaf, self).__init__()
        self.value = value
        self.immutable = is_immutable(value)
        self.own = own or self.immutable
        self.hash = secure_hash_digest(value)

    def compute(self):
        return self.value

    def format(self, varnames, ):
        name = varnames.get(self, None)
        if name is None:
            if should_inline_repr(self.value):
                name = repr(self.value)
            else:
                name = varnames[self] = 'v%d' % len(varnames)
        return name

    def serialize_for_hash(self, values):
        return values[self].hash

    def gather_program(self, exprs, leafs):
        leafs.append(self)


def binop_impl(name, func):
    def method(self, other):
        return Lazy(InfixExpr(name, name, func, (self._expr, lazy(other)._expr)))
    return method


def rbinop_impl(name, func):
    def method(self, other):
        return Lazy(InfixExpr(name, name, func, (lazy(other)._expr, self._expr)))
    return method


class Lazy(object):

    def __init__(self, expr):
        self._expr = expr

    def _secure_hash(self):
        return self._expr.hash

    def __compute__(self):
        leaf_to_varname, statements = get_program(self._expr)
        return evaluate_program(self._expr, leaf_to_varname, statements)

    def __repr__(self):
        if isinstance(self._expr, Leaf):
            return '<lazy %s %s>' % (hexlify(self._expr.hash[:3]), short_repr(self._expr.value))
        else:
            hashval = self._secure_hash()
            varnames = {}
            leaf_to_varname, statements = get_program(self._expr)
            lines = ['<lazy {hash}'.format(hash=hexlify(hashval[:3])), '  input:']
            for varname, node in sorted([(name, leaf) for leaf, name in leaf_to_varname.items()]):
                lines.append('    %s: %s %s' % (varname, hexlify(node.hash[:3]), short_repr(node.value)))
            lines.extend(['  program:'])
            for varname, node, arg_reprs in statements:
                lines.append('    %s: %s %s' % (varname, hexlify(node.hash[:3]), node.format_single(arg_reprs)))
            lines.append('>')
            return '\n'.join(lines)
            
            #expr_repr = self._expr.format(varnames)
            #vars_reprs = []
            #leaf_and_name_list = varnames.items()
            #leaf_and_name_list.sort(key=lambda tup: tup[1])  # sort by varname
            #for leaf, name in leaf_and_name_list:
            #    vars_reprs.append('  %s: %s %s' % (name, hexlify(leaf.hash[:3]), short_repr(leaf.value)))
            #return '  {expr}\n\nwith:\n\n{vars}\n)'.format(
            #    hash=hexlify(hashval[:3]), expr=expr_repr, vars='\n'.join(vars_reprs))

    __add__ = binop_impl('+', operator.add)
    __sub__ = binop_impl('-', operator.sub)
    __mul__ = binop_impl('*', operator.mul)
    __div__ = binop_impl('/', operator.div)

    __radd__ = rbinop_impl('+', operator.add)
    __rsub__ = rbinop_impl('-', operator.sub)
    __rmul__ = rbinop_impl('*', operator.mul)
    __rdiv__ = rbinop_impl('/', operator.div)


def get_program(root):
    def format_arg(arg):
        return node_to_varname.get(arg, None) or repr(arg.value)
            
    exprs, leafs = [], []
    root.gather_program(exprs, leafs)
    leafs.sort(key=lambda x: x.stamp)
    exprs.sort(key=lambda x: x.stamp)

    leaf_to_varname = {}
    node_to_varname = {}
    idx = 0
    for leaf in leafs:
        if leaf not in leaf_to_varname and not should_inline_repr(leaf.value):
            leaf_to_varname[leaf] = node_to_varname[leaf] = 'v%d' % idx
            idx += 1

    statements = []  #  (target, exprnode, (argname1, argname2, ...))
    idx = 0
    for expr in exprs:
        if expr not in node_to_varname:
            node_to_varname[expr] = varname = 'e%d' % idx
            statements.append((varname, expr, tuple(format_arg(arg) for arg in expr.args)))
            idx += 1

    return leaf_to_varname, statements


def evaluate_program(root, leaf_to_varname, statements):
    vars = {}
    for target, exprnode, argnames in statements:
        args = tuple(arg.value if isinstance(arg, Leaf) else vars[argname]
                     for argname, arg in zip(argnames, exprnode.args))
        result = exprnode.func(*args)
        if exprnode is root:
            return result
        else:
            vars[target] = result
    raise AssertionError()


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
