from ..palate import _formula_hash
import sympy as sp

def test_hash_consistency():
    x = sp.symbols('x')
    expr = x**2 + 2*x + 1
    h1 = _formula_hash(expr)
    h2 = _formula_hash(expr)
    assert h1 == h2, "Hash should be consistent for the same expression"


def test_hash_diff_for_different_exprs():
    x = sp.symbols('x')
    expr1 = x**2 + 2*x + 1
    expr2 = x**2 + 3*x + 1
    assert _formula_hash(expr1) != _formula_hash(expr2), "Different expressions should have different hashes"


def test_hash_equivalent_expr_structure():
    x = sp.symbols('x')
    expr1 = x + x
    expr2 = 2*x
    # These are mathematically equivalent but structurally different
    assert _formula_hash(expr1) != _formula_hash(expr2), "Hashes depend on structure, not simplification"


def test_hash_handles_functions():
    x = sp.symbols('x')
    expr = sp.sin(x) + sp.cos(x)
    h = _formula_hash(expr)
    assert isinstance(h, str)
    assert len(h) == 12, "Hash should be a 12-character string"


def test_hash_with_nested_expr():
    x, y = sp.symbols('x y')
    expr = (x + y)**2 + sp.exp(x*y)
    h = _formula_hash(expr)
    assert isinstance(h, str)
    assert len(h) == 12
