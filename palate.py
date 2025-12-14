import hashlib
from dataclasses import dataclass

import numpy as np
import sympy as sp
from jaxlib.xla_client import Array
import logging
from dmmd import dmmd_blockwise
logger = logging.getLogger(__name__)
dmmd_test, dmmd_train, denominator_scale = sp.symbols(
    "dmmd_test dmmd_train denominator_scale"
)
palate = sp.symbols("palate")

PALATE_EXPR: sp.Expr = dmmd_test / (dmmd_test + dmmd_train)
M_PALATE_EXPR: sp.Expr = (
    dmmd_test / (2 * denominator_scale) + sp.Rational(1, 2) * palate
)


# Convert the sympy expressions to functions runnable using jax
MODULE_FOR_SYMPY = "jax"
PALATE_FN = sp.lambdify(
    (dmmd_test, dmmd_train),
    PALATE_EXPR,
    modules=MODULE_FOR_SYMPY,
)

M_PALATE_FN = sp.lambdify(
    (dmmd_test, denominator_scale, palate),
    M_PALATE_EXPR,
    modules=MODULE_FOR_SYMPY,
)

# Get plain-text representation of the functions
PALATE_FORMULA = str(PALATE_EXPR)
M_PALATE_FORMULA = str(M_PALATE_EXPR)


def _formula_hash(expr: sp.Expr) -> str:
    """
    Structural hash of the symbolic expression.
    Robust to formatting but sensitive to math changes.
    Detects meaningful change to the math formula.
    """
    return hashlib.sha256(sp.srepr(expr).encode()).hexdigest()[:12]


PALATE_FORMULA_HASH = _formula_hash(PALATE_EXPR)
M_PALATE_FORMULA_HASH = _formula_hash(M_PALATE_EXPR)


@dataclass(frozen=True)
class PalateComponents:
    # computed
    m_palate: Array
    palate: Array

    # raw
    dmmd_train: Array
    dmmd_test: Array
    denominator_scale: Array

    # formulas
    palate_formula: str
    m_palate_formula: str
    palate_formula_hash: str
    m_palate_formula_hash: str


def compute_palate(
    train_representations: np.ndarray,
    test_representations: np.ndarray,
    gen_representations: np.ndarray,
) -> PalateComponents:
    dmmd_train_val, _ = dmmd_blockwise(train_representations, gen_representations)
    dmmd_test_val, denominator_scale_val = dmmd_blockwise(
        test_representations, gen_representations
    )

    return _compute_palate(dmmd_train_val, dmmd_test_val, denominator_scale_val)


def _compute_palate(
    dmmd_test_val: Array,
    dmmd_train_val: Array,
    denominator_scale_val: Array,
) -> PalateComponents:
    palate_val = PALATE_FN(dmmd_test_val, dmmd_train_val)
    m_palate_val = M_PALATE_FN(dmmd_test_val, denominator_scale_val, palate_val)
    logger.info(f"Palate computed: {m_palate_val=}, {palate_val=}.")
    return PalateComponents(
        denominator_scale=denominator_scale_val,
        dmmd_test=dmmd_test_val,
        dmmd_train=dmmd_train_val,
        palate=palate_val,
        m_palate=m_palate_val,
        palate_formula=PALATE_FORMULA,
        m_palate_formula=M_PALATE_FORMULA,
        palate_formula_hash=PALATE_FORMULA_HASH,
        m_palate_formula_hash=M_PALATE_FORMULA_HASH,
    )
