import hashlib
import logging
import time
from dataclasses import dataclass

import numpy as np
import sympy as sp
<<<<<<< Updated upstream
from jaxlib.xla_client import Array

=======
from jax import Array
import logging
>>>>>>> Stashed changes
from dmmd import dmmd_blockwise

logger = logging.getLogger(__name__)
dmmd_test_sym, dmmd_train_sym, denominator_scale_sym = sp.symbols(
    "dmmd_test dmmd_train denominator_scale"
)
palate_sym = sp.symbols("palate")

PALATE_EXPR = dmmd_test_sym / (dmmd_test_sym + dmmd_train_sym)
M_PALATE_EXPR = (
    dmmd_test_sym / (2 * denominator_scale_sym)
    + sp.Rational(1, 2) * palate_sym
)

MODULE_FOR_SYMPY = "numpy"
PALATE_FN = sp.lambdify(
    (dmmd_train_sym, dmmd_test_sym),
    PALATE_EXPR,
    modules=MODULE_FOR_SYMPY,
)

M_PALATE_FN = sp.lambdify(
    (dmmd_test_sym, denominator_scale_sym, palate_sym),
    M_PALATE_EXPR,
    modules=MODULE_FOR_SYMPY,
)

PALATE_FORMULA = str(PALATE_EXPR)
M_PALATE_FORMULA = str(M_PALATE_EXPR)


def formula_hash(expr: sp.Expr) -> str:
    """Structural hash of a symbolic expression."""
    return hashlib.sha256(sp.srepr(expr).encode()).hexdigest()[:12]


PALATE_FORMULA_HASH = formula_hash(PALATE_EXPR)
M_PALATE_FORMULA_HASH = formula_hash(M_PALATE_EXPR)

@dataclass(frozen=True)
class DmmdValues:
    train: Array
    test: Array
    denominator_scale: Array


@dataclass(frozen=True)
class PalateMetric:
    palate: Array
    m_palate: Array


@dataclass(frozen=True)
class PalateComponents:
    """Store the partial results of the calculations along with additional data for reproducibility."""
    # computed
    m_palate: Array
    palate: Array

    # raw
    dmmd_train: Array
    dmmd_test: Array
    denominator_scale: Array
    sigma: float

    # formulas
    palate_formula: str
    m_palate_formula: str
    palate_formula_hash: str
    m_palate_formula_hash: str


def compute_palate(
    *,
    train_representations: np.ndarray,
    test_representations: np.ndarray,
    gen_representations: np.ndarray,
    sigma: float,
) -> PalateComponents:
    logger.info("Computing DMMD values...")
    t0 = time.time()

    dmmd_train, _ = dmmd_blockwise(
        x=train_representations,
        y=gen_representations,
        sigma=sigma,
    )
    dmmd_test, denominator_scale = dmmd_blockwise(
        x=test_representations,
        y=gen_representations,
        sigma=sigma,
    )

    logger.info("DMMD computed in %.3fs", time.time() - t0)

    dmmd = DmmdValues(
        train=dmmd_train,
        test=dmmd_test,
        denominator_scale=denominator_scale,
    )

    palate_metric = _compute_palate_from_dmmd(dmmd)

    return PalateComponents(
        palate=palate_metric.palate,
        m_palate=palate_metric.m_palate,
        dmmd_train=dmmd.train,
        dmmd_test=dmmd.test,
        denominator_scale=dmmd.denominator_scale,
        sigma=sigma,
        palate_formula=PALATE_FORMULA,
        m_palate_formula=M_PALATE_FORMULA,
        palate_formula_hash=PALATE_FORMULA_HASH,
        m_palate_formula_hash=M_PALATE_FORMULA_HASH,
    )


def _compute_palate_from_dmmd(dmmd: DmmdValues) -> PalateMetric:
    logger.info("Computing palate metrics...")
    t0 = time.time()

    palate_val = PALATE_FN(dmmd.train, dmmd.test)
    m_palate_val = M_PALATE_FN(
        dmmd.test,
        dmmd.denominator_scale,
        palate_val,
    )

    logger.info(
        "Palate computed in %.3fs (m_palate=%.6f, palate=%.6f)",
        time.time() - t0,
        m_palate_val,
        palate_val,
    )

    return PalateMetric(
        palate=palate_val,
        m_palate=m_palate_val,
    )