import hashlib
import time
from dataclasses import dataclass
import dataclasses
import numpy as np
import sympy as sp
import logging

from jax import Array
from dmmd import dmmd_blockwise

logger = logging.getLogger(__name__)

# =========================
# Symbolic definitions
# =========================

dmmd_test_sym, dmmd_train_sym, denominator_scale_sym = sp.symbols(
    "dmmd_test dmmd_train denominator_scale"
)

# palate uses sigma / 3
PALATE_EXPR = dmmd_test_sym / (dmmd_test_sym + dmmd_train_sym)

# m_palate:
#   first term → sigma
#   second term → sigma / 3
M_PALATE_EXPR = (
    dmmd_test_sym / (2 * denominator_scale_sym)
    + sp.Rational(1, 2) * PALATE_EXPR
)

MODULE_FOR_SYMPY = "numpy"

PALATE_FN = sp.lambdify(
    (dmmd_train_sym, dmmd_test_sym),
    PALATE_EXPR,
    modules=MODULE_FOR_SYMPY,
)

M_PALATE_FN = sp.lambdify(
    (dmmd_test_sym, denominator_scale_sym, dmmd_train_sym),
    M_PALATE_EXPR,
    modules=MODULE_FOR_SYMPY,
)

PALATE_FORMULA = str(PALATE_EXPR)
M_PALATE_FORMULA = str(M_PALATE_EXPR)


def formula_hash(expr: sp.Expr) -> str:
    return hashlib.sha256(sp.srepr(expr).encode()).hexdigest()[:12]


PALATE_FORMULA_HASH = formula_hash(PALATE_EXPR)
M_PALATE_FORMULA_HASH = formula_hash(M_PALATE_EXPR)

# =========================
# Dataclasses
# =========================

@dataclass(frozen=True)
class IterableDataclass:
    def __iter__(self):
        for field in dataclasses.fields(self):
            yield field.name, getattr(self, field.name)


@dataclass(frozen=True)
class DmmdValues(IterableDataclass):
    train_gen_sigma: Array
    test_gen_sigma: Array
    test_gen_sigma3: Array
    train_gen_sigma3: Array
    denominator_scale_sigma: Array


@dataclass(frozen=True)
class PalateMetrics(IterableDataclass):
    m_palate: Array
    palate: Array


@dataclass(frozen=True)
class PalateComponents(IterableDataclass):
    palate_metrics: PalateMetrics
    dmmd_values: DmmdValues
    sigma: float

    palate_formula: str
    m_palate_formula: str
    palate_formula_hash: str
    m_palate_formula_hash: str


# =========================
# Main computation
# =========================

def compute_palate(
    *,
    train_representations: np.ndarray,
    test_representations: np.ndarray,
    gen_representations: np.ndarray,
    sigma: float,
) -> PalateComponents:

    logger.info("Computing DMMD values...")
    t0 = time.time()

    sigma3 = sigma / 3

    # --- sigma ---
    dmmd_train_gen_sigma, _ = dmmd_blockwise(
        x=train_representations,
        y=gen_representations,
        sigma=sigma,
    )

    dmmd_test_gen_sigma, denom_sigma = dmmd_blockwise(
        x=test_representations,
        y=gen_representations,
        sigma=sigma,
    )

    # --- sigma / 3 ---
    dmmd_test_gen_sigma3, _ = dmmd_blockwise(
        x=test_representations,
        y=gen_representations,
        sigma=sigma3,
    )

    dmmd_train_gen_sigma3, _ = dmmd_blockwise(
        x=train_representations,
        y=gen_representations,
        sigma=sigma3,
    )

    logger.info("DMMD computed in %.3fs", time.time() - t0)

    dmmd_values = DmmdValues(
        train_gen_sigma=dmmd_train_gen_sigma,
        test_gen_sigma=dmmd_test_gen_sigma,
        test_gen_sigma3=dmmd_test_gen_sigma3,
        train_gen_sigma3=dmmd_train_gen_sigma3,
        denominator_scale_sigma=denom_sigma,
    )

    palate_metrics = _compute_palate_from_dmmd(dmmd_values)

    return PalateComponents(
        palate_metrics=palate_metrics,
        dmmd_values=dmmd_values,
        sigma=sigma,
        palate_formula=PALATE_FORMULA,
        m_palate_formula=M_PALATE_FORMULA,
        palate_formula_hash=PALATE_FORMULA_HASH,
        m_palate_formula_hash=M_PALATE_FORMULA_HASH,
    )


# =========================
# Final metric
# =========================

def _compute_palate_from_dmmd(dmmd: DmmdValues) -> PalateMetrics:
    logger.info("Computing palate metrics...")
    t0 = time.time()

    # palate → sigma / 3
    palate_val = PALATE_FN(
        dmmd.train_gen_sigma3,
        dmmd.test_gen_sigma3,
    )

    # m_palate:
    #   first term → sigma
    #   second term → sigma / 3
    m_palate_val = M_PALATE_FN(
        dmmd.test_gen_sigma,
        dmmd.denominator_scale_sigma,
        dmmd.train_gen_sigma3,
    )

    logger.info(
        "Palate computed in %.3fs (m_palate=%.6f, palate=%.6f)",
        time.time() - t0,
        m_palate_val,
        palate_val,
    )

    return PalateMetrics(
        palate=palate_val,
        m_palate=m_palate_val,
    )
