import hashlib
import time
from dataclasses import dataclass
import dataclasses
import numpy as np
import sympy as sp

from jax import Array
import logging
from dmmd import dmmd_blockwise

logger = logging.getLogger(__name__)
dmmd_test_sym, dmmd_train_sym, denominator_scale_sym = sp.symbols(
    "dmmd_test dmmd_train denominator_scale"
)
palate_sym = sp.symbols("palate")

PALATE_EXPR = dmmd_test_sym / (dmmd_test_sym + dmmd_train_sym)
M_PALATE_EXPR = (
    dmmd_test_sym / (2 * denominator_scale_sym) + sp.Rational(1, 2) * palate_sym
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
class IterableDataclass:
    def __iter__(self):
        fields = dataclasses.fields(self)
        for field in fields:
            yield field.name, getattr(self, field.name)


@dataclass(frozen=True)
class DmmdValues(IterableDataclass):
    train_gen: Array
    test_gen: Array
    test_train: Array
    denominator_scale: Array


@dataclass(frozen=True)
class PalateMetrics(IterableDataclass):
    m_palate: Array
    palate: Array


@dataclass(frozen=True)
class PalateComponents(IterableDataclass):
    """Store the partial results of the calculations along with additional data for reproducibility."""

    # computed
    palate_metrics: PalateMetrics

    # raw
    dmmd_values: DmmdValues
    sigma: float

    # formulas
    m_palate_formula: str
    palate_formula: str
    m_palate_formula_hash: str
    palate_formula_hash: str


def flatten_dataclass(data_class: IterableDataclass) -> dict[str, float | str]:
    field_to_value = {}
    for field, value in data_class:
        if isinstance(value, IterableDataclass):
            sub_field_to_value = flatten_dataclass(value)
            field_to_value = {**field_to_value, **sub_field_to_value}
        else:
            field_to_value[field] = value
    return field_to_value


def compute_palate(
    *,
    train_representations: np.ndarray,
    test_representations: np.ndarray,
    gen_representations: np.ndarray,
    sigma: float,
) -> PalateComponents:
    logger.info("Computing DMMD values...")
    t0 = time.time()

    dmmd_train_gen, _ = dmmd_blockwise(
        x=train_representations,
        y=gen_representations,
        sigma=sigma,
    )
    dmmd_test_gen, denominator_scale = dmmd_blockwise(
        x=test_representations,
        y=gen_representations,
        sigma=sigma,
    )

    dmmd_test_train, _ = dmmd_blockwise(
        x=test_representations,
        y=train_representations,
        sigma=sigma,
    )

    logger.info("DMMD computed in %.3fs", time.time() - t0)

    dmmd_values = DmmdValues(
        train_gen=dmmd_train_gen,
        test_gen=dmmd_test_gen,
        test_train=dmmd_test_train,
        denominator_scale=denominator_scale,
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


def _compute_palate_from_dmmd(dmmd: DmmdValues) -> PalateMetrics:
    logger.info("Computing palate metrics...")
    t0 = time.time()

    palate_val = PALATE_FN(dmmd.train_gen, dmmd.test_gen)
    m_palate_val = M_PALATE_FN(
        dmmd.test_gen,
        dmmd.denominator_scale,
        palate_val,
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
