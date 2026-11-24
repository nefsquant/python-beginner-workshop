from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

TRADING_DAYS: int = 252
ASSETS: tuple[str, ...] = ("ALPHA", "BETA", "GAMMA")


def synthetic_prices(days: int = 756, seed: int = 7) -> pd.DataFrame:
    """Generate three correlated price paths, including an initial price of 100."""
    if days < 2:
        raise ValueError("At least two price observations are needed.")

    rng = np.random.default_rng(seed)
    drift: FloatArray = np.array([0.07, 0.09, 0.05])
    volatility: FloatArray = np.array([0.14, 0.22, 0.30])
    correlation: FloatArray = np.array([[1.0, 0.55, 0.15], [0.55, 1.0, 0.25], [0.15, 0.25, 1.0]])
    covariance: FloatArray = np.outer(volatility, volatility) * correlation
    log_returns: FloatArray = rng.multivariate_normal(
        (drift - 0.5 * volatility**2) / TRADING_DAYS,
        covariance / TRADING_DAYS,
        size=days - 1,
    )
    cumulative: FloatArray = np.vstack([np.zeros(len(ASSETS)), np.cumsum(log_returns, axis=0)])
    return pd.DataFrame(
        100.0 * np.exp(cumulative),
        index=pd.bdate_range("2022-01-03", periods=days, name="date"),
        columns=list(ASSETS),
    )
