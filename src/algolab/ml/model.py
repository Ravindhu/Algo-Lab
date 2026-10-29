"""Trade-quality model: estimates P(win) for a signal from features known at the signal bar.
Used as a filter: a trade is taken only when the probability reaches the strategy's ml_threshold."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from algolab.ml.features import FEATURES
from algolab.profile import ROOT

MODEL_FILE: Path = ROOT / "models" / "trade_filter.joblib"
MIN_TRADES = 60


def dataset(trades: pd.DataFrame, feats_by_symbol: dict[str, pd.DataFrame]):
    """X indexed like `trades`, y = 1 for winning trades. Rows with missing features are dropped."""
    if trades.empty:
        return pd.DataFrame(columns=FEATURES), pd.Series(dtype=int)
    parts = []
    for sym, g in trades.groupby("symbol"):
        f = feats_by_symbol[sym].reindex(pd.to_datetime(g["signal_time"]))
        f.index = g.index
        parts.append(f)
    X = pd.concat(parts).sort_index()[FEATURES]
    X = X[X.notna().all(axis=1)]
    y = (trades.loc[X.index, "pnl"] > 0).astype(int)
    return X, y


def fit(X: pd.DataFrame, y: pd.Series):
    if len(X) < MIN_TRADES or y.nunique() < 2:
        raise ValueError(f"need at least {MIN_TRADES} trades with both outcomes to fit")
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          l2_regularization=1.0, random_state=0).fit(X[FEATURES], y)


def proba(model, X: pd.DataFrame):
    return model.predict_proba(X[FEATURES])[:, 1]


def save(model) -> None:
    """Write to a temp file and rename, so a crash never leaves a half-written model."""
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_FILE.with_suffix(".tmp")
    joblib.dump(model, tmp)
    os.replace(tmp, MODEL_FILE)


def load():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
