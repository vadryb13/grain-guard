"""Обучение ML-моделей (baseline №2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelBundle:
    model: object
    feature_cols: list[str]
    threshold: float
    metrics: dict[str, float] = field(default_factory=dict)
    importance: pd.DataFrame | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_cols])[:, 1]


def split_by_group(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть (train_pos, test_pos) — позиции строк в исходном df без утечки по silo_id."""
    keep = meta["keep"].to_numpy(dtype=bool)
    valid_pos = np.where(keep)[0]
    groups = meta["silo_id"].to_numpy()[valid_pos]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(splitter.split(valid_pos, groups=groups))
    return valid_pos[tr], valid_pos[te]


def _best_threshold(y_true: np.ndarray, proba: np.ndarray, min_precision: float) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    best_thr = 0.5
    best_rec = -1.0
    for p, r, t in zip(prec[:-1], rec[:-1], thr):
        if p >= min_precision and r > best_rec:
            best_rec = r
            best_thr = float(t)
    return best_thr


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    if len(y_true) == 0:
        return {"roc_auc": float("nan"), "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "roc_auc": float(auc),
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def fit_baselines(
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    Xte: pd.DataFrame,
    yte: pd.Series,
    seed: int = 42,
    min_precision: float = 0.7,
) -> dict[str, ModelBundle]:
    """Обучить LogReg и бустинг и оценить на отложенных данных."""
    feature_cols = list(Xtr.columns)
    ytr_np = ytr.to_numpy()
    yte_np = yte.to_numpy()

    candidates = {
        "logreg": Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "gbm": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6, random_state=seed
        ),
    }

    bundles: dict[str, ModelBundle] = {}
    for name, model in candidates.items():
        model.fit(Xtr, ytr_np)
        proba = model.predict_proba(Xte)[:, 1]
        thr = _best_threshold(yte_np, proba, min_precision)
        bundle = ModelBundle(model=model, feature_cols=feature_cols, threshold=thr)
        bundle.metrics = _metrics(yte_np, proba, thr)
        bundles[name] = bundle

    best_name = max(bundles, key=lambda k: (bundles[k].metrics["roc_auc"] or 0.0))
    best = bundles[best_name]
    n = len(Xte)
    sample = min(n, 20000)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample, replace=False) if n > sample else np.arange(n)
    if sample > 0:
        imp = permutation_importance(
            best.model, Xte.iloc[idx], yte_np[idx], n_repeats=5, random_state=seed, scoring="roc_auc"
        )
        best.importance = (
            pd.DataFrame({"feature": feature_cols, "importance": imp.importances_mean})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    bundles["best"] = ModelBundle(
        model=best.model,
        feature_cols=feature_cols,
        threshold=best.threshold,
        metrics={**best.metrics, "selected": best_name},  # type: ignore[dict-item]
    )
    return bundles


def train_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
    min_precision: float = 0.7,
) -> dict[str, ModelBundle]:
    """Удобная обёртка: сплит по группам + обучение."""
    tr, te = split_by_group(X, y, meta, test_size=test_size, seed=seed)
    return fit_baselines(
        X.iloc[tr].reset_index(drop=True),
        y.iloc[tr].reset_index(drop=True),
        X.iloc[te].reset_index(drop=True),
        y.iloc[te].reset_index(drop=True),
        seed=seed,
        min_precision=min_precision,
    )
