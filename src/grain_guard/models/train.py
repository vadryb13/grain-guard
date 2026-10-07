"""Обучение ML-моделей (baseline №2): LogReg и HistGradientBoosting."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelBundle:
    """Модель с порогом, метриками и важностью признаков."""

    name: str
    model: object
    threshold: float = 0.5
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    importance: pd.DataFrame | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model.predict_proba(X)[:, 1]
        return np.asarray(proba, dtype=np.float64)


def split_by_group(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Сплит по silo_id без утечки: силос целиком попадает в train или test."""
    rng = np.random.default_rng(seed)
    silos = meta["silo_id"].drop_duplicates().to_numpy(dtype=object)
    rng.shuffle(silos)
    n_test = max(1, int(round(len(silos) * test_size)))
    test_silos = set(silos[:n_test])
    is_test = meta["silo_id"].isin(test_silos).to_numpy()
    train_idx = np.where(~is_test)[0]
    test_idx = np.where(is_test)[0]
    if len(np.unique(y.iloc[train_idx])) < 2 or len(np.unique(y.iloc[test_idx])) < 2:
        raise ValueError("В train или test только один класс — увеличьте выборку")
    return train_idx, test_idx


def _importance(model: object, feature_names: list[str]) -> pd.DataFrame | None:
    est = model
    if hasattr(est, "named_steps"):
        est = est.named_steps.get("clf", est)
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(est.coef_).ravel()
    else:
        return None
    df = pd.DataFrame({"feature": feature_names, "importance": np.asarray(values)})
    return df.sort_values("importance", ascending=False, ignore_index=True)


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        pred = proba >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def _make_candidates(seed: int) -> dict[str, object]:
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    LogisticRegression(max_iter=1000, C=1.0),
                    method="sigmoid",
                    cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                ),
            ),
        ]
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, random_state=seed
    )
    return {"logreg": logreg, "hgb": hgb}


def fit_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = 42,
) -> dict[str, ModelBundle]:
    """Обучить LogReg и HistGB, выбрать лучшую по AUC на отложенной выборке."""
    bundles: dict[str, ModelBundle] = {}
    for name, model in _make_candidates(seed).items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
        threshold = _best_threshold(np.asarray(y_test), proba)
        bundles[name] = ModelBundle(
            name=name,
            model=model,
            threshold=threshold,
            metrics={"roc_auc": round(auc, 4), "threshold": round(threshold, 3)},
            importance=_importance(model, list(X_train.columns)),
        )

    best_name = max(bundles, key=lambda k: bundles[k].metrics["roc_auc"])
    best = bundles[best_name]
    best.metrics["selected"] = best_name
    bundles["best"] = best
    return bundles


def serialize_bundles(bundles: dict[str, ModelBundle]) -> bytes:
    """Сериализовать набор моделей в байты (joblib)."""
    buf = io.BytesIO()
    joblib.dump(bundles, buf, protocol=5)
    return buf.getvalue()


def deserialize_bundles(data: bytes) -> dict[str, ModelBundle]:
    buf = io.BytesIO(data)
    return joblib.load(buf)
