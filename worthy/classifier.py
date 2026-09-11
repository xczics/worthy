# -*- coding: utf-8 -*-
"""线性分类器的加载、预测与增量重训逻辑"""

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .config import Thresholds
from .store import SampleStore


def _fit_quietly(clf: LogisticRegression, X, y) -> LogisticRegression:
    # scikit-learn's lbfgs solver can emit spurious "overflow"/"divide by zero"
    # RuntimeWarnings from transient line-search overshoots on well-separated
    # data — the fit still converges correctly. Silencing keeps a normal
    # retrain() call from looking like something broke.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.*")
        clf.fit(X, y)
    return clf


class ClassifierManager:
    def __init__(self, thresholds: Thresholds, store: SampleStore, model_dir: str):
        self.th = thresholds
        self.store = store
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[LogisticRegression] = None
        self.version = 0
        self._load_latest_model()

    def _model_path(self, version: int) -> Path:
        return self.model_dir / f"clf_v{version}.joblib"

    def _load_latest_model(self):
        latest = self.store.get_latest_model_version()
        if latest > 0 and self._model_path(latest).exists():
            self.model = joblib.load(self._model_path(latest))
            self.version = latest
        else:
            self.model = None
            self.version = 0

    def has_model(self) -> bool:
        return self.model is not None

    def predict_proba(self, embedding: np.ndarray) -> float:
        """返回"值得入库"(正类=1)的概率"""
        assert self.model is not None
        p = self.model.predict_proba(embedding.reshape(1, -1))[0]
        classes = list(self.model.classes_)
        idx = classes.index(1) if 1 in classes else int(np.argmax(p))
        return float(p[idx])

    def is_uncertain(self, prob: float) -> bool:
        return self.th.uncertain_low <= prob <= self.th.uncertain_high

    def current_ambiguous_threshold(self) -> float:
        n_retrain = self.store.get_retrain_count()
        threshold = (self.th.ambiguous_threshold_start
                     - n_retrain * self.th.ambiguous_threshold_decay_per_retrain)
        return max(threshold, self.th.ambiguous_threshold_floor)

    def should_retrain(self) -> bool:
        new_samples = self.store.count_new_samples_since_last_train()

        if new_samples < self.th.min_new_samples_for_retrain:
            return False
        if new_samples >= self.th.max_new_samples_for_retrain:
            return True

        ambiguous_rate = self.store.get_recent_ambiguous_rate(self.th.ambiguous_rate_window)
        threshold = self.current_ambiguous_threshold()
        return ambiguous_rate > threshold

    def retrain(self) -> bool:
        """返回是否真的训练成功(数据不足/单一类别会跳过)"""
        X, y = self.store.get_training_data(self.th.max_training_samples)
        if X is None or len(np.unique(y)) < 2:
            return False

        val_acc = None
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            _fit_quietly(clf, X_tr, y_tr)
            val_acc = clf.score(X_val, y_val)
        except ValueError:
            # 数据量太小无法做分层切分时,直接用全量数据训练,不做验证集评估
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            _fit_quietly(clf, X, y)

        new_version = self.version + 1
        joblib.dump(clf, self._model_path(new_version))

        threshold_used = self.current_ambiguous_threshold()
        self.store.save_model_meta(
            version=new_version,
            n_samples=len(y),
            ambiguous_threshold_used=threshold_used,
            val_accuracy=val_acc,
        )
        self.store.mark_all_used_in_training()

        self.model = clf
        self.version = new_version
        return True
