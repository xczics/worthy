# -*- coding: utf-8 -*-
"""Worthy(掌眼) 主入口:输入文本 -> 输出 embedding + 值不值得入库的裁定"""

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .config import LLMConfig, EmbeddingConfig, Thresholds, ClassificationCriteria
from .llm_client import LLMScorer
from .embedding_client import EmbeddingClient
from .store import SampleStore
from .classifier import ClassifierManager


@dataclass
class Verdict:
    """一次"鉴定"的完整结果"""
    text: str
    embedding: np.ndarray
    should_index: bool                # 最终裁定:值不值得入库
    source: str                       # "rule" / "specialist"(专用小模型) / "general_llm"(大模型)
    general_llm_score: Optional[int] = None          # 若由大模型裁定,原始 0-10 分数
    specialist_confidence: Optional[float] = None   # 若由专用小模型裁定,置信概率
    escalated: bool = False           # 专用小模型是否因拿不准而上报给大模型复核
    specialist_version: int = 0       # 当前专用小模型的版本号


class Worthy:
    """
    用法:
        from worthy import Worthy, LLMConfig, EmbeddingConfig

        w = Worthy(
            llm_config=LLMConfig(base_url=..., api_key=..., model=...),
            embedding_config=EmbeddingConfig(base_url=..., api_key=..., model=...),
            criteria="判断文本是否为学术论文的正文段落。是则值得入库；"
                     "参考文献、标题、[数字]、Fig、网址、乱码或无意义片段，不值得入库。",
        )

        verdict = w.judge("这是一段正文...")
        verdict.embedding      # np.ndarray
        verdict.should_index   # bool
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        criteria: str,
        thresholds: Thresholds = None,
        db_path: str = "worthy.db",
        model_dir: str = "worthy_models",
        rule_filter_fn: Optional[Callable[[str], Optional[bool]]] = None,
        prompt_template: Optional[str] = None,
    ):
        self.thresholds = thresholds or Thresholds()

        crit_kwargs = {"criteria": criteria}
        if prompt_template:
            crit_kwargs["prompt_template"] = prompt_template
        self.criteria = ClassificationCriteria(**crit_kwargs)

        self.embedder = EmbeddingClient(embedding_config)
        self.general_llm = LLMScorer(llm_config, self.criteria)

        self.store = SampleStore(db_path)
        self.specialist = ClassifierManager(self.thresholds, self.store, model_dir)

        self.rule_filter_fn = rule_filter_fn  # text -> True/False/None(不适用规则)

        self._retrain_lock = threading.Lock()
        self._retrain_in_progress = False

    def judge(self, text: str) -> Verdict:
        """对一段文本做出"值不值得入库"的裁定"""

        # 1. 可选的规则前置过滤(零成本,确定性 case 直接返回;仍会计算 embedding 便于后续入库)
        if self.rule_filter_fn is not None:
            rule_result = self.rule_filter_fn(text)
            if rule_result is not None:
                embedding = self.embedder.embed(text)
                return Verdict(
                    text=text, embedding=embedding, should_index=bool(rule_result),
                    source="rule",
                )

        embedding = self.embedder.embed(text)

        # 2. 冷启动:还没有专用小模型时,全部交给大模型裁定
        if not self.specialist.has_model():
            return self._judge_via_general_llm(text, embedding, escalated=False)

        # 3. 专用小模型打分
        prob = self.specialist.predict_proba(embedding)
        uncertain = self.specialist.is_uncertain(prob)
        self.store.log_routing(was_uncertain=uncertain)

        if not uncertain:
            return Verdict(
                text=text, embedding=embedding,
                should_index=(prob >= 0.5),
                source="specialist",
                specialist_confidence=prob,
                specialist_version=self.specialist.version,
            )

        # 4. 专用小模型拿不准,升级(escalate)给大模型复核,结果追加进训练库
        return self._judge_via_general_llm(text, embedding, escalated=True)

    def _judge_via_general_llm(self, text: str, embedding: np.ndarray,
                                escalated: bool) -> Verdict:
        raw_score = self.general_llm.score(text)
        label = 1 if raw_score >= self.thresholds.score_binarize_threshold else 0
        self.store.add_sample(
            text, embedding, label, source="llm",
            raw_score=raw_score, was_uncertain=escalated,
        )
        self._maybe_trigger_retrain()
        return Verdict(
            text=text, embedding=embedding,
            should_index=bool(label),
            source="general_llm",
            general_llm_score=raw_score,
            escalated=escalated,
            specialist_version=self.specialist.version,
        )

    def _maybe_trigger_retrain(self):
        if self._retrain_in_progress:
            return
        if not self.specialist.should_retrain():
            return

        with self._retrain_lock:
            if self._retrain_in_progress:
                return
            self._retrain_in_progress = True

        def _do_retrain():
            try:
                self.specialist.retrain()
            finally:
                self._retrain_in_progress = False

        threading.Thread(target=_do_retrain, daemon=True).start()

    def force_retrain(self) -> bool:
        """手动立即触发一次专用小模型重训(同步阻塞),返回是否训练成功"""
        return self.specialist.retrain()

    def status(self) -> dict:
        return {
            "specialist_version": self.specialist.version,
            "specialist_ready": self.specialist.has_model(),
            "new_samples_since_last_train": self.store.count_new_samples_since_last_train(),
            "recent_escalation_rate": self.store.get_recent_ambiguous_rate(
                self.thresholds.ambiguous_rate_window
            ),
            "current_escalation_trigger_threshold": self.specialist.current_ambiguous_threshold(),
            "retrain_count": self.store.get_retrain_count(),
        }
