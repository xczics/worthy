# -*- coding: utf-8 -*-
"""大模型调用封装:OpenAI 兼容接口 + <score> 格式解析 + 重试"""

import re
from typing import Optional

from openai import OpenAI

from .config import LLMConfig, ClassificationCriteria

_SCORE_PATTERN = re.compile(r"<score>\s*(-?\d+)\s*</score>", re.IGNORECASE)


class LLMScorer:
    def __init__(self, llm_config: LLMConfig, criteria: ClassificationCriteria):
        self.cfg = llm_config
        self.criteria = criteria
        self.client = OpenAI(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
            timeout=llm_config.timeout,
        )
        self._system_prompt = criteria.build_system_prompt()

    def _parse_score(self, content: str) -> Optional[int]:
        content = content.strip()
        m = _SCORE_PATTERN.search(content)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        # 兜底:模型偶尔只输出裸数字
        if 0 < len(content) <= 2 and content.lstrip("-").isdigit():
            try:
                return int(content)
            except ValueError:
                return None
        return None

    def _call_once(self, text: str) -> Optional[int]:
        user_prompt = (
            f"输入文本：\n{text}\n\n"
            f"再次强调，只需按判断标准打分，严格按照 \"<score>数字</score>\" 格式输出，"
            f"不要输出任何其他文本。"
        )
        kwargs = dict(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        if self.cfg.extra_body:
            kwargs["extra_body"] = self.cfg.extra_body

        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        return self._parse_score(content)

    def score(self, text: str) -> int:
        """
        对文本打分,返回 0-10 的整数。
        最多重试 max_retries 次,全部失败则抛出 ValueError。
        """
        last_raw_error = None
        for attempt in range(self.cfg.max_retries):
            try:
                score = self._call_once(text)
            except Exception as e:  # noqa: BLE001 - 需要兜底各种网络/接口异常并重试
                last_raw_error = e
                continue
            if score is not None:
                return max(0, min(10, score))
        raise ValueError(
            f"LLM 在 {self.cfg.max_retries} 次尝试后仍未输出有效分数"
            f"{f'，最后一次异常: {last_raw_error}' if last_raw_error else ''}"
        )
