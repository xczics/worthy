# -*- coding: utf-8 -*-
"""Embedding 调用封装:支持 OpenAI 兼容接口 与 Ollama 原生接口"""

import numpy as np
import requests
from openai import OpenAI

from .config import EmbeddingConfig


class EmbeddingClient:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        if cfg.provider == "openai_compatible":
            self._client = OpenAI(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                timeout=cfg.timeout,
            )
        elif cfg.provider == "ollama_native":
            self._client = None  # 用 requests 直接调用
        else:
            raise ValueError(f"不支持的 embedding provider: {cfg.provider}")

    def embed(self, text: str) -> np.ndarray:
        if self.cfg.provider == "openai_compatible":
            return self._embed_openai_compatible(text)
        else:
            return self._embed_ollama_native(text)

    def _embed_openai_compatible(self, text: str) -> np.ndarray:
        kwargs = dict(model=self.cfg.model, input=text)
        if self.cfg.dimensions:
            kwargs["dimensions"] = self.cfg.dimensions
        resp = self._client.embeddings.create(**kwargs)
        vec = resp.data[0].embedding
        return np.array(vec, dtype=np.float32)

    def _embed_ollama_native(self, text: str) -> np.ndarray:
        url = self.cfg.base_url.rstrip("/") + "/api/embeddings"
        resp = requests.post(
            url,
            json={"model": self.cfg.model, "prompt": text},
            timeout=self.cfg.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        vec = data.get("embedding")
        if vec is None:
            raise ValueError(f"Ollama embedding 接口返回格式异常: {data}")
        return np.array(vec, dtype=np.float32)
