# -*- coding: utf-8 -*-
"""用 mock 替代真实网络调用,验证 Worthy 全流程能跑通"""

import random
import time
from unittest.mock import MagicMock, patch

import numpy as np

from worthy import Worthy, LLMConfig, EmbeddingConfig, Thresholds


def fake_embed(text: str) -> np.ndarray:
    is_ref = ("[" in text and "]" in text and len(text) < 50)
    base = np.array([1.0, 0.0] if not is_ref else [0.0, 1.0], dtype=np.float32)
    noise = np.random.RandomState(abs(hash(text)) % (2**32)).normal(0, 0.1, size=2).astype(np.float32)
    vec = base + noise
    pad = np.random.RandomState(abs(hash(text)) % (2**32) + 1).normal(0, 0.05, size=14).astype(np.float32)
    return np.concatenate([vec, pad])


def fake_llm_score(text: str) -> int:
    is_ref = ("[" in text and "]" in text and len(text) < 50)
    return 0 if is_ref else 10


class FakeEmbeddingResp:
    def __init__(self, vec):
        self.data = [MagicMock(embedding=vec.tolist())]


class FakeChatChoice:
    def __init__(self, content):
        self.message = MagicMock(content=content)


class FakeChatResp:
    def __init__(self, content):
        self.choices = [FakeChatChoice(content)]


def run_smoke_test():
    with patch("worthy.embedding_client.OpenAI") as MockEmbedOpenAI, \
         patch("worthy.llm_client.OpenAI") as MockLLMOpenAI:

        embed_instance = MockEmbedOpenAI.return_value
        embed_instance.embeddings.create.side_effect = (
            lambda model, input, **kw: FakeEmbeddingResp(fake_embed(input))
        )

        llm_instance = MockLLMOpenAI.return_value
        def fake_chat_create(model, messages, **kw):
            user_msg = messages[-1]["content"]
            text = user_msg.split("输入文本：\n", 1)[1].split("\n\n再次强调")[0]
            score = fake_llm_score(text)
            return FakeChatResp(f"<score>{score}</score>")
        llm_instance.chat.completions.create.side_effect = fake_chat_create

        w = Worthy(
            llm_config=LLMConfig(base_url="http://fake", api_key="fake", model="fake-model"),
            embedding_config=EmbeddingConfig(base_url="http://fake", api_key="fake", model="fake-embed"),
            criteria="判断文本是否为学术论文的正文段落，是则值得入库，否则不值得入库。",
            thresholds=Thresholds(
                min_new_samples_for_retrain=40,
                max_new_samples_for_retrain=80,
            ),
            db_path="/home/claude/worthy/tests/_smoke.db",
            model_dir="/home/claude/worthy/tests/_smoke_models",
        )

        long_texts = [f"这是第{i}段完整的叙述性正文,内容描述了实验过程与结论分析。" * 2 for i in range(60)]
        ref_texts = [f"[{i}] Smith et al., J. Astrochem., {2000+i}" for i in range(60)]
        all_texts = long_texts + ref_texts
        random.shuffle(all_texts)

        sources_seen = set()
        for t in all_texts:
            v = w.judge(t)
            sources_seen.add(v.source)
            assert isinstance(v.should_index, bool)
            assert v.embedding.shape[0] == 16

        time.sleep(1.5)  # 等后台重训线程

        status = w.status()
        print("judge 来源分布:", sources_seen)
        print("状态:", status)
        assert status["specialist_ready"] is True, "重训应已触发,专用小模型应已生成"

        post_sources = set()
        for t in random.sample(all_texts, 20):
            v = w.judge(t)
            post_sources.add(v.source)
        print("重训后 judge 来源分布:", post_sources)
        assert "specialist" in post_sources, "重训后应有请求命中专用小模型"

        print("\n✅ 冒烟测试通过")


if __name__ == "__main__":
    run_smoke_test()
