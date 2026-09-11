# -*- coding: utf-8 -*-
"""
快速上手示例

场景一:云平台 OpenAI 兼容接口(大模型 + Embedding 都走同一类接口)
场景二:本地 Ollama(大模型走 OpenAI 兼容层,Embedding 走 Ollama 原生接口)
"""

from worthy import Worthy, LLMConfig, EmbeddingConfig, Thresholds


# ------------------------------------------------------------------
# 场景一:云平台(以 OpenAI 兼容接口为例,替换成你自己的云平台 base_url/key/model)
# ------------------------------------------------------------------
def build_cloud_worthy() -> Worthy:
    llm_config = LLMConfig(
        base_url="https://your-cloud-platform.example.com/v1",
        api_key="sk-xxxxxxxx",
        model="your-model-name",
        temperature=0.05,
        max_tokens=50,
        # 如果你的平台需要类似 reasoning/thinking 关闭字段,原样透传:
        extra_body={"reasoning": None, "reasoning_effort": "low"},
    )

    embedding_config = EmbeddingConfig(
        provider="openai_compatible",
        base_url="https://your-cloud-platform.example.com/v1",
        api_key="sk-xxxxxxxx",
        model="your-embedding-model-name",
    )

    criteria = """判断文本是否为学术论文的正文段落。
规则：
    - 如果是完整的叙述性句子（像在讲故事或讲原理），值得入库。
    - 图注和表标题也识别为正文段落，值得入库。
    - 如果是参考文献、标题、[数字]、Fig、网址、乱码或无意义片段，不值得入库。
"""

    # 可选:零成本的确定性规则,先于模型判断
    def rule_filter(text: str):
        if len(text) > 100 and (text.count("<td>") > 8 or text.count("<tr>") > 10):
            return False
        if "![](images/" in text:
            return False
        return None  # 交给 专用小模型/大模型 继续判断

    return Worthy(
        llm_config=llm_config,
        embedding_config=embedding_config,
        criteria=criteria,
        thresholds=Thresholds(
            min_new_samples_for_retrain=500,
            max_new_samples_for_retrain=1000,
            ambiguous_threshold_start=0.10,
            ambiguous_threshold_floor=0.03,
        ),
        db_path="paper_worthy.db",
        model_dir="paper_worthy_models",
        rule_filter_fn=rule_filter,
    )


# ------------------------------------------------------------------
# 场景二:本地 Ollama(大模型用 Ollama 的 OpenAI 兼容层,Embedding 用原生接口)
# ------------------------------------------------------------------
def build_local_ollama_worthy() -> Worthy:
    llm_config = LLMConfig(
        base_url="http://localhost:11434/v1",   # Ollama 的 OpenAI 兼容端点
        api_key="ollama",                        # Ollama 不校验,随便填非空字符串
        model="qwen2.5:3b",
    )

    embedding_config = EmbeddingConfig(
        provider="ollama_native",
        base_url="http://localhost:11434",       # 注意没有 /v1,原生接口在 /api/embeddings
        model="bge-m3",
    )

    criteria = "判断文本是否为学术论文的正文段落，是则值得入库，否则不值得入库。"

    return Worthy(
        llm_config=llm_config,
        embedding_config=embedding_config,
        criteria=criteria,
        db_path="local_worthy.db",
        model_dir="local_worthy_models",
    )


if __name__ == "__main__":
    w = build_cloud_worthy()

    text = "在本实验中，我们首先制备了样品并在低温环境下进行了光谱测量，结果表明该反应路径与理论预测一致。"
    verdict = w.judge(text)

    print("值不值得入库:", verdict.should_index)
    print("这次是谁裁定的:", verdict.source)   # rule / specialist / generalist
    print("embedding 维度:", verdict.embedding.shape)
    print("当前状态:", w.status())
