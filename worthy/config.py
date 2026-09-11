# -*- coding: utf-8 -*-
"""配置类定义:大模型、Embedding、阈值、分类标准"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any


@dataclass
class LLMConfig:
    """
    大模型调用配置,走 OpenAI 兼容接口(适用于 OpenAI 官方、
    大部分云平台的兼容端点、vLLM、以及 Ollama 的 OpenAI 兼容层 /v1)。
    """
    base_url: str = "https://api.openai.com/v1"
    api_key: str = "EMPTY"          # 本地 vLLM/Ollama 通常随便填一个非空字符串即可
    model: str = "gpt-4o-mini"
    temperature: float = 0.05
    max_tokens: int = 50
    timeout: float = 30.0
    # 有些提供商(如支持 reasoning/thinking 开关的模型服务)需要额外字段,
    # 原样透传到 chat.completions.create(extra_body=...)
    extra_body: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 5


@dataclass
class EmbeddingConfig:
    """
    Embedding 调用配置。
    provider="openai_compatible": 走 OpenAI 兼容的 /v1/embeddings 接口
        (OpenAI 官方、大部分云平台、Ollama 新版的 OpenAI 兼容层均可用这个)
    provider="ollama_native": 走 Ollama 原生 /api/embeddings 接口
    """
    provider: Literal["openai_compatible", "ollama_native"] = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = "EMPTY"
    model: str = "text-embedding-3-small"
    timeout: float = 30.0
    dimensions: Optional[int] = None  # 部分 OpenAI 兼容模型支持指定输出维度


@dataclass
class Thresholds:
    """分类器行为与自学习触发相关的阈值"""

    # 不确定区间:线性分类器输出概率落在这个区间内,视为"拿不准",路由回大模型
    uncertain_low: float = 0.35
    uncertain_high: float = 0.65

    # 触发重训的样本数量条件
    min_new_samples_for_retrain: int = 500   # 少于这个数,不训练(防止小样本过拟合)
    max_new_samples_for_retrain: int = 1000  # 达到这个数,不管模糊率多少,强制重训一次

    # 模糊率阈值:随重训次数递减,下限为 ambiguous_threshold_floor
    ambiguous_threshold_start: float = 0.10
    ambiguous_threshold_floor: float = 0.03
    ambiguous_threshold_decay_per_retrain: float = 0.01

    # 近期模糊率的统计窗口(最近多少次调用)
    ambiguous_rate_window: int = 1000

    # 训练时最多用多少条历史数据(None = 全部)
    max_training_samples: Optional[int] = 20000

    # LLM 打分(0-10)转二值标签的分界:score >= 这个值记为 1(值得入库)
    score_binarize_threshold: int = 8


DEFAULT_PROMPT_TEMPLATE = """任务：根据下方【判断标准】，判断输入文本是否值得被索引入库。

【判断标准】
{criteria}

【打分规则】
- 完全符合判断标准 → 10
- 完全不符合判断标准 → 0
- 如果判断标准本身是非此即彼的二元标准，只输出 0 或 10；只有标准描述了"程度"时才使用 1-9 之间的分数

【输出格式】
<score>0-10之间的整数</score>

【输出示例】
<score>0</score>
<score>10</score>

注意：只输出上述格式，不要输出任何解释、前后缀或其他文本。
"""


@dataclass
class ClassificationCriteria:
    """
    用户用自然语言描述的分类标准。
    例如:
        "判断文本是否为学术论文的正文段落。规则：
         - 如果是完整的叙述性句子（像在讲故事或讲原理），判定为值得入库。
         - 图注和表标题也算作正文段落，值得入库。
         - 如果是参考文献、标题、[数字]、Fig、网址、乱码或无意义片段，不值得入库。"
    """
    criteria: str
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE

    def build_system_prompt(self) -> str:
        return self.prompt_template.format(criteria=self.criteria.strip())
