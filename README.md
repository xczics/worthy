# Worthy · 掌眼

**中文** | [English](#english)

> "掌眼"是古玩行话:懂行的人帮你看一件东西是真是假、值不值得收。
> Worthy 做的是同一件事——只不过鉴定的对象是文本,标准由你说了算,值不值得放进你的知识库。

---

## 还在无脑 RAG 吗?

"把所有能切出来的文本块统统塞进向量库"是很多 RAG 项目的默认做法——看起来省事,实际上埋了好几个坑:

- **检索不准**:参考文献列表、页眉页脚、图表编号、HTML残留、乱码片段,这些噪声一样会被embedding进库。一旦它们恰好和某次查询的向量距离更近,就会顶掉真正相关的内容,拉低召回质量
- **上下文污染**:检索到的垃圾chunk被塞进大模型的context窗口,不仅浪费token,还会分散模型注意力,让最终回答变得答非所问、甚至引用错误信息
- **存储和算力被白白消耗**:每一条参考文献、每一段表格残留、每一个孤立URL,都要占一份embedding的存储、一次检索时的计算,却对最终答案毫无贡献
- **向量空间被稀释**:垃圾向量越多,最近邻检索的"信噪比"越低——同样召回10条,有效内容的占比会越来越小
- **出问题难排查**:等你发现回答质量下降,想回头定位"到底是哪些垃圾内容混进去了",往往要在几十万条向量里大海捞针

这些问题的根源其实都一样:**入库这道门槛,从来没人真正把过关。**

## Worthy 能帮你做什么

在你做 embedding 的时候,顺手挂上 Worthy——它会按你自己定义的标准(用自然语言描述即可)对每一段文本做一次"值不值得入库"的裁定,只放真正有价值的内容进知识库。

- **从源头保证检索质量**:垃圾从没进过库,检索时自然不会被垃圾干扰
- **不多花一分钱**:判断复用的正是你本来就要计算的embedding向量,不需要额外的模型调用
- **越用越便宜、越用越准**:接入时全靠大模型逐条把关,但系统会自动学习,训练出一个专属你语料的轻量判断模型,后续大部分case交给它秒级完成,只有真正拿不准的边界case才会麻烦大模型复核——语料库越大,系统越省、越准

## 它是怎么工作的

```
文本进来
   │
   ▼
[可选] 你写的零成本规则,能一眼认定的直接返回
   │
   ▼
专用小模型打分(如果还没训练出来,这一步跳过)
   │
   ├─ 有把握 ──► 直接给出裁定
   │
   └─ 拿不准 ──► 升级给大模型裁定,结果记入训练库
                      │
                      ▼
              (后台异步)攒够数据 + 拿不准的比例超过阈值
                      │
                      ▼
              自动重新训练专用小模型,不阻塞当前请求
```

也就是**大模型 + 专用小模型动态路由**:大模型判断力强但慢、贵、偶尔输出格式不稳;专用小模型是基于你的embedding训练出的轻量分类器,判断快(毫秒级)、免费(本地推理)、稳定(输出就是个概率,不存在解析失败)。Worthy 让两者各司其职——专用小模型扛下大多数一眼能判断的case,大模型只处理真正的边界情况,而且专用小模型会随着你的使用自动训练、自动迭代,不需要你手动标注一条数据。

每次调用返回:

```python
verdict.embedding              # 这段文本的 embedding 向量,可直接拿去入库,不用重复计算
verdict.should_index           # 值不值得入库
verdict.source                 # 这次是谁做的裁定: "rule" / "specialist"(专用小模型) / "generalist"(大模型)
```

## 安装

```bash
pip install worthy
```

想跟着 main 分支实时同步、不等正式版发布,可以直接从 GitHub 装:

```bash
pip install "git+https://github.com/xczics/worthy.git@main"

# 想同步到最新 push,重新装一次即可
pip install --upgrade --force-reinstall "git+https://github.com/xczics/worthy.git@main"
```

本地开发/改代码见「贡献」一节。

## 快速上手

```python
from worthy import Worthy, LLMConfig, EmbeddingConfig

w = Worthy(
    llm_config=LLMConfig(
        base_url="https://your-endpoint/v1",   # OpenAI 兼容接口,云平台/本地部署均可
        api_key="sk-xxxx",
        model="your-model-name",
    ),
    embedding_config=EmbeddingConfig(
        provider="openai_compatible",            # 或 "ollama_native",本地 Ollama 原生接口
        base_url="https://your-endpoint/v1",
        api_key="sk-xxxx",
        model="your-embedding-model",
    ),
    criteria="""判断文本是否为学术论文的正文段落。
如果是完整的叙述性句子，或图注/表标题，值得入库；
如果是参考文献、标题、[数字]、Fig、网址、乱码或无意义片段，不值得入库。""",
    db_path="paper_worthy.db",          # 标注样本存这里(SQLite),不填默认当前目录下 worthy.db
    model_dir="paper_worthy_models",    # 训练出的专用小模型存这里,不填默认当前目录下 worthy_models/
)

verdict = w.judge("这是一段正文...")
verdict.should_index   # True / False
verdict.embedding       # np.ndarray
```

`criteria` 直接用自然语言描述你的判断标准即可——不用写 prompt 工程,Worthy 会帮你拼装成完整、带格式约束的指令。

更完整的示例(含本地 Ollama 配置)见 `examples/quickstart.py`。

## 查看运行状态 / 手动触发重训

```python
w.status()          # 专用小模型版本、最近的"拿不准"比例、距下次自动重训还差多少数据……
w.force_retrain()   # 不等自动触发条件,立即同步重训一次
```

## 主要配置项

| 配置 | 作用 |
|---|---|
| `LLMConfig` | 大模型接口地址/密钥/模型名,走 OpenAI 兼容协议,覆盖云平台、vLLM、Ollama 兼容层 |
| `EmbeddingConfig` | Embedding 接口配置,支持 `openai_compatible` 和 `ollama_native` 两种 |
| `Thresholds` | 控制"何时算拿不准""何时自动重训"的一组阈值,均有默认值,可按需覆盖 |
| `criteria` | 用自然语言描述的判断标准,核心配置项 |

详细字段说明见 `worthy/config.py` 内的注释。

## 数据存到哪了

标注样本(文本、embedding、标签、来源)和历次训练出的模型版本,分别存在你指定的 `db_path`(SQLite文件)和 `model_dir`(模型文件目录)里,换语料库时指向新路径即可从零积累,不会互相污染。

## 贡献

```bash
git clone https://github.com/xczics/worthy.git
cd worthy
pip install -e ".[dev]"
pytest tests/
```

---

<a name="english"></a>
## English

[中文](#worthy--掌眼) | **English**

> Any antiques dealer worth their salt has someone they trust to vet a piece before it goes on the shelf — real or fake, worth having or not. Worthy is that person for your knowledge base: tell it what "worth keeping" means, and it renders a verdict on every chunk before it ever gets indexed.

---

### Still indexing everything and hoping for the best?

Chunk it, embed it, index it — that's the default RAG pipeline, and it's the easy part. It's also how most retrieval quality problems get baked in on day one:

- **Retrieval gets noisy** — reference lists, headers/footers, figure numbers, stray HTML, garbled OCR — all of it gets embedded right alongside your real content. When one of those happens to sit close to a query vector, it bumps out something that actually mattered
- **Context gets polluted** — junk chunks that make it into a retrieval call land straight in the LLM's context window. That's wasted tokens at best; at worst, the model anchors on the noise and the answer drifts off-topic or gets facts wrong
- **You're paying to store and search garbage** — every mangled table, every orphaned URL, every duplicate boilerplate line costs you an embedding, a slot in your vector store, and compute at query time — for a chunk that was never going to help anyone
- **Signal-to-noise keeps dropping** — the more junk piles up, the more of any top-10 result is dead weight, even though nothing about your retrieval logic changed
- **And when it breaks, good luck finding why** — by the time someone notices answers going sideways, "which chunk did this" means grepping through a vector store with hundreds of thousands of entries

It's all one root cause: **nobody ever put a gate in front of the index.**

### What Worthy actually does

Drop Worthy into your embedding step and every chunk gets a real verdict — worth indexing or not — checked against a standard you write in plain English, before it ever touches your vector store.

- **Fixes retrieval quality at the source** — junk that never gets indexed can't crowd out the good stuff later
- **Doesn't cost you anything extra to run** — it judges off the same embedding you were already computing for indexing. No second model call, no extra API bill
- **Gets cheaper and sharper the more you use it** — day one, every call goes to your large model. But Worthy is quietly learning in the background, training a lightweight specialist on your actual corpus. Within a few hundred calls, that specialist is handling the obvious cases in milliseconds, and the large model only sees the genuinely ambiguous ones. The bigger your corpus, the less you pay and the better it gets

### How it works

```
Text comes in
   │
   ▼
[Optional] your zero-cost rules — anything obvious is resolved here
   │
   ▼
Specialist scores it (skipped if no specialist has been trained yet)
   │
   ├─ Confident ──► verdict returned directly
   │
   └─ Unsure ──► escalated to the generalist, result logged for training
                      │
                      ▼
              (in background) enough new data + escalation rate over threshold
                      │
                      ▼
              specialist retrains automatically, without blocking the current request
```

In other words: **dynamic routing between a large model and a specialist you grow yourself.** The large model (the generalist) has good judgment but is slow, costs money per call, and occasionally hands back output that doesn't parse cleanly. The specialist is a lightweight classifier trained on embeddings you were already computing — millisecond inference, runs locally for free, and its output is just a probability, so there's nothing to fail to parse. Worthy routes to whichever one fits: the specialist takes the obvious majority of cases, the generalist only gets pulled in for genuine edge cases, and the specialist keeps retraining itself on what it learns from those escalations — no manual labeling, ever.

Every call returns:

```python
verdict.embedding              # the embedding for this text — reuse it for indexing, no need to recompute
verdict.should_index           # worth indexing or not
verdict.source                 # who made this call: "rule" / "specialist" / "generalist"
```

### Install

```bash
pip install worthy
```

Want to track `main` in real time instead of waiting for tagged releases? Install straight from GitHub:

```bash
pip install "git+https://github.com/xczics/worthy.git@main"

# to pick up whatever's newest on main, just reinstall
pip install --upgrade --force-reinstall "git+https://github.com/xczics/worthy.git@main"
```

See Contributing below for a local dev setup.

### Quickstart

```python
from worthy import Worthy, LLMConfig, EmbeddingConfig

w = Worthy(
    llm_config=LLMConfig(
        base_url="https://your-endpoint/v1",   # OpenAI-compatible, works with cloud or self-hosted
        api_key="sk-xxxx",
        model="your-model-name",
    ),
    embedding_config=EmbeddingConfig(
        provider="openai_compatible",            # or "ollama_native" for Ollama's native API
        base_url="https://your-endpoint/v1",
        api_key="sk-xxxx",
        model="your-embedding-model",
    ),
    criteria="""Judge whether the text is a body paragraph from an academic paper.
Full narrative sentences, figure captions, and table captions are worth indexing.
References, titles, bracketed numbers like [3], "Fig.", URLs, garbled text, or
meaningless fragments are not worth indexing.""",
    db_path="paper_worthy.db",          # labeled samples live here (SQLite) — defaults to ./worthy.db
    model_dir="paper_worthy_models",    # trained specialist versions live here — defaults to ./worthy_models/
)

verdict = w.judge("This is a body paragraph...")
verdict.should_index   # True / False
verdict.embedding       # np.ndarray
```

Just describe your criteria in plain language — no prompt engineering needed. Worthy assembles it into a complete, format-constrained instruction for you.

See `examples/quickstart.py` for a fuller example, including a local Ollama setup.

### Check status / trigger a retrain manually

```python
w.status()          # specialist version, recent escalation rate, how much data until next auto-retrain...
w.force_retrain()   # retrain immediately, without waiting for the automatic trigger
```

### Key config objects

| Config | Purpose |
|---|---|
| `LLMConfig` | Generalist endpoint / key / model name, OpenAI-compatible, works with cloud platforms, vLLM, Ollama's compatible layer |
| `EmbeddingConfig` | Embedding endpoint config, supports `openai_compatible` and `ollama_native` |
| `Thresholds` | The set of thresholds controlling "when is the specialist unsure" and "when to auto-retrain," all with sane defaults |
| `criteria` | Your judgment standard, written in plain language — the main thing you actually configure |

See the comments in `worthy/config.py` for full field details.

### Where your data lives

Labeled samples (text, embedding, label, source) and every trained model version are stored under the `db_path` (a SQLite file) and `model_dir` (a folder of model files) you provide. Point to a new path when you switch corpora, and it starts fresh without mixing with old data.

### Contributing

```bash
git clone https://github.com/xczics/worthy.git
cd worthy
pip install -e ".[dev]"
pytest tests/
```
