---
title: "DeepSeek 系列 LLM 架构（一）：从 Dense 到 MLA 与 MoE"
date: 2026-10-10
permalink: /posts/2026/10/deepseek-llm-architecture-01/
project: deepseek-study
series: deepseek-study
excerpt: "梳理 DeepSeek LLM 7B/67B 与 DeepSeek-V2 的架构演进，理解 MHA、GQA、MLA 的 KV cache 组织，以及细粒度专家与共享专家组成的 DeepSeekMoE。"
tags:
  - DeepSeek
  - 大模型
  - Attention
  - MoE
  - 学习笔记
---

{% include toc icon="list" title="目录" position="inline" %}

本文记录 DeepSeek 早期模型的架构演进：先以 DeepSeek LLM 7B/67B 的 Dense Transformer 为起点，再看 DeepSeek-V2 如何通过 **MLA 压缩 KV cache**，通过 **DeepSeekMoE 降低每个 token 的计算开销**。

系列入口：[DeepSeek 系列文章集]({{ '/portfolio/deepseek-study/' | relative_url }})。

## DeepSeek LLM 7B/67B

[论文：DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954)

这是 DeepSeek 早期的通用语言模型系列，采用 Dense 架构，沿用 LLaMA 风格的 Decoder-only Transformer。这里的 Dense 指每个 token 都经过各层完整的 FFN，与后面的稀疏 MoE 相对应。

![DeepSeek LLM 67B 与 LLaMA 的 Transformer 架构比较]({{ "/images/42.png" | relative_url }})
<p class="image-caption">图 1：DeepSeek LLM 67B 与 LLaMA 的 Transformer 架构比较</p>

### 模型规模

7B/67B 的主要配置如下：

|                       | DeepSeek 7B | DeepSeek 67B |
| --------------------- | ----------- | ------------ |
| Transformer layers    | **30**      | **95**       |
| hidden size `d_model` | 4096        | 8192         |
| Attention Head 数量   | 32          | 64           |
| KV heads              | 32          | **8**        |
| Context length        | 4096        | 4096         |
| Training tokens       | 2T          | 2T           |

### Transformer 模块

两个模型的主要差别之一是注意力头的组织方式：**7B 使用 MHA，67B 使用 GQA**。67B 的 64 个 Query heads 共享 8 个 KV heads，即每 8 个 Query heads 共用一组 K、V。

其余主要模块包括：

- **RoPE**：为 Q、K 引入旋转位置编码。
- **RMSNorm / Pre-Norm**：在注意力和 FFN 子层之前进行归一化。
- **SwiGLU FFN**：通过带门控的前馈网络进行非线性变换。

## DeepSeek-V2

[论文：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)

[前置研究：DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)

相比前面的 Dense 模型，DeepSeek-V2 的两项主要架构变化是 **MLA（Multi-head Latent Attention）** 和 **DeepSeekMoE**。前者优化注意力中的 KV cache，后者用稀疏激活的专家替换大部分 Dense FFN。DeepSeekMoE 论文介绍了专家设计思路，V2 论文则给出 MLA 与 MoE 组合后的完整模型。

![DeepSeek-V2 的 MLA 与 MoE 架构]({{ "/images/43.png" | relative_url }})
<p class="image-caption">图 2：DeepSeek-V2 的 MLA 与 MoE 架构</p>

### MLA 注意力

#### MHA、MQA 与 GQA

在传统注意力层中，输入隐状态 X 分别投影得到 Q、K、V，再完成缩放点积注意力。自回归推理会缓存历史 token 的 K、V，避免每生成一个 token 都重新计算整个前缀。

MHA、MQA 和 GQA 的主要区别是 **Query heads 如何共享 KV heads**：

![传统注意力在 Prefill 阶段的 Q、K、V 投影与缓存数据流]({{ "/images/44.png" | relative_url }})
<p class="image-caption">图 3：传统注意力在 Prefill 阶段的 Q、K、V 投影与缓存数据流</p>

| 机制 | Query heads 与 KV heads 的关系 | KV cache 特点 |
| --- | --- | --- |
| MHA（Multi-head Attention） | 每个 Query head 对应独立的 K、V head | KV heads 最多，缓存开销较大 |
| MQA（Multi-query Attention） | 所有 Query heads 共享一组 K、V | KV heads 最少，缓存开销较小 |
| GQA（Grouped-query Attention） | 一组 Query heads 共享一组 K、V | 在 MHA 与 MQA 之间折中 |

![MHA、MQA 与 GQA 的 Query heads 和 KV heads 共享关系]({{ "/images/45.png" | relative_url }})
<p class="image-caption">图 4：MHA、MQA 与 GQA 的 Query heads 和 KV heads 共享关系</p>

#### 低秩压缩 Q、K、V

MLA 的目标是在显著压缩 KV cache 的同时保持较好的模型能力。它不只减少 KV heads 的数量，而是将 K、V 联合压缩到一个低维潜在向量中。

![MLA 的低秩投影与解耦 RoPE 数据流（简化示意，具体维度以正文配置表为准）]({{ "/images/46.png" | relative_url }})
<p class="image-caption">图 5：MLA 的低秩投影与解耦 RoPE 数据流（简化示意，具体维度以正文配置表为准）</p>

下面按单个 token 描述数据流，省略 batch 和序列维度：

```text
c_KV = W_DKV · X
K_C  = W_UK  · c_KV
V    = W_UV  · c_KV

c_Q  = W_DQ  · X
Q_C  = W_UQ  · c_Q
```

- **K、V 分支**：先得到联合压缩的 `c_KV`，再分别上投影得到内容 Key `K_C` 和 Value `V`。
- **Q 分支**：先得到压缩的 `c_Q`，再上投影得到内容 Query `Q_C`。Q 的压缩也有助于减少训练时的激活开销，但 `c_Q` 不属于历史 KV cache。

#### 解耦 RoPE

如果直接对内容 Key 应用 RoPE，随位置变化的旋转会妨碍将 Key 的上投影矩阵吸收到 Query 侧的计算中。因此 MLA 将位置编码分支与内容分支分开：

```text
K_R = RoPE(W_KR · X)
Q_R = RoPE(W_QR · c_Q)

K = concat(K_C, K_R)
Q = concat(Q_C, Q_R)
```

**`K_R` 在各个注意力头之间共享，`Q_R` 则为各个 Query head 分别生成。** 对 DeepSeek-V2，每个 token 的 `K_R` 维度是 64；128 个 Query heads 对应的 `Q_R` 形状为 `[128, 64]`。这里省略了 batch 和序列维度。

最终使用组合后的 Q、K 和内容分支 V 计算多头注意力。上述展开形式便于理解；高效推理实现会利用矩阵吸收等方式，避免把历史 token 的完整 K、V 都重新展开并缓存。

#### KV cache 实际存什么

MLA 每层、每个历史 token 主要缓存两部分：

| 缓存项 | DeepSeek-V2 维度 | 用途 |
| --- | --- | --- |
| `c_KV` | 512 | 压缩后的 K、V 内容信息 |
| `K_R` | 64 | 各个注意力头共享的位置 Key |
| 合计 | 576 | 历史 token 的缓存元素数量 |

因此，MLA 的 KV cache 包括 **压缩内容向量和位置 Key**，不能只按 `c_KV` 的 512 维计算。这个元素数量还需乘以层数、序列长度、batch size 和每个元素的字节数，才能估算实际显存占用。

### DeepSeekMoE

DeepSeek-V2 将大部分 Transformer 层的 FFN 替换为 DeepSeekMoE。其专家分为两类：

- **Routed Experts（路由专家）**：将专家划分得更细，每个 token 根据路由结果选择其中一部分，有利于形成更细致的专业分工。
- **Shared Experts（共享专家）**：每个 token 都经过共享专家，用于学习不同 token 共同需要的知识，减少路由专家之间的重复学习。

![DeepSeekMoE 的路由专家与共享专家组织]({{ "/images/47.png" | relative_url }})
<p class="image-caption">图 6：DeepSeekMoE 的路由专家与共享专家组织</p>

DeepSeek-V2 的每个 MoE 层包含 **160 个路由专家和 2 个共享专家**，每个 token 激活 **6 个路由专家，并经过 2 个共享专家**。因此，模型的总参数量与每个 token 实际激活的参数量不同。

### 模型配置

下面列出完整 DeepSeek-V2 的主要配置，不包含 V2-Lite：

| 参数                          | DeepSeek-V2 配置 |
| ----------------------------- | ---------------- |
| Transformer 总层数            | 60               |
| 隐藏维度 `d_model`            | 5120             |
| Attention Head 数量           | 128              |
| Head的内容维度                | 128              |
| Head的RoPE维度                | 64               |
| Head的运算时的维度            | 128+64           |
| KVcache内容维度（c_KV）       | 512              |
| KV cache 的 RoPE Key 维度（`K_R`） | 64          |
| Q的压缩维度（c_Q）            | 1536             |
| MoE 层数                      | 59（首层为 Dense FFN） |
| 每层共享专家数量              | 2                |
| 每层routed专家数量            | 160              |
| 每个token选择的routed专家数量 | 6                |
| 每个专家的中间维度            | 1536             |

## 架构演进小结

| 模型 | 注意力机制 | FFN 组织 | 主要关注点 |
| --- | --- | --- | --- |
| DeepSeek LLM 7B | MHA | Dense FFN | 建立 Dense Transformer 基线 |
| DeepSeek LLM 67B | GQA | Dense FFN | 减少 KV heads，降低缓存开销 |
| DeepSeek-V2 | MLA | 首层 Dense FFN，其余为 DeepSeekMoE | 压缩 KV cache，并减少每个 token 激活的专家参数 |

从硬件和推理实现角度看，MLA 主要改变 **缓存的数据形式与注意力计算的数据流**；MoE 则改变 **FFN 的参数访问、路由和计算分配**。后续阅读可以分别沿着 KV cache 带宽、矩阵计算和专家通信这几条线继续展开。

## 参考资料

1. [DeepSeek LLM](https://arxiv.org/abs/2401.02954)
2. [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
3. [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
4. [DeepSeek-V2 官方代码仓库](https://github.com/deepseek-ai/DeepSeek-V2)

