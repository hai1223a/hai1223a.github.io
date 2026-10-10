---
title: "DeepSeek系列LLM架构（一）"
date: 2026-10-10
permalink: /posts/2026/10/deepseek-llm-architecture-01/
project: deepseek-study
series: deepseek-study
excerpt: "DeepSeek LLM 7B/67B、DeepSeek V2架构学习笔记。"
tags:
  - DeepSeek
  - 大模型
  - Attention
  - MoE
  - 学习笔记
---

{% include toc icon="list" title="目录" position="inline" %}

## DeepSeek LLM 7B/67B

[[论文地址] DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954)

这是DeepSeek首个LLM系列，Dense架构，继承自LLAMA经典架构。下图展示了67B的架构与LLAMA经典架构比较：

<img src="{{ '/images/42.png' | relative_url }}" alt="DeepSeek LLM 67B 与 LLaMA 架构比较" style="width: 500px; max-width: 100%;" />
<p class="image-caption">图 1：DeepSeek LLM 67B 与 LLaMA 架构比较</p>

其中7B/67B的规模数据如下：

|                       | DeepSeek 7B | DeepSeek 67B |
| --------------------- | ----------- | ------------ |
| Transformer layers    | **30**      | **95**       |
| hidden size `d_model` | 4096        | 8192         |
| Attention Head 数量   | 32          | 64           |
| KV heads              | 32          | **8**        |
| Context length        | 4096        | 4096         |
| Training tokens       | 2T          | 2T           |

7B模式使用MHA注意力机制，67B使用GQA注意力机制。使用RoPE对Q，K进行旋转位置编码后参与运算。使用RMSNorm进行Pre-Norm归一化操作。使用SwiGLU激活的FFN层注入非线性信息。

## DeepSeek V2

[[论文地址] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)

Deepseek第二代模型，相比第一代模型采用了MoE架构，并设计了MLA注意力机制大幅压缩KV cache。

<img src="{{ '/images/43.png' | relative_url }}" alt="DeepSeek V2 模型架构" style="width: 280px; max-width: 100%;" />
<p class="image-caption">图 2：DeepSeek V2 模型架构</p>

### MLA 注意力

在attention层中，由输入向量X得到Q，K，V三个矩阵，并完成对应的计算。其中针对QKV矩阵的不同处理，衍生出了MHA，MQA和GQA等不同的处理机制。

<img src="{{ '/images/44.png' | relative_url }}" alt="Prefill 阶段的 Q、K、V 投影与缓存数据流" style="max-width: 100%;" />
<p class="image-caption">图 3：Prefill 阶段的 Q、K、V 投影与缓存数据流</p>

* MHA表示将K，V，Q切分为相同数量，相同大小的组，然后一个K，V，Q完成运算，最后再将各组结果拼接得到结果；
* MQA表示将K，V尺寸缩小，将Q切分为和K，V相同大小的若干组，让唯一一组K，V与所有组的Q完成运算，最后将结果拼接；
* GQA表示K，V尺寸比Q较小，将K，V，Q切分为不同数量，相同大小的组。其中K，V组数相同，由一组K，V和若干组Q完成运算，最后将结果拼接。

<img src="{{ '/images/45.png' | relative_url }}" alt="MHA、MQA 和 GQA 的 KV head 组织" style="max-width: 100%;" />
<p class="image-caption">图 4：MHA、MQA 和 GQA 的 KV head 组织</p>

本文提出的MLA目标是同时达到MHA的推理效果，和GQA，MQA的KV cache压缩效果。其改变了attention的产生机制，具体如图：

<img src="{{ '/images/46.png' | relative_url }}" alt="MLA 注意力机制" style="max-width: 100%;" />
<p class="image-caption">图 5：MLA 注意力机制</p>

* K，V矩阵。MLA通过`W_DKV`权重生成压缩后的`c_KV`矩阵，然后通过`W_UV`权重生成解压后的`V`矩阵，通过`W_UK`权重生成解压后的`K_C`矩阵。

  Q矩阵。MLA通过`W_DQ`权重生成压缩后的`c_Q`矩阵，然后由`W_UQ`权重生成解压后的`Q_C`矩阵。

  RoPE编码。

  - 在原本设计中需要对`Q`，`K`进行RoPE编码添加位置信息后参与attention运算，但是MLA改变了attention的计算范式，KVcache缓存`c_KV`而不是原始的`K`、`V`矩阵，且旋转位置信息并不容易被`W_DKV`这样的低秩权重表达，因此设计了专门的应对机制。
  - 由`W_KR`权重生成压缩后的`k_R`矩阵，对`k_R`应用RoPE添加位置信息得到`K_R`，将`K_R`和`K_C`进行拼接后得到完整的`K`矩阵。其中`K_R`由`K_C`拆分后的多个head共享，故其尺寸只有`[1,32]`。
  - 由`W_QR`权重生成压缩后的`q_R`矩阵，对`q_R`应用RoPE添加位置信息得到`Q_R`，将`Q_R`和`Q_C`进行拼接后得到完整的`Q`矩阵。其中`Q_R`和`K_C`对应的多个head都不同，故其尺寸是`[1,128]`。

  最终由完整的`K`，`V`，`Q`完成MHA运算。

### DeepSeek MoE

DeepSeek V2将FFN替换成了DeepSeek MoE。该MoE由细粒度的Routed Experts，和Shared Experts组成，具体配置和结构如图。

<img src="{{ '/images/47.png' | relative_url }}" alt="DeepSeek MoE 路由专家与共享专家" style="max-width: 100%;" />
<p class="image-caption">图 6：DeepSeek MoE 路由专家与共享专家</p>

DeepSeek V2的数据规模如下：

| 参数                          | DeepSeek-V2 配置    |
| ----------------------------- | ------------------- |
| Transformer 总层数            | 60（前一层为FFN层） |
| Hidden Dimension              | 5120                |
| Attention Head 数量           | 128                 |
| Head的内容维度                | 128                 |
| Head的RoPE维度                | 64                  |
| Head的运算时的维度            | 128+64              |
| KVcache内容维度（c_KV）       | 512                 |
| KVcache RoPE维度（q_K）       | 64                  |
| Q的压缩维度（c_Q）            | 1536                |
| MoE层数                       | 59                  |
| 每层共享专家数量              | 2                   |
| 每层routed专家数量            | 160                 |
| 每个token选择的routed专家数量 | 6                   |
| 每个专家的中间维度            | 1536                |
| 模型总参数量                  | 236B                |
| 每个 Token 激活的参数量       | 21B                 |
| 最大上下文                    | 128K                |