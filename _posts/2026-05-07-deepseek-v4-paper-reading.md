---
title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence 论文解读"
date: 2026-05-07
permalink: /posts/2026/05/deepseek-v4-paper-reading/
project: paper-study
excerpt: "阅读 DeepSeek-V4 论文，梳理 Hybrid Attention、mHC、Muon Optimizer、state cache 与稀疏 Attention Kernel 协同设计。"
tags:
  - DeepSeek
  - 论文学习
  - 大模型
  - Attention
---

{% include toc icon="list" title="目录" position="inline" %}

![image-20260507111847279](/images/3.png)

Deepseek V4问世，在不同测试中取得了较好的效果，本文阅读论文记录学习Deepseek V4新架构。

## 创新点

论文中一共给出了三个创新点，分别是Hybrid Attention，mHC以及Muon Optimizer。

1. Hybrid Attention是指压缩的KV cache，包括CSA和HCA两种技术。CSA 即 `Compressed Sparse Attention`，先把 KV cache 沿序列维压缩，再做稀疏 attention；HCA 即 `Heavily Compressed Attention`，用更大的压缩率压缩 KV，但仍保留 dense attention。

2. mHC即`Manifold-Constrained Hyper-Connections`，它是对传统残差连接的增强。普通 Hyper-Connection 会扩展 residual stream，但深层堆叠时可能不稳定；mHC 通过把 residual mapping 约束到双随机矩阵流形上，使其谱范数受控，从而提高前向传播和反向传播的数值稳定性。

3. DeepSeek-V4 大部分模块使用 Muon优化器，而 embedding、prediction head、RMSNorm、mHC 的部分静态参数仍用 AdamW优化器。Muon 通过矩阵更新的近似正交化来改善收敛速度和训练稳定性。

## CAS
![image-20260507143638628](/images/6.jpg)

传统的 **attention** 机制中，输入 $H$ 经过变换得到 $Q, K, V$。

$$
H \in \mathbb{R}^{n \times d},
\quad
Q, K, V \in \mathbb{R}^{n \times \mathrm{hidden}}
$$

```text
Q = H W_Q
K = H W_K
V = H W_V
```

然后将其中的 **KV** 矩阵存储起来就是 **KV cache**，然后通过下面的 attention 公式得到输出。

$$
\operatorname{Attention}(Q, K, V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

而 **Deepseek V4** 没有生成传统的 **KV** 矩阵，而是设计了一种新的结构——**compressed KV entry**

------

### 1. 生成压缩 KV 候选内容和压缩权重分值

每个 token 生成压缩 **KV** 候选 $C$，以及对应压缩权重分值 $Z$

$$
C^a = H \cdot W^{aKV}, \quad C^b = H \cdot W^{bKV}
$$

$$
Z^a = H \cdot W^{aZ}, \quad Z^b = H \cdot W^{bZ}
$$

------

### 2. 压缩得到 Compressed KV Entry

将 $m$ 个 token 对应候选进行压缩得到 **compressed KV entry**。**其值 = 当前 block 的 $C^a$ 的加权和 + 下一个 block 的 $C^b$ 的加权和**。权重来自 $Z^a, Z^b$ 以及位置偏重 $B^a, B^b$。这样在进行压缩时也缓解了边界效应效应，最终将 **KV entry** 的规模降低：

$$
\mathbb{R}^{n \times d}
\rightarrow
\mathbb{R}^{n/m \times d}
$$

$$
\left[
S^a_{mi:m(i+1)-1};
S^b_{m(i-1):mi-1}
\right]
=
\operatorname{Softmax}_{\mathrm{row}}
\left(
\left[
Z^a_{mi:m(i+1)-1} + B^a;
Z^b_{m(i-1):mi-1} + B^b
\right]
\right)
$$

$$
C^{\mathrm{Comp}}_i
=
\sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j
+
\sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j
$$

------

### 3. 生成 Compressed Indexer Keys 并计算相关性分数

需要进行稀疏选择，即每次生成新 token 时只选择少量 **compressed KV entry** 进行计算注意力分数。

首先会根据压缩 **KV** 候选用类似的方式生成 **compressed indexer keys**

$$
K^{IComp} \in R^{n/m \times c_I}
$$

为当前 token 生成 **indexer query** -- $\mathbf{q}_t^I$

$$
c_t^Q = \mathbf{h}_t \cdot W^{DQ}
$$

$$
\left[
\mathbf{q}_{t,1}^I;
\mathbf{q}_{t,2}^I;
\dots;
\mathbf{q}_{t,n_h^I}^I
\right]
=
\mathbf{q}_t^I
=
c_t^Q \cdot W^{IUQ}
$$

用 $q^I_t$ 和所有历史 $K^{IComp}_s$ 计算相关性分数 $I_{t,s}$，这个分数 $I_{t,s}$ 可以理解为当前 **query** 与第 $s$ 个 **compressed block** 的匹配程度。

$$
\left[
w^I_{t,1};
w^I_{t,2};
\dots;
w^I_{t,n^I_h}
\right]
=
\mathbf{w}^I_t
=
\mathbf{h}_t \cdot W^w
$$

$$
I_{t,s}
=
\sum_{h=1}^{n^I_h}
w^I_{t,h}
\cdot
\operatorname{ReLU}
\left(
\mathbf{q}^I_{t,h}
\cdot
K^{\mathrm{IComp}}_s
\right)
$$

------

### 4. Top-k 稀疏检索

当前 **query** 做 **top-k** 检索，选出分数最高的 $k$ 个 **compressed block**，得到 $C^{SprsComp}$，也就是当前 token 真正要看的 **top-k** 个远程压缩 **KV**。

------

### 5. 拼接局部 Sliding Window KV

拼接局部窗口，**CSA** 不只看远程压缩块，还保留最近 $n_{win}$ 个 token 的未压缩 **KV**。最终 **attention** 对象是：

**top-k compressed KV + recent sliding-window KV**

------

### 6. 执行 Shared Key-Value MQA

执行 **Shared Key-Value MQA**，对 **compressed KV** 部分 $C^{SprsComp}$ 同时作为 **Key** 和 **Value**。即：

$$
\operatorname{Attention}
\left(
Q,
C^{SprsComp},
C^{SprsComp}
\right)
$$

------

### 7. Grouped Output Projection

最终需要将注意力分数投影回到 $d$ 维，但不是用直接投影（扇出太大），而是使用 **grouped output projection**，即分为多组进行投影到 $d_g$ 维，再拼接投影回 $d$ 维。

## HCA

![image-20260507160013638](/images/7.jpg)

**HCA** 拥有和 **CSA** 类似的整体流程，但压缩率更高，并且不做 **top-k sparse selection**。

------

### 1. 生成 **KV 候选内容**和**压缩权重**

$$
C = H \cdot W^{KV},
\quad
C_i^{\mathrm{Comp}}
=
\sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j
$$

------

### 2. 重压缩远程 KV

HCA 将每 $m'$ 个 token 压缩成 1 个 **compressed KV entry**：

$$
C^{Comp} \in \mathbb{R}^{\frac{n}{m'} \times c}
$$

其中，$m'$ 远大于 CSA 中的压缩率 $m$，即：

$$
m' \gg m
$$

因此，HCA 的压缩更激进，同时，HCA 不做 **overlapped compression**，也就是每个压缩块之间没有重叠。

------

### 3. 不进行稀疏选择

CSA 的流程是：

> 先压缩成 $\frac{n}{m}$ 个 **compressed entries**，再通过 **indexer** 选择 **top-k** 个 entries。

而 HCA 的流程是：

> 直接把长度为 $n$ 的 KV 序列压缩成 $\frac{n}{m'}$ 个 **compressed entries**，然后让这些 entries 全部参与 attention。

------

### 4. 拼接局部窗口

和 CSA 一样，HCA 也会额外保留最近 $n_{win}$ 个 token 的未压缩 KV。

最终 attention 的对象是：

> **heavily compressed KV + recent sliding-window KV**

------

### 5. 使用 **Shared Key-Value MQA**

HCA 也使用 **Shared Key-Value Multi-Query Attention, MQA**。

------

### 6. **Grouped Output Projection**

HCA同样执行这一步

------

## mHC

mHC是Deepseek V4对HC技术的改进，HC是对普通残差连接的一种升级。

普通残差连接是将**当前层输出**+**当前层输入**作为**下一层的输出**

### Hyper-Connections

![image-20260507170629918](/images/8.jpg)

首先将输入h复制n份，这里以2份为例，得到$h_1, h_2$

1. 首先从多个h中混合得到当前层的输入：

   $$
   h_0 = \alpha_{10} h_1 + \alpha_{20} h_2
   $$

   这里就是$A_m$对应权重

2. 将当前层的输出写回成多条stream：

   $$
   h'_1 += \beta_1 T(h_0), \quad h'_2 += \beta_2 T(h_0)
   $$

   这里就是$B$对应权重

3. 不同h之间彼此混合：

   $$
   h'_1 += \alpha_{11} h_1 + \alpha_{21} h_2,
   \quad
   h'_2 += \alpha_{12} h_1 + \alpha_{22} h_2
   $$

   这里就是$A_r$对应权重

而下一层就不需要进行复制了，直接使用$h'_1，h'_2$就可以了

### Dynamic HC

HC包括Static HC和Dynamic HC，static HC的$A_m, A_r, B$都是训练好固定的，而Dynamic HC的三个权重则是部分依赖输入H，其中静态部分是预先训练好的，而微小扰动则是根据输入H得到的。

$$
\begin{aligned}
B(H) &= B + \Delta B(H) \\
A_m(H) &= A_m + \Delta A_m(H) \\
A_r(H) &= A_r + \Delta A_r(H)
\end{aligned}
$$

$$
\begin{aligned}
B(H) &= s_\beta \circ \tanh(HW_\beta)^T + B \\
A_m(H) &= s_\alpha \circ \tanh(HW_m) + A_m \\
A_r(H) &= s_\alpha \circ \tanh(HW_r) + A_r
\end{aligned}
$$

### Manifold-Constrained Hyper-Connections

![image-20260507173936220](/images/9.jpg)

mHC是在DHC上的改良，包括： 
1. 将$A_r$，也就是论文中的$B_l$约束为双随机矩阵——$B_l$ 每个元素 ≥ 0；$B_l$ 每一行之和 = 1； $B_l$ 每一列之和 = 1 
2. 将$A_m$，也就是论文中的$A_l$，使用了sigmod约束，保证其值在0-1之间。 
3. 将$B$，也就是论文中的$C_l$，使用了2*sigmod约束，保证其值在0-2之间。
4. mHC生成三个矩阵动态部分的方法是： 将输入$X_l$（n个d维变量）进行展平后**RMSNorm**归一化，然后线性变化生成原始动态参数，与静态参数相加，最后对原始三个矩阵进行上面的约束投影。

## Muon

Muon 和 AdamW 最大的不同在于：

> **Muon 不只是按照梯度方向更新参数，而是会对更新矩阵做近似正交化。**

论文中对每个权重矩阵 $W \in \mathbb{R}^{n \times m}$，先计算梯度：

$$
G_t = \nabla_W \mathcal{L}_t(W_{t-1})
$$

然后维护 momentum：

$$
M_t = \mu M_{t-1} + G_t
$$

接着用 **Nesterov trick** 得到更新方向，并对它做 **Hybrid Newton-Schulz** 正交化：

$$
O'_t = \operatorname{HybridNewtonSchulz}(\mu M_t + G_t)
$$

最后再进行尺度调整和权重更新：

$$
O_t = O'_t \cdot \max(n,m) \cdot \gamma
$$

$$
W_t = W_{t-1}(1-\eta\lambda)-\eta O_t
$$

这里可以理解为：

```
梯度 → momentum → 近似正交化 → 尺度调整 → 参数更新
```

## state cache和Sparse Attention Kernel Co-Design

![image-20260507175127721](/images/10.jpg)

由于deepseek的kv结构的多样性，市面上paged的kv cache结构难以覆盖，所有Deepseek V4设计了新的cache结构--**state cache**，和正常kv cache一同使用。

- 其中state cache针对每个request分配固定长度（比较长的），用于存放（Sliding Window Attention）SWA （这个就是存放局部kv的地方）和 CSA/HCA 里还没准备好压缩的尾部 token。

- 而kv cache用于存放已经压缩好的CSA/HCA历史KV。

Sparse Attention Kernel Co-Design只说明了设计约束：
传统 attention kernel 通常假设固定 B 个 token/block，但 CSA/HCA 中 compressed entry 对应的原始 token 数不同，因此需要 KV cache layout 与 sparse attention kernel 协同设计，使不同层支持 variable tokens per block。具体做法没有展开，只举例说明可通过 padding block 对齐 cache line 来提升性能，并要求 original tokens per block 取 lcm(m,m') 的倍数，以保证 CSA/HCA 的 compressed entries 都能整齐落在 block 边界内。
