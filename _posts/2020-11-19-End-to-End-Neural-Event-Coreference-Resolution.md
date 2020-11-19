---
layout: post
title: 2020《End-to-End Neural Event Coreference Resolution》
categories: [Note, Event Coreference]
description: 端到端事件同指消解
keywords: Coreference Resolution
mathjax: true
original: true
author: 黄琮程
authorurl: https://github.com/Ottohcc
---

> 2020《端到端事件同指消解》[(End-to-End Neural Event Coreference Resolution)](https://arxiv.org/pdf/2009.08153.pdf) 的阅读笔记

## 一、问题

传统的事件同指消解通常依赖管道框架和手工构建的特征，常常面临错误传播问题，泛化能力差。KBP 2017 中最好的事件检测系统只有 56 的 F1，这无疑会限制后续事件共指任务的性能。

本文提出了一种端到端的神经网络事件同指消解模型，对事件检测和事件同指消解进行联合建模，并学习如何自动从原始文本中提取特征。

由于事件的高度多样性和事件间的关联性受到长距离、语义依赖性等的控制，本文在神经网络中进一步提出了一种类型引导的事件同指消解机制。

+ type-informed antecedent network

  先行词（antecedent）是出现在前面文档的同指 mention，使模型能够同时预测事件同指分数和事件类型分数，从而获得更多事件 mention 的语义信息

+ type-refined mention representation

  使用事件类型信息增强 mention 表示，因此，即使在词汇上不相似的 mention 也可以识别在一起

+ type-guided decoding algorithm

  利用全局事件类型一致性来获得更精确的事件链

## 二、方案

<div style="text-align: center;">
<img src="/images/blog/End-to-End-Neural-Event-Coreference-Resolution-1.png" width="400px"/>
</div>


### 1.Mention Proposal Network（事件抽取）

#### 上下文词嵌入层

首先基于预训练的 BERT 嵌入得到每个 token 的独立表示，具体操作如下。 

$$\mathbf h_i = \gamma\sum^L_{j=1}\alpha_j\mathbf x_i^{(j)}$$

$$x_i^{(j)}$$ 代表第 $$i$$ 个 token 的第 $$j$$ 层 BERT 的输出。$$\alpha_j$$ 是一个 softMax-normalized 权重，$$\gamma$$ 是一个参数。

$$\mathbf H = \{\mathbf h_1,\mathbf h_2,...,\mathbf h_n\}$$

$$\mathbf C = \mathbb {softmax}(\cfrac {\mathbf {HH}^T}{\sqrt d}+\mathbf M)\mathbf H$$

$$\mathbf M_{ij}=\begin{cases}0,\ \ \ \ \ \ \ \ \ |i,j<c|\\-\infty,\ \ \ \ \mathrm{otherwise}\ \end{cases}$$

常数 c 代表一个上下文 window，本文设置为 10，最后得到每句话的上下文表示 $$\mathbf C = \{\mathbf c_1,\mathbf c_2,...,\mathbf c_n\}$$。

#### mention 抽取层

在已得上述 token 表达的情况下，对于每一个 span 经过一个打分器 $$s_m(i)$$ ，输出是这个 span 成为一个事件mention 的可能性。在本文中设置 span 长度为 1，也就是 span $$\mathbf g_i = \mathbf c_i$$ 。

$$s_m(i) =\mathbf {FFNN}_m(\mathbf g_i)$$

其中 FFNN 代表标准前馈神经网络，最后取得打分最高的 $$l=0.1*\bf len(doc)$$ 个 mention。

### 2.Type-informed Antecedent Network（先行词神经网络）

#### 先行词得分

对于给定一个事件 mention $$m_i$$ ，计算其与文章中的 mention $$m_j $$ $$(j<i)$$ ，之间的先行词得分 $$s(i,j)$$

$$s(i,j)=s_m(i)+s_m(j)+s_a(i,j)$$

$$s_a(i,j)$$ 代表 $$m_i$$ 和 $$m_j$$ 之间的语义相似度。

$$s_a(i,j) = \mathbf {FFNN}_a([\mathbf g_i,\mathbf g_j,\mathbf g_i\circ \mathbf g_j,\Phi(i,j)])$$

其中 $$\Phi(i,j)$$ 代表两个向量之间的距离。

#### 事件类型得分

为了使得同指链中的事件 mention 保持事件类型一致性，定义以下打分器。

对于所有的事件类型 $$\mathcal T=\{t_1,...,t_t\}$$ 经过一个分层嵌入算法（BERT）得到 $$\mathbf e_{type}(t_k)$$

$$\mathbf g_{t_k} = \mathbf W_e \cdot [\mathbf e_{event},\mathbf e_{type}(t_k)]$$

$$\mathbf e_{event}$$ 是所以事件类型共享的，最后得到的向量维度与 mention 的维度相同。

$$s_m(t_k) =\mathbf {FFNN}_m(\mathbf g_{t_k})$$

$$s_a(i,t_k) = \mathbf {FFNN}_a([\mathbf g_i,\mathbf g_{t_k},\mathbf g_i\circ \mathbf g_{t_k},\Phi(i,t_k)=0])$$

$$s(i,t_k)=s_m(i)+s_m(t_k)+s_a(i,t_k)$$

当所有的 $$s(i,j)\le 0$$，且 $$s(i,t_k)\le 0$$ 时，设定该 mention 不是事件。

这样，就获得了先行词得分和类型得分。

### 3.Type-based Refining（事件类型信息增强）

为了加强事件类型对于同指识别的帮助，识别出形态上不一样，但语义上相似的 mention（如 leave、depart），做以下操作。

$$Q(t_k)=\cfrac {e^{s(i,t_k)}} {\sum_{t_k'\in\mathcal T \cup\{\varepsilon\}}e^{s(i,t_k')}} $$

$$s(i,\varepsilon)=0$$

$$\tilde {\mathbf g}_i=\sum_{t_k'\in \mathcal T}Q(t_k = t_k')\cdot \mathbf g_{t_k'}+Q(t_k=\varepsilon)\cdot \mathbf g_i$$

$$\mathbf f_i = \sigma(\mathbf W_f\cdot[\mathbf g_i,\tilde {\mathbf g_i} ])$$

$$\mathbf g_i' =\mathbf f_i \circ \mathbf g_i+(1-\mathbf f_i)\mathbf g_i$$

通过上式可以得到被事件类型信息加强的 mention 表达 $$\mathbf g_i'$$，然后再用前面介绍过的神经网络计算事件对的同指得分 $$s'(i,j)$$。

### 4.Type-guided Decoding（后处理）

为了保证同指事件链上的时间类型保持一致性，设定

$$a_i = \mathrm {arg\ max}_{m_j,j<i}s(i,j)$$

对于 $$t_i=\mathrm {arg\ max}_{k\in\mathcal T}s(i,t_k)$$，如果 $$s(i,a_i)>s(i,t_i)$$ 则认为这两个事件的类型是一样的。

若 $$s(i,a_i)\le s(i,t_i)$$ 则认为这两个事件不是同一种类型，该事件 mention $$i$$ 将开启一条新的同指链。

### 5.Model

本文将**最大化**以下的损失函数得到最终结果。

$$\mathcal L(\Theta)=\mathcal L_{antecendent}(\Theta)+\mathcal L_{proposal}(\Theta) $$

#### Mention Proposal Loss

$$\mathcal L_{proposal}(\Theta) = \sum_{i=1}^n y_ilog\sigma(s_m(i))+(1-y_i)log(1-\sigma(s_m(i)))$$

#### Antecedent Loss

$$\mathcal L_{antecendent}(\Theta) = log \prod_{i=1}^l\sum_{\hat y\in\mathcal Y(i)\cap \mathrm{GOLD}(i)}P(\hat y|D)$$

$$\mathrm{GOLD}(i)$$ 代表 $$i$$ 所在同指链所有标注的同指 mention，$$\mathcal Y(i)$$ 代表所有在 $$i$$ 之前的合理的可能先行词。

$$P(y_i|D)=\cfrac {\mathrm {exp}(s(i,y_i))}{\sum_{y'\in\mathcal Y(i)}\mathrm {exp}(s(i,y'))}$$

具体例子，如图 3 所示

<div style="text-align: center;">
<img src="/images/blog/End-to-End-Neural-Event-Coreference-Resolution-2.png" width="400px"/>
</div>

## 三、结果及分析

### 1.结果

<div style="text-align: center;">
<img src="/images/blog/End-to-End-Neural-Event-Coreference-Resolution-3.png" width="700px"/>
</div>

其中 Type-F1 是事件检测的准确度。

### 2.消融分析

#### 事件类型影响

<div style="text-align: center;">
<img src="/images/blog/End-to-End-Neural-Event-Coreference-Resolution-4.png" width="400px"/>
</div>

Type Rule 是一种简单的启发式方法，它认为一篇文章中同一类型的所有事件 mention 都是同指的。

图 2 体现了事件类型信息增强（Type-Refined）和事件类型一致后处理（Type-Guided）的作用。

#### 端到端训练影响

<div style="text-align: center;">
<img src="/images/blog/End-to-End-Neural-Event-Coreference-Resolution-5.png" width="400px"/>
</div>


two stage 代表把事件抽取和同指检测分开训练，pipeline 式的操作。

w/o Proposal Loss 指的是把 loss 函数中的 Proposal 部分去掉进行训练。

Gold mention 使用的是有标注的 mention 进行同指推断。

可以发现事件抽取任务在某种程度上还是在限制同指模型的性能。此外，即使修正了预测事件检测结果中的所有同指链的错误，AVG-F 的增长仅从 40.85 到 42.80，说明本文的事件抽取模块还有提升空间。

#### 预训练模型影响

<div style="text-align: center;">
<img src="/images/blog/End-to-End-Neural-Event-Coreference-Resolution-6.png" width="400px"/>
</div>


### 3.总结

#### 事件抽取模块限制

作者认为事件抽取模块主要难在以下两个方面：

+ 事件 mention 具有多样性和模糊性，检测事件需要对上下文有深入的理解。
+ 许多时间 mention 是 multi-tagged 的，即一个触发词触发多个事件。本文没有考虑这一问题，因此遗漏了一些内容。据作者统计在 KBP 2016 有 10.18%，在 KBP 2017 有 8.4% 符合上述情况。

#### 文本体裁限制

<div style="text-align: center;">
<img src="/images/blog/End-to-End-Neural-Event-Coreference-Resolution-7.png" width="400px"/>
</div>


作者的模型在新闻体裁上的表现好于在论坛体裁上的表现。

主要是因为论坛体裁上的文章没有显著的文本结构性特征，且在论坛中的事件可能不仅与 mention 的内容有关，与说话人和文本主题也有很大的关系。