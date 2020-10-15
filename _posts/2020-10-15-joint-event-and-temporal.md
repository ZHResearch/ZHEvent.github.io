---
layout: post
title: 2019 EMNLP《Joint Event and Temporal Relation Extraction with Shared Representations and Structured Prediction》
categories: [Note, Temporal Relation, Event Extraction]
description: 事件和事件时序关系的联合抽取
keywords: Joint Model, Temporal Relation, Event Extraction
mathjax: true
original: true
author: 黄琮程
authorurl: https://github.com/Ottohcc
---

> 2019 EMNLP《事件和事件时序关系的联合抽取》[(Joint Event and Temporal Relation Extraction with Shared Representations and Structured Prediction)](https://www.aclweb.org/anthology/D19-1041/)的阅读笔记

## 一、问题

事件时序关系抽取是一项重要的自然语言理解（NLU）任务。

事件时序关系抽取一般包含两个任务，即事件抽取和事件时序关系分类，现有的系统 （Verhagen et al., 2007, 2010; UzZaman et al., 2013; Chambers et al., 2014; Ning et al., 2017; Meng and Rumshisky, 2018）一般将这两个子任务视作在一个 pipeline 下的两个独立任务，这些系统都是先提取事件，然后再预测事件之间的时序关系。事件抽取工作往往会有不少错误，而在这些系统中事件抽取的错误将会传播到时序关系分类的任务中，且难以更正。

本文作出两点贡献：

+ 提出了一个同时提取事件和时间关系的联合模型。其动机是，如果训练的关系分类器预测非事件之间的关系为 NONE，那么它就可能具有纠正事件提取错误的能力。比如，在时序关系分类时，事件 M 和事件 N 有极大的可能关系是 NONE，那么 M、N 中很有可能有一个不是事件。
+ 通过在事件提取和时间关系提取模块之间共享相同的 context embedding 和 neural representation learner 来提升事件表示。该模型在共享 context embedding 和 neural representation learner 的基础上，生成一个表示给定语句中所有事件和关系的图结构输出。

有效的图预测应满足两个结构约束。

首先，两个非事件之间的关系，或一个事件和一个非事件之间的关系的时序关系一定是 NONE。

其次，对于事件之间的时间关系，由于时间的传递性，不应存在任何循环（例如，如果 A 在 B 之前，B 在 C 之前，则 A 必须在 C 之前）。

本文通过求解具有结构约束的整数线性规划（ILP）优化问题，保证了图的有效性，并用结构支持向量机（SSVM）对联合模型进行端到端训练。

## 二、方案

将所有可能的关系标签（包括 NONE）的集合表示为 $$\mathcal{R}$$，所有可能的事件候选（包括非事件）的集合表示为 $$\mathcal{E}$$，所有关系候选表示为 $$\mathcal{E}\mathcal{E}$$。

### 1.ssvm

ssvm 的 loss 函数为：

$$\mathcal{L}=\sum_{n=1}^l\frac{C}{M^n}[max_{\widehat{\boldsymbol{y}}^n\in \mathcal{Y}}(0,\Delta(\boldsymbol{y}^n,\widehat{\boldsymbol{y}}^n)+\overline{S}_\mathcal{R}^n+C_\mathcal{E}\overline{S}_{\mathcal{E}}^n)]+||\Phi||^2$$

$$\Phi$$ 指模型参数。

$$C$$ 和 $$C_\mathcal{E}$$ 是平衡 loss 函数的正则化超参数，$$C = 1$$，$$C_\mathcal{E}$$ 会被训练。

$$\boldsymbol{y}^n$$ ，$$\widehat{\boldsymbol{y}}^n$$ 分别代表实例 $$n$$ 的事件和关系 gold 和预测结果，包含$$\boldsymbol{y}_\mathcal{E}^n,\widehat{\boldsymbol{y}}_\mathcal{E}^n\in\{0,1\}$$	,$$\boldsymbol{y}_\mathcal{R}^n,\widehat{\boldsymbol{y}}_\mathcal{R}^n\in\{0,1\}$$，需要找到一个最大后验概率（MAP）推理来找到 $$\widehat{\boldsymbol{y}}^n$$ ，将其作为一个整数线性规划（ILP）问题，此处将在第 3 点中详解。

$$\Delta(\boldsymbol{y}^n,\widehat{\boldsymbol{y}}^n)$$ 表示 gold 和预测结果之间的距离，使用的是 hamming 距离。

$$ {S}_{\mathcal{E}}^n$$ 和 $${S}_\mathcal{R}^n$$​ 分别代表判断是否为事件、是否存在时序关系的打分器：

$$\overline{S}_{\mathcal{E}}^n=S(\widehat{\boldsymbol{y}}^n_\mathcal{E};\boldsymbol{x^n})-S({\boldsymbol{y}}^n_\mathcal{E};\boldsymbol{x^n})$$

$$\overline{S}_\mathcal{R}^n=S(\widehat{\boldsymbol{y}}^n_\mathcal{R};\boldsymbol{x^n})-S({\boldsymbol{y}}^n_\mathcal{R};\boldsymbol{x^n})$$

$$M^n$$ 代表事件数和关系数的总和：

$$M^n=|\mathcal{E}|^n+|\mathcal{E}\mathcal{E}|^n$$

本文中的 SSVM 与 传统的 SSVM 的最大区别在于 Scorer，传统的 SSVM 往往采用人工设计的线性函数，而本文中的 Scorer 采用的是 RNN 的神经网络模型，并且通过训练整个端到端的结构来训练 Scorer。

### 2.Multi-Tasking Neural Scoring Function

<div style="text-align: center;">
<img src="/images/blog/joint-event-and-temporal-1.png" width="350px"/>
</div>


底层的 $$v_i$$ 表示包含上下文信息的词向量。使用预训练 BERT 作为 word emmbeding。将其结果输入 BiLSTM 层对每个 token 都进行正向和反向的编码，得到 $$f_i$$，$$b_i$$，$$f_j$$，$$b_j$$ 再加上他们的语言学特征向量 $$L_{ij}$$ （token 距离、时态和事件极性）。最后，将它们连起来形成输入，以计算成为事件的可能性或可能的关系的softmax函数的分布。

### 3.MAP Inference

本文将预测问题表示为 ILP 问题，本文通过构建一个全局目标函数，从局部 scorer 和以下几点约束来获得总得分：

+ One-label 分配
+ 事件关系一致性
+ 对称性和传递性

#### 目标函数：

$$\widehat{y}=arg max\sum_{(i,j)\in\mathcal{E}\mathcal{E}}\sum_{r\in\mathcal{R}}y_{i,j}^rS(y_{i,j}^r,x)+C_{\mathcal{E}}\sum_{k\in\mathcal{E}}\sum_{e\in{\{0,1\}}}y_k^eS(y_k^e,x)$$

其中

$$y_{i,j}^r,y_k^e \in\{0,1\},\sum_{r\in\mathcal{R}}y_{i,j}^r=1,\sum_{e\in{\{0,1\}}}y_k^e=1$$

$$\widehat y$$ 是上下文中所有事件和关系候选的最佳标签分配情况。

#### 限制条件：

**事件关系限制**：当且仅当一对输入的 token 都是事件时，这两个 token 间才可能有时序关系。

$$\forall(i,j)\in\mathcal{E}\mathcal{E},e_i^P\ge r_{i,j}^P,e_j^P\ge r_{i,j}^P \ and\ e_j^N+e_i^N\ge r_{i,j}^N$$

$$r_{i,j}^P$$ 代表所有可能的 positive relations 包括：before、after、simultaneous、includes、is_included、vague

$$r_{i,j}^N$$ 代表所有可能的 positive relation 即 NONE。

两个 token 分别是事件的得分一定要高于他们之间存在时序关系的得分，即完成了限制。

**对称性和传递性限制**：

$$\forall (i,j),(j,k)\in\mathcal{E}\mathcal{E},y_{i,j}^r=y_{i,j}^r$$

$$y_{i,j}^{r_1}+y_{j,k}^{r_2}-\sum_{r_3 \in Trans(r_1,r_2)}y_{i,k}^{r_3}\le 1$$

在学习该模型时，作者曾经尝试直接训练整个模型，最后发现效果不佳，之后作者采用两段式训练：

+ 作者首先采用交叉熵 loss 函数，在训练的前几个 epoch 中，先使用 gold 的事件和关系分别训练 scorers，以得到相对准确的事件模型，然后转为 pipeline 的方式训练，在这一步中先前的限制条件并没有被加入。

+ 然后在使用全局 loss 函数结合限制条件重新训练整个上述模型。



## 三、结果和分析

### 1.结果

本文的实验选择的语料库是 TB-Dense 和 MATRES。 

![result](/images/blog/joint-event-and-temporal-2.png) 

Table 3 为消融实验：

Single-task 指的是将两块模型分别使用 gold 数据进行训练，并且相互之间无影响的端到端模型，BiLSTM 层不共用。

Multi-task 指的是结构和 Single-task 相似，但是共用同一个 BiLSTM 层的模型。

Pipeline Joint Model 结构与 Multi-task 相似，模型架构和多任务模型一样，区别在于 pipeline 的联合模型在训练阶段，使用事件模型来构建关系候选，以用于训练关系模型。使用这一策略，在训练阶段若一个候选关系的元素不是事件，则会生成 NONE 对，这些 NONE 对会帮助关系模型分辨出是否存在关系。

### 2. 分析

**标签不平衡**

<div style="text-align: center;">
<img src="/images/blog/joint-event-and-temporal-3.png" width="400px"/>
</div>

本文作者的解决办法是加大训练时样本量较小的标签的权值，图中显示，在相对加大权值时性能会有一定的提高。

**全局限制的表现**

<div style="text-align: center;">
<img src="/images/blog/joint-event-and-temporal-4.png" width="400px"/>
</div>

事件关系限制对于两个数据集都有 1% 左右的提高，但是时序关系传递性对于模型的提升很有限，作者觉得是因为 BERT 在编码词向量时已经包含了大量上下文信息，可能其中就有时序传递关系，并且由于 NONE 关系的存在，时序传递性的传播也会收到阻碍。

**错误分析**

<div style="text-align: center;">
<img src="/images/blog/joint-event-and-temporal-5.png" width="400px"/>
</div>

作者列出了主要的三类错误：事件没有被识别、存在包含非事件的时序关系（即 NONE 被识别有关系）、VAGUE 关系的识别错误。

对于第一种和第二种错误，作者提出的办法是需要构建更强的事件抽取模型，对于第三种错误，作者认为可以加入常识知识，或者创建更好的有利于区分 VAGUE 和其他种类时序关系的数据集。