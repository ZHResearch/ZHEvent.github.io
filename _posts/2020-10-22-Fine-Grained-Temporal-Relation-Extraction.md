---
layout: post
title: 2019 ACL《Fine-Grained Temporal Relation Extraction》
categories: [Note, Temporal Relation]
description: 细粒度时序关系抽取
keywords: Fine-Grained,Temporal Relation
mathjax: true
original: true
author: 黄琮程
authorurl: https://github.com/Ottohcc
---

> 2019 ACL《细粒度时序关系抽取》[(Fine-Grained Temporal Relation Extraction)](https://www.aclweb.org/anthology/P19-1280/) 的阅读笔记

## 一、问题

事件时序关系抽取是一个很有意义的研究方向，该领域目前的工作还是将其视为一个分类问题，标注成对的事件表达和时间表达，并进行时序关系分类。这种方法的缺点是时间表达往往依赖于显式存在的持续时间信息，但是，几乎所有的时间持续信息都可以不直接存在于句中。所以在这种方法下，一般只能对可以被识别的时间持续信息进行编码。

这篇文章中，作者开发了一种新型的框架将事件持续时间放在时序关系表示的首部或中部，将事件映射射到可能的持续时间上，并将事件对直接映射到实际值的相对应时间线。在这种做法下，不仅提升了事件持续时间在判断时序关系时的重要性，同时也帮助我们更好的判断在一整段文字下的多个复杂事件的时序结构。

<div style="text-align: center;">
<img src="/images/blog/Fine-Grained-Temporal-Relation-Extraction-1.png" width="400px"/>
</div>


## 二、方案

### 1.数据收集

<div style="text-align: center;">
<img src="/images/blog/Fine-Grained-Temporal-Relation-Extraction-2.png" width="400px"/>
</div>


作者收集了 Universal Decompositional Semantics Time（UDS-T）数据集，该数据集在 Universal Dependents English Web Treebank（UD-EWT）之上标注。

具体数据标注步骤如下：

+ 作者首先使用 Predpatt 从 UD-EWT 中提取谓词
+ 将其中相邻句子连接起来
+ 给标注者两个连续的句子，并带有两个突出显示的事件引用表达式（谓词），要求标注者做到以下三点：
  + 为突出显示的谓词所指的事件对提供有界刻度的相对时间线
  + 从以下列表中得出谓词所指事件的可能持续时间： instantaneous, seconds, minutes, hours, days, weeks, months, years, decades, centuries, forever
  + 标注者须对其上述两点标注行为作出打分（0-4，分数越高代表越有信心）
+ 对标注者做出的标注进行归一化（Normalization）如图 3 所示，先发生事件 $$e_1$$ 的开始在 0 一端，后发生事件$$e_2$$的结束在 1 一端。

<div style="text-align: center;">
<img src="/images/blog/Fine-Grained-Temporal-Relation-Extraction-3.png" width="400px"/>
</div>


作者将每对事件对滑块（silder）定义为四个维度

（1）先后顺序（PRIORITY），当 e1 早于 e2 开始和/或结束时为正，否则为负；
（2）包含量（CONTAINMENT），当 e1 含有更多的 e2 时越大；
（3）相等性（EQUALITY），当 e1 和 e2 的时间范围相同时，相等性最大，当它们最不相等时最小；
（4）SHIFT，即事件在时间轴上向前或向后移动。

并通过以下这个方程中 $$\mathbf R$$ 的解得出这四个维度

$$\mathbf R \begin{bmatrix}-1&-1&1&1\\-1&1&1&-1\\-1&1&-1&1\\1&1&1&1\end{bmatrix} = 2\mathbf S -1$$

$$\mathbf S =  \begin{bmatrix}beg(e_1)&end(e_1)&beg(e_2)&end(e_2)\end{bmatrix}$$

$$\mathbf S$$ 中的元素由归一化后的标注数据决定，即 $$\mathbf S \in [0,1]^{N*4}$$ ，$$N$$ 为事件对数目。

为帮助理解求出了上述给定矩阵的逆矩阵：

$$\begin{bmatrix}-0.25&-0.25&-0.25&0.25\\-0.25&0.25&0.25&0.25\\0.25&0.25&-0.25&0.25\\0.25&-0.25&0.25&0.25\end{bmatrix}$$

作者最终收集的数据分布如图：
<div style="text-align: center;">
<img src="/images/blog/Fine-Grained-Temporal-Relation-Extraction-4.png" width="400px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/Fine-Grained-Temporal-Relation-Extraction-5.png" width="400px"/>
</div>


### 2.模型

作者的想法是：对于句子中提到的每一对事件，共同预测这些事件的相对时间线以及它们的持续时间，然后使用一个单独的模型从相对时间轴中归纳出整篇文章的时间线。

<div style="text-align: center;">
<img src="/images/blog/Fine-Grained-Temporal-Relation-Extraction-6.png" width="450px"/>
</div>

Tuner 是一个降维器，把 ELMo 降到 256 维。


#### 相对时间线（Relative timelines）

相对时间线模型包含三个组件：事件模型（Event model）、持续时长模型（Duration model）、关系模型（Relation model）。

这些组件在 $$\mathbf H \in \mathbb R^{N\times D}$$ 这个 embedding 上使用多层点积注意力机制（dot-product attention）。

$$\mathbf H = \tanh(\mathrm{ELMo}(\mathbf s)\mathbf W^{\mathrm{TUNE}}+\mathbf b^\mathrm{TUNE})$$

其中

$$\mathbf s = [w_1,...,w_N]$$ 代表一个句子，这个句子会经由 ELMo 产生三个 M 维上下文 embedding，并进行串联。

$$D$$ 代表 tuned embedding 的维数，$$\mathbf W^{\mathrm{TUNE}}\in \mathbb R^{3M\times D}$$ ，$$\mathbf b^\mathrm{TUNE}\in\mathbb R^{N\times D}$$

#### 事件模型（Event model）

定义 $$\mathbf {g}_{\mathrm{pred}_k} \in \mathbb R^D$$ 表示谓词 $$k$$ 所指代的事件，使用点积注意力机制（dot-product attention）的变体建立。

$$\mathrm {\mathbf a^{SPAN}_{pred_k}=tanh(\mathbf A^{SPAN}_{PRED}\mathbf h_{ROOT(pred_k)}+\mathbf b^{SPAN}_{PRED})}$$

$$\alpha_{\mathrm{pred}_k} = \mathrm{softmax}(\mathbf H_{\mathrm {span}_{(\mathrm{pred}_k)}}\mathrm{\mathbf a^{SPAN}_{pred_k}})$$

$$\mathbf {g}_{\mathrm{pred}_k} = [\mathbf h_{\mathrm{ROOT(pred}_k)};\alpha_{\mathrm {pred}_k}\mathbf H_{\mathrm {span}_{(\mathrm{pred}_k)}}]$$

$$\mathrm {\mathbf A^{SPAN}_{PRED}}\in \mathbb R^{D\times D}$$ 、$$\mathbf b^\mathrm {SPAN}_\mathrm{PRED} \in\mathbb R^D$$

Eg. My dog has **been *sick* for** about 3 days **now**.

$$\mathbf h_{\mathrm{ROOT(pred}_k)}$$ 代表第 $$k$$ 个谓词短语的 root 的隐层表达，在例句中就是 sick 的隐层表示。

$$\mathbf H_{\mathrm {span}_{(\mathrm{pred}_k)}}$$ 代表第 $$k$$ 个谓词短语所在 mention 中每个词隐层表达的堆叠，在例句中就是 been sick for now 的隐层表示。

#### 持续时长模型（Duration model）

$$\mathrm {\mathbf a^{SENT}_{dur_k}=tanh(\mathbf A^{SENT}_{DUR}\mathbf g_{pred_k}+\mathbf b^{SENT}_{DUR})}$$

$$\alpha_{\mathrm{dur}_k} = \mathrm{softmax}(\mathbf H\mathrm{\mathbf a^{SENT}_{dur_k}})$$

$$\mathbf {g}_{\mathrm{dur}_k} = [\mathbf g_{\mathrm{pred}_k};\alpha_{\mathrm {dur}_k}\mathbf H]$$

$$\mathrm {\mathbf A^{SENT}_{DUR}}\in \mathbb R^{D\times \mathrm {size}(g_{\mathrm{pred}_k})}$$ 、$$\mathbf b^\mathrm {SENT}_\mathrm{DUR} \in\mathbb R^D$$

$$\mathbf H$$ 指的是整句话的隐层表达。

关于持续时间的分类，作者提出两个方法：softmax 法、二项式分布法。二者之间的区别主要是二项式分布法会强制分布为凸状分布。

softmax 法：

$$\mathbf v_{\mathrm {dur}_k} = \mathrm{ReLU(\mathbf W_\mathrm{DUR}^{(1)}}\mathbf {g}_{\mathrm{dur}_k}+\mathbf b^{(1)}_\mathrm {DUR})$$

$$\mathbf p = \mathrm{softmax}(\mathbf W_\mathrm{DUR}^{(2)}+\mathbf b^{(1)}_\mathrm {DUR})$$

二项式分布法：

$$\mathbf v_{\mathrm {dur}_k} = \mathrm{ReLU(\mathbf W_\mathrm{DUR}^{(1)}}\mathbf {g}_{\mathrm{dur}_k}+\mathbf b^{(1)}_\mathrm {DUR})$$

$$\pi=\sigma(\mathbf w_\mathrm{DUR}^{(2)}\mathbf {v}_{\mathrm{dur}_k}+\mathbf b^{(2)}_\mathrm {DUR})$$

$$p_c=\dbinom{n}{c}\pi^n(1-\pi)^{(n-c)}$$

在使用这种方法时得到的 $$\pi$$ 是一个值。$$c\in\{0,1,..,10\}$$ 代表着从瞬间到永久的持续时间分布，在此模型中 $$n=10$$。

以上这两个方法得到的结果都用交叉熵函数作为 loss 函数进行训练 $$\mathbb L_\mathrm{dur}(d_k;\mathbf p) = -log\ p_{d_k}$$。

#### 关系模型（Relation model）

对于第 $$i$$ 和 $$j$$ 个谓词短语，依旧用相似的 attention 机制。

$$\mathbf a^{\mathrm {SENT}}_{\mathrm{rel}_{i,j}}=\tanh(\mathrm{A^{SENT}_{REL}}[\mathbf g_{\mathrm{pred}_i};\mathbf g_{\mathrm{pred}_j}]+\mathbf b^{\mathrm {SENT}}_\mathrm{REL})$$

$$\alpha_{\mathrm{rel}_{i,j}} = \mathrm{softmax}(\mathbf H\mathrm{\mathbf a^{SENT}}_{\mathrm{rel}_{i,j}})$$

$$\mathbf {g}_{\mathrm{rel}_{i,j}} = [\mathbf g_{\mathrm{pred}_i};\mathbf g_{\mathrm{pred}_j};\alpha_{\mathrm {rel}_{i,j}}\mathbf H]$$

$$\mathrm {\mathbf A^{SENT}_{REL}}\in \mathbb R^{D\times 2\mathrm {size}(g_{\mathrm{pred}_k})}$$ 、$$\mathbf b^\mathrm {SENT}_\mathrm{REL} \in\mathbb R^D$$

作者在此时序模型中的核心思想就是将事件及其状态直接映射到时间轴 $$[0,1]$$ 上，且对于每个事件 $$k$$ ，开始端一定要小于结束端，即 $$e_k\ge b_k$$。

$$[\widehat\beta_i,\widehat\delta_i,\widehat\beta_j,\widehat\delta_j]=\mathrm {ReLU}(\mathrm{MLP}(\mathbf {g}_{\mathrm{rel}_{i,j}} ))$$

对于每一个事件的开始及结束状态由下式得到：

$$[\widehat b_k,\widehat e_k]=[\sigma(\widehat\beta_k),\sigma(\widehat\beta_k+｜\widehat\delta_k｜)]$$

可以计算出一对事件所形成的 slider $$\widehat {\mathbf s}_{i,j}=[\widehat b_i,\widehat e_i,\widehat b_j,\widehat e_j]$$

定义 loss 函数

 $$\mathbb L_\mathrm{rel}({\mathbf s}_{i,j};\widehat {\mathbf s}_{i,j}) = |(b_i-b_j)-(\widehat b_i-\widehat b_j)|+|(e_i-b_j)-(\widehat e_i-\widehat b_j)|+$$

$$|(e_j-b_i)-(\widehat e_j-\widehat b_i)|+|(e_i-e_j)-(\widehat e_i-\widehat e_j)|$$

所以总的 loss 函数为：$$\mathbb L = \mathbb L_\mathrm{dur}+2\mathbb L_\mathrm{rel}$$

#### 持续时间模型与时序关系模型的联系（Duration-relation connections）

在 Dur->Rel 中，可采用两种方式：

$$\mathbf {g}_{\mathrm{rel}_{i,j}} = [\mathbf g_{\mathrm{pred}_i};\mathbf g_{\mathrm{pred}_j};\alpha_{\mathrm {rel}_{i,j}}\mathbf H;\mathbf p_i;\mathbf p_j]$$

$$\mathbf {g}_{\mathrm{rel}_{i,j}} = [\mathbf p_i;\mathbf p_j]$$

在 Rel->Dur 中，可采用两种方式：

$$\mathbf {g}_{\mathrm{dur}_k} = [\mathbf g_{\mathrm{pred}_k};\alpha_{\mathrm {dur}_k}\mathbf H;\widehat b_k;\widehat e_k]$$

$$\pi_{\mathrm{dur}_k} = \widehat e_k-\widehat b_k$$

#### 文本时间线（Document timelines）

假设有一个文本隐层事件线 $$T\in\mathbb R^{n_d\times 2}_+$$，其中 $$n_d$$ 代表文档中事件总数，2 代表每个事件的开始和持续时长。

通过确定所有谓词发生的起点及时长，使本文潜在时间线与事件对的相对时间线相连，且文档中始终存在以 0 为起点的谓词，并为文档内每个事件的所有组合 $$i$$ 和 $$j$$ 定义辅助变量：

$$\tau_{i,j} = [t_{i1},t_{i2},t_{j1},t_{j2}]$$

$$\widehat {\mathbf s}_{i,j}=\cfrac{\tau_{i,j}-min(\tau_{i,j})}{max(\tau_{i,j}-min(\tau_{i,j}))}$$

并且通过 $$\mathbb L_\mathrm{rel}({\mathbf s}_{i,j};\widehat {\mathbf s}_{i,j}) $$ 来学习 T，结果就是一条文本时间线。

## 三、结果及分析

### 1.结果

![result1](/images/blog/Fine-Grained-Temporal-Relation-Extraction-7.png)

以上结果是对自建的 UDS-T 测试集，大多数模型可以很好地预测事件开始和结束的相对位置（在 Relation 中的 $$\rho$$ 值高）事件的相对持续时间也能很好地预测（相对较低的 rank diff）。总的来说使用二项分布的结果好。

$$\mathrm{R1=1-\cfrac{MAE{model}}{MAE_{baseline}}}$$

<div style="text-align: center;">
<img src="/images/blog/Fine-Grained-Temporal-Relation-Extraction-8.png" width="450px"/>
</div>


上图是在 Timebank-Dense 上的测试结果。

本文的模型是建立在自己的标注数据上的，所以在其他语料库上进行测试时使用了迁移学习的方法，并取得了不错的成果。

迁移学习的大概步骤是，首先使用在 UDS-T 集上性能最佳的模型，获取 TE3 和 TD 中每对带注释的事件关系的关系表示 $$\mathbf {g}_{\mathrm{rel}_{i,j}}$$，然后，将此向量用作具有高斯核的 SVM 分类器的输入，并训练整个模型。

### 2.分析

![](/images/blog/Fine-Grained-Temporal-Relation-Extraction-9.png)

在本文的 Attention 机制下，在持续时长（Duration）模型中，表示某个时间段的词，如月、分、时、年、日、周是平均权重最高的词，前 15 个词中有 7 个直接表示持续时间类别，表示本文的系统在预测持续时间上有较好的表现。

在关系（Relation）模型中，平均注意权重最高的大多数词都是时态信息的连词或者包含时态信息的词汇动词和助动词，如 or、and、were 等。

<div style="text-align: center;">
<img src="/images/blog/Fine-Grained-Temporal-Relation-Extraction-10.png" width="450px"/>
</div>


文本事件线（Document timelines）的表现并不理想，对于开始点的得分略高于持续时长，该模型仅基于每个slider 来确定相对的事件时间，在构建整条文本事件线时效果不理想可以理解，需要加入更多信息。