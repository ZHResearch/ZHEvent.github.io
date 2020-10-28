---
layout: post
title: 2019 NAACL《Improving Event Coreference Resolution by Learning Argument Compatibility from Unlabeled Data》
categories: [Note, Event Coreference]
description: 从未标记数据中学习论元兼容性改进事件同指识别
keywords: Argument Compatibility, Coreference Resolution
mathjax: true
original: true
author: 黄琮程
authorurl: https://github.com/Ottohcc

---

> 2019 NAACL《从未标记数据中学习论元兼容性改进事件同指识别》[(Improving Event Coreference Resolution by Learning Argument Compatibility from Unlabeled Data)](https://www.aclweb.org/anthology/N19-1085/) 的阅读笔记

## 一、问题

当两个论元在句子中的地位相同、描述的具体程度相同，但是对应描述的现实世界的实体不同则称它们是不兼容的，其余情况皆为兼容。

例：对于时间论元来说，”2012年“和”星期三“是兼容的，因为这两个时间描述的具体程度不一样。而“2012年”与“2005年“就是不兼容的。

其他句内地位（包括施事者、受施者、地点等）也如此。

作者认为如果两个事件中存在论元不兼容的情况发生，那么它们一定不是同指的。尽管论元兼容性很重要，但由于缺乏足够的标记数据，将论元兼容性引入事件同指是一项具有挑战性的工作。

许多现有的工作都将论元抽取作为上游组件，并设计论元特征来实现事件共指识别中的论元兼容性识别。然而，在每个步骤中引入的错误会通过这些解析器传播，并极大地阻碍了它们的性能。

本文将论元兼容性知识推理迁移到事件同指的框架中，具体地说是采用交互式推理网络作为模型结构。

+ 训练一个模型——确定事件对的对应论元是否兼容。

+ 为了将论元兼容性的知识迁移到事件同指中，以前一步训练的模型为起点，训练它来确定两个事件提及是否在人工标注的事件共指语料库上是共指的。

+ 迭代地重复上述两个步骤，使用学习的同指模型来重新标记论元兼容性（同指论元一定兼容），重新训练该模型以确定论元兼容性，再使用生成的预训练模型学习事件同指识别模型。本质上是相互引导论元兼容性确定任务和事件同指识别任务。

## 二、方案

<div style="text-align: center;">
<img src="/images/blog/Event-Coreference-Resolution-by-Learning-Argument-Compatibility-1.png" width="650px"/>
</div>


本文的模型训练分为两步：1.预训练论元兼容性模型；2.微调该模型成为事件共指识别模型

### 1.论元兼容性识别

在此部分数据不是 gold 的，来自于语料库 English Gigaword 包含五处不同资源的新闻。

**Related trigger extraction**

当两个事件 mention 中的触发词对是相关的时，我们无法通过触发词来判断两个 mention 是否同指，此时，论元兼容性会起到相对关键的作用。

所以本文作者首先分析事件同指语料库，提取训练数据中同指次数超过 10 次的触发词对，并以有着相关触发词的事件 mention 对来训练论元兼容性模型。

**Compatible samples extraction**

为了获得更优质的兼容性样本，对于以上提取出的事件 mention 对（来自同一篇文章），作者加了五条限制。

其中第一条到第四条是对于（DATE、PERSON、NUMBER、LOCATION）这四种不同的论元短语，使用命名实体识别技术（NER）进行抽取，如果存在，必须要有一个重复的单词，如果在某一事件 mention 中有两个或以上相同类别的论元时则取其中距离触发词近的那一个。

第五条限制是除了虚词外，上下文的单词重复率不能超过 30%。

目的是既要排除一眼就能看出来不一样的（一到四），也要排除一眼看起来就一样的（五）防止模型学习时认为单词重复率越高兼容/同指率越高。

**Incompatible sample extraction**

对于非兼容性样本，本文加了两条限制。

首先，两端文本的产生时间必须间隔一个月以上，防止这两个事件 mention 是同指的；

其次，除了触发词和虚词，两段事件 mention 必须有一个以上的重复单词，也是为了防止模型学习的结果基于两段 mention 中是否存在重复单词。

**Argument compatibility classifier**

本文将在以上抽取出的数据集上训练一个二分类器，即为论元兼容性分类器，具体模型在第 4 点中阐述。

### 2.事件同指

这部分中训练数据都是 gold 的，是 KBP 2015 和 KBP 2016，且将 KBP 2017 作为测试。

**Event Mention Detection**

在事件检测任务中，作者训练一个单独的模型，抽取出事件 mention 以及对应的事件类型。将事件检测视为一个多分类问题。规定触发词一定是在上一步的训练集中出现过的词，且不考虑多词触发词的情况（触发词只能是一个词）。

对于每一句话，首先串联这句话的 word embedding 和 character embedding，将结果输入到 biLSTM 层，最后通过一个 inference 层预测结果。

**Mention-Pair Event Coreference Model**

在之前的论元兼容性识别模型基础上，进行微调。首先对每对事件 mention 进行打分，再将同指于同一事件的 mention 生成同指链。

### 3.迭代重标记训练

最开始那个兼容性样本尽管经过一些规则限制，但一定会有很多噪声。作者提出以下的优化算法。

首先计算在兼容性样本中事件 mention 的同指可能性，如果可能性高于 $$\theta_M$$ 就被加入新的兼容性样本中，同时，如果同指可能性低于 $$\theta_m$$ 就将该事件 mention 对加入原有的非兼容性样本。然后再次训练论元兼容性识别模型。

在本文中 $$\theta_M$$ 取 0.8，$$\theta_m$$ 取 0.2。

### 4.具体模型

<div style="text-align: center;">
<img src="/images/blog/Event-Coreference-Resolution-by-Learning-Argument-Compatibility-2.png" width="650px"/>
</div>


**输入**

$$m_a = \{w_a^1,w_a^2,...,w_a^N\}$$

$$m_b = \{w_b^1,w_b^2,...,w_b^N\}$$

输入是是两个 mention 包括触发词加其上下文，上下文的范围是一个 n-word 窗口，n 为 10。

**Embedding 层**

包含 Word embedding（GloVe）和 Character embedding（CNN）。

在得到的输出中，Exact match 指得是给定的 token 是否在两个 mention 中都存在，Triger 指的是该 token 是否是触发词。

**编码层**

采用一个 biLSTM 进行编码：

$$\mathbf h_a^i = \mathrm {biLSTM}(\mathrm{emb(}w_a^i),\mathbf h_a^{i-1})$$

$$\mathbf h_b^i = \mathrm {biLSTM}(\mathrm{emb(}w_b^i),\mathbf h_b^{i-1})$$

**Interaction 层**

$$I_{ij}=\mathbf h_a^i\circ\mathbf h_b^j$$

然后通过一个多层的卷积神经网络来提取事件对特征向量 $$f_{ev}$$。

**推理层**

在第一个模型（论元兼容性模型）中采用一个全联接层，输出是一个二分类。

在第二个模型（事件同指识别模型）中加了一个辅助向量 $$f_{aux}$$，包括两个句子之间的距离信息和两个触发词之间word embedding 的差别。之后也是一个全联接层，输出一个多分类。

## 三、结果及分析

### 1. 结果

<div style="text-align: center;">
<img src="/images/blog/Event-Coreference-Resolution-by-Learning-Argument-Compatibility-3.png" width="800px"/>
</div>

表中 standard 代表直接训练同指模型，transfer 代表使用迁移学习（加入了论元兼容性特征），表中显示，迁移学习效果不错。然而多次迭代从第三次开始，效果就微乎其微了，作者推断是因为训练集合的标签在之后改变不大。

#### 2.分析

<div style="text-align: center;">
<img src="/images/blog/Event-Coreference-Resolution-by-Learning-Argument-Compatibility-4.png" width="700px"/>
</div>


当句中包含显式的时间、地点、名字等信息时，作者的模型的效果往往会比较好（训练集的数据是基于 NER 的）。

当不包含显式的命名实体时，本文模型的效果就没那么好，但是作者认为，其效果优于那些基于论元抽取及实体同指识别的模型的效果。

概括性的事件是最难以预测的，在本文中的导致的最多的错误。





