---
layout: post
title: 2020 ACL《Discourse as a Function of Event——Profiling Discourse Structure in News around the Main Event》
categories: [Note, Event Coreference]
description: 分析新闻文章与其主事件的篇章结构
keywords: Coreference Resolution,Discourse Structure
mathjax: true
original: true
author: 黄琮程
authorurl: https://github.com/Ottohcc
---

> 2020 ACL《分析新闻文章与其主事件的篇章结构》[(Discourse as a Function of Event: Profiling Discourse Structure in News Articles around the Main Event)](https://www.aclweb.org/anthology/2020.acl-main.478) 的阅读笔记

## 一、问题

虽然学术界对新闻篇章进行了大量的理论研究，但对新闻篇章结构的计算建模和自动构造的研究却很少。

本文提出了一个新的任务和一个新语料库，围绕新闻文章的主事件对新闻文章内容进行分类。

这个新的文章语料库包含 802 篇新闻文章（包含 18155 个句子），这些文章提取于三种来源（纽约时报（102 篇来自 KBP 2015、 100 篇来自 NYT corpus）、新华社、路透社）包含四种新闻类型（商业、犯罪、灾祸、政治）。本文对这个语料库中每一个句子进行标注，分成八种种类。

然后本文提供了一些方法来识别这些句子的种类，发现使用一个基础的神经网络就可以有一个不错的结果。通过对文章中句子之间的联系进行建模，并根据文章中的主要事件识别句子类型，可以进一步提高句子分类的性能。

作者认为他们所做的这两个工作对很多下游任务都有用，并以事件同指消解为例做了实验。作者分析了事件共指链在不同句子类型上的生命周期和传播广泛程度，并设计了约束条件来捕获事件共指解析的几个显著特征。

## 二、方案

### 1.语料库建立

<div style="text-align: center;">
<img src="/images/blog/Profiling-Discourse-Structure-in-News-Articles-around-the-Main-Event-1.png" width="700px"/>
</div>


在该语料库中，每句话都会被定义为一个种类，分为以下几类型：

#### 主要内容（Main Contents）

（1）主要事件（Main Events）

表示在一篇文章中最主要的与文章主题相关的事件，其他种类的句子都是为了对主要事件进行解释或补充。

（2）结果（Consequences）

指主事件发生后造成的结果，一般在时序上与主事件的发生时间有重合或在主事件发生之后。

#### 上下文通知内容（Context-informing Contents）

（3）先前事件（Previous Event）

指的是发生在主事件之前的真实事件，一般是导致主事件发生的原因或前提。

（4）当前上下文（Current Context）

涵盖提供主事件上下文的所有信息，是与主事件同时发生的事件或是可以帮助理解主事件的时代、政治背景。它们在时间上与主要事件同时发生或是描述主事件发生的背景环境。

#### 其他有帮助的内容（Additional Supportive Contents）

（5）历史事件（Historical Event）

在主事件发生的数月或数年以前，一般对现在的主事件发生环境有所影响，但不是直接导致主事件发生的原因。

（6）轶事（Anecdotal Event） 

包括具有难以验证的特定参与者的活动，具有不确定性，可能是虚构的情况或对一个未知的人的事件的个人描述。

（7）评价（Evalution）

表式有影响力的人对主事件的评估或看法。

（8）推测（Expectation）

主要表示主事件可能的结果。本质上是观点，但包含着更强烈的含义，一般是作者试图预测未来可能发生的事件。

#### 是否为言语

如果一句话是直接或间接引用他人的陈述的句子，并将其标记为言语，是独立于上述八种之外的注释。

最后标注结果如图：

<div style="text-align: center;">
<img src="/images/blog/Profiling-Discourse-Structure-in-News-Articles-around-the-Main-Event-2.png" width="750px"/>
</div>


### 2.对篇章概述进行神经网络建模

<div style="text-align: center;">
<img src="/images/blog/Profiling-Discourse-Structure-in-News-Articles-around-the-Main-Event-3.png" width="400px"/>
</div>


首先，对输入的每个句子做 biLSTM（ELMo）得到词向量，然后做 Attention 得到句子向量。

对句子向量再做 biLSTM，得到含有上下文信息的句子向量，对句子向量做Attention 得到文章向量。

句子类型是根据主要事件来决定的。虽然句子水平的 biLSTM（ELMo）增加了句子的局部语境表征，但仍然不知道整篇文章的主题。因此，本文计算文章向量和句子向量之间的元素乘和差来度量它们的相关性，并进一步将乘和差与句子向量连接起来，以获得用于预测其句子类型的最终句子表示。

<div style="text-align: center;">
<img src="/images/blog/Profiling-Discourse-Structure-in-News-Articles-around-the-Main-Event-4.png" width="750px"/>
</div>


Basic Classifier 只是用了单词级别的 biLSTM 得到句子向量，然后直接接全联接层进行分类。

Document LSTM 相较于 Basic 加了句子级别的 biLSTM。

Document encoding 指的是加入了文章向量和句子向量之间的元素乘和差等信息，即上文模型中的最后形态。

Headline 指的是将上述的文章向量换成标题句的向量表示（也从句子级别的 biLSTM 中得到），可以看到这一步对于判断主事件有很大的帮助。

最后加入的是 CRF 层，分别为细粒度（八种内容类型）和粗粒度（三种大的内容类型）之间的依赖性进行建模，并输出最后的类别预测。

CRF 很复杂，作用是对输出进行自动化的约束。

> 具体可以见 https://blog.csdn.net/Suan2014/article/details/89419283

<div style="text-align: center;">
<img src="/images/blog/Profiling-Discourse-Structure-in-News-Articles-around-the-Main-Event-5.png" width="450px"/>
</div>



### 3.运用于事件同指消解（新闻）

<div style="text-align: center;">
<img src="/images/blog/Profiling-Discourse-Structure-in-News-Articles-around-the-Main-Event-6.png" width="450px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/Profiling-Discourse-Structure-in-News-Articles-around-the-Main-Event-7.png" width="460px"/>
</div>


表 7 表示各种类句子中的事件是单独事件（无同指）的概率。

表 8 表示各种类句子中包含主事件（假定标题句对应事件为主事件）的概率，M1 仅为 58% 的原因作者给出两条：1.主事件抽取不准确，本身就是一个挑战；2.KBP 语料库中有些主事件句中的事件不在其规定的 7 种事件类型内，所以被标注为无事件。

表 9 表示每种句子包含的同指事件所在的同指链上的事件全部在该种句子中的概率。

作者使用上文中提到的模型（ELMo + biLSTM）来预测句子类型。并且建立一个非同指事件句分类器，结合上述模型输出的句子向量和文章向量，加上全联接层，输出是否是非同指事件句（类似迁移学习）。

然后利用上述所得的句子类型和是否是非同指事件句，建立 ILP 限制。

事件抽取和事件对打分器和上次的论文中的方法一样。

<div style="text-align: center;">
<img src="/images/blog/Profiling-Discourse-Structure-in-News-Articles-around-the-Main-Event-8.png" width="700px"/>
</div>


作者认为自己的工作对于时间时序和因果识别也有帮助。

#### 附录

附录部分介绍了 ILP 限制的具体方法。

**Base Function：**

$$\Theta_B = \sum_{i,j\in\Lambda}-log(p_{ij})x_{ij}-log(1-p_{ij})(\neg x_{ij})$$

$$s.t. x_{ij}\in\{0,1\}$$

$$\Lambda$$ 代表所有可能的事件 mention 对。

**非同指事件句限制：**

$$\Theta_S=\sum_{i\in \lambda,j\in \lambda,i\or j \in S}x_{ij}$$

$$\Theta_N=-\sum_{i\in \lambda,j\in \lambda,i\or j \in N}x_{ij}$$

 $$\lambda$$ 代表一篇文章中的所有事件 mention，

$$S、N$$  分别代表是否是同指事件句。

**句子类型限制**

根据之前的统计，如果事件链以 C1-D4 句子类型开始，则它往往在相同的句子类型或主事件句中有同指事件。

主事件句限制:

$$\Theta_M=-\sum_{i\in \xi_H, j\in\xi_M}x_{ij}$$ 

$$H、M$$ 分别代表的是标题句和主事件句中的 mention 集合。

$$\Theta_C=-\sum_{i\in \xi_H, j\in\xi_R}x_{ij}$$ 

$$R$$ 表示的是 M2、C1、C2 这几种句子类型中的 mention 集合。

同指链上的事件全部为一种类型的句子限制：

$$\Theta_L = \sum_{i\in\xi_T}Y_i$$

$$Y_i\ge\Gamma_i- M \gamma_{ij}$$

$$\gamma_i = \sum_{k\notin \xi_T,i\in\xi_T}x_{ki}\ ;\ \Gamma_i = \sum_{i\in \xi_T,j\notin (\xi_M \cup \xi_T )}x_{ij}$$

$$T$$ 表示的是 C1-D4 这几种句子类型中的 mention 集合。

$$\gamma_i$$ 表示在 mention $$i$$ 前面与 $$i$$ 同指的 mention 数。$$\Gamma_i$$ 表示在 mention $$i$$ 后面与 $$i$$ 同指的 mention 数。

目的是不鼓励以 C1-D4 内容类型句子开头的事件链与其他非主类型中的后续事件 mention 形成共指。