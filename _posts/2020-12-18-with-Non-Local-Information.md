---
layout: post
title: 2020 AACL-IJCNLP《Event Coreference Resolution with Non-Local Information》
categories: [Note, Event Coreference]
description: 基于非局部信息的事件同指消解
keywords: Joint Model,Cross-document,Coreference Resolution
mathjax: true
original: true
author: 黄琮程
authorurl: https://github.com/Ottohcc
---

> 2020 AACL-IJCNLP《基于非局部信息的事件同指消解》[(Event Coreference Resolution with Non-Local Information)](https://www.aclweb.org/anthology/2020.aacl-main.66.pdf) 的阅读笔记

## 一、问题

现有的事件共指解决方案主要利用的还是局部上下文中提取的信息。

假设非局部信息也可以用于事件共指消解，本文提出了两个目前最先进的事件同指联合模型的扩展，包括：

（1）一个有监督的篇章主题模型，基于全局上下文，来提升触发词识别

（2）一个预处理模块，使用基于显著实体计算的篇章语境，舍弃不可能出现的事件先行事（antecedent）的同指

## 二、方案

### 数据集设置

#### 英语数据集

**训练**

使用 LDC2015E29, E68, E73, E94 和 LDC2016E64 作为训练。训练集中共含 817 篇文章（646 训练，171调参），22894 个事件，13146 条同指链。

**测试**

使用 KBP 2017 作为测试。测试集中共含 167 篇文章，4375 个事件，2963 条同指链。

#### 中文数据集

**训练**

使用 LDC2015E78, E105, E112 和 LDC2016E64 作为训练。训练集中共含 548 篇文章（438 训练，110 调参），7388 个事件，5526 条同指链。

**测试**

使用 KBP 2017 作为测试。测试集中共含 167 篇文章，3884 个事件，2558 条同指链。

### 模型

构建候选触发词集合：训练集中作为触发词出现过至少一次的所有单字或多字的名词和动词。

（目的：方便对文章中的候选事件 mention 进行联合预测。）

对于事件候选集中的每一个事件 mention 需要做一下三点预测：

+ 判断是否是事件，及其事件类型
+ 该 mention 所引出的话题
+ 它的先行事（antecedent）

基于以上的想法，对于每一篇文章定义三种输出变量：

+ $$\mathbf s = (s_1,...,s_n)$$  其中 $$s_i$$ 代表 mention i 的事件类型（18 种 subtype 或 none）
+ $$\mathbf c = (c_1,...,c_n)$$  其中 $$c_i \in \{1,...,i-1,\mathrm {NEW}\}$$  代表 mention i 的先行事（antecedent）的 ID。如果 i 是一个同指链的开头，这个值为 NEW。
+ $$\mathbf t = (t_1,...,t_n)$$   其中 $$t_i$$ 是 19 个数中的一个值，每一个只代表一个主题，其中主题与上面定义的事件子类型标签一一对应。虽然一一对应但是并不一样， $$\mathbf t$$ 值会受到本文定义的有监督模型所影响。

本文模型的可能性分布如下：

$$p(\mathbf {s,c,t}|x;\Theta \propto \mathrm {exp}(\sum_i\theta_if_i(\mathbf{s,c,t},x)))$$

其中，$$\theta_i\in\Theta$$ 代表与特征函数 $$f_i$$ 相关的权值，$$x$$ 代表输入的文章。

#### 事件特征表示

每个候选触发词 m 都使用以下的特征来表示：m 的单词、m 的词根（lemma）、m 的 bigrams。

将 m 的词根（lemma）分别与以下每种特征进行配对（拼接）：在语法上最接近 m 的实体的头单词，语篇上最接近 m 的实体的头单词，句法上最接近 m 的实体的实体类型，以及语篇上最接近 m 的实体的实体类型（使用基于 CRF 的实体抽取模型）。

对于触发词是动词的 mention，使用头词及其主语和宾语的实体类型（Stanford CoreNLP 抽取）作为特征。

对于触发词是名词的 mention，使用启发式提取施事者和受施者作为特征。

将所有特征进行拼接作为特征。

#### 主题模块

每个候选触发词使用 19 个特征表示，这些特征对应于 19 个主题标签。特征值来自于 LabeledLDA 模型的输出，分别对应候选触发词属于相应主题的概率。

本文使用 Mallet toolkit 到训练文章中，它会学习到每篇文章的主题分布 $$\beta$$。使用文章中具有区别文章主题能力的候选触发词及其上下文代表训练集中的每篇文章。

为了得到具有区别文章主题能力的候选触发词及其上下文，统计

$$P(w_i|m_j,v_k)\mathrm {log} \cfrac {P(w_i|m_j,v_k)} {P(w_i|m_j,\neg v_k)}$$

$$w_i$$ 代表在词表的第 i 个词，$$m_j$$ 代表第 $$j$$ 个候选触发词，$$v_k$$ 代表第 k 类子类（subtype）。

如果单词 $$w_i$$ 经常与 $$v_k$$ 子类型的 $$m_j$$ 一起出现，并且很少与其他子类型一起出现，那么 $$w_i$$ 相对于 $$v_k$$ 子类型的候选触发词 $$m_j$$ 具有较高的相关度。

采用前 125 个词成为具有区别文章主题能力的候选触发词。

训练后，得到 LabeledLDA model。对于每一个测试样例，首先判断其文章主题可能性分布，然后使用贝叶斯规则计算给定的主题的后验分布

$$P(z|m)\propto P(m|z :\beta)P(z)$$

$$P(z)$$ 代表文章为主题 z 的概率

$$P (m|z : β)$$  文章主题为 z 时，候选触发词为 m 的概率（即候选触发词 m 和文章主题 z 的相关度，之前已算出）。

$$P(z|m)$$ 是指给定一个候选触发词其文章主题为 z 的概率。

#### 同指模块

本文使用两种类型的特征，来表示待解决事件 $$m_j$$ 的候选先行事。

第一种类型：候选先行事是 NULL，即 $$m_j$$ 开启一条新的同指链。则此时特征为 $$m_j$$ 的词、$$m_j$$ 的词根、一个通过将 $$m_j$$ 的词根与 $$mj$$ 前面的句子数配对而创建的一个联合特征，以及另一个通过将 $$m_j$$ 的词根与文章中 $$m_j$$ 前面的事件 mention 的个数配对而创建的一个联合特征。

第二种类型：候选先行事不是 NULL。此时特征包括 $$m_i$$ 的词，$$m_i$$ 的词根，$$m_i$$ 和 $$m_j$$ 是否有相同的词根，以及以下特征：（1）$$m_i$$ 的词与 $$m_j$$ 的词对，（2）$$m_i$$ 的词根与 $$m_j$$ 的词根对，（3）$$m_i$$ 和 $$m_j$$ 之间的句子距离与 $$m_i$$ 的词根和 $$m_j $$ 的词根的配对（pair），（4） $$m_i$$ 和 $$m_j$$ 之间的 mention 距离与 $$m_i$$ 引理和 $$m_j$$ 词根配对（pair），（5）由 $$m_i$$ 和 $$m_j$$ 的主语及其词根组成的四元组，（6）由 $$m_i$$ 和 $$m_j$$ 的宾语及其词根组成的四元组。

本文引入了一个事件共指消解的预处理模块，在该组件中，我们根据篇章级上下文对不太可能是其正确先行事的事件mention 的候选先行事进行删减（对于不属于同一篇章级上下文的事件 mention 进行剪枝）。

本文的目的是使用具有显著（salient）篇章信息的实体对每一个事件 mention 进行篇章级的上下文编码，因此需要计算关于每个事件 mention m 的每个实体（E），的显著性得分，以如下公式获得。

$$\sum_{e\in E}g(e)\times decay(e)$$

公式中 e 是出现在与 m 相同的句子中，或者出现在其前面的句子中的实体 mention（同一实体可能有多种mention 形式）。

g(e) 是一个基于实体语法结构的打分函数：如果 e 是主语，则记 4 分，如果 e 是宾语则记 2 分，其余情况记 1 分。

decay(e) 是一个衰减函数，具体为 $$0.5^{dis}$$，dis 代表实体 e 和 mention m 之间的句子距离。

过程中使用的是 CoreNLP 的命名实体识别技术，语法结构采用 CoreNLP 的 sdp 技术识别。

对于出现在先行事 c 和 mention m 的篇章上下文中的每个实体 E，首先计算 E 的 SSR（Salience score ratios）即 $$E_m$$ 及 $$E_c$$ 的显著性得分的比值 （如果这个比率小于1，取其倒数）。然后，对于每个（c，m）对，创建了五个特征来编码其 SSR 属于这五个区间的实体数量：[1,1]，（1,2]，（2,3]，（3,4]，（4,5]和[5，inf]）。如果 c 和 m 的语篇语境往往更相似，如果他们有更多的实体在较小的区间里。

对于候选先行事 c 的篇章上下文中每个实体的提及 em1 和 m 的篇章上下文中每个实体的提及 em2，创建了一个将em1 的头部和 em2 的头部配对的词汇特征。

本文使用了一个对数线性模型来训练这个打分器。

在训练之后，应用这个打分器来筛选掉测试文档中每个事件除了得分前 k 候选先行事的其他 mention。这 k 个候选先行事以及 NULL 将通过事件共指模型进行排序，并选择排名最高的候选项作为所该事件 mention 的先行事。k 是一个超参数，在开发集上对其进行调整。

<div style="text-align: center;">
<img src="/images/blog/with-Non-Local-Information-1.png" width="500px"/>
</div>


该部分的结果如图所示。

### 联合学习

#### 跨任务特征

**触发词识别和同指**

在子类型变量 $$s_i$$ 和 $$s_j$$ 上定义的特征只有在当前的 $$m_j$$与文章前面的 $$m_i$$ 被判为同指时才会被激活。

这些特征是：

（1）$$m_i$$ 和 $$m_j$$ 的子类型对

（2）$$m_j$$的子类型和 $$m_i$$ 的词对

（3）$$m_i$$ 的子类型和 $$m_j$$ 的词对

**触发词识别和主题模型**

将每个候选事件mention 的事件子类型、主题和触发词词根拼接。以二进制因子来控制是否激活。

**主题模型和同指**

与触发词识别和同指部分相同。

<div style="text-align: center;">
<img src="/images/blog/with-Non-Local-Information-2.png" width="500px"/>
</div>


#### 训练

最大化目标函数：

$$L(\Theta)=\sum_{i=1}^dlog\sum_{\mathbf c^*\in A(C_i^*)}p'(\mathbf {s_i^*,t_i^*,c^*}|x_i;\Theta)+\lambda||\Theta||_1$$

其中 d 是训练集的文章数，$$x_i$$ 代表第 i 篇文章的输入内容，$$\mathbf {s^*_i}$$ 表示 gold 的触发词标注信息，$$\mathbf {t^*_i}$$ 代表由 LabeledLDA 模型推理出的对应文章主题，$$\mathbf {C^*_i}$$ 表示 gold 的同指标注信息，

$$p'(\mathbf {s_i^*,t_i^*,c^*}|x_i;\Theta) \propto p(\mathbf {s_i^*,t_i^*,c^*}|x_i;\Theta)\exp[\alpha_sl_s(\mathbf {s,s^*})+\alpha_tl_t(\mathbf {t,t^*})+\alpha_sl_s(\mathbf {c},C^*)]$$

$$l_s、l_t、l_c$$ 是三个任务分别的损失函数。$$\alpha_s、\alpha_t、\alpha_c$$ 是对应的权重。


## 三、结果及分析

<div style="text-align: center;">
<img src="/images/blog/with-Non-Local-Information-3.png" width="750px"/>
</div>


可以看出两部分都有作用。

<div style="text-align: center;">
<img src="/images/blog/with-Non-Local-Information-4.png" width="750px"/>
</div>


上表解释了为什么篇章级上下文有作用，在过滤了大量无关信息的同时保留了大部分有用的先行事件。

