---
layout: post
title: 2020《Paired Representation Learning for Event and Entity Coreference》
categories: [Note, Entity Coreference, Event Coreference]
description: 通过成对特征学习解决事件与实体同指
keywords: Joint Model,Cross-document,Coreference Resolution
mathjax: true
original: true
author: 黄琮程
authorurl: https://github.com/Ottohcc
---

> 2020《通过成对特征学习解决事件与实体同指》[(Paired Representation Learning for Event and Entity Coreference)](https://arxiv.org/pdf/2010.12808.pdf) 的阅读笔记

## 一、问题

本文研究了事件和实体的共指消解问题。共指消解通常被建模为二元分类问题，且一般分为如下两个步骤：首先，学习并得到每个（事件或实体）mention 的特征（向量），然后对每两个 mention 进行分类。其中，获取高质量的 mention 特征是至关重要的。作者认为前人的工作并没有获得强有力的 mention 特征，主要原因为以下两点：

+ 点方式（Point-wise）的特征学习

  大多数的工作试图通过仅仅从每个独立事件句提取特征来学习 mention 向量。在判断是否同指时是以同指对的方式进行判断，而不是单个 mention，并且，在不同的上下文中，两个 mention 可以成为同指事件，也可能不同指。

+ 非结构化（Unstructured）的特征学习

  事件 mention 会包含一些论元，大多数工作会将所有这些论元编码并拼接成一个向量，然后比较每对论元向量。用一个单一的拼接向量来表示论元，会让机器失去进行细粒度推理的机会，也不容易解释模型的预测。某些论元的不匹配可能是决定性的，或者比其他论元的不匹配更具决定性。例如，“四川省”与“神户”不匹配时，可以直接认定这两个事件是不同指的。
  
  <div style="text-align: center;">
  <img src="/images/blog/Paired-Representation-Learning-for-Event-and-Entity-Coreference-1.png" width="400px"/>
  </div>

## 二、方案

为解决以上问题，本文提出如下方式。

### 成对特征学习

本文把每一对 mention 对而不是一个单一的 mention 作为特征学习的对象。

具体来说，本文将把两个事件句作为一个完整的序列，并将其输入到 RoBERTa 系统中。RoBERTa 将两句话中的每个 token（包括 mention span）在编码时就能够互相进行比较。作者认为这比在分别编码成一个向量之后比较两个 mention 的 token 要好。本文将这种**成对表示学习**应用于事件和实体同指消解任务。然后二元分类器将采用这对上下文化的 mention 中的每对论元信息判断最后结果。

### 模型

本文使用的是 gold 的事件句，并没有事件抽取这一步。

对于每一个事件句使用 SRL 模型（https://demo.allennlp.org/semantic-role-labeling/）去抽取预测论元包括施事者、受施者、地点、时间。

然后将两个事件句用 [SEP] 相隔，输入 RoBERTa，得到一对事件句的编码。

<div style="text-align: center;">
<img src="/images/blog/Paired-Representation-Learning-for-Event-and-Entity-Coreference-2.png" width="400px"/>
</div>


如图在得到事件对编码后，首先找到触发词的对应位置，将其词向量相加，压缩成一维后相拼接，得到一对触发词的表示。

$$v_t(i,j)=[v_t^i, v_t^j,v_t^i\circ v_t^j]$$

再找每个论元的位置，并将每个论元的词向量相加，压缩成一维。对于每一种论元，通过以下式子得到每对论元的表示 $$\rm {role} \in \{arg0, arg1, loc, time\}$$，再将上述得到的向量经过一个多层感知机（$$\rm MLP_1$$）得到每对论元的表示$$a_{\mathrm {role}}(i,j)$$，之后再将上述的所有表示拼接起来，得到一对事件 mention 的最后表示 $$v(i,j)$$。

$$v_{\mathrm {role}}(i,j)=[v_{\mathrm {role}}^i, v_{\mathrm {role}}^j,v_{\mathrm {role}}^i\circ v_{\mathrm {role}}^j]$$

$$a_\mathrm{role}(i,j)=\mathrm{MLP}_1(v_\mathrm {role}(i,j))$$

$$v(i,j)=[v_t(i,j),a_{arg0},a_{arg1},a_{loc},a_{time}]$$

如果是实体同指，则 $$v(i,j)=v_t(i,j)$$。

<div style="text-align: center;">
<img src="/images/blog/Paired-Representation-Learning-for-Event-and-Entity-Coreference-3.png" width="400px"/>
</div>


如图，再将上一步中得到的事件 mention 对表示输入到 $$\mathrm {MLP}_2$$ 中，得到最后得到一个关于是否同指的二分类答案。

$$p(i,j)=\mathrm {MLP}_2(v(i,j))$$

## 三、结果及分析

### 跨文档（ECB+）

<div style="text-align: center;">
<img src="/images/blog/Paired-Representation-Learning-for-Event-and-Entity-Coreference-4.png" width="350px"/>
</div>


以上是 ECB+ 上的数据分布。对于 ECB+ 作者首先使用 SRL 系统对序列中的论元进行标注（训练时使用 gold 数据），然后使用 K-Means 算法对文章主题进行聚类（训练使用 gold topic），最后使用前文所描述的模型进行打分判断。

<div style="text-align: center;">
<img src="/images/blog/Paired-Representation-Learning-for-Event-and-Entity-Coreference-5.png" width="600px"/>
</div>


### 文档内（KBP）

<div style="text-align: center;">
<img src="/images/blog/Paired-Representation-Learning-for-Event-and-Entity-Coreference-6.png" width="350px"/>
</div>


以上是 KBP 语料库上的数据分布。在本实验中，事件 mention 使用的是 gold 的。也是先使用 SRL 系统对序列中的论元进行标注，然后对每篇文章中的每一对事件评分，然后得到结果。

<div style="text-align: center;">
<img src="/images/blog/Paired-Representation-Learning-for-Event-and-Entity-Coreference-7.png" width="550px"/>
</div>

