---
layout: post
title: 2019 ACL《Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution》
categories: [Note, Entity Coreference, Event Coreference]
description: 跨文档实体和事件同指消解的联合建模
keywords: Joint Model,Cross-document,Coreference Resolution
mathjax: true
original: true
author: 黄琮程
authorurl: https://github.com/Ottohcc
---

> 2019 ACL《跨文档实体和事件同指消解的联合建模》[(Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution Event Detection)](https://www.aclweb.org/anthology/P19-1409.pdf) 的阅读笔记

## 一、问题

为解决跨文档实体和事件同指解析，本文提出一种实体、事件的联合模型，同时提高了实体同指和事件同指的识别正确率，是第一个在数据集 ECB+ 上进行实体同指识别的模型。

本文通过结合各实体和事件的 mention、上下文（ELMo）、与其他 mention 的相互联系，提高了 Lee. 提出的联合模型的准确率。

## 二、方案

### 1.Span

结合单词级和字符级的特征。

单词级使用预训练 embedding（表达 event 时，使用 event mention 的 head word 的 embedding，表达 entity时，使用 span 中所有 word embedding 的平均值）。

字符级与单词级互补，使用 LSTM。

串联单词级和字符级的向量表示为： $$\vec{s}(m)$$。

### 2.Context (上下文)

使用 ELMo，取结果的 head word，将上下文向量表示为： $$\vec{c}(m)$$。

### 3.Semantic dependency to other mentions（对于其他指代的语义依赖）

对于一个给定事件 mention $$v_i$$ ，提取四个论元：$$Arg0$$，$$Arg1$$，$$location$$，$$time$$。如果 $$Arg1$$ 所在的 slot 与实体mention $$e_j$$  所在的 slot 相同，且存在  $$e_j$$  于实体簇 $$c$$ 中，则将 $$Arg1$$ 的向量设置为 $$c$$ 中所有 span 向量的平均值：

$$\vec{d}_{Arg1}(m_{vi}) = \frac{1}{|c|}\sum_{m\in c}\vec{s}(m)$$

否则，$$Arg1$$ 的向量为0。

$$\vec{d}_{Arg1}(m_{vi}) = \vec{0}$$

将上述四个论元的向量串联得：

$$\vec{d}(m_{vi}) = [\vec{d}_{Arg0}(m_{vi});\vec{d}_{Arg1}(m_{vi});\vec{d}_{loc}(m_{vi});\vec{d}_{time}(m_{vi})]$$

最后，一个 mention 的向量表示为含有上述三个特征的向量：

$$\vec{v}(m) = [\vec{c}(m);\vec{s}(m);\vec{d}(m)]$$

### 4.Scorer

<div style="text-align: center;">
<img src="/images/blog/joint-modeling-coreference-resolution-1.png" width="400px"/>
</div>

图中 Scorer 输入为：

$$[\vec{v}(m_i);\vec{v}(m_j);\vec{v}(m_i)\circ\vec{v}(m_j)]$$

圈乘代表的是按元素乘法。$$f(i,j)$$ 是一个 50 维的二进制向量，表示两个 mention 是否有同指的参数或谓词。

损失函数为二元交叉熵函数。

### 5.算法

本质上是进行聚类。

<div style="text-align: center;">
<img src="/images/blog/joint-modeling-coreference-resolution-2.png" width="600px"/>
</div>

## 三、结果及分析

### 1.结果

<div style="text-align: center;">
<img src="/images/blog/joint-modeling-coreference-resolution-3.png" width="650px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/joint-modeling-coreference-resolution-4.png" width="350px"/>
</div>

### 2.错误分析

<div style="text-align: center;">
<img src="/images/blog/joint-modeling-coreference-resolution-5.png" width="700px"/>
</div>

（其中 Patial argument coreference 主要源于相似事件发生时间不同）

### 3.分离分析

<div style="text-align: center;">
<img src="/images/blog/joint-modeling-coreference-resolution-6.png" width="350px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/joint-modeling-coreference-resolution-7.png" width="350px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/joint-modeling-coreference-resolution-8.png" width="350px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/joint-modeling-coreference-resolution-9.png" width="350px"/>
</div>

此处作者肯定了上下文对于同指识别的重要性；并且认为在脱离自身 span 向量及上下文的同时，仅靠语义相关性向量就获得相当程度的精度已经不易。
