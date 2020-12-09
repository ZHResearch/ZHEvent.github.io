---
layout: post
title: 2019 IJCNN《Selective Expression For Event Coreference Resolution on Twitter》
categories: [Note, Event Coreference]
description: 针对 Twitter 事件同指消解的选择性表达
keywords: Joint Model,Cross-document,Coreference Resolution
mathjax: true
original: true
author: 黄琮程
authorurl: https://github.com/Ottohcc
---

> 2019 IJCNN《针对 Twitter 事件同指消解的选择性表达》[(Selective Expression For Event Coreference Resolution on Twitter)](https://www.researchgate.net/publication/325556729_Selective_Expression_For_Event_Coreference_Resolution_on_Twitter) 的阅读笔记

## 一、问题

现有的事件共指消解研究中，常常需要从已有的 NLP 工具和各种知识库中获得丰富的语言特征集。这种方法限制了域的可伸缩性，并会传播由各种工具导致的错误。

论坛形式文章的事件同指是更加困难的，主要是因为有表情符号，URL，垃圾短信和拼写错误的文本等问题。

本文首先使用 BiLSTM 来获取句子级别的推特文本特征。

在一句话中可能有多个触发词，而对于不同的触发词，句中的每个词扮演的角色很可能是不一样的。本文采用了一种选择门结构，根据触发词从句子级特征中过滤掉不重要或不相关单词的信息，得到事件 mention 特征。

然后，本文使用 attention 机制，对每个句子级向量做 attention，得到潜在（latent）向量。

最后，本文将潜在（latent）特征、mention 特征和局部特征串联起来，作为最终每个 mention 的特征进行通知判断。

本文作者建立了一个新的语料库 EventCoreOnTweet (ECT)。

## 二、方案

本文的模型分为两部分，mention 特征生成和同指识别。

### mention 特征生成

首先针对推特文本特点过滤表情符号，URL 等信息。

<div style="text-align: center;">
<img src="/images/blog/Selective-Expression-For-Event-Coreference-1.png" width="600px"/>
</div>


对被处理的文本本文使用 word2vec 做词嵌入，对于 mention 中的各词如图进行距离编码，触发词位置的编码为 0。

首先，通过 LSTM 获得 Sentence/mention 级别的向量。

$$h_{f,i}=LSTM_f[x_i,h_{f,i-1}]$$

$$h_{b,i}=LSTM_b[x_i,h_{f,i+1}]$$

$$h_i=[h_{f,i},h_{b,i}]$$

$$Sent_{level}=\{h_0,h_1,...,h_n\}$$

$$Ment_{level}=[h_{f,m},h_{b,0}]$$

mention 级别的向量取的是 LSTM 中正向和反向的最后一个向量的拼接。

然后经过一个选择门结构，根据触发词从句子级特征中过滤掉不重要或不相关单词的信息

$$R_c=h_i*Ment_{level}$$

$$\alpha_i=tanh(W_s\cdot R_c+b_s)$$

$$Select_i=\alpha_i*h_{i}$$

$$Select=(Select_0,Select_1,...,Select_n)$$

接下来是一个 Attention 机制

$$u_i=V_a^Ttanh(W_aSelect_i+b_a)$$

$$\beta_i=\cfrac {e^{u_i}}{\sum_{i=1}^ne^{u_i}}$$

$$latent=\sum_{i=1}^n\beta_iSelect_i$$

最后，将 latent 和 mention 的向量拼接起来，得到最终的事件 mention 向量。

$$V_{em}=(latent,Ment_{level})$$

### 同指识别

<div style="text-align: center;">
<img src="/images/blog/Selective-Expression-For-Event-Coreference-2.png" width="600px"/>
</div>


首先得到事件对的向量表示

$$V_{pair}=(V_{em}^1,V_{em}^2,V^{1,2}_{local})$$

$$V_{local}^{1,2}=(V_W,V_D)$$

$$V_W$$ 是指两个 mention 中处罚词的重叠部分的 embedding。

$$V_D$$ 表示两条 twitter 发送的天数间隔。

$$V_{ds}=relu(W_{ds}\cdot V_{pair}+b_{ds})$$

$$Score = Softmax(W_{pro}\cdot V_{ds} + b_{pro})$$

使用交叉熵和 Adam 做参数更迭。

### 语料库

本文从推特的官方借口获取推特文本。

本文设置关键字（“特朗普”、“希拉里”、“总统大选”）来过滤垃圾信息和不相关的推文。本文使用 twitter nlp 来提取所有的触发词，提取后按频率排序，最后进行筛选，获取待标注语料。

注释者首先需要判断一条 tweet 是否提到某个事件。然后对于那些提到事件的 tweet，需要注释事件触发词和同指链索引。本文使用 Cohens Kappa 来衡量注释者之间的注释者之间的一致性。在标注事件时，两个注释者达到了0.78。在标注同指时，注释者达到了 0.84。最后选择了一致的 2994 条推特作为黄金标准语料库。

<div style="text-align: center;">
<img src="/images/blog/Selective-Expression-For-Event-Coreference-3.png" width="300px"/>
</div>


## 三、结果及分析

### 数据集设置

<div style="text-align: center;">
<img src="/images/blog/Selective-Expression-For-Event-Coreference-4.png" width="400px"/>
</div>


### 结果

<div style="text-align: center;">
<img src="/images/blog/Selective-Expression-For-Event-Coreference-5.png" width="600px"/>
</div>


