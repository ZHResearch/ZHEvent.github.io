---
layout: post
title: 2019 EMNLP《An Improved Neural Baseline for Temporal Relation Extraction》
categories: [Note, Temporal Relation, Event Extraction]
description: 一种用于时序关系提取的改进神经
keywords: Joint Model, Temporal Relation, Event Extraction
mathjax: true
original: true
author: 李婧
authorurl: https://github.com/LevyLee18
---

> 2019 EMNLP《一种用于时序关系提取的改进神经》[(An Improved Neural Baseline for Temporal Relation Extraction)](https://www.aclweb.org/anthology/D19-1642/)的阅读笔记

## 一、目的

事件时序抽取任务是一项有挑战的自然语言理解任务。由于其语料库的大小和质量都十分有限而导致神经网络的方法无法很好地应用。

本文提出一种神经网络系统，使用常识编码器（Common Sense Encoder，CSE）来对抽取出的外部知识进行编码，以达到通过外部知识来加强事件时序关系分类的性能。

## 二、模型

整体模型图如下：

<div style="text-align: center;">
<img src="/images/blog/An-Improved-Neural-Baseline-for-Temporal-Relation-Extraction-model.jpg" width="750px"/>
</div>

### 1.常识编码器（CSE）

人们在判断两个事件的发生先后关系时常常需要借助一些时间连词如：when, until等等，这些词通常表达的时序关系都较为明确。然而，即使在没有这些明确的时间连词的情况下，人们也可以使用常识来很容易地进行判断。比如，我们知道大部分情况下attack（攻击）是发生在die（死亡）之前的。

此前，[Ning等人](https://www.aclweb.org/anthology/N18-1077/)已经尝试从外部知识中来获取类似的知识，他们从大的语料库中抽取出时序关系库TEMPROB，其中的数据形式为：*($$v_1$$, $$v_2$$, $$r$$)*，表示verb1和verb2有关系r。但是TEMPROB只是简单的计数模型，因此其对一些出现概率较小的事件对不敏感。比如：*(attack, die, before)*出现的概率远远大于*(ambush, die, before)*，而attack和ambush（伏击）是近义词，对于人类来说，可以很好地进行推论ambush发生在die之前的概率也和前者类似。

作者提出一种常识编码器（CSE）来解决同义词问题。仍然使用TEMPROB这个外部事件时序关系知识库，利用Siamese network（即模型图中的（c）部分）训练外部知识，该网络可以使得相似的词语有相似的表示。CSE提前训练好，且在后续使用过程中不再调整网络权重。

#### Siamese network

Siamese network就是“连体的神经网络”，神经网络的“连体”是通过共享权值来实现的，如下图所示：

<div style="text-align: center;">
<img src="/images/blog/An-Improved-Neural-Baseline-for-Temporal-Relation-Extraction-Siamese.jpg" width="550px"/>
</div>



其中，共享权值的意思是两边的神经网络其实就是同一个。用于计算两个输入的相似程度。度量相似度可以用Cosine，距离等，目的就是让两个相似的输入距离尽可能的小，两个不同类别的输入距离尽可能的大。神经网络可以根据具体任务来选择，如RNN，CNN，LSTM都可以。

### 2.模型主体部分

如整体模型图中的（a）和（b）所示，使用LSTM进行编码。作者使用了两种方法来获取目标事件词编码后的向量。其中（a）是在输入时对事件句里的事件词加入\<e\>和\</e\>进行标记，在输出端取最后一个时间步的表示$$h_out$$。（b）的做法是在编码器的输出端取两个时间词对应的向量$$h_e1$$，$$h_e2$$，将两者进行拼接。后续的实验结果证明这两种处理方式结果相似，并无太大区别。

LSTM输出端的结果再与两个事件词在CSE中的向量表示进行拼接。最后通过Softmax进行分类预测。

## 三、结果和分析

### 1.结果

本文的实验选择的语料库是 MATRES。 其中MATRES的组成部分如下：

![dataset](/images/blog/An-Improved-Neural-Baseline-for-Temporal-Relation-Extraction-dataset.jpg) 

MATRES：TB+AQ for train, PT for test，除了使用PT作为test之外，还是用TCR进行测试。测试结果如下：

**PT for test**

![dataset](/images/blog/An-Improved-Neural-Baseline-for-Temporal-Relation-Extraction-res1.jpg) 

其中P.I.表示在LSTM输入端加入标签来区别事件词，Concat表示在LSTM输出端直接将两个事件词的向量进行连接。Concat+CSE表示再加入CSE编码器的结果。CogCompTime是此前在该语料库上性能最好的模型。

**TCR for test**

![dataset](/images/blog/An-Improved-Neural-Baseline-for-Temporal-Relation-Extraction-res2.jpg) 

### 2. 分析

**Embedding**

通过两个测试集结果可以看出，使用ELMO和BERT作为词向量，可以获得很大的提升（4%左右）。

**外部知识**

加入外部知识后（CSE），整体效果提升并不是很大（2%）左右。由此可见，虽然使用了神经网络来处理外部知识解决开头提出的同义词问题，但对于外部知识是类似于强规则的，或许可以探寻更好的办法来根据语境意思使用外部知识。