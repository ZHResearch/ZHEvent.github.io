---
layout: post
title: 2020 《Effective Distant Supervision for Temporal Relation Extraction》
categories: [Note, Temporal Relation]
description: 远程监督方法在时序关系抽取上的应用
keywords: Temporal Relation, Distant Supervision
mathjax: true
original: true
author: 王亮
authorurl: https://github.com/NeoAlice

---

> 2020《基于时间锚标记推理时序关系》[(Effective Distant Supervision for Temporal Relation Extraction)](https://arxiv.org/abs/2010.12755)的阅读笔记

# Abstract

在新领域训练时序关系抽取模型的一个主要障碍是缺乏不同的、高质量的例子。本文提出了一种自动收集远距离监督的时序关系样本的方法。自动标注具有显式时序关系的事件对，然后掩盖那些显式的cue，使得在这些数据上训练的模型学习其他信号。实验表明预训练的 Transformer 模型能够将弱标注的样本转化为人类标注基准，无论是以零样本还是少样本的设置。且 mask 方法可以提高泛化性能。

Code and dataset are available at https://github.com/xyz-zy/xdomain-temprel



# **1 Introduction**

过去关于时序关系标注的工作一直难以设计出既全面又易于判断的标注方案，且很难获得对模糊现象的高精度判断，导致高质量标记数据的稀缺。与句法分析自然语言推理等任务相比，在其他领域的时序关系抽取的资源较少。利用外部数据资源来提高泛性是一种平凡的想法，但很难高效地将其结合到模型中。

本文提出一种自动收集远标的时序关系样本的方法。使用自动系统收集和标注这些样本。关键的是，盖住显式的时序指示器：在这些示例上训练的模型不再学习基于时间的规则，而是注意更一般的时序上下文线索。以这种方式训练的模型能够学习一般的、隐含的转化为人类标注的基准的线索。 值得注意的是，这些基准数据集包含的用我们的弱标记方法所依赖的启发式方法的示例非常少。

本模型使用预训练的 Transformers ,该模型能够有效地从远标的数据集转移到更广泛的人类标注基准，提高了 MATRES 数据集上的性能，只需要少量的域内或域外样本。



# **2 Classification Model**

使用 RoBERTa（强于 BERT） and ELECTRA。一个样本由文中的一个事件对（一句句子或者两句句子）和一个标签组成，与 MATRES 一致。分词后得到词嵌，令 $$c=[e_i;e_j;e_i\bigodot e_j;e_i-e_j]$$。最后，线性分类器层产生 4 个关系标签的分布。训练是通过最大化样本标签的可能性来完成的。



<div style="text-align: center;">
<img src="/images/blog/distant-supervision-for-temprel-1.png" width="350px"/>
</div>



在 RoBERTa 和 ELECTRA 上的 F1 分别为：79.8 和  80.3。



# **3 Learning from Distant Data**

目的是自动收集能被应用于未标注文本的高质量的数据。首先，识别含有显式话语连接词的事件对被自动标注的单句样本。其次，爬取事件对的出现，这些事件对可以锚定到确定它们之间关系的时间序列。第二种技术实际上更好，并分析了导致性能增量的一些因素。

对于这两种技术，从英语 Gigaword 第五版中爬取远程样例，从数据集中可用的不同新闻源的平衡中提取样本。



## **Temporal Connectives**

如 *before*, *after*, *during*, *until*, *prior to*，以及其他在文中显式地指示事件的时序地位的连接词。过去的工作表明，在非时间环境中，复杂的关系可以从话语连接词中学习，所以这样的连接词可以是强有力的指标。

在这项工作中，专注于 *before*, *after*，因为这些是最常见和最直接的时序关系映射。为了识别连接的事件对，我们使用 Stanford CoreNLP LexicalizedParser 生成解析树。 然后，通过识别连接词、找到最接近的父动词短语和找到最接近的子动词短语来搜索相关的事件对。这些成为样例的事件。当此识别为修饰语或助动词时，取相应的主动词。样例的标签只是基于连接词是 before or after。附录中列出了例子；在检查时，我们发现这种方法是可靠的。





##  **Events Anchored to Time Expressions**

第二个线索是事件发生的显式时间，如图 2。



<div style="text-align: center;">
<img src="/images/blog/distant-supervision-for-temprel-2.png" width="350px"/>
</div>



利用  CAEVO 来检测事件对和链接事件和事件对应的发生时间，包括显式日期，相关事件（明天），以及其他自然语言指示（*now*, *until recently*）。首先，CAEVO 标注输入文档的事件和时间，ADJACENTVERBTIMEX sieve 通过句法分析树中的一条直接路径识别锚定到时间表达式上的事件。TIMETIMESIEVE 使用一组小的确定性规则来标注时间之间的关系。这两个 sieve 有很高的精确度，分别是 0.74 和 0.9。图 2 展示了使用这两个 sieve 之后的结果。最后，该系统能够推理锚定在可比时间节点（(i.e *finished* before *published*）的事件之间的关系，得到用于训练的事件对。

输出的数据集拥有较为平衡的 before 和 after 标签，以及较稀少的 equal，没有 vague。



## **Example Masking**

这些远程样例是由文本中一些零碎的指示器收集而来，因此 BERT 会产生过拟合问题。因此必须对识别的显式时序线索进行 mask，目的是从剩余的 token（包括事件实例自身和更远的上下文）中学习标签，mask 操作在分词之前进行，所以每个单词或时间有一个 mask token。对于本文的 BeforeAfter 样例，只 mask 时序连接词；对于远程时间样例，使用 CAEVO 生成的 时间 tag  mask 所有识别的文中出现的时间。这样做除了显式的日期外，其他非显式的时间也都能 mask。这可能导致“稀疏”的训练样本有很高比例的 mask token。



# **4 Results**

+ **Comparison of Distant Datasets** 

  <div style="text-align: center;">
  <img src="/images/blog/distant-supervision-for-temprel-3.png" width="350px"/>
  </div>

+ **Using Fewer Labeled Examples** 

  <div style="text-align: center;">
  <img src="/images/blog/distant-supervision-for-temprel-4.png" width="350px"/>
  </div>

+ **Effect of Masking**

  <div style="text-align: center;">
  <img src="/images/blog/distant-supervision-for-temprel-5.png" width="350px"/>
  </div>

+ **Understanding the data distribution**

  <div style="text-align: center;">
  <img src="/images/blog/distant-supervision-for-temprel-6.png" width="350px"/>
  </div>

