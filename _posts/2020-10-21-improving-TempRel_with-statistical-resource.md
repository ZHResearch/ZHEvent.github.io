---
layout: post
title: 2018 NAACL《Improving Temporal Relation Extraction with a Globally Acquired Statistical Resource》
categories: [Note, Temporal Relation, Probabilistic Resource]
description: 利用统计资源提高时序关系抽取
keywords: Temporal Relation, Probabilistic knowledge Base
mathjax: true
original: true
author: 王亮
authorurl: https://github.com/NeoAlice
---

> 2018 NAACL《利用统计资源提高时序关系抽取》[(Improving Temporal Relation Extraction with a Globally Acquired Statistical Resource)](https://www.aclweb.org/anthology/N18-1077/)的阅读笔记

# 一、**Abstract**



本文从蕴含事件通常遵循的时序的形式的先验知识的可用资源中获益，具体从 20年 间 （1987-2007） 的纽约时报文章中开发该种资源，获取概率知识库。这种资源可以改进现有的时序抽取系统，且有希望改进其他时间感知任务。



# 二、**Introduction**



<div style="text-align: center;">
<img src="/images/blog/improving-TRE-1.png" width="350px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/improving-TRE-2.png" width="350px"/>
</div>

时序关系抽取的一个挑战是其需要事件通常遵循的时序顺序的高级先验知识，即常识，否则即使是人类也无法判断该关系，对比例 1 和例 2 很容易发现。故先验知识非常重要。但是大多数现有的系统只利用事件的局部特征，即使对于例 2 这样的输入，而处理的却是相当例 1 的问题。

本文的第一个贡献就是基于纽约时报语料构建一个以概率知识库形式的资源，命名为 TEMPROB，样例如表 1。

 <div style="text-align: center;">
<img src="/images/blog/improving-TRE-3.png" width="350px"/>
</div>



其次，在 TimeBank-Dense 数据集上使用 TEMPROB 可以显著提高其表现。



# 三、**TEMPROB: A Probabilistic Resource for TempRels**



构建期望的资源，需要抽取事件、从一个未标记的大型语料中抽取时序关系。



## **Event Extraction**

事件指的是一个行为和对应行为中的参与对象。本文使用现有的 semantic role labeling（SRL） 工具直接检测事件。由于名词性事件的时序关系标记的困难，我们仅考虑动词语义框架。



## **TempRel Extraction**

+ **Features**

  采用TempRel提取中常用的特征集

  - 每个单独动词及其相邻三个词的词性（POS）标签。
  - 两者之间的距离，以 tokens 数量表示。
  - 事件提及之间的情态动词。
  - 事件提及之间的时间联系词。
  - 这两个动词是否在 WordNet 的同义词集中具有共同的同义词。
  - 输入事件提及是否具有从 WordNet 派生的通用派生形式。
  - 介词短语的主词分别覆盖每个动词。

+ **Learning**

  训练一个可以在每个文档标记时序关系的系统。

  采用 TB-Dense 数据集：

  + 标签集 R ：before, after, includes, included, equal, vague
  + 训练集：抽取所有动词语义框架，且仅保留与事件匹配的部分

  averaged perceptron algorithm：

  + two classififiers：分别是相同的和相邻的句子关系。

+ **Inference**

  采用贪心推断策略：先使用相同的句子关系分类器，然后使用相邻的句子关系分类器。每当文章中加入新的关系，就立即执行传递图闭包（如果一条边已经在闭包过程中被标记，则不会被重复标记，避免了冲突）。

+ **Corpus**

  在 NYT 语料库中发现了 51K 个独特的动词语义框架和 80M 个关系（其中 15K 个动词框架提取了 20 多个关系，而 9K 个动词框架则提取了100多个关系）。



## **Interesting Statistics**

+ **Extreme cases**

  定义如下比率：

  $$\eta_b=\frac{C(v_i,v_j,before)}{C(v_i,v_j,before)+C(v_i,v_j,after)},\eta_a=1-\eta_b$$			(1)

  在表 2，展示了一些 $$\eta_b>0.9$$ 和 $$\eta_a>0.9$$ 的事件对。

  

   <div style="text-align: center;">
  <img src="/images/blog/improving-TRE-4.png" width="350px"/>
  </div>



​	我们认为这些例子与直觉一致：“切”在“品尝”之前，而“清洁”在“污染”之后，等等。

​	更有趣的是，在这些极端例子中，存在时序上处于后面的动词，在文本当中出现的物理位置却是前面。这也符合直觉。因此对于理解动词的时序含义，很有必要 	获取动词的时间顺序，而非简单地从文本出现顺序来判断。

+ **Distribution of Following Events**

  对每一个动词 v，$$C(v,r)=\sum_{v_{i\in V}C(V,V_i,r)}$$,则

  $$P(v\ T-Before\ v'\vert v\ T-Before)=\frac{C(v,v',before)}{C(v,before)}$$	(2)

  表示 v 发生在任何动词前面的条件下 v 发生在 v' 前面的条件概率，类似地，

  $$P(v\ T-After\ v'\vert v\ T-After)=\frac{C(v,v',after)}{C(v,after)}$$		  (3)

  如图 1，对于每个单词按照上述 2 个条件概率排序，我们得到一些合理的事件序列，如：

  **{involve, kill,suspect, steal}→investigate→{report, prosecute,pay, punish}**

  这表明使用 TEMPROB 可能适用于事件序列预测或者完型填空任务。

  同样存在一些反直觉的序列，如 **know** 于 **investigate** 之前，**report** 于 **bomb** 之前和 **play** 在 **mourn** 之后，这来自于系统错误还是特殊的上下文还有待进一步研究。



 <div style="text-align: center;">
<img src="/images/blog/improving-TRE-5.png" width="350px"/>
</div>



# 四、**Experiments**



 首先量化上文获得的 TEMPROB 的正确性，然后显示其在提高现有时序关系抽取系统的作用。



## **Quality Analysis of TEMPROB**

取阈值  $$\tau\in[0.5,1)$$,对于一组事件 $$v_i,v_j$$，如果 $$\eta>\tau$$，则预测 $$v_i\ is\ T-Before\ v_j$$,反之亦然。否则视为 $$v_i\ is\ T-Vague\ v_j$$。



 <div style="text-align: center;">
<img src="/images/blog/improving-TRE-6.png" width="350px"/>
</div>



表 3 显示，使用 $$\eta$$ 作为指标是合适的，其值越高，则准确率越高，同时造成更多的情况被预测为 T-Vague,故返回值越低。



 <div style="text-align: center;">
<img src="/images/blog/improving-TRE-7.png" width="350px"/>
</div>



表 4 使用了一个非 TempRel 的因果关系数据集，将因视为 T-Before,果视为 T-After,其 label 是二元的，对比只预测一种结果，有较大提升。



## **Improving TempRel Extraction**

分别利用局部和全局方法提高时序抽取效果。

### **Improving Local Methods**

作为局部方法中的特征，即 $$\eta_b$$ 和所有标签的先验概率分布。

 <div style="text-align: center;">
<img src="/images/blog/improving-TRE-8.png" width="350px"/>
</div>



其中，$$f_r(v_i,v_j)=\frac{C(v_i,v_j,r)}{\sum_{r'\in R}C(v_i,v_j,r')},r\in R$$

对于相邻句子关系的提升更大，因为距离更远的动词的词汇依赖性更低，故而需要更多的先验知识。



### **Improving Global Methods**

作为全局方法中的正则化条件，即在目标函数中加入标签的先验分布作为正则项。

$$\hat{I}=\mathop{\arg\max}\limits_{I}\sum\limits{i,j\in \epsilon}(x_r(ij)+\underline{\lambda f_r(ij))}I_r(ij)$$		(4)

$$I_r(ij)$$ 是一个对于事件 i , j 是否具有关系 r 的指示器。且具有传递性、对称性和唯一性。



 <div style="text-align: center;">
<img src="/images/blog/improving-TRE-9.png" width="350px"/>
</div>

  <div style="text-align: center;">
<img src="/images/blog/improving-TRE-10.png" width="350px"/>
</div>

  <div style="text-align: center;">
<img src="/images/blog/improving-TRE-11.png" width="350px"/>
</div>



# 五、Conclusions

时序关系抽取在 NLP 任务中挑战性很大，部分原因是其非常依赖先验知识，因此本文提出一种包含事件通常遵循的时间顺序的资源是有助于这一任务的。为了构建这一资源，处理了 NYT 的超过 1000000 个文档语料，得到时序关系概率知识库（TEMPROB）,其很好地体现了时序关系的先验知识，且在具体时序关系抽取系统上取得了较大的效果。
