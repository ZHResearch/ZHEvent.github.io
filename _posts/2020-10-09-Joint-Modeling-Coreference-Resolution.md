```
layout: post
title: 跨文档实体和事件同指解析的联合建模_论文笔记
categories: Review
description: Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution
keywords: Joint Model,Cross-document,Coreference Resolution
mathjax: true
original: true
```

# 跨文档实体和事件同指解析的联合建模

**Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution**

**ACL 2019**

### 一、问题

​		为解决跨文档实体和事件同指解析，本文提出一种实体、事件的联合模型，同时提高了实体同指和事件同指的识别正确率，是第一个在数据集ECB+上进行实体同指识别的模型。

​		本文通过结合各实体和事件的mention、上下文（ELMo）、与其他mention的相互联系，提高了Lee等提出的联合模型的准确率。

 

### 二、方案

#### 1.Span

​		结合单词级和字符级的特征。

​		单词级使用预训练embedding（表达event时，使用event mention的head word的embedding，表达entity时，使用span中所有word embedding的平均值）。

​		字符级与单词级互补，使用LSTM。

​		串联单词级和字符级的向量表示为: 
$$
\vec{s}(m)
$$


#### 2.Context (上下文)

​		使用ELMo，取结果的head word，将上下文向量表示为:
$$
\vec{c}(m)
$$




#### 3.Semantic dependency to other mentions（对于其他指代的语义依赖）

​		对于一个给定事件mention vi ，提取四个论元：Arg0,Arg1,location,time。如果Arg1所在的slot与实体mention ej 所在的slot相同，且存在 ej 于实体簇c中，则将Arg1的向量设置为c中所有span向量的平均值：

$$
\vec{d}_{Arg1}(m_{vi}) = \frac{1}{|c|}\sum_{m\in c}\vec{s}(m)
$$
否则，Arg1的向量为0。

$$
\vec{d}_{Arg1}(m_{vi}) = \vec{0}
$$
将上述四个论元的向量串联得

$$
\vec{d}(m_{vi}) = [\vec{d}_{Arg0}(m_{vi});\vec{d}_{Arg1}(m_{vi});\vec{d}_{loc}(m_{vi});\vec{d}_{time}(m_{vi})]
$$
最后，一个mention的向量表示为含有上述三个特征的向量：

$$
\vec{v}(m) = [\vec{c}(m);\vec{s}(m);\vec{d}(m)]
$$


#### 4.Scorer

<img width = 400px src="/images/blog/notes_pics1/scorer.png"  align=center />

 

​		图中Scorer输入为：
$$
[\vec{v}(m_i);\vec{v}(m_j);\vec{v}(m_i)\circ\vec{v}(m_j)]
$$
​		圈乘代表的是按元素乘法。f(i,j)是一个50维的二进制向量，表示两个mention是否有同指的参数或谓词。

损失函数为二元交叉熵函数. 



#### 5.算法

​		本质上是进行聚类。

<img width = 600px src="/images/blog/notes_pics1/algorithm.png"  align=center />



### 三、结果及分析

#### 1.结果

<img width = 650px src="/images/blog/notes_pics1/result1.png"  align=center />



<img width = 350px src="/images/blog/notes_pics1/result2.png"  align=center />



 #### 2.错误分析

<img width = 700px src="/images/blog/notes_pics1/analysis.png"  align=center />

（其中Patial argument coreference 主要源于相似事件发生时间不同）

 

#### 3.分离分析

<img width = 350px src="/images/blog/notes_pics1/component1.png"  align=center />

<img width = 350px src="/images/blog/notes_pics1/component2.png"  align=center />

<img width = 350px src="/images/blog/notes_pics1/component3.png"  align=center />

<img width = 350px src="/images/blog/notes_pics1/figure4.png"  align=center />

​		此处作者肯定了上下文对于同指识别的重要性；并且认为在脱离自身span向量及上下文的同时，仅靠语义相关性向量就获得相当程度的精度已经不易。

 

 

 



 
