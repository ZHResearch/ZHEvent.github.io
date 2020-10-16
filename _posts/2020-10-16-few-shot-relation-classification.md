---
layout: post
title: 2019 ACL《Multi-Level Matching and Aggregation Network for Few-Shot Relation Classifification》
categories: [Note, Relation Classification, Few-Shot]
description: 小样本关系分类的多级匹配聚合网络
keywords: Relation Classification, Multi-Level Matching, Aggregation Network
mathjax: true
original: true
author: 王亮
authorurl: https://github.com/NeoAlice
---

> 2019 ACL《小样本关系分类的多级匹配聚合网络》[(Multi-Level Matching and Aggregation Network for Few-Shot Relation Classification)](https://www.aclweb.org/anthology/P19-1277)的阅读笔记

# 一、**Abstract**



前人主要采用原型神经网络，将查询实例的 embedding 向量和支撑集中的原型向量单独计算。本文提供的 MLMAN 模型，考虑局部和实例层的匹配信息，以交互式方式编码查询实例和每个支撑集。每个支撑集的类原型通过其支持实例向量的聚合获得，其中，权重由查询实例计算。



# 二、Introduction



关系分类（RC），即找到文本中两个实体之间的语义关系。例如 **London** is the **capital o**f the **UK**.

传统的分类方法采用监督学习,受制于大规模的人工标记数据。为此，又提出远程监督方法，即通过对齐知识库（KBs）和文本来标注训练集，然而长尾现象导致在有限的训练样本中难以很好地区别这些关系。

本文主要的任务是处理小样本分类中的长尾现象，每个关系只给定 1-5 个支持实例，如下表。



<div style="text-align: center;">
<img src="/images/blog/MLMAN-5.png" width="350px"/>
</div>



在局部层，查询实例和支持集的本地上下文表示将按照句子匹配框架进行软匹配，然后使用最大和平均池将匹配的局部表示聚合为每个查询和支持实例的嵌入向量。在实例层，通过多层感知机计算查询和支持实例之间的匹配度。将该匹配度作为权重，将支持集中的实例聚类为最终分类的原型类。利用训练集中的数据对模型中的匹配层和聚类层进行联合评估。由于每个支持集中的支持实例表示期望为相互接近，从而设计一个辅助损失函数来度量每个类中支持向量的不一致性。



# 三、Task Defifinition



给定 2 个关系标签不相联的数据集：$$\mathscr{D}_{meta-train} $$ 和 $$\mathscr{D}_{meta-test} $$，每个数据集由一组样本(x,p,r)组成，其中，x 表示句子，$$p=（p_{1},p_{2}）$$ 表示2个实体的位置，r 为关系标签。$$\mathscr{D}_{meta-train} $$ 被分为 $$\mathscr{D}_{train-support} $$ 以及 $$\mathscr{D}_{train-query} $$。

每次迭代，从 $$\mathscr{D}_{train-support} $$随机挑选 N 个类，每个类对应随机 K 个实例，构建 train-support set $$S=\{s_{k}^{i};i=1,...,N,k=1,...,K\}$$ ,同时，才刚才 N 个类里剩余的样本中随机挑选 R 个，构建 tarin-query set $$Q=\{(q_{j},l_{j});j=1,...,R\}$$ ，在训练时最小化（1）式：

$$J_{match}=-\frac{1}{R}\sum\limits_{(q,l)\in Q}P(l\mid S,q)$$， **(1)**

其中，

$$P(l\mid S,q)=\frac{exp(f(\{s_{k}^{l}\}_{k=1}^{K},q))}{\sum_{i=1}^{N}exp(f(\{s_{k}^{i}\}_{k=1}^{K},q))}$$ , **(2)**

$$f(\{s_{k}^{i}\}_{k=1}^{K},q)$$  用于计算查询实例 q 和支持实例 $$\{s_{k}^{l}\}_{k=1}^{K}$$ 之间的匹配度。这是本文所需要设计的函数。



# 四、**Methodology**



<div style="text-align: center;">
<img src="/images/blog/MLMAN-1.png" width="350px"/>
</div>



+ **Context Encoder**

  给定句子和其中的 2 个实体所在的位置，使用 CNNs 获得句中每个单子的局部上下文表示。

  $$w_t=[e_t;p_{1t};p_{2t}]$$

+ **Local Matching and Aggregation**

  使用 attention 方法获取给定的查询实例的局部表示和 K 个支持实例的局部表示之间的局部匹配信息，然后将匹配的局部表示聚类为嵌入向量。

  $$C=concat(\{s_{k}^{i}\}_{k=1}^{K})$$,			**(3)**

  $$\alpha_{mn}=q_m^Tc_n$$,			    			**(4)**

  $$\tilde{q}_m=\sum \limits_{n=1}^{T_s}\frac{exp(\alpha_{mn})}{\sum_{n'=1}^{T_s}exp(\alpha_{mn^{'}})}c_n$$,	 **(5)**

  $$\tilde{c}_n=\sum \limits_{n=1}^{T_s}\frac{exp(\alpha_{mn})}{\sum_{m'=1}^{T_q}exp(\alpha_{m^{'}n})}q_m$$,    **(6)**

  $$\bar{Q}=ReLU([Q;\tilde{Q}\mid Q-\tilde{Q}\mid ;Q\bigodot\tilde{Q}]W_1)$$,**(7)**

  $$\bar{C}=ReLU([C;\tilde{C}\mid C-\tilde{C}\mid ;C\bigodot\tilde{C}]W_1)$$,**(8)**

  $$\hat{s}=[max(\hat{S}_k;ave(\hat{S}_k)]$$,		 **(9)**

  $$\hat{q}=[max(\hat{Q};ave(\hat{Q})]$$,			**(10)**

+ **Instance Matching and Aggregation**

  使用多层感知机计算查询实例和 K 个支持实例之间的匹配信息，然后将匹配度作为权重累加到支持实例的表示上，从而获得原型类。

  权重： q 和 $$s_k$$ 在实例层之间的匹配度。

  $$\beta_k=v^T=(ReLU(W_2[\hat{s};\hat{q}]))$$,	**(11)**

  所有的 $$\hat{s}_k$$ 聚类为类原型：

  $$\hat{s}=\sum \limits_{k=1}^{K}\frac{exp(\beta_k)}{\sum_{k'=1}^{K}exp(\beta_{k}^{'})}\hat{s}_k$$,			 	  **(12)**

+ **Class Matching**

  建立一个多层感知机计算查询实例的表示和原型类之间的匹配得分。
  
  类层匹配函数:
  
  $$f(\{s_k\}_{k=1}^{K},q)=v^T(ReLU(W_2[\hat{s};\hat{q}]))$$,	**(13)**
  
  在式 (11) 和 (13) 中共享权重 $$W_2$$ 和 $$v$$ ,即在 class-level 和 instance-level 的每一轮迭代利用相同的函数。

+ **Joint Training with Inconsistency Measurement**

  采用支持实例和类原型之间的欧式距离平均值度量不一致性。

  $$J_{incon}=\frac{1}{NK}\sum \limits_{i=1}^{N}\sum \limits_{k=1}^{N} \Vert\hat{s}_k^i-\hat{s}^i\Vert_2^2$$,				**(14)**

  结合等式 (1) 与 (14),

  $$J=J_{match}+\lambda J_{incon}$$,								 **(15)**



# 五、**Experiments and Analysis**



<div style="text-align: center;">
<img src="/images/blog/MLMAN-2.png" width="350px"/>
</div>



<div style="text-align: center;">
<img src="/images/blog/MLMAN-3.png" width="350px"/>
</div>



+ **Instance Matching and Aggregation**

  attentive pooling 较于 max pooling 和 average pooling 的优势是对于不同的查询，能够动态地改变权重,即与查询实例越相似的实例，帮助越大，权重就越大。

  在实例层和类层共享权重参数对于计算匹配度贡献很大。

+  **Inconsistency Measurement**

  对比是否使用 $$J_{incon}$$ ,来计算机欧氏距离平均，数据分别是 0*.*0199 以及 0*.*0346（模型 1、2）。故 $$J_{incon}$$ 能够使同类的支持实例更接近。

  模型 5 至模型 6 数据的下降程度高于 1 至 2，显示 $$J_{incon}$$ 与 支持实例的 attentive aggregation 之间的依赖和关联。

+  **Local Matching**

  + 相较于模型 6 ,模型 7 免去了拼接操作(拼接所有支持实例的向量到一个矩阵里)，单独对查询和支持实例进行局部匹配，导致准确率下降。可能的原因是：拼接操作能够抑制查询和支持实例相似度较低时的影响。

  + 在模型 9 进一步移除局部匹配模块、拼接和 attentive aggregation,可见局部匹配操作显著影响结果。

    图 2 显示由查询和支持实例计算的 attention weight 矩阵，可以看出基于 attention 的局部匹配能够捕获局部上下文的匹配关系。



<div style="text-align: center;">
<img src="/images/blog/MLMAN-4.png" width="350px"/>
</div>



+ **Class Matching**

  对比两类匹配函数： ED 以及 MLP 。

  为了忽略实例层 attentive aggregation 的影响，两张匹配函数的比较基于模型 6 和模型 9 ，将这两个模型中的 MLP 函数转化为欧式距离之后，得到模型 8 和模型 10 ，通过对比，发现：

  + 当采用局部匹配时，learnable MLP 比 ED 度量方法在类型匹配任务上有巨大提高。
  + 而移除局部匹配后，所得结果与上述相反。

  原因：局部匹配过程提高了查询实例和支持集之间的交互，因此，简单的欧式距离难以描述这两者之间复杂的关联和依赖关系;另一方面，MLP 映射在使用局部匹配时更加适合于类型匹配。



# 六、Conclusions



首先，查询和支持实例通过局部匹配和聚合进行交互性地编码，然后，一个类中的支持实例进一步聚合成类原型的形式，并且通过基于 attention 的实例匹配来计算权重。最后，利用 learnable MLP 匹配函数计算查询实例和每个候选类之间的匹配分，进而设计一个附加的目标函数来提升同一类支持实例的向量之间的一致性。































