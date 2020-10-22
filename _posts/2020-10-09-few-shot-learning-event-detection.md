---
layout: post
title: 2020 ACL《Extensively Matching for Few-shot Learning Event Detection》
categories: [Note, Event Detection]
description: 用于小样本学习事件检测的广泛匹配
keywords: few-shot, event, detection
mathjax: true
original: true
author: 王亮
authorurl: https://github.com/NeoAlice
---

> 2020 ACL《用于小样本学习事件检测的广泛匹配》[(Extensively Matching for Few-shot Learning Event Detection)](https://www.aclweb.org/anthology/2020.nuse-1.5.pdf) 的阅读笔记

## 一、现状和问题

目前典型的事件检测方法是基于特征工程的传统监督学习和神经网络，但是监督学习模型在处理未知类别的事件时效率较差，通常使用标注、再训练的方法，其代价较大。

## 二、方法

1.支撑集中添加NULL类 **N+1-way K-shot**，如下：

$$S={\{(s_{1}^{1},a_{1}^{1},t_{1}),...,(s_{1}^{K},a_{1}^{K},t_{1}),...\\(s_{N}^{1},a_{N}^{1},t_{N}),...,(s_{N}^{K},a_{N}^{K},t_{N}),...,\\(s_{N+1}^{1},a_{N+1}^{1},t_{null}),...,(s_{1}^{K},a_{N+1}^{K},t_{null})\}}$$

+ $$(t_{1},...,t_{N})$$ positive labels
+ $$t_{null}$$ a special label for non-event

2.利用查询实例和支撑集之间与支撑集内样本自身之间的匹配信息来训练ED模型，可以显著减少标注和训练代价，同时维持高准确率。具体的方法是通过在损失函数中添加辅助参数来抑制学习过程。

+ 最大似然估计值

  $$L_{query}(x,S)=-logP(y=t|x,S) \tag{1}$$

  where x,t,S are query instance,ground true label,and support set​

+ Intra-cluster matching

  相同类之间的向量是类似的，因此最小化它们的间距

  $$L_{intra}=\sum\limits_{i=1}^{N}\sum\limits_{k=1}^{K}\sum\limits_{j=k+1}^{K}mse(v_{i}^{j},v_{i}^{k}) \tag{2}$$

+ Inter-cluster information

  最大化类之间的距离

  $$L_{inter}=1-\sum\limits_{i=1}^{N}\sum\limits_{j=i+1}^{N}cosine(c_{i},c_{j}) \tag{3}$$

+ 损失函数

  由(1)、(2)、(3)

  $$L=L_{query}+\beta \hat{L}_{intra}+\gamma \hat{L}_{inter} \tag{4}$$

## 三、实验

![1](/images/blog/few-shot-learning-event-detection-1.jpg)

+ 表1显示：
  + 5+1-Way 5-shot的表现总是优于10+1-Way 10-shot，因为后者中需要被分类的类的数量是前者的2倍之多
  + Proto和Proto+Att模型的表现均最好
  + 在10+1-Way 10-shot中Proto+Att的表现略好于Proto 

+ 表2显示：
  + 使用给出的损失函数后，所有的神经网络模型中F都明显提高了

## 四、消融实验

![2](/images/blog/few-shot-learning-event-detection-2.jpg)

上表显示了各个模型中未加入损失函数、只加入Inter、只加入Intra和同时加入Inter和Intra的结果，表明这两个损失函数对于结果都有明显的提升，且缺失任何一个，都会对结果造成较大精度损失。
