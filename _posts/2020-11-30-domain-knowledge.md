---
layout: post
title: 2020 EMNLP《Domain Knowledge Empowered Structured Neural Net for End-to-End Event Temporal Relation Extraction》
categories: [Note, Temporal Relation, Domain Knowledge]
description: 基于域知识加强结构化神经网络的端到端事件时序关系抽取
keywords: Temporal Relation, Domain Knowledge
mathjax: true
original: true
author: 王亮
authorurl: https://github.com/NeoAlice
---

> 2020 EMNLP《基于域知识加强结构化神经网络的端到端事件时序关系抽取》[(Domain Knowledge Empowered Structured Neural Net for End-to-End Event Temporal Relation Extraction)](https://www.aclweb.org/anthology/2020.emnlp-main.461/)的阅读笔记

# **Abstract**

目前的系统利用深度学习和预训练的语言模型来提高任务的性能。存在两个缺点：1) 在根据神经模型执行最大后验 (MAP) 推理时，以前的系统只使用被认为是绝对正确的结构化知识，即硬约束；2) 在有限数据量的训练中有偏差的预测。本文提出了一个框架，它增强了由概率域知识构造的具有分布约束的深层神经网络。通过拉格朗日松弛来解决约束推理问题，并将其应用于端到端事件时序关系抽取任务。实验结果表明能够改进基线神经网络模型，在新闻和临床领域两个广泛使用的数据集上具有很强的统计意义。 



# **1 Introduction**

<div style="text-align: center;">
<img src="/images/blog/domain-knowledge-1.png" width="350px"/>
</div>



在 SOTA 系统中使用的硬约束只能在几乎 100% 正确的情况下构建，从而使知识的采用具有限制性。时间关系传递性是一种常用的硬约束，它要求如果 A Before B和 B Before C，必须A Before C。然而，在实际应用中，约束通常不是确定性的。例如，临床治疗和测试更有可能发生在医疗问题之后，但并不总是如此。这种概率约束不能像以前的系统那样用硬约束编码。

此外，深度神经模型对优势类有偏差预测，尤其是事件时序抽取中的小数据集和偏置数据集。利用 **headed** and **say** 分别具有事件类型occurrence and reporting 的领域知识，可以为这对找到一个新的标签概率分布 (Type Pair (G))。分配给 VAGUE 的概率将减少10%，INCLUDES 增加7.2%，这大大增加了正确标签预测的机会。

本文提出在模型推理中加入语料库统计等领域知识，并利用拉格朗日松弛法解决约束推理问题，从而改进深层结构神经网络。这个框架从对预训练的语言模型的强大上下文理解中受益，同时基于先前深度模型未能考虑的概率结构化知识优化模型输出。实验结果证明了该框架的有效性。总之：

+ 本文将概率知识作为一个约束推理问题，并利用它来优化强神经模型的结果。

+ 端到端时序关系抽取任务中应用拉格朗日松弛和事件类型与关系约束。 
+ 该框架在不采用知识的情况下显著优于基线系统，并在新闻和临床领域的两个数据集上取得了新的 SOTA 结果。



# **2 Problem Formulation**

我们关注的问题是端到端事件时序关系抽取，它以一个原始文本作为输入，首先识别所有事件，然后对所有预测的事件对进行时序关系分类。 图 2 的左列显示了

一个示例。在实际环境中，端到端系统是实用的，在输入中没有标注事件，并且具有挑战性，因为在事件抽取过程中引入噪声后，时序关系更难预测。

<div style="text-align: center;">
<img src="/images/blog/domain-knowledge-2.png" width="350px"/>
</div>



#　**3 Method**

首先，描述端到端事件时序关系抽取系统的深层神经网络的细节，然后展示了如何在整数线性规划 (ILP) 中将事件类型和关系之间的域知识表示为分布约束，最后应用拉格朗日松弛来解决约束推理问题。基本模型是训练端到端的交叉熵损失和多任务学习，以获得关系分数。 最后执行一个额外的推理步骤，以便将领域知识作为分布约束结合起来。



## **End-to-end Event Relation Extraction**

如图 2 左列所示，端到端模型与 Han et al. (2019b) 的管道模型具有类似的工作流，其中使用具有共享特征提取器的多任务学习来训练管道模型。组合训练损失为 $$L=c_{\epsilon}L_{\epsilon}+L_R$$，$$L_{\epsilon},L_R$$分别是事件提取器和关系模块的损失，$$c_{\epsilon}$$ 是平衡两个损失的超参数。

+ **Feature Encoder.**

  输入实例首先放入 BERT ,然后进去 Bi-LSTM 层。将编码的特征作为事件抽取器和关系模块的输入。

+ **Event Extractor.**

  事件抽取器首先预测每个输入 token 的事件类的分数，然后根据这些分数检测事件跨度。如果事件有多个 tokens，则将其开始向量和结束向量拼接为最终事件表示。事件分数定义为事件类的预测概率分布。 非事件的对被自动标记为 NONE，而有效的候选事件对被输入到关系模块中以获得它们的关系分数。

+ **Relation Module.** 

  关系模块的输入是一对事件，它们与事件抽取器具有相同的编码特征。只需在将它们输入到关系模块之前将它们拼接起来，生成由Softmax 计算的关系分数 $$S(y_{i,j}^r,x^n)$$。



##  **Constrained Inference for Knowledge Incorporation**

如图 2 所示，一旦通过关系模块计算关系分数，就会执行 MAP 推理，以纳入分布约束，从而利用结构化知识来调整神经基线模型分数并优化最终模型输出。将具有分布约束的 MAP 推理作为 LR 问题，并用迭代算法求解。以下是 MAP 推理中每个组件的细节。

+ **Distributional constraints**

  图 2 右列说明了如何利用语料库统计来构造分布约束。设 P 是一组事件属性，在语料库中计数：

  $$C(P^m,P^n,r)=\sum\limits_{i,j\in \epsilon\epsilon}c(P_i=P^m;P_j=P^n;r_{i,j}=r)$$

  $$C(P^m,P^n)=\sum\limits_{i,j\in \epsilon\epsilon}c(P_i=P^m;P_j=P^n)$$

  令 $$t=(P^m,P^n,r)$$,则先验三重概率定义为：$$p_t^*=\frac{C(P^m,P^n,r)}{C(P^m,P^n)}$$

  $$\hat{p_t}$$ 为预测的三重概率，分布约束为：

  $$p_t^*-\theta\le\hat{p_t}\le p_t^*+\theta$$				(1)

  其中 $$\theta$$ 是先验概率和预测概率之间的容差。

+ **Integer Linear Programming with Distributional Constraints**

  定义 ILP 为：

  $$L=\sum\limits_{(i,j)\in \epsilon\epsilon}\sum\limits_{r\in R}y_{i,j}^rS(y_{i,j}^r,x)$$		(2)

  $$s.t.\ p_t^*-\theta\le\hat{p_t}\le p_t^*+\theta,\forall t\in T$$

  S 是关系模块获得的打分函数，$$\hat{p_t}=\frac{\sum_{(i:P^m,j:P^n)}^{\epsilon\epsilon}y_{i,j}^r}{\sum_{(i:P^m,j:P^n)}^{\epsilon\epsilon}\sum_{r'}^{R}y_{i,j}^{r'}}$$

  MAP 推理的输出 $$\hat{y}$$ 是输入实例 $$x^n$$中所有关系候选的最优标签分配的集合。 $$\sum\limits_{r\in R}y_{i,j}^r=1$$ 确保每个事件对获得一个标签分配，这是唯一硬约束。

  对于每个三元组 t,其等式约束可以重写为：

  $$F(t)=(1-p_t^*)\sum\limits_{(i:P^m,j:P^n)}^{\epsilon\epsilon}y_{i,j}^r-p_t^*\sum\limits_{(i:P^m,j:P^n,r'\neq r)}^{\epsilon\epsilon}\sum y_{i,j}^{r'}=0$$		(3)

  目标是满足方程的同时最大化方程（2）定义的目标函数。

  

+ **Lagrangian Relaxation**

  解决式（2）是一个 NP-hard 问题，因此，为每个分布约束引入拉格朗日乘子 $$\lambda_t$$，将之规约为拉格朗日松弛问题：
  
  $$L(y,\lambda)=\sum\limits_{(i,j)\in\epsilon\epsilon}\sum\limits_{r\in R}y_{i,j}^rS(y_{i,j}^r,x)+\sum\limits_{t\in T}\lambda_tF(t)$$  					(4)
  
  上式的解如 Algorithm 1：
  
  + 每轮迭代 k 获取 MAP 推理的最佳关系指派 $$\hat{y_k}=arg\ max\ L(y,\lambda)$$
  + 更新拉格朗日乘子，使预测概率更接近先验。
  
  α 是步长。第一步通过固定 λ 来选择最大似然赋值；第二步搜索最小化目标函数的 λ 值。
  
  

<div style="text-align: center;">
<img src="/images/blog/domain-knowledge-3.png" width="350px"/>
</div>



# **4 Constrained Inference Implementation**

构造分布约束以及用 LR 进行推理的实现细节。



## **Distributional Constraint Selection**

分布约束的选择对于算法至关重要。如果事件类型和关系三元组的概率在不同的数据划分中是不稳定的，可能会过度修正预测的概率。 使用以下带有启发式规则的搜索算法来确保约束稳定性。

+ **TimeBank-Dense**

  根据 $$C(P^m,P^n)=\sum_{\hat{r}\in R}C(P^m,P^n,\hat{r})$$ 值排序候选约束，表1中列出了在开发集中具有最大预测数及百分比的 $$C(P^m,P^n)$$。

  设置 3% 作为阈值。表 1 底部的约束被过滤。式（3）表明在三元组 $$(P^m,P^n,r)$$ 上定义的一个约束影响所有的 $$(P^m,P^n,r') \ for\ r'\in R/r$$，也就是说降低 $$\hat{p}_{(p^m,p^n,r)}$$ 相当于提高 $$\hat{p}_{(p^m,p^n,r’)}$$，反之亦然。因此，启发式地将 $$(P^m,P^n,VAGUE)$$ 作为约束三元组的默认值。

  最后，采用贪婪搜索规则来选择最终的约束集。从表 1 中的顶部约束三元组开始，然后继续添加下一个，只要它不会伤害开发集上的网格搜索 F1 分数。 最后，选择了四个约束三元组，如表 3。

  

<div style="text-align: center;">
<img src="/images/blog/domain-knowledge-4.png" width="350px"/>
</div>



+  **I2B2-TRMPORAL**

  将训练集以相同的大小随机划分为 5 个子集，保证 $$\frac{1}{5}\sum_{k=1}^5\vert p_{t,s_k}-p_t^*\vert<0.001\ and\ \vert\hat{p}-p_t^*\vert>0.1$$

  第一个规则确保约束三元组在随机划分的数据上是稳定的；第二个规则确保预测和黄金之间的概率差距很大，这样就不会过度修正它们。 最终，四个约束满足这些规则，如表 9中，对于这些约束只运行一个最终的网格搜索。



<div style="text-align: center;">
<img src="/images/blog/domain-knowledge-11.png" width="350px"/>
</div>



## **Inference**

利用现有的 Gurobi optimizer 实施 ILP。超参数如表 6。



<div style="text-align: center;">
<img src="/images/blog/domain-knowledge-5.png" width="350px"/>
</div>



# 5  **Results and Analysis**

+ TB-Dense

  利用拉格朗日松弛来结合领域知识，即使对于强神经网络模型也是非常有益的。

  表 3 中的消融研究显示了分布约束是如何工作的，以及约束的个体贡献。 对于所选择的三个约束，预测的概率差距分别缩小 0.15、0.24 和0.13，同时为关系提取提供了0.91%、0.65%和0.44%的改进。我们还在表 4 中显示了每个关系类的性能细目。总体 F1 的改进主要是由正相关类 (BEFORE、After和INCLUDES) 中的召回分数驱动的，这些类的样本大小比VAGUE小得多。这些结果与表 3 中的消融研究是一致的，其中端到端基线模型对 VAGUE 的预测过多，LR 算法根据其关系分数将对 VAGUE 的不太自信的预测分配给正类和少数类来纠正它。



<div style="text-align: center;">
<img src="/images/blog/domain-knowledge-6.png" width="350px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/domain-knowledge-7.png" width="350px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/domain-knowledge-8.png" width="350px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/domain-knowledge-9.png" width="350px"/>
</div>



除了有必要为事件时序关系抽取创建高质量的数据外，还可以纳入额外的信息，如篇章关系 （尤其是对于(occur., occur., VAGUE)），和其他关于事件属性的先验知识，以解决事件时序推理中的歧义性。



<div style="text-align: center;">
<img src="/images/blog/domain-knowledge-10.png" width="350px"/>
</div>



尝试放宽阈值，F1分数在开发集上继续提高，但在测试集上，F1分数最终会下降。 当三元组计数较小时，基于该计数计算的比率不那么可靠，因为该比率在开发和测试集之间可能会有很大的变化。由此产生过拟合。



# Conclusion

本文提出了利用概率域知识构造了具有分布约束的深层神经网络。 将其应用于具有事件类型和关系约束的端到端时间关系提取任务中，具有分布约束的MAP推理可以显著提高结果。后续计划将所提出的框架应用于各种事件推理任务，并构建新的分布约束，这些约束可以利用语料库统计之外的领域知识，例如更大的未标记数据和知识库中包含的丰富信息。