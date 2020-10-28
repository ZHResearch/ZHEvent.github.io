---
layout: post
title: 2018 ACL《A Multi-Axis Annotation Scheme for Event Temporal Relations》
categories: [Note, Temporal Relation, Multi-Axis Annotation]
description: 事件时序抽取的一种多轴标注方法
keywords: Temporal Relation, Multi-Axis Annotation
mathjax: true
original: true
author: 王亮
authorurl: https://github.com/NeoAlice
---

> 2018 ACL《事件时序抽取的一种多轴标注方法》[(A Multi-Axis Annotation Scheme for Event Temporal Relations)](https://www.aclweb.org/anthology/P18-1122/)的阅读笔记

# **Abstract**



现有的时序关系标注体系的即使在专家间的标注一致程度  (IAA) 也常常较低，这说明当前的标注任务需要一个更好的定义。本文提供一种多维模型来更好地获取事件的时序结构。另外，事件的 end-points 是主要的标注混淆的根源，因此，提出只基于 start-points 来标注时序关系。



# **1 Introduction**



TB 等专家标注的数据集的 IAA 仍然较低，例如，TB-Dense,RED,THYME-TimeML 的 IAA 只有 60 %，一个低 IAA 通常意味着该任务是非常困难的，即使人类也难以分辨。简化任务的方式有减少标签集，分解为子任务等。Bethard et al. (2007) 取得了90%的一致度，但是其标注规模被压缩到了一种非常特殊的动词从句结构。



<div style="text-align: center;">
<img src="/images/blog/Multi-model-temprel-1.png" width="350px"/>
</div>



对于上述的问题，本文提出：

第一，引入多轴模型来表示时间的时序结构（通过将事件锚定到不同的轴上），只有来自相同轴的时间会被对比。例子 1-3 中的事件对之所以特别难以确定，是因为它们代表的是不同的语义现象，而属于不同的轴。

第二，对于一个事件对的 2 个时间区间 $$[t_{start}^1,t_{end}^1],[t_{start}^2,t_{end}^2]$$,本文认为对比 2 个 end-points 显然比 start-points 难的多，原因是表达的含糊和不同的人对于事件持续时间的不同理解。因此，时序标注应当更加关注 start-points。

使用这种标注体系，在 TB-Dense 的子集上取得了 84 的 IAA。

除了低 IAA 问题，时序标注还非常依赖人工。本文的第三个贡献就是首次使用众包（解决了标注质量问题）来收集一个新的、高质量的 TempRel 数据集。



# **2 Temporal Structure of Events**

## **Motivation**

目前的 TB-Dense 迫使标注者使用许多歧义的标签，导致了较低的 IAA。



## **Multi-Axis Modeling**

按理说，一个理想的标注者应当靠自己处理歧义，或者将其标注为 Vague,但是让所有标注者都不出错是不现实的，更不必说众包的情况。更有甚者，经过努力解决了那些困难的案例之后，无论各个标注者持不同意见还是一致标为 Vague,这样的案例最终还是会被标为 Vague。

TB-Dense 采用了80%的置信度规则，即标注者允许使用 80 %确定程度的标签。但是 80 %的置信度对于不同标注者的理解不同，故而最终结果仍然常常不一致。

相比于这些困难，人们很容易理解新闻的含义。这说明标注任务和文本的真实表达的含义之间存在距离。比如例 1.人们关注的是：“Serbian police **tried** to restore order but **killed** 51 people”,而不关心“Serbian police tried to **restore** order but **killed** 51 people”。

因此，单一的轴对于表示复杂的 NON-GENERIC events 而言过于局限了，所以需要一个比通用图更具限制性的模型来让标注者集中在关系标注上，而非总是先寻找事件对。同时，歧义的关系不强制标注。具体而言，需要诸如意图、观点、假设之类的轴，如表 1，而例 1-3 可以被表示为图 1。该模型的目的是让标注者获取作者清晰的表达，而非强迫其在常见的歧义队上做选择。



<div style="text-align: center;">
<img src="/images/blog/Multi-model-temprel-2.png" width="350px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/Multi-model-temprel-3.png" width="350px"/>
</div>



在实践中，一次标注一个轴：首先对一个事件按照是否可锚定在给定轴上进行分类（可锚定标注步骤)；然后我们标注每一对可锚定事件(即关系标注步骤）；最后，移动到另一个轴并重复上面的两个步骤。然而排除跨轴关系只是将良定义关系与歧义的关系分开而采用的一种策略，并非认为跨轴关系不重要，相反，如图 2 所示，跨轴关系是一种不同的语义现象，需要进一步的研究。



<div style="text-align: center;">
<img src="/images/blog/Multi-model-temprel-4.png" width="350px"/>
</div>

## **Axis Projection**

TB-Dense 处理主轴上的事件和 HYPOTHESIS/NEGATION 事件的不可比较性，是通过标注后者为已经发生。本文的模型中，TB-Dense 采用的是一种更一般的策略，即“轴投影”，在不同的轴上投影事件，以处理任意两个轴之间的不可比性（不限于假设和否定的情况）。针对例 2 的例子特别有效，**Asian crsis** 在 **expected** 之前，而 **expected** 又在 **hardest hit** 之前，故 **Asian crsis**  before **hardest hit** 。

然而，一般来说，没有直接的指导下，标注者可能会产生不同的投影，这导致需要特别设计的准则和强外部知识，标注者需要遵循有时违反直觉的准则或者“猜”一个标签，而非去文中寻找证据。

当强知识参与轴投影中时，则成为了一个推理过程。



## **Introduction of the Orthogonal Axes**

另一个创新点是引入正交轴，两个轴的交叉事件可以同两个轴的事件都进行比较，这有时可以桥接事件。



## **Differences from Factuality**

主轴上的事件同事实有以下区别：

+ 未来事件可能在主轴上，但属于 *Non-Actual*
+ 不在主轴上的事件也可以是 *Actual* events,例如已经实现的意图和真实的意见。
+ 识别事件是否可锚定相对简单，但是判断其是否确实发生很难，需要外部知识或者对全文的理解。



# **3 Interval Splitting**

显示地比较时间点，如图 3。



<div style="text-align: center;">
<img src="/images/blog/Multi-model-temprel-5.png" width="350px"/>
</div>



当两个事件之间的关系模糊时，区间分割可以提供更多的信息。 在常规设置中，假设标注者发现两个事件之间的关系可以是在之前或之前和重叠的。 然后，结果将不得不是 vague，尽管标注者实际上认同$$t_{start}^1$$ 和 $$t_{start}^2$$ 之间比较得到的关系。使用间隔分裂，这样的信息可以被保留。



## **Ambiguity of End-Points**

比较 $$t_{start}^1$$ 和 $$t_{start}^2$$ 以及 $$t_{end}^1$$ 和 $$t_{end}^2$$ 的标注正确率，如表 2。



<div style="text-align: center;">
<img src="/images/blog/Multi-model-temprel-6.png" width="350px"/>
</div>



本文假设任务的困难来自持续时间的表达（作者）和感知（读者）。在认知心理学中，人类感知的持续时长往往比真实的长。因此，本文忽略结束节点，尽管事件持续时间是另一项重要任务。



# **4 Annotation Scheme Design**

第一步，所有事件候选标记为可锚定或者不可锚定（基于正在处理的时间轴）。第二步，采用 dense annotation scheme 来标记可锚定事件。

本文仅处理动词事件，故非动词事件在预处理时被删除。为这两个步骤设计众包任务。



## **Quality Control for Crowdsourcing**

利用 CrowdFlower 控制众包质量。对于任何工作，专家都会事先标注一组例子（gold），并将起到两个目的：

+ 资格测试：70 %的精确度视为通过
+ 留存检验：必须维持 70 %的精确度，否则 kick 且舍弃之前的标注

每次判断需要至少 5 名标注者，且结果由多数票决定。



## **Vague Relations**

对于 $$t_{start}^1$$ 和 $$t_{start}^2$$,提出 2 个问题：

Q1=Is it possible that *t*1 *start* is before *t*2 *start*? 

Q2=Is it possible that *t*2 *start* is before *t*1 *start*?

回答记为 A1 , A2 ,则

+ if A1=A2=yes, then vague.

+ if A1=A2=no , then equal.

+ if A1=yes and A2=no , then before.
+ if A1=no and A2=yes , then after.

 一个好处是，人们会被提示思考所有的可能性，从而减少忽视的机会。



# **5 Corpus Statistics and Quality**



首先只标注主轴，检查两个专家在 TB-Dense 子集的 IAA。

首先，可锚定性标注，如果两个专家都认为事件是可锚定的，转到关系标注。如表 3。



<div style="text-align: center;">
<img src="/images/blog/Multi-model-temprel-7.png" width="350px"/>
</div>



再实行 2 步众包任务，结果如表 4。



<div style="text-align: center;">
<img src="/images/blog/Multi-model-temprel-8.png" width="350px"/>
</div>



接着标注正交轴上的 INTENTION\OPINION ,第一步，众包实现了 gold 上的 .82 的精确度和 .89 的 WAWA。因为只有 16 %的事件属于这一类，且轴通常很短，标注规模相对较小。两个专家获得了 .86（F1）的一致度。

本数据集命名为 *MATRES* for Multi-Axis Temporal RElations for Start-points。

每个独立的分类花费 0.01 \$，总共36篇文档花费 400 $。



## **Comparison to TB-Dense**

TB-Dense 有 1.1K 个动词事件，其中有 3.4K 对 EE 关系。

本数据集有 72 %的事件（0.8K）锚定到主轴，从而有 1.6K 对 EE 关系。以及 16 %（0.2K）锚定到了正交轴，产生了 0.2K EE 关系。

下述比较基于 2 个数据集共有的 1.8K 个 EE 关系。由于 TB-Dense 是基于区间而非 start-points 的，我们将其转化为 start-points relations (eg.,if A includes B ,then $$t_{start}^A\ is\ before\ t_{start}^B$$)。

混淆矩阵如表 5 所示。



<div style="text-align: center;">
<img src="/images/blog/Multi-model-temprel-9.png" width="350px"/>
</div>



+ 当 TB-Dense 标记 before or after 时，MATRES 有较高的概率有相同的标记 (b=455/513=.89, a=309/438=.71)；当 MATRES 标记 vague 时，TB-Dense 同样很可能标记为 vague (v=192/312=.62)。这表明 2 个数据集有高度的一致性。

+ 许多 TB-Dense 中的 vague 标记在 MATRES 中被标记为 before,after or equal（如例7）。因为前者采用的时间区间使得问题变得更加困难，故而容易被标记为 vague。

+ equal 似乎是这两个数据集主要不一致的关系，这可能是由于众包商在时间粒度和事件同指方面缺乏理解。 虽然 equal 关系在所有关系中只占很小的一部分，但需要进一步调查。

  

# **6 Baseline System**

<div style="text-align: center;">
<img src="/images/blog/improving-TRE-10.png" width="350px"/>
</div>

# **7 Conclusion**

本文提出了一种新的事件时序关系标注方法，即每次在一个独立的时间轴上标注。同时，确定了事件的 end-points 是主要的标注混淆的根源，因此，提出只基于 start-points 来标注时序关系。 专家标注者的初步研究表明，与文献值相比，IAA 有显著的提升，表明在所提出的方案下能得到更好的任务定义。 这进一步使众包的使用能够以较低的时间成本收集新的数据集 MATRES，且仍然能达到较高的一致度和 WAWA。









