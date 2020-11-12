---
layout: post
title: 2018 NAACL《Inducing Temporal Relations from Time Anchor Annotation》
categories: [Note, Temporal Relation]
description: 利用统计资源提高时序关系抽取
keywords: Temporal Relation, Time Anchor, TORDER
mathjax: true
original: true
author: 王亮
authorurl: https://github.com/NeoAlice

---

> 2018 NAACL《基于时间锚标记推理时序关系》[(Inducing Temporal Relations from Time Anchor Annotation)](https://www.aclweb.org/anthology/N18-1166/)的阅读笔记

# **Abstract**

判断时序关系的常规标注给标注者带来了沉重的负担。 在现实中，现有的标注语料库只包括“显著”事件对上的标注，或在句子中固定窗口中的事件对上的标注。本文提出一种从绝对时间值中获取时序关系的方法（time anchors），这对于包含丰富时序信息的文本非常有效，例如新闻。从事件和时间表达式的时间锚开始，通过计算两个时间锚的相对顺序自动诱导时序关系标注。几个优点：较少的标注工作量，容易诱导句子间关系，增加了时序关系的信息量。将经验统计和自动识别结果与我们的数据基于先前的时序关系语料库进行比较。我们的数据有助于显著改善下游时间锚预测任务，显示出 14.1% 的总体精度提升。



# **1 Introduction**

大多数时序信息语料库采用时序链接 (TLINKs) 对文档中的时序信息进行编码。TLINK 表示 mention 之间的时序关系，即事件、时间表达式和文档创建时间，然而，标注 TLINKs 是一项痛苦的工作，因为候选项的数量是文档中 mention 数量的二次。原始的 TimeBank 只标注那些有标注者判断为“显著”的 mention 对，而“显著”的定义不一定清楚。因此，出现了 “vague”,"no-link" 占据很高比例的 TB-Dense。

在本工作中，提出了一种从时间锚获取时序关系的新方法，即所有 mention 的绝对时间值。我们假设通过比较时间轴上的两个时间锚可以推断出一个时序关系。输入两个标注好的时间锚，我们使用预定义的规则（第3节）生成 TORDER 关系 (e.g. BEFORE,AFTER, SAME DAY, etc.)。这需要标注时间锚，其工作量和 mention  数量是线性的。这是第一次从单个 mention 的标注中获取时序关系的工作，这与大多数手动标注 mention 对的工作不同。

几点相较目前时序标注的优势：

+ 只要时间锚给定，预定义规则可以推导出 2 次数量的 mention 对的时序关系，这样做可以略过判断识别“显著”对的工作。
+ 标注时间锚相对简单，工作量线性于 mention 数量。
+ 自动生成规则可以根据我们的定义提供灵活的关系类型，且这种增加的信息量可能有助于下游任务。



# **2 Automatic generation of TORDERs**

TORDER 方案的设计是为了解决识别“显著”对的不稳定问题并且减少标注工作量。我们假设通过比较时间轴上的两个时间锚的相对顺序自动计算时序关系。我们提出了一组预定义的生成规则（表 1 和表 2），该规则具有通过将两个标注好的时间锚作为输入来严格诱导 TORDER 的能力。

TimeBank 包含时间表达式和 DCT 的标准化日期 “YYYY MM-DD”，但不包括事件的时间。 我们的方案是通过比较两个时间锚来诱导一个 TORDER，这需要在与时间表达式和 DCT 相同的粒度中对事件进行时间锚定。 因此，用 “YYYY-MM-DD” 标注事件是一个合理的设置，天被用作标注的最小粒度。我们选择事件的日级别时间锚的标注作为我们的自动 TORDER 生成器的来源。在语料库可以提供更具体的时间信息 “YYYY-MM-DD，hh-mm-ss” 的情况下 (e.g. this morning, three o’clock in the afternoon)，TORDER 生成器可以灵活地处理这些信息，只要所有 mention 的时间锚都以相同的粒度标注。获取时间表达式时，采用现成的 temporal taggers。



<div style="text-align: center;">
<img src="/images/blog/temprel-from-time-anchor-annotation-2.png" width="350px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/temprel-from-time-anchor-annotation-1.png" width="350px"/>
</div>



# **3 Comparison of TORDERs and TLINKs**

##  **Correspondences and Differences**

TORDERs 和 TLINKs 都表示两个 mention 之间的时序关系。

一天的最小粒度是造成 TORDER 和 TLINK 类型差异的主要原因。 对于：*I went to* **sleep** *after taking a* **bath**. 会被标记为 **SAME_DAY**，而 TLINK 一定会标记 **after**。这使得直接比较分类结果难以衡量二者信息量的多少。

TORDER 可以捕获一些 TLINK 无法得到的关系。如：Stocks **rose**, **pushing** the Dow Jones industrial average up 72.24 points, to 8,189.49, **leaving** the index within 70 points of its record.TB-Dense 将这些事件标记为 vague, 而我们可以轻易地得到  SAME_DAY 这个标记。

不精确地表示时间锚 (e.g. after YYYY-MM-DD) 是导致丢失时序信息的主要缺点。 例如：America’s economic stamina has **withstood** any **disruption**... TLINK 标记为 after，而由于两个事件锚定都是（begin=before 1998-02-06, end= before 1998-02-06），故 TORDER 生成规则将其推断为 Pvague 关系，时序信息从而丢失。

之前假设跳过手动识别“显著”对可以减少 vague 关系，如果能找到被 TLINK 标记为 vague 而 TORDER 标记为 non-vague 的对，就可以作为证明。

TLINK 和 TORDER 在不同的文本域中表现出优势。小说和叙事文本中经常缺少时间表达式，而 TLINK 仍然可以捕获其中的时序关系，但是其标注代价较大，且由于人工识别”显著“对造成许多对的关系丢失。反之，TORDER 可以获取很多这些丢失的关系，但是其需要从文档（通常是新闻）中锚定事件到时间轴上，且不精确的锚定会导致信息丢失。



## **Empirical Comparison**

调查自动生成的 TORDERs 的质量对于本研究非常重要，我们以经验方法比较自动生成的 TORDERs 和人工标注的 TLINKs 的统计数据。理论上，在文档中任意距离的两个 mention 之间的 TORDER 可以自动计算。然而，新数据与现有数据具有可比性很重要。在本文中，遵循 TB-Dense 的过程，在相同和相邻的句子中生成10007 个 mention 对的完整图。

表 3 为混淆矩阵。如表所示，许多人工标注为 v 的数据在自动生成的 TORDER 中被标记为 non-vague（一大部分是 SAME_DAY），但是，一部分 non-vague 也被 TORDER 标记为 v，这符合之前描述的不精准锚定问题。因此，需要一个下游任务（下一节中的时间锚定预测）。



<div style="text-align: center;">
<img src="/images/blog/temprel-from-time-anchor-annotation-4.png" width="350px"/>
</div>



图 2 展示了两张标注的标签分布。TB-Dense 明显由于更高比例的 vague 而更加稀疏。而 TORDERs 更加平衡，这表明其能够编码更多信息。对于 Event-DCT 对，vague 是非常少的，原因在于给定 *Single-Day* DCT 后，其与 event 比较时序顺序时可以避免不稳定的人工判断。虽然 TLINK 和 TORDER 的不同定义使得直接比较变得困难，但 TORDER 的更平衡分布可能提供更多信息的分类结果，有利于下游任务。



<div style="text-align: center;">
<img src="/images/blog/temprel-from-time-anchor-annotation-3.png" width="350px"/>
</div>



## **Classification Results**

表 4 展示了数据 (27 training/validation documents, 9 testing documents) 上的 Bi-LSTM 的分类结果。

最特别的是对于 non-vague 标签，TLINK 的表现下降的很快。



<div style="text-align: center;">
<img src="/images/blog/temprel-from-time-anchor-annotation-5.png" width="350px"/>
</div>



# **4 Evaluation in Time Anchor Prediction**

##  **Task Defifinition**

从新闻文章中预测事件的时间是一个有吸引力的目标，这是走向自动事件时间线提取的必要步骤。事件锚定预测，目的是对于给定文档的每一个 *Single-Day* event，预测其时间锚。决定事件锚定的两部过程如图 3。给定一组具有已标注的事件和时间表达式的文档，系统首先获得每个事件的可能时间列表。 然后，为每个事件选择最精确的时间。



<div style="text-align: center;">
<img src="/images/blog/temprel-from-time-anchor-annotation-7.png" width="350px"/>
</div>



## **The Two-step System in Experiments**

第一步,设计时序分类器提供文档中 mention 对的时序关系。

第二步，输入一个目标的事件所有 Event-Time and Event-DCT 关系，采用微调的 (Reimers et al., 2016) 选择算法，选择最精确的时间。



## **Main Results**

Event-DCT, Event-Time 对是预测时间锚的时序信息源。只用 DCT 就得到很高的结果，原因是新闻报道往往和事件是同一天发生。而且 SAME_DAY 的引入存在不精准的问题，因为人们往往在意一些同一天事件的先后发生顺序。如表 5。



<div style="text-align: center;">
<img src="/images/blog/temprel-from-time-anchor-annotation-6.png" width="350px"/>
</div>



## Comparison to a state-of-the-art dense TLINK classifier

比较第一步中的分类器。如表 6：



<div style="text-align: center;">
<img src="/images/blog/temprel-from-time-anchor-annotation-8.png" width="350px"/>
</div>



# 5 **Conclusion**

在本文中，我们提出了一种新的方法来获得基于**新闻报道**中 mention 的时间锚的时序关系(即时间绝对值）。我们预先定义的生成规则可以通过比较时间线中两个时间锚的时间顺序自动诱导 **TORDER** 关系。与传统方法相比，我们的标注时间锚要容易得多，因为标注工作与mention 的数量是线性的。本文中使用的 TORDER 数据是公开的。新的 TORDER 和 TB-Dense 中 TLINK 的分析、实证比较和分类结果表明，我们的新数据实现了低 VAGUE 比例、信息关系类型和平衡的标签分布。 我们对使用时序关系分类器完成新闻文章中时间锚预测的下游任务进行了第二次评估。主结果表明，我们的 TORDER 在这项任务中明显优于 TLINK，这表明我们的方法具有编码时序顺序信息的能力，并且标注工作较少。

TORDER 的主要限制是要求将事件锚定在时间线中。 Strotgen 和 Gertz（2016）介绍了四个文本领域中时间表达的高度不同的特征。 这表明我们的建议很难在某些领域应用。一种可能的解决方案是采用混合标注方法，在上下文中没有时序信息时，将目标事件标注到最相关的事件( TLINK)。 虽然这项工作的动机是为时间线应用程序作出贡献，但在时序问题回答中评估这一方案也是有价值的。 SAME_DAY 可能是有害的，因为这个任务可能需要知道同一天发生的两个事件之间的确切顺序。 值得设想一个更通用的解决方案，以改善 TORDER 在未来工作中的局限性。

