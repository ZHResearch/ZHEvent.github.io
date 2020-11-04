---
layout: post
title: 2018 ACL《Temporal Event Knowledge Acquisition via Identifying Narratives》
categories: [Note, Temporal Relation, Knowledge]
description: 通过识别叙事文获取事件时序知识
keywords: Temporal Relation, Narratives, Weakly Supervised
mathjax: true
original: true
author: 王亮
authorurl: https://github.com/NeoAlice
---

> 2018 ACL《通过识别叙事文获取事件时序知识》[(Temporal Event Knowledge Acquisition via Identifying Narratives)](https://www.aclweb.org/anthology/P18-1050/)的阅读笔记

# **Abstract**

受记叙文本双重时序特性的启发，本文提出一种从记叙文句子中获取时序关系知识的方法。双重时序特征表示事件的时序顺序和文本出现顺序一致。 本文探索了叙事学的原理，并建立了一种弱监督的方法，从三个大文本语料库中识别出 287k 个叙事段落。然后从这些叙事段落中抽取了丰富的时序事件知识。 这些事件知识被证明有助于改进时序关系分类，并在叙事完形任务优于最近的几个神经网络模型。



# **1 Introduction**

事件的发生，指的是变化和行动，表现出规律性。 具体来说，某些事件往往同时发生，并以特定的时间顺序发生。 例如，people often go to *work* after *graduation* with a degree。 这种 “after/before” 事件时序知识可以用来识别文档中事件之间的时序关系，即使它们的本地上下文不表示任何时间关系。 事件时序知识也有助于预测给定上下文中其他几个事件的事件。改进事件时序关系识别和事件预测能力可以使各种 NLP 应用程序受益，包括事件时间线生成、文本摘要和 QA 。

虽然需求很高，但是事件时序知识非常缺乏且难以获得，现有的知识库，包括 Freebase ,Probase ,通常包含实体的丰富知识，例如某个人的出生地，但包含的事件知识很少。现有的从文本语料中获取事件时序知识的方法有利用文本模式和建立时序关系标识符等。但是这些方法往往局限在句内。

本文提出了一种新的方法，通过识别叙事故事来获得丰富的 “before/after” 事件时序知识。双重时序特征表示事件的时序顺序和文本出现顺序一致。因此，如果我们已经识别的大量的叙事文本，可以很容易获取事件时序知识。如图 1 获取的事件序列：

***{*graduated, marry, attend,receive, work, take over, expand, increase*}***  

***{*pay, jump out, head,reach into, enter, undress, shower, change, grab,leave*}***



<div style="text-align: center;">
<img src="/images/blog/temporal-knowledge-narrative-1.png" width="350px"/>
</div>



最近对博客中的叙事识别的研究中，以监督的方式建立了文本分类器。然而，叙事文本在其他体裁中也很常见，包括新闻文章和小说书籍，其中几乎没有注释数据。因此，为了从丰富的来源识别叙事文本，我们开发了一种弱监督的方法，通过大量探索用于描述叙事学研究中叙事结构的原则，可以快速地适应和识别来自不同体裁的叙事文本。叙事学普遍认为，叙事是一种话语，呈现一系列事件，按时间顺序排列（情节），涉及特定人物（人物）。首先，我们推导出特定的语法和实体共引用规则，以识别每个包含一系列句子的叙事段落，这些句子具有相同的句法结构且提及了同一个人物，即 NP,VP 表示一个人物做了某事。然后，我们使用最初识别的种子叙事文本和一组语法、共引用和语言特征来训练分类器，这些特征捕捉了叙事的两个关键原则和其他文本要素。 接下来，将分类器应用于从原始文本中识别新的叙述。 新确定的叙述将被用来增加种子叙述文本和引导学习过程迭代，直到没有足够的新的叙述可以找到。

 然后利用叙事段落的双重时序性特征，提炼出一般的事件时序知识。 具体来说，我们通过计算事件之间的因果可能，提取事件对以及由在叙述段落中经常以特定文本顺序出现的强关联事件组成的较长事件序列。

我们从 287k 叙事段落中获得了 19k 事件对和 25k 事件序列，其中有 3 到 5 个事件，我们确定了三种类型，新闻文章，小说书籍和博客。 我们的评价表明，自动识别的叙事段落和提取的事件知识都是高质量的。 此外，学习的事件时序知识被证明在用于时序关系识别和叙事完形任务时会产生额外的性能增益。 获取的事件时序知识和知识获取系统是公开的。



# **2 Key Elements of Narratives**

叙事学普遍认为，叙事呈现一系列事件，按时间顺序排列（情节），涉及特定人物（人物）。

+ **Plot.**

  情节由一系列密切相关的事件组成。 叙事中的事件通常描述“从一个阶段过渡到另一个阶段，由行为者引起或经历”。 此外，叙事往往是“对某人生活中或某物发展中的过去事件的描述”。 这些先前的研究表明，包含情节事件的句子很可能具有动作句法 “NP VP”，主要动词在过去时。

+ **Character**.

  叙事通常描述由行为者引起或经历的事件。 因此，一个叙事故事往往有一两个主要人物，称为主角，他们参与多个事件，把事件联系在一起。 主要人物可以是一个人，也可以是一个组织。

+ **Other Textual Devices**.

  包括时间、地点、人物的情感和心理状态等。使用 Linguistic Inquiry and Word Count (LIWC) 特征来获取这些内容。



# **3 Phase One: Weakly Supervised Narrative Identification**

 为了获得丰富的事件时序知识，开发了一种弱监督的方法，可以快速适应从各种文本来源识别叙事段落。



## **System Overview**

弱监督方法旨在每两个阶段捕捉叙事的关键要素。如图 2，在第一阶段， 确定满足严格规则和叙述关键原则的第一批叙述段落。第二阶段， 使用最初识别的种子叙事文本和一组软特征来训练统计分类器，以捕捉相同的关键规则和其他叙事的文本要素。接下来，应用分类器再次从原始文本中识别新的叙述。新确定的叙述将被用来增加种子叙述文本和引导学习过程迭代，直到没有足够（低于 2000）的新的叙述可以找到。

在这里，为了将统计分类器专门用于每种类型，分别在新闻、小说和博客上进行学习过程。



<div style="text-align: center;">
<img src="/images/blog/temporal-knowledge-narrative-2.png" width="350px"/>
</div>



## **Rules for Identifying Seed Narratives**

+ **Grammar Rules for Identifying Plot Events**.

  在先前的叙事学研究和我们的观察的指导下，使用上下文无关的语法生成规则来识别在角色语法结构中描述事件的句子。具体来说，使用三组语法规则来指示句子的整体句法结构。首先，我们要求一个句子具有基本的主动结构 “S*→* NP VP”，或者由之产生的并列连词（CC），状语短语（ADVP），介词短语（PP）。例如，*“Michael Kennedy earned a bachelor’s degree from Harvard University in 1980.”*具有 “S *→* NP VP” 结构 ，其中 “NP” 控制了提到“迈克尔·肯尼迪”的角色，“VP”控制了句子的其余部分，并描述了一个情节事件。

  此外，考虑到叙事通常是“对某人生活中或某物发展中的过去事件的描述”，我们要求 VP 的首词用过去式。此外，句子的主题代表一个人物。 因此，我们指定了12个语法规则，要求句子主语名词短语有一个简单的结构，并有一个专有名词或代词作为它的头词。

  对于种子叙事，每个段落包含至少四个句子，且需要 60% 以上的句子来满足上面指定的句子结构。 还要求叙事段落包含不超过20%的疑问句、感叹句或对话句，这些句子通常不包含任何情节事件。 具体的参数设置主要是根据我们对叙事样本的观察和分析来确定的。设定60%的阈值来反映叙事段落中的句子通常（超过一半）具有角色结构。 允许一小部分（20%）的疑问句、感叹句或对话句反映这样的观察，即许多段落是整体叙述，即使它们可能包含1或2个这样的句子，这样我们在叙事识别中就能得到很好的覆盖。

+ **The Character Rule**.

  叙事通常具有出现在多个句子中的主角，并将一系列事件联系起来，因此，我们还指定了一条规则，要求叙事段落具有主角。 具体而言，应用了核心 CoreNLP 工具包中的命名实体识别和实体同指消解，以确定段落中至少有一个提到被确认为人或组织或性别代词的最长实体链。 然后，通过将实体提到的数量除以段落中的句子数量来计算这个实体链的**归一化长度**，设定为 0.4 以上，意味着叙事中 40% 以上的句子提到了一个角色。



## **The Statistical Classifier for Identifying New Narratives**

利用第一阶段确定的种子叙事段落作为正例，训练一个统计分类器，以继续识别更多可能不满足特定规则的叙事段落。 同时加入负例，以便在训练中与正叙述段落竞争。 负例是指不太可能是叙事的段落，不呈现情节或主角人物，但在其他方面类似于种子叙事。具体来说，类似于种子叙事，我们要求非叙事段落包含至少四个句子，不超过 20% 的句子是疑问句、感叹句或对话；但与种子叙事相比，非叙事段落应该包含 30% 以下的角色结构句子，其中最长的角色实体链不应跨越 20% 以上的句子。随机抽样这样的非叙事段落，五倍于叙事段落。

此外，由于将训练后的分类器应用于大文本语料库中的所有段落是不可行的，例如 Gigaword。首先确定候选叙事段落，并且只将统计分类器应用于这些候选段落。 具体来说，需要一个候选段落来满足用于识别种子叙事段落的所有约束，其包含 30% 或更多具有角色结构的句子，并且具有跨越 20% 以上句子的最长角色实体链。

我们选择“最大熵”作为分类器。 具体来说，我们使用具有默认参数设置的 LIBLINEAR 库中的 MaxEnt 模型实现。接下来描述用于捕捉叙述的关键要素的功能。

+ **Features for Identifying Plot Events:**

  认识到语法生成规则在识别包含情节事件的句子时是有效的，将所有生成规则编码为统计分类器中的特征。具体来说，对于每个叙事段落，使用所有句法生成规则的频率作为特征。 注意到底层句法生成规则具有 POS tag → WORD 的形式，并包含一个 lexical word，这使得这些规则依赖于段落的特定上下文。 因此，将这些底层生产规则排除在特征集之外，以模拟可泛化的叙事元素，而不是段落的特定内容。

  此外，为了捕捉新叙事和已经学习的叙事之间潜在的事件序列重叠，使用从学习的叙事段落中提取的动词序列构建了一个动词 Bigram 语言模型并计算候选叙事段落中动词序列的 perplexity score 作为特征：

  $$PP(e_1,...,e_N)=\sqrt[N]{\prod_{i=1}^{N}\frac{1}{P(e_i\vert e_{i-1})}}$$

  where $$e_i$$ is a event word, $$P(e_i\vert e_{i-1})=\frac{C(e_{i-1,e_i})}{C(e_{i-1})}$$.

  计数基于已知的叙事段落。

+ **Features for the Protagonist Characters:** 

  一个段落中最长的三个同指实体链，其中至少有一个 mention 被认为是一个人或组织，或一个性别代词。类似于种子叙事识别阶段，我们通过将实体 mentions 的数量与段落中的句子数量相除，得到每个实体链的归一化长度。此外，我们还观察到，主角角色也经常出现在周围的段落中，因此，基于实体出现的目标段落和前后一个段落计算每个实体链的归一化长度。 我们使用 6 个归一化长度（3 来自目标段落和 3 来自周围段落）作为特征。

+ **Other Writing Style Features:**

  我们在语言查询和单词计数 (LIWC) 字典中为每个语义类别创建一个特征，特征值是该类别中所有单词出现的总数。 这些LIWC的特点是捕捉某些类型的单词的存在，例如表示相对性的单词（例如运动、时间、空间）和指心理过程的单词（例如情感和认知）。 此外，我们还将 POS 标记频率编码为特征，其在识别文本类型和写作风格方面被证明是有效的。



## **Identifying Narrative Paragraphs from Three Text Corpora**

+ **News Articles**. 

  10 million articles from English Gigaword 5th edition

+ **Novel Books**.

  BookCorpus which contains 11,038 books of 16 different sub-genres (e.g., Romance, Historical,Adventure, etc.).

+ **Blogs**.

  Blog Author-ship Corpus  which consists of 680k posts written by thousands of authors.

使用 Stanford CoreNLP tools 获取 POS tags, parse trees, named entities, coreference chains, etc.

为了应对自举学习中的语义漂移 (McIntosh和Curran，2009)，将统计分类器产生的初始选择置信度评分设置为 0.5，并在每次迭代后增加 0.05。 引导系统运行四次迭代，总共学习 287k 个叙述段落。表 1 显示了在种子阶段和每个文本语料库的每次引导迭代中获得的叙述的数量。



<div style="text-align: center;">
<img src="/images/blog/temporal-knowledge-narrative-3.png" width="350px"/>
</div>



# 4 **Phase Two: Extract Event Temporal Knowledge from Narratives**

从第一阶段获得的叙述可以描述特定的故事，并包含不常见的事件或事件转换。 因此，我们应用基于逐点互信息 (PMI) 来度量事件时序关系的相关性，以便识别不特定于任何特定故事的一般知识。目标是学习事件对和较长的事件链，且事件在“前后”关系中完全有序。

首先，通过利用叙事的双重时序性特征，只考虑事件对和较长的事件链。 具体来说，从叙事段落中提取事件序列（情节），方法是在每个句子中找到主要事件，并根据其文本顺序链接主要事件。

然后，根据 2 个指标对候选事件对排名，分别是事件对的关联性和出现频率，计算其 Causal Potential (CP)。

$$cp(e_i,e_j)=pmi(e_i,e_j)+log\frac{P(e_i\rightarrow e_j)}{P(e_j\rightarrow e_i)}$$

where $$pmi(e_i,e_j)=log\frac{P(e_i,e_j)}{P(e_i)P(e_j)},P(e_i)=\frac{C(e_i)}{\sum_xe_x},P(e_i,e_j)=\frac{C(e_i,e_j)}{\sum_x\sum_yC(e_x,e_y)}$$

同时考虑不相邻的事件对，即中间存在其他事件，通过加权平均 CP 值计算：

$$CP(e_i,e_j)=\sum_{d=1}^3\frac{cp_d(e_i,e_j)}{d}$$

然后，根据包含在事件序列中的单个事件对的 CP 分数对事件序列排序：

$$CP(e_1,e_2,...,e_n)=\frac{\sum_{d=1}^3\sum_{j=1}^{n-d}\frac{CP(e_j,e_{j+d})}{d}}{n-1}$$



# 5 **Evaluation**

## **Precision of Narrative Paragraphs**

两个专家标注者的标注一致度为 0.77，如果两人都标记为叙事文本，则认为该文本是叙事。



<div style="text-align: center;">
<img src="/images/blog/temporal-knowledge-narrative-4.png" width="350px"/>
</div>



使用基于叙事学的特征使统计分类器能够广泛地学习新的叙事，同时保持较高的精度。



## **Precision of Event Pairs and Chains**

所有事件对和所有事件链的平均 CP 分数分别为 2.9 和 5.1。

两名裁判被要求判断事件是否可能按照所显示的时间顺序发生。 对于事件链，我们有一个额外的标准，要求事件形成一个整体的连贯序列。两人同时标记为正则视为正确。事件对和事件链的标注一致性为 0.71 和 0.66。

对已获得知识的覆盖往往很难评估，因为我们没有一个完整的知识库可供比较。因此，提出伪召回率来评价(基于 Event Sequence Descriptions (ESDs))。



<div style="text-align: center;">
<img src="/images/blog/temporal-knowledge-narrative-5.png" width="350px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/temporal-knowledge-narrative-6.png" width="350px"/>
</div>

<div style="text-align: center;">
<img src="/images/blog/temporal-knowledge-narrative-7.png" width="350px"/>
</div>



## **Improving Temporal Relation Classifification by Incorporating Event Knowledge**



<div style="text-align: center;">
<img src="/images/blog/temporal-knowledge-narrative-8.png" width="350px"/>
</div>



## **Narrative Cloze**



<div style="text-align: center;">
<img src="/images/blog/temporal-knowledge-narrative-9.png" width="350px"/>
</div>



# 6 Conclusions

本文提出了一种新的方法来利用叙事文本的双重时序性特征，并在叙事段落中获取跨句子的时间事件知识。 我们开发了一个弱监督系统，探索叙事学原理，并从三个不同体裁的文本语料库中**识别叙事文本**。 从叙事文本中**提取的事件时序知识**有助于改进时间关系分类，并在叙事完形任务上优于几种神经语言模型。

