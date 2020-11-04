---
layout: post
title: 2018 ACL《Improving Event Coreference Resolution by Modeling Correlations between Event Coreference Chains and Document Topic Structures》
categories: [Note, Event Coreference]
description: 根据同指链和文本主题结构改进事件同指
keywords: Coreference Resolution
mathjax: true
original: true
author: 黄琮程
authorurl: https://github.com/Ottohcc
---

> 2019 ACL《根据同指链和文本主题结构改进事件同指》[(Improving Event Coreference Resolution by Modeling Correlations between Event Coreference Chains and Document Topic Structures)](https://www.aclweb.org/anthology/P18-1045) 的阅读笔记

## 一、问题

事件共指消解是一个独特的挑战。

现有的方法无法处理复杂和精细的情况，如出现部分事件（partial event）、论元不兼容和缺乏局部上下文的信息的情况，这些都会导致同指链破碎或缺失。

在一个文档中，事件的同指比实体更少，并且在句子中分布得更为稀疏。

<div style="text-align: center;">
<img src="/images/blog/event-coreference-chains-and-topic-structure-1.png" width="400px"/>
</div>


如图，就实体而言，实体同指的分布大概率在同一句话或相邻句中，而事件同指的出现则比较分散，间隔多句出现的可能性也很大。

就以上问题，作者提出一张基于文章主题结构的同指识别方法。主要的理论依据是以下两点：

+ 一个事件同指的出现往往是为了描述这个事件的新的方面或者新进展。
+ 一个事件的重复提及主要是为了组织文章内容，所以一定与该文的文章主题有高相关度。

作者结合了事件同指链和文章主题结构之间的多方面关系，在线性整数规划（ILP）的框架下进行了建模。

## 二、方案

### 1.事件同指链与文章主体结构的关系

作者对以下四个方面的关系进行了建模。

#### 1.1 主同指链与主题过渡句的相关性

文章的主事件通常有多个相关事件 mention，这些事件描述占文章的很大一部分，并且与文章主题过渡句的布局结构很一致。

所以作者通过设计约束条件和修改目标函数来建立文章主题过渡句中事件 mention 之间的同指关系。此外，为了避免部分事件链支离破碎，并建立主要事件的完整链，鼓励将更多的相关事件 mention 关联到一个较大延伸的链。

#### 1.2 语义关联的同指链之间的关联

语义相关的事件一般出现在同一个句子中。换句话说，语义相关的两个事件的同指链很有可能相似。比如“逮捕”和“拘留”。所以，作者认为如果一些句子中存在已知的同指链，那么这些句子中的其他事件也可能成为同指链。

#### 1.3 文章体裁的特有分布方式

不同类型的文本都有其相似的行文形式。就新闻文本来说，文章的开始往往是对整个事件的总结，然后介绍主要事件及其密切相关的事件。因此，大多数事件同指链倾向于在文本的早期建立。在后面的段落中的事件 mention 一般会是前文已建立同指链的延续，或者作为单独的事件而存在，然而，它们不太可能建立一个新的共指链。所以，作者认为要使同指链的产生尽量来自于文本开头部分。

#### 1.4 子事件

子事件一般是父事件的补充，子事件可能与父事件共享相同的词法形式，并导致错误的事件共指链接。子事件一般是详细的动作描述事件，且往往是独立的，所以，作者尽可能不让描述详细动作的事件出现在已有其他的同指链中。

### 2.建模

#### 2.1 局部事件对分类器

本文使用基于事件 mention 对特征的神经网络模型。

第一层是一个 347  个神经元的共享层，以生成词向量（300 维）和词性标记（POS）（47 维）。公共层的目的是用 POS 标记丰富事件词向量。

第二层由 380 个神经元组成，用来嵌入事件词的后缀（-ing 等）和前缀（re- 等）、两个事件 mention 的词向量之间的距离（欧几里德、绝对和余弦）以及两个事件 mention 之间的公共论元。第二层的输出被连接起来，并被送入第三层。

第三层有10个神经元，然后经过一个全联接层，产生一个分数，表示该对事件 mention 的同指可能性。

三个层和输出层都使用 sigmoid 激活函数。

先用 KBP 2015 训练这个分类器，训练的结果要在后面使用。

#### 2.2 事件同指的基本 ILP 

设 $$\lambda$$ 代表一篇文章中的所有事件 mention，$$\Lambda$$ 代表所有可能的事件 mention 对，$$p_{ij}$$ 代表 $$i,j$$ 两个事件 mention 成为同指的可能性，即上一步分钟分类器的输出。

我们可以通过最小化 $$\Theta_B$$，来获得 baseline 的目标函数。

$$\Theta_B = \sum_{i,j\in\Lambda}-log(p_{ij})x_{ij}-log(1-p_{ij})(\neg x_{ij})$$

$$\neg x_{ij}+\neg x_{jk}\ge\neg x_{ik}$$

$$s.t. x_{ij}\in\{0,1\}$$

其中，第二个式子是对每三个事件 mention 所加的限制，是为了保证事件同指的传递性，即当 $$x_{ij}=x_{jk}=1$$ 时 $$x_{ik}=1$$。

接着，在此基础上，作者对每一种第 1 点中的关系增加不同的目标函数。

#### 2.3 主同指链与主题过渡句的相关性建模

设 $$\Omega$$  代表文章中所有满足条件的句子对集合，$$s_{mn}$$ 代表两个句子之间的相似程度，$$w_{mn}$$ 代表这两个句子是否是文章的主题过渡句。

$$(n-m)\ge\cfrac{|S|}{\theta_s}$$

$$\Theta_T = \sum_{m,n\in\Omega}-log(s_{mn})w_{mn}-log(1-s_{mn})(\neg w_{mn})$$

$$s.t. w_{mn}\in\{0,1\}$$

$$\sum_{i'\in \xi_m, j'\in\xi_n}x_{i',j'}\ge w_{mn}$$

$$n-m$$ 代表两个句子之间的距离，｜S｜代表文章中句子的总数，本文中 $$\theta_s$$ 取 5。

当两个句子之间的相似程度大于 0.5 时，倾向于设置 $$w_{mn}=1$$。

且需要满足最后一个式子，其中 $$\xi_m$$ 代表在句子 $$m$$ 中事件 mention 的集合。这样就保证了两个主题过渡句中至少有一个同指事件对。

**避免事件链破碎的约束**

以上的限制虽然可以保证主题过渡句内有同指事件对，但很有可能造成事件链破碎。

为解决此问题，建立如下限制：

$$\Theta_G = -\sum_{i,j\in\mu} \gamma_{ij}$$

$$\sigma_{ij} = \sum_{k<i}\neg x_{ki} \land \sum_{j<l}\neg x_{jl}\land x_{ij} \  \  \ ,  \  \  \ \sigma_{i,j}\in\{0,1\}$$

$$\Gamma_i = \sum_{k,i\in \Lambda}x_{ki}+\sum_{i,j\in \Lambda}x_{ij}$$

$$M(1-y_{ij})\ge(\varphi[j]-\varphi[i])\cdot \sigma_{ij}-\left \lceil 0.75(|S|) \right \rceil\  \ , \  \  \  y_{ij}\in\{0,1\}$$

$$\gamma_{ij}-\Gamma_i-\Gamma_j\ge M\cdot y_{ij}$$

$$\Gamma_i,\Gamma_j,\gamma_{ij}\in Z;\ \Gamma_i,\Gamma_j,\gamma_{ij}\ge0$$

对于 $$\sigma_{ij}$$ 当（1）事件 mention $$i,j$$ 是共指的（2）在 $$i$$ 之前没有与其共指的 mention（3）在 $$j$$ 之后没有与其共指的 mention 时为 1。这个值可以判断这条同指链是否是开始于 $$i$$  并结束于 $$j$$，这个变量的作用时找到一个全局的共指链，全局共指链的头尾之间相差的句子要超过文章总句的 75%。

$$\varphi[j]-\varphi[i]$$ 代表 mention $$i,j$$ 之间的句子个数。

$$M$$ 是一个很大的正数，如果 $$\sigma_{ij}$$ 表示的事件链是全局链，则变量 $$y_{ij}$$ 的值为 0。

在这一步中，首先，我们希望找到一个 $$\sigma_{ij}$$ 使得 $$y_{ij}=0$$ ，目的是找到一个全局的链。

然后，式子 $$\gamma_{ij}-\Gamma_i-\Gamma_j\ge M\cdot y_{ij}$$ 就变成了 $$\gamma_{ij}\ge \Gamma_i+\Gamma_j$$，由于我们希望目标函数最小化，即 $$\gamma_{ij}$$ 越大越好，所以，在优化的过程中，会希望  $$ \Gamma_i+\Gamma_j$$ 越大越好，也就是希望与mention $$i,j$$（共指链的开头和结尾）同指的事件越多越好，就达到了避免事件链破碎的目的。

#### 2.4 跨同指链相关约束

如果已知两个同指链中的事件同时出现在多个句子中，若其中有一个同指链中的 mention 出现在一个句子中，则倾向于这句话中存在其他 mention 且被链在另外一个同指链上。

作者在本文中将此限制简化成了希望每对句子中的同指事件对越多越好，这里的句子对需要满足两个句子都要包含两个及以上的事件 mention。

$$\Theta_C = -\sum_{m,n\in\Omega}\Phi_{mn}$$

$$\Phi_{mn}=\sum_{i\in\xi_m,j\in\xi _n}x_{ij}$$

$$|\xi_m|>1;|\xi_n|>1;\Phi_{mn}\in Z;\Phi_{mn}\ge 0$$

其中 $$\Phi_{mn}$$ 指的是在两个句子中同指对的个数。

#### 2.5 文章体裁的特有分布方式

事件在文档中的位置对事件共指链有直接的影响。在前几个段落中提到的事件更有可能建立一条事件链，文档后部分中提到的事件可能与文本前部的事件同指，但极不可能建立新的同指链。这种分布情况在新闻文章中最为普遍。

$$\Theta_D = -\sum_{i\in\xi_m,j\in\xi_n}x_{ij} + \sum_{k\in\xi_p,l\in\xi_q}x_{kl}$$

$$m,n< \left \lfloor \alpha|S| \right \rfloor;p,q>\left \lfloor \beta|S| \right \rfloor;\alpha \in[0,1];\beta\in[0,1]$$

以上的公式希望更多的文章前部的 mention 之间同指，而不希望文章后部的 mention 之间同指。且以上的公式不会影响文章后部的 mention 与文章前部的 mention 同指，所以可以根据传递性的限制，再去建立那些在既在文章前部有同指而在文章后部也有同指的事件 mention，达到了文章后部的事件 mention 不建立新的同指链的目的。

#### 2.6 子事件同指限制

$$\Theta_S=\sum_{s\in\mathbb S}\Gamma_s $$ 

$$\mathbb S$$ 代表文章中所有子事件的集合，$$\Gamma_s$$ 代表与子事件 $$s$$ 同指的 mention 个数。

作者通过使用表面句法线索来识别可能的子事件，主要是识别句子中事件序列。特别是，一个连词结构前后有两个或两个以上动词的一系列事件被提取为子事件。

#### 2.7  整个 ILP 模型及其参数

结合以上提到的所有方案，得到如下的全局目标函数。

$$\Theta = \kappa_B\Theta_B+\kappa_T\Theta_T+\kappa_G\Theta_G+\kappa_C\Theta_C+\kappa_D\Theta_D +\kappa_S\Theta_S $$

经过测试，设置 $$\kappa_B=\kappa_T = 1.0;\kappa_G=\kappa_C = 0.5 ;\kappa_D = 2.5;\kappa_S =10$$

## 三、结果及分析

### 1.结果及分析

<div style="text-align: center;">
<img src="/images/blog/event-coreference-chains-and-topic-structure-2.png" width="750px"/>
</div>

本文选用 KBP 2015 作为训练集。KBP 2015 中包含 181 篇论坛讨论帖和 179 篇新闻文章。作者随机挑选出 50 篇新闻文章来微调 ILP 中的参数，其他的 310 篇文章用来训练局部事件对分类器。

由于论坛文章杂乱无章，没办法分析文章体裁的特有分布方式，所以测试时只使用 KBP 2016 和 KBP 2017 中的新闻类文章，在比较时 baseline 的分数也只测了在新闻文章上的效果。

可以看到本文的 ILP 模型有较好的效果。然而，在 KBP 2017 中最后一条限制，有了一些下降，主要是因为本文模型在 KBP 2017 中只提取到 31 个子事件，而在 KBP 2016 中提取到了 211 个子事件。

作者认为本文的模型是具有普遍利用价值的，在文中有一条限制是根据本文（新闻）的特征提出的，作者觉得很多文本都是有其自己的特征的，举了诊断报告（clinic notes）的例子。