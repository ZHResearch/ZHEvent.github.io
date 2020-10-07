---
layout: post
title: 从word2vec开始，说下GPT庞大的家族系谱
categories: Review
description: 从word2vec开始，从头到尾梳理了GPT的家谱
keywords: word2vec, bert, gpt, attention
mathjax: true
---

> 转载自[机器之心](https://mp.weixin.qq.com/s/dKbGR4sCkNpik0Xw41QLVw)，作者：王子嘉，编辑：H4O。部分内容有修改。

> 本文从老祖级别的 word2vec 开始，从头到尾梳理了 GPT 的 「家谱」 和 word2vec 领衔的庞大的 NLP「家族集团」。

GPT 不是凭空而出，它是经过了很多人的努力，以及很长一段时间的演化得来的。因此，梳理一下 GPT 的庞大 “家族” 还是很有必要的，看看他继承了什么，学习了什么，又改进了什么，这样也能更好地理解 GPT 各个部分的原理。

现在很多人把 2018 年（BERT 提出）作为 NLP 元年（类似于当时 ImageNet 的提出），其趋势与当年的图像领域也极其类似——模型越来越大。2018 年的 BERT-large（最大的 BERT 模型）的参数量是 340M，而到了 2020 年的 GPT-3，这个数字已经翻了无数倍了。很多人第一次了解到 GPT 大概是 2018 年，那个时候 GPT 还是个配角（被其兄弟 BERT 拉出来示众），当时的主角是 BERT，BERT 的成功让当时论文中作为前身的 ELMo 和 GPT 也火了一把。其实当时的 GPT 也没有 BERT 第一版论文中说的那么差，现在的 BERT 论文也已经没有了当时的对比图片，而最近的 GPT-3 总结了当时的失败经验，也开始重视媒体宣传，让 GPT 成功以 C 位出道，结结实实当了次主角。

一提到 GPT3，大家第一印象大概就是异常庞大的参数量——1750 亿，比其前身多 100 倍，比之前最大的同类 NLP 模型要多 10 倍。事实上，如今的 GPT-3 是在很长一段时间的演变后得到的（汇聚了老祖宗们的优秀智慧），从 word2vec 开始，各式各样的语言模型就开始变得让人眼花缭乱，也有很多给 GPT 的诞生提供了很大的启发，我们今天就从老祖级别的 word2vec 开始，从头到尾梳理一下 GPT 的 “家谱” 和 word2vec 领衔的庞大的 NLP“家族集团”。

值得注意的是，这里列出的家族成员都是跟 GPT 关系比较近的，所以本文列举的内容并不能完全囊括所有语言模型的发展，本文的主要目的是为了梳理 GPT 的原理脉络，并与一些类似的模型做必要的对比以加深理解。

## 家谱总览

为了更好地给 GPT 建立一个“家谱”，也让你们知道这篇文章会涉及什么内容，首先要宏观的比较一下这个庞大的家族各个成员的出生时间（图 1）。

<div style="text-align: center;">
<img src="/images/blog/from-word2vec-to-gpt-1.png" width="500px"/>
</div>

<center>图 1：家族成员出生日期。</center>

有了这个出生时间表，再对他们有一定的了解（本文的主要目的），它们的关系其实就很好确定了，所以这个庞大家族的族谱大概可以画成图 2 的这个样子。

<div style="text-align: center;">
<img src="/images/blog/from-word2vec-to-gpt-2.png" width="600px"/>
</div>

<center>图 2：GPT 族谱。</center>

读到这里对这些模型不够了解或者有完全没有听过的也没有关系，细心的同学可能会发现 Attention 的出生日期并没有列在图 1 中，因为 Attention 算是 GPT 的一个远方表亲，因为 Attention 业务的特殊性（主要是外包工作，后面会详细说），GPT 对其没有完全的继承关系，但是 GPT 和他的兄弟姐妹们都有 Attention 的影子。

对 GPT 族谱有了宏观的了解后，就可以开始正式进入正题了。

## Word Embedding [1,2]

Word Embedding（词嵌入）作为这个庞大家族集团的创始人，为整个 “集团” 的蓬勃发展奠定了坚实的基础。到目前为止，词嵌入一直是 NLP 集团的中坚力量。Word2Vec 和 Glove 等方法就是很好的例子，为了避免对 “集团” 的根基不明白，这里先对词嵌入进行简要介绍。

对于要被机器学习模型处理的单词，它们需要以某种形式的数字表示，从而在模型中使用这些数字（向量）。Word2Vec 的思想就是我们可以用一个向量（数字）来表征单词的语义和词间的联系（相似或相反，比如 “斯德哥尔摩” 和 “瑞典” 这两个词之间的关系就像 “开罗” 和 “埃及” 之间的关系一样），以及语法联系（如英文中的 ‘had’ 和 ‘has’ 的关系跟 ‘was’ 和 ‘is’ 的关系一样）。

这位创始人很快意识到，他可以用大量文本数据对模型进行预训练从而得到嵌入，这样的效果比在特定的任务（数据量较少）上训练更好。所以 word2vec 和 Glove 这种可供下载的预训练词向量表（每个词都有自己对应的词向量）就出现了，图 3 展示了 GloVe 中 ‘stick’ 这个词的对应的词嵌入（部分）。

![3](/images/blog/from-word2vec-to-gpt-3.png)

<center>图 3：“stick”的词向量 （图源：[15]）</center>

## ELMo [3] —— 语境很重要！

在爷爷创建了这个家族企业之后，后续也有很多很多后代在对其进行发展，GPT 的近亲中也有这么一位——ELMo (2018 年 2 月)。这位 GPT-3 的叔叔在 2018 年跟 GPT-1 一起被 BERT 拉出来示众（作比较），所以大家应该也比较耳熟。ELMo 创业的时候，Transformer 还未经打磨，Transformer 的儿子 Transformer-decoder（2018 年 1 月）同样还没有火起来，所以他还没有用 Transformer（也因此被 BERT 在最开始的论文里拉出来作对比），但是他注意到了词向量不能是不变的，比如一开始学习词嵌入的时候是用下面这两句话：

- “哦！你买了我最爱的披萨，我爱死你了！”
- “啊，我可真爱死你了！你把我最爱的披萨给蹭到地上了？”

这里的 “爱” 明显意思是不同的，但是因为训练的时候没有看到 “爱” 后面的话，所以 “爱” 就在词嵌入的空间里被定义为褒义词了。首先我们要知道的是 ELMo（Embedding from Language Model）中也是使用了 “语言模型” 任务来完成语境的学习，这也是我在这篇文章里提到 ELMo 的一个重要原因（另一个就是其为了解决上面提出的问题的方法），为了防止有人对语言模型不熟悉，这里给出一个语言模型的定义——**语言模型其实就是给定一个模型一串词，然后让模型预测下一个词**。

了解了语言模型的概念，上面问题出现的原因就不难理解了——模型看到的是前文，看不到后文。**为了解决这个问题，ELMo 就用双向 LSTM 来获取双向语境**。同时，上面涉及的问题不只是双向语境问题，还有一个很严重的问题——词嵌入不应该是不变的。也就是说，不同的句子里同一个词可能有不同的意思，那么词嵌入肯定也该不一样。

因此 **ELMo 又提出要在看完整个句子的前提下再给定这个词的嵌入**。也就是说词嵌入的来源不再是过去的查表了，而是通过预训练好的模型来获得（是不是很像图像领域的 transfer learning?）。看一下原论文中对 ELMo 的定义：

> Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pretrained on a large text corpus.

最终，这些思想都给他的侄子们（GPT，BERT 等）带来了很大的启发。如果对 ELMo 的计算细节感兴趣，本文最后也贴了 reference，可以去看一下原论文，还是有很多很聪明的想法的，不过它不是我们今天的主角，因此这里就不多做赘述了。

## 旁支 Attention

在说完 ELMo 之后，本来就应该开始介绍现在家族集团的中流砥柱 BERT 和 GPT 了，但是在这之前还是要简要回顾一下 attention 和 self attention，我猜很多加入 NLP 不久的人应该跟我一样，一上来从各大科普文中接收到的概念就是 self-attention 和 self-attention 的计算过程，对于 self-attention 名字的由来还是很迷糊，甚至很多科普文的作者对这个概念都是迷迷糊糊的，导致我在求证自己的理解的时候都发现很多不同的版本，不过我们还是要忠于原论文，因此这个问题还是从最开始论文对 Attention 的定义开始说起，很多 attention 的科普文的第一部分都会对 attention 的原理进行很形象的描述，顾名思义就是我们希望我们的模型在给我们结果的时候不要傻乎乎的给它什么他就都看，而是只看重要的那一点，机器之心对于注意力还有很多很有趣的解释，这里就不浪费篇幅做重复的工作了，直接上正式的定义：

> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 
>
> <div style="text-align:right">——Attention is all you need</div>

这段话的意思就是说，attention 机制里主要有三个向量：key, query 和 value，其实可以将 Attention 机制看作一种软寻址（Soft Addressing）：Source 可以看作一个中药铺子的储物箱，储物箱里的药品由地址 Key（药品名）和值 Value（药品）组成，当前有个 Key=Query（药方）的查询，目的是取出储物箱里对应的 Value 值（药品），即 Attention 数值。通过 Query 和储物箱内元素 Key 的地址进行相似性比较来寻址，之所以说是软寻址，指的是我们不只从储物箱里面找出一中药物，而是可能从每个 Key 地址都会取出内容，取出内容的重要性（量的多少）根据 Query 和 Key 的相似性来决定，之后对 Value 进行加权求和，这样就可以取出最终的 Value 值（一副中药），也即 Attention 值。所以不少研究人员将 Attention 机制看作软寻址的一种特例，这也是非常有道理的[12]。只说理论没有细节就是耍流氓，所以下面看一下作为外包公司的 Attention 的甲方们到底是谁。

**RNN & LSTM**

本节我们以机器翻译任务为例，以介绍甲方 RNN 提出的问题以及 Attention 给出的解决方案。首先看一下 RNN 原创的解决方案（图 4）。

![4](/images/blog/from-word2vec-to-gpt-4.png)

<center>图 4：RNN 原始方案 （图源：[12]）</center>

在原始的方案中，待翻译的序列（X）的信息被总结在最后一个 hidden state（hm）中，身负重任（带着原始句子的所有信息）的 hm 最终被用来生成被翻译的语言（Y），这个方案的问题很明显，hm 的能力有限，当句子很长的时候，很容易丢失重要信息。问题提出来了，Attention 给出了怎样的解决方案呢？

<div style="text-align: center;">
<img src="/images/blog/from-word2vec-to-gpt-5.png" width="280px"/>
</div>

<center>图 5：Attention 给出的最终方案 （图源：[13]）</center>

在正式介绍 Attention 给出的方案之前，还是简单回顾一下 Attention 的计算过程（这里提到的 Attention 在 Transformer 中被称为 Scaled Dot-Product Attention）。

<div style="text-align: center;">
<img src="/images/blog/from-word2vec-to-gpt-6.png" width="250px" />
</div>

<center>图 6：Attention 计算过程 （图源：[4]）</center>

如图 6 所示，Q 和 K 首先会计算联系程度（这里用的是 dot-product），然后通过 scale 和 softmax 得到最后的 attention 值，这些 attention 值跟 V 相乘，然后得到最后的矩阵。

回顾了 Attention 的计算过程后，图 5 所示的方案就好理解多了，它还是一个 encoder-decoder 结构，上半部分是 decoder，下半部分是 encoder（双向 RNN）。开始的流程跟原始版本的 RNN 一样，首先 Encoder 获得了输入序列 X 的 hidden state (h)，然后 docoder 就可以开始工作了。从这里开始，docoder 的工作因为 Attention 的加入开始发生了变化，在生成 $$s_t$$ 和 $$y_t$$ 时，decoder 要先利用 $$s_{t-1}$$ 和各个 hidden state 的**联系度**（利用加权点乘等方式获得）来获得 attention ($$\alpha$$)，这些 attention 最终作为权重对各个 h 进行**加权求和**从而得到背景向量 (context)，后面就没什么变化了，$$y_t$$ 基于 $$s_{t-1},y_{t-1}$$ 和 context 来生成。

为了更好地理解图 5 在做什么，我们可以再将上图中的各个符号跟我们前面的 Attention 中的三类向量联系起来：

- 在查询过程中，我们的目的是为了通过 h 和 s 的相关性来确定 h 在 context 矩阵中的权重，所以最上面的 $s_t$ 就是 query 向量，用来做检索的；
- 如果理解了上一点和前面对 Attention 机制的解读，因此这里的 $h_t$ 就很好理解了，它就是上文中的 key 和 value 向量。

LSTM 公司中的 Attention 机制虽然没有那么明显，但是其内部的 Gate 机制也算一定程度的 Attention，其中 input gate 选择哪些当前信息进行输入，forget gate 选择遗忘哪些过去信息。LSTM 号称可以解决长期依赖问题，但是实际上 LSTM 还是需要一步一步去捕捉序列信息，在长文本上的表现是会随着 step 增加而慢慢衰减，难以保留全部的有用信息。

**总的来说，Attention 机制在外包阶段就是对所有 step 的 hidden state 进行加权，把注意力集中到整段文本中比较重要的 hidden state 信息**。Attention 除了给模型带来性能上的提升外，这些 Attention 值也可以用来可视化，从而观察哪些 step 是重要的，但是要小心过拟合，而且也增加了计算量。

**自立门户 —— Self-attention**

Attention 在外包自己的业务的时候，其优秀的外包方案引起了 Transformer 的注意，Transformer 开始考虑 Attention 公司的核心思想能不能自立为王呢？一不做二不休，Transformer 向自己的远房表亲 Attention 表达了这一想法，两者一拍即合，经过了辛苦的钻研后，他们兴奋地喊出了自己的口号——“Attention is all you need!” 

然后，Transformer 公司应运而生了！

## 中兴之祖 Transformer：Attention 就够了！

**承前启后 —— self-attention**

他们到底做了什么呢？简单来说，就是用了改良版的 self-attention 将 attention 从配角位置直接带到了主角位置。为了防止我的转述使大家的理解出现偏差，这里还是先贴上原文对于 Transformer 中各个组件的 attention 机制的介绍（为了方便解释，我稍微调整了一下顺序）：

- The encoder contains self-attention layers. In a self-attention layer, all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to $$-\infty$$) all values in the input of the softmax which correspond to illegal connections. 
- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. 

可以看出 Transformer 主要提使用了两种 Attention——Self-attention 和 encoder-decoder attention。这里的 Self-attention 主要是为了抛掉外力（LSTM，RNN，CNN 等），encoder-decoder attention 则延续了前面的外包方案（图 5），作用跟过去相同，主要是为了连接 encoder 和 decoder。**这里的 Self Attention 顾名思义，指的不是 Target 和 Source 之间的 Attention 机制，而是 Source 内部元素之间或者 Target 内部元素之间发生的 Attention 机制，也可以理解为 Target=Source 这种特殊情况下的注意力计算机制[12]**。

![7](/images/blog/from-word2vec-to-gpt-7.png)

<center>图 7：self-attention 可视化（图源：http://jalammar.github.io/illustrated-transformer/）</center>

图 7 可视化了某一层的 self-attention 的结果，Attention 值都是相对于这个句子本身的（包括这个词本身），这个图很生动的表现了 self 的含义。同时我们可以看到相较于传统的时间序列模型，self-attention 的优势还是很明显的——可以很好地注意到该注意的部分，不会受文章长度的影响。原论文 [4] 中的第四章有具体的抛掉外力之后的好处（总不能只是为了自立门户而自立门户，自立门户的时候一定是有自己的优势的），但是这里的原因与主线关系不太大，这里就不做搬运工了。

总的来说，self-attention 的引入让 Attention 机制与 CNN、RNN 等网络具有了一样的地位（都开了公司），但是可以看到的是，这样做的限制还是很大的，所以 Transformer 的儿子们几乎都没有完整的引入 Transformer，都是有选择的保留一部分。

**Self-attention 计算细节**

但是，对于 self-attention 理解在后面理解这些兄弟们企业核心的区别很重要，所以这里我们占用篇幅搬运一个具体的计算例子（来自 Jalammar 的多篇文章，如果已经理解了可以跳过）：第一步，先是通过 $$X_i$$ 和各个 $$W$$（可训练的权重矩阵）的相乘得到 query, key 和 value 矩阵（如图 8 所示）：

<div style="text-align: center;">
<img src="/images/blog/from-word2vec-to-gpt-8.png" width="700px" />
</div>

<center>图 8：self-attention 原始矩阵（图源：http://jalammar.github.io/illustrated-transformer/）</center>

然后就是按照图 9 所示的步骤一步一步先计算 score，再 normalize (divide 那一步)，最后用 softmax 得到 attention score，然后用这个 attetion 作为权重求 $$v_1$$ 和 $$v_2$$ 的加权和，就得到最终的 self-attention 在这个位置（thinking 这个词）的输出值了。

<div style="text-align: center;">
<img src="/images/blog/from-word2vec-to-gpt-9.png" width="600px" />
</div>

<center>图 9：self-attention 计算流程（图源：http://jalammar.github.io/illustrated-transformer/）</center>

图 10 是一个具体的 self-attention 的例子，可以看到 it 这个词对自己的 attention 其实很小，更多的注意力放在了 a robot 上，因为 it 本身没有意思，它主要是指代前面的 a robot。**所以一个词的 query 和 key 是不同的**（相同的话相似度肯定是最大的，这样百分百的注意力都在自己身上），在做计算的时候是一视同仁的，虽然都是来自于 it 这个词，但是这里 it 的 key 告诉我们的信息就是它并不重要。

<div style="text-align: center;">
<img src="/images/blog/from-word2vec-to-gpt-10.png" width="700px" />
</div>

<center>图 10：self-attention 矩阵细节（图源：http://jalammar.github.io/）</center>

**绝不微不足道的区别**

前面介绍了那么多，甚至贴上了论文的原文，一个很重要的目的是为了强调 self-attention 层在 encoder 和 decoder 中是不一样的！encoder 在做 self-attention 的时候可以“attend to all positions”，而 decoder 只能“attend to all positions in the decoder up to and including that position”（划重点，涉及到 BERT 和 GPT 的一个重要区别）。简单来说，就是 decoder 跟过去的 Language model 一样，只能看到前面的信息，但是 encoder 可以看到完整的信息（双向信息）。具体细节在介绍到 BERT 和 GPT 的时候会详细介绍。

## 后浪时代 BERT & GPT & 其他

如果你足够细心的话，可以看到前面我提到的例子几乎都是机器翻译相关的，这是因为 Transformer 的 encoder-decoder 结构给其带来了很大的局限性。如果你想要做文本分类任务，使用 Transformer 的困难就很大，你也很难预训练好这个模型然后再在各种下游任务上 fine-tune。因此，Transformer 的儿子们给我们带来了令人惊喜的后浪时代。

**大儿子 Transformer-decoder [6] —— 语言模型回来了！**

Transformer 的大儿子首先发现了父亲公司的冗余机制，然后打出了自己的两块主要招牌：

> 翻译成中文就是：“我们的模型在裁员（去除 encoder）后，看的更远了，也能 pretrain 了，还是儿时的味道（Language Modelling）！”

可以看出大儿子是个黑心老板，他发现只需要用一部分 Transformer，就可以做成他想做的 language modelling，因此它只保留了 decoder，因为 decoder 在 Transformer 里的工作就是根据前面的词预测后面的词（跟 Language modelling 的任务一样）。但是如前文所述（图 11），Transformer 除了其提出的 self-attention 以外，还保留了过去的 encoder-decoder attetion，而现在因为没有 encoder 了，所以 encoder-decoder attention 层这里就没有了。

<div style="text-align: center;">
<img src="/images/blog/from-word2vec-to-gpt-11.png" width="700px" />
</div>

<center>图 11：Transformer 的 encoder-decoder 结构细节（图源：http://jalammar.github.io/）</center>

如果你仔细读了上面那一句话，然后看了图 11 并相应地移除了 encoder-decoder attention 层，你就会发现，encoder 和 decoder 的结构一模一样了！那为什么我还要提 BERT 和 GPT 使用的 Transformer 不同呢？先看一下图 12 中对于 transformer decoder 的描述：

![12](/images/blog/from-word2vec-to-gpt-12.png)

<center>图 12：transformer decoder 细节（图源: [6]）</center>

就是说本来 Transformer 中的 decoder 是要接收 encoder 那边对输入（m）处理后得到的信息，然后得到最终输出（y），而且在得到后面的 y 的时候，要考虑前面已经生成的 y，因此在去掉 encoder 之后，decoder 在考虑 y 的同时也得考虑 m 的信息，所以公司既然裁员了，那么 decoder 的活就得多一些，它的输入就是 m 和 y 的结合体了，同时中间还有 $\delta$ 来分隔开输入输出（黑心老板无疑了），当然这样做的时候，给的工资也是更高的（输入序列长度从 512 变成了 1024）。

基于回归语言模型的特性，Transformer-decoder 为自己赢下了极好的声誉，而这良好的声誉引起了他的弟弟——GPT-1 的注意。

**二儿子 GPT-1： NLP 也能迁移学习**

GPT-1 从大哥 Transoformer-decoder 的成功中看到了机会，并且挖掘出了更本质的商机——用预训练好的模型做下游任务。基于这个想法，他然后打出了自己的招牌：

> We demonstrate that large gains on these tasks (downstream tasks) can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task.

这块招牌里的 Generative Pre-Training（GPT）其实本质上就是大哥 Transformer-decoder 做的事，但是真的在 finetune 的时候，其实有很多麻烦，于是 GPT-1 公司体贴的在这个基础上做了很多很灵活的调整，给出了很多方案：

> we make use of task-aware input transformations during fine-tuning to achieve effective transfer while requiring minimal changes to the model architecture. 

具体 GPT-1 的具体工作流程如图 13 所示：

<div style="text-align: center;">
<img src="/images/blog/from-word2vec-to-gpt-13.png" width="700px" />
</div>

<center>图 13：如何使用 GPT（OpenAI Transformer）进行 Finetune（图源：http://jalammar.github.io/）</center>

图 14 展示了 GPT-1 给出的 finetune 方案，也就是前面说的对应不同任务的 input transformation，这些方案非常巧妙，也取得了很多成功，同样也使其获得了广泛的应用。但是 GPT-1 的热度并没有 BERT 高，因为当时的 GPT-1 没有足够商业头脑，媒体宣传度不够，从而在其兄弟 BERT 公司开张的时候被当做 “反面典型” 示众。当然，当时的 GPT 家族野心也不够大，他的 few-shot learning 的强大力量还没有展现出来（GPT-2 开始在做的事，后面详述）。

![14](/images/blog/from-word2vec-to-gpt-14.png)

<center>图 14：如何进行 Finetune（图源：http://jalammar.github.io/）</center>

**小儿子 BERT —— Encoder 也能撑起半边天**

在 GPT 的公司开的如日中天的时候，小儿子 BERT 也悄悄地长大了。叔叔 ELMo 最喜欢他兄弟的这个小儿子，所以常常带他来公司玩，也会给他讲一讲他们公司的业务，因此 “双向信息很重要” 这个概念也在小 BERT 的脑海中深深烙下了烙印。当他长大后，看到哥哥 GPT 公司的宣传标语时，觉得这不就是语言模型吗？不行，双向信息都没得到，得改！可是他不想直接进叔叔 ELMo 的公司，父亲 Transformer 又直接抛弃了叔叔公司的核心技术之一——LSTM，双向信息无法直接应用在 transformer 中（看一下 LSTM 和基于 self attention 的 Decoder 的工作机制就可以发现 Decoder 没办法像 LSTM 那样获得反向的信息）。

冥思苦想之后，他突然发现，父亲的 Encoder 不正是最好的选择吗？哥哥们用了 Decoder 做语言模型，那他用 Encoder 不也可以吗，而且还能获得双向信息（Masked language Model, MLM）。MLM 的大概思想就是本来自注意力机制不是主要注意自己嘛（类似于照镜子），那我就挡住你自己的脸，让你自己根据兄弟姐妹的样子（前后文信息）来猜自己的样子，等你能猜得八九不离十了，你就出师了，可以干活了。

但是小 BERT 还是太天真了，哥哥们选择 decoder 不是没有理由的，比如一个很实际的问题就是，BERT 既然用的是 Encoder，因为 encoder 输入的就是一个带 mask 的句子，那么怎么去做“双句问题”（如给定两个句子，说明是否是表达同一个意思）呢？经过仔细的考量，BERT 决定再学习一下哥哥们语言模型的特性，在预训练的时候加入了 Next sentence prediction 任务——就是给定句子 A，让你猜句子 B 是不是 A 后面的句子，这样句间关系也学到了。这个时候，BERT 公司就可以正式开业了。具体业务和工作方式如图 15 所示：

![15](/images/blog/from-word2vec-to-gpt-15.png)

<center>图 15：BERT 业务（图源：[10]）</center>

最后，还是要说一下 Encoder 和 Decoder 的区别，其实本质上是自回归模型（Auto regression）和自编码模型（Auto Encoder）的区别，他们并不是谁比谁更好的关系，而是一种要做权衡的关系。BERT 选择了 Encoder 给其带来的一个很重要的问题，Encoder 不具备 Decoder 的自回归特性（Auto Regressive），而自回归特性可以让模型有很明确的概率依据。这个区别在 XLNet（不重要的儿子们之一，后面会再稍微提一下）提出的时候尤为明显，因为 XLNet 属于 Auto Regressive 模型，而 BERT 属于 Auto Encoder 模型，为了更好地理解 AR 和 AE 模型的差异，我们来看一下 BERT 和 XLNet 的目标函数：

这是 XLNet（AR）的：

$$\max_{\theta}\quad \log p_\theta(\textbf{x}) = \sum_{t=1}^T \log p_\theta(x_t\mid \textbf{x}_{<t}) = \sum_{t=1}^T \log \frac{\exp\big(h_\theta(\textbf{x}_{1:t-1})^\top e(x_t)\big)}{\sum_{x'}\exp \big(h_\theta (\textbf{x}_{1:t-1})^\top e(x')\big)}$$

这是 BERT（AE）的：

$$\max_{\theta}\quad \log p_\theta(\bar{\textbf{x}}\mid \hat{\textbf{x}}) \approx \sum_{t=1}^T m_t\log p_\theta(x_t\mid\hat{\textbf{x}}) = \sum_{t=1}^T m_t\log\frac{\exp(H_\theta\big(\hat{\textbf{x}})_t^\top e(x_t)\big)}{\sum_{x'}\exp\big(H_\theta(\hat{\textbf{x}})_t^\top e(x')\big)}$$

这两个公式的具体意思我这里就不详细讲了，感兴趣的可以去看原论文，我之所以贴出这两个目标函数，重点在于 XLNet 的 “=” 和 BERT 的 “≈”。而这个“≈”，就是抛弃自回归特性带来的代价。至于具体原理，由于输入中预测的 token 是被 mask 的，因此 BERT 无法像自回归语言建模那样使用乘积法则（product rule）对联合概率进行建模，他只能假设那些被 mask 的 token 是独立的，而这个“≈” 就是来自于这个假设。同时，因为模型微调时的真实数据缺少 BERT 在预训练期间使用的 [MASK] 等人工符号，也就是在输入中加入了部分噪音，这会导致预训练和微调之间出现差异。而 AR 模型不需要在输入中加入这些噪音，也就不会出现这种问题了。

他的哥哥们之所以选择 AR 模型，是因为 AR 模型在生成任务中表现得更好，因为生成任务一般都是单向且正向的，而且 GPT 的招牌中就明确写出了他们的主要工作是 Gnereative pretraining。因此 AR 和 AE，具体来说就是选择 encoder 还是 decoder，其实最本质还是一种权衡利弊，最适合自己的就是最好的。

## 外传 Transformer 的私生子们

私生子们因为不受到重视，反而就会格外努力去遵循父亲的指导，尽心尽力去改进父亲的不足。Transformer 的其中一个私生子，transformer XL[17]（XL 的意思是 extra long），表现很出色（主要是他的儿子 XLNet[16]出色），让 transformer 回归了 AR 的本性，也让 Transformer 可以处理的序列更长了，前面也说过 AR 模型的优势。但是 AR 模型之所以没有这些缺陷是因为他们没有使用 Masked LM 方法，而 Masked LM 的提出正是为了解决 AR 模型的限制之一——AR 语言模型仅被训练用于编码单向语境（前向或后向），而下游语言理解任务通常需要双向语境信息。可见 AR 阵营本身有自己的优势，但如果想要获得更大的胜利，就必须找到一个比 Masked LM 更好的办法来解决双向语境问题。

可是他究其一生也没能完成这个任务，他的儿子 XLNet 弥补了它的遗憾，解决了双向语境问题，大概的方案就是

![16](/images/blog/from-word2vec-to-gpt-16.png)

<center>图 16：XLNet 提出的双向语境方案（图源：[16]）</center>

一开始 AR 模式不是只能单向工作吗？那就把输入乱序，找到所有的排列组合，并按照这些排列组合进行因式分解。当然这样计算量很大，XLNet 也想到了很好的方式来解决，具体怎么解决的这里就不说了，可以去原论文看一下，但是 XLNet 确实在当时也引起了不小的波澜，也算是私生子们一场不小的胜利了。

## 现代 GPT2 & 3 —— 大即正义 (Bigger than bigger)

最后回到当下，也是现在的 GPT2 和 GPT3，读到这里，其实 GPT2 和 GPT3 就没什么技术细节可以讲了，他们发现父辈们已经开创了一个时代，比如 BERT 打破了自然语言处理领域模型在各项任务中的记录，而且在描述模型的论文发布后不久，该团队还开源了该模型的代码，并提供了已经在大量数据集上进行了预训练的模型。这是一项重要的发展，因为它使任何构建涉及语言处理的机器学习模型的人都可以使用此强大的功能作为随时可用的组件，从而节省了从训练语言处理模型上来的时间，精力，知识和资源（如图 17 所示）。

![17](/images/blog/from-word2vec-to-gpt-17.png)

<center>图 17：新时代（图源：http://jalammar.github.io/）</center>

回到 GPT，在介绍 GPT-1 的后代们做了什么之前（主要是扩大模型），先看一下 GPT-2 和 GPT-3 的论文名字：

- Language models are unsupervised multitask learners. (GPT-2)
- Language Models are Few-Shot Learners. (GPT-3)

看到这些名字，第一感觉大概就是“一脉相承”。实际上 GPT 的后代们也是这么做的，GPT-1 的后代们的目标是实现 zero-shot learning，取消 fine-tune 机制！这也是为什么 GPT-3 能够大火，给人以经验的原因。到了 GPT-2，就开始跨出了创造性的一步——去掉了 fine-tuning 层，再针对不同任务分别进行微调建模，而是不定义这个模型应该做什么任务，模型会自动识别出来需要做什么任务。这就好比一个人博览群书，你问他什么类型的问题，他都可以顺手拈来，GPT-2 就是这样一个博览群书的模型 [18]。其他的特征主要就是扩大了公司规模（扩大数据及，增加参数，加大词汇表，上下文大小从 512 提升到了 1024 tokens），除此之外，也对 transformer 进行了调整，将 layer normalization 放到每个 sub-block 之前，并在最后一个 Self-attention 后再增加一个 layer normalization。

总的来说 GPT-2 跟 GPT-1 的区别如 GPT-2 的名字所示，他要让语言模型变成 unsupervised multitask learner，[19]给了一个很简洁的对比，我搬运过来供大家参考理解：

- 数据质量：GPT 2 更高，进行了筛选
- 数据广度：GPT 2 更广， 包含网页数据和各种领域数据
- 数据数量：GPT 2 更大，WebText，800 万网页
- 数据模型：模型更大，15 亿参数
- 结构变化：变化不大
- 两阶段 vs 一步到位：GPT 1 是两阶段模型，通过语言模型预训练，然后通过 Finetuning 训练不同任务参数。而 GPT 2 直接通过引入特殊字符，从而一步到位解决问题

到了 GPT-3，如果去看一下论文就发现其实 GPT-3 更像一个厚厚的技术报告，来告诉大家 GPT-3 怎么做到 few-shot 甚至 zero-shot learning，他的内核细节这里已经没有什么要单独提及的了，他的庞大和财大气粗就是他最大的特色（整个英语维基百科（约 600 万个词条）仅占其训练数据的 0.6％），如果有机会，还是希望大家可以自己去试一下这个模型，去体验一下 GPT-3 带来的魅力。

## 总结

读完这篇文章，估计就可以发现，所有的技术都不是凭空而来的，都是一点一点进步得来的，从源头开始，梳理一下一个模型的“集团成员”，不仅仅可以对这个领域有更深刻的理解，对于这个模型的每一块技术，都能有更加深刻的理解。

同时，在实际应用的时候，不是最新的模型就是最好的，还要考虑这个模型的大小是否合适，模型在你特定所需的任务上表现是否优秀等等等等，对整个 NLP 领域有更广泛的理解，你在做选择的时候就更能做出更好地选择，而不是在别人问到你为什么选择 BERT 的时候说一句，“哦，我只会 BERT。”

## 参考文献

[1] Mikolov, Tomas; et al. (2013). "Efficient Estimation of Word Representations in Vector Space". arXiv (https://en.wikipedia.org/wiki/ArXiv_(identifier)):1301.3781 (https://arxiv.org/abs/1301.3781) [cs.CL (https://arxiv.org/archive/cs.CL)].   
[2]Mikolov, Tomas (2013). "Distributed representations of words and phrases and their compositionality". Advances in neural information processing systems.  
[3] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, & Luke Zettlemoyer. (2018). Deep contextualized word representations.  
[4] Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin (2017). Attention Is All You NeedCoRR, abs/1706.03762.  
[5] Zihang Dai and Zhilin Yang and Yiming Yang and Jaime G. Carbonell and Quoc V. Le and Ruslan Salakhutdinov (2019). Transformer-XL: Attentive Language Models Beyond a Fixed-Length ContextCoRR, abs/1901.02860.  
[6] P. J. Liu, M. Saleh, E. Pot, B. Goodrich, R. Sepassi, L. Kaiser, and N. Shazeer. Generating wikipedia by summarizing long sequences. ICLR, 2018.  
[7] Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.  
[8] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI Blog, 1(8), 9.  
[9] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, & Dario Amodei. (2020). Language Models are Few-Shot Learners.  
[10]Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language UnderstandingCoRR, abs/1810.04805.  
[11]Zhilin Yang and Zihang Dai and Yiming Yang and Jaime G. Carbonell and Ruslan Salakhutdinov and Quoc V. Le (2019). XLNet: Generalized Autoregressive Pretraining for Language UnderstandingCoRR, abs/1906.08237.  
[12] attention 机制及 self-attention(transformer). Accessed at: https://blog.csdn.net/Enjoy_endless/article/details/88679989  
[13] Attention 机制详解（一）——Seq2Seq 中的 Attention. Accessed at: https://zhuanlan.zhihu.com/p/47063917  
[14]一文看懂 Attention（本质原理 + 3 大优点 + 5 大类型. Accessed at:https://medium.com/@pkqiang49/%E4%B8%80%E6%96%87%E7%9C%8B%E6%87%82-attention-%E6%9C%AC%E8%B4%A8%E5%8E%9F%E7%90%86-3%E5%A4%A7%E4%BC%98%E7%82%B9-5%E5%A4%A7%E7%B1%BB%E5%9E%8B-e4fbe4b6d030  
[15]The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning). Accessed at:http://jalammar.github.io/illustrated-bert/  
[16] Yang, Zhilin, et al. "Xlnet: Generalized autoregressive pretraining for language understanding." Advances in neural information processing systems. 2019.  
[17] Dai, Zihang, et al. "Transformer-xl: Attentive language models beyond a fixed-length context." arXiv preprint arXiv:1901.02860 (2019).  
[18] NLP——GPT 对比 GPT-2. Accessed at: https://zhuanlan.zhihu.com/p/96791725  
[19] 深度学习：前沿技术 - GPT 1 & 2. Accessed at: http://www.bdpt.net/cn/2019/10/08/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%EF%BC%9A%E5%89%8D%E6%B2%BF%E6%8A%80%E6%9C%AF-gpt-1-2/

> 转载自[机器之心](https://mp.weixin.qq.com/s/dKbGR4sCkNpik0Xw41QLVw)，作者：王子嘉，编辑：H4O。部分内容有修改。