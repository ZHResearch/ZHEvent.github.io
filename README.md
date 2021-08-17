# 事件论文阅读笔记

专注于事件以及事件关系抽取任务：

- [事件抽取（Event Detection）](EventDetection.md)
- [事件同指消解（Event Coreference Resolution）](EventCoreference.md)
- [事件时序关系（Temporal Relation Extraction）](#temporal-relation-extraction)
- [事件因果关系（Causal Relation Extraction）](#causal-relation-extraction)
- [事件包含关系（Subevent Relation Extraction）](#subevent-relation-extraction)
- [事件空间关系（Spatial Relation Extraction）](#spatial-relation-extraction)
- [事件相关及应用（Other）](#other)

## Temporal Relation Extraction

| 年份 | 来源 | 名称 | 作者 | 笔记 |
| :- | :-: | :- | :- | :- |
| 2021 | arxiv | [DEER: A Data Efficient Language Model for Event Temporal Reasoning](https://arxiv.org/pdf/2012.15283) | Rujun Han |  |
| 2021 | arxiv | [Clinical Temporal Relation Extraction with Probabilistic Soft Logic Regularization and Global Inference](https://arxiv.org/pdf/2012.08790) | Yichao Zhou |  |
| 2021 | NAACL | [Temporal Reasoning on Implicit Events from Distant Supervision](https://www.aclweb.org/anthology/2021.naacl-main.107/) | Ben Zhou |  |
| 2021 | NAACL | [EventPlus: A Temporal Event Understanding Pipeline](https://www.aclweb.org/anthology/2021.naacl-demos.7/) | Mingyu Derek Ma |  |
| 2020 | EMNLP | [Joint Constrained Learning for Event-Event Relation Extraction](https://www.aclweb.org/anthology/2020.emnlp-main.51/) | Haoyu Wang |  |
| 2020 | EMNLP | [TORQUE: A Reading Comprehension Dataset of Temporal Ordering Questions](https://www.aclweb.org/anthology/2020.emnlp-main.88/) | Qiang Ning |  |
| 2020 | EMNLP | [Dynamically Updating Event Representations for Temporal RelationClassification with Multi-category Learning](https://www.aclweb.org/anthology/2020.findings-emnlp.121/) | Fei Cheng |  |
| 2020 | ACL | [Temporal Common Sense Acquisition with Minimal Supervision](https://www.aclweb.org/anthology/2020.acl-main.678/) | Ben Zhou |  |
| 2020 | EMNLP  | [Domain Knowledge Empowered Structured Neural Net for End-to-End Event Temporal Relation Extraction](https://www.aclweb.org/anthology/2020.emnlp-main.461/) | Rujun Han | [王亮](https://zhevent.github.io/2020/11/30/domain-knowledge/) |
| 2020 | EMNLP  | [Severing the Edge Between Before and After:Neural Architectures for Temporal Ordering of Events](https://www.aclweb.org/anthology/2020.emnlp-main.436/) | Hayley Ross | [王亮](https://zhevent.github.io/2020/12/09/tdp/) |
| 2020 | EMNLP  | [Exploring Contextualized Neural Language Models for Temporal Dependency Parsing](https://www.aclweb.org/anthology/2020.emnlp-main.689/) | Miguel Ballesteros | [王亮](https://zhevent.github.io/2020/12/03/smtl/) |
| 2020 | EACL | [Effective Distant Supervision for Temporal Relation Extraction](https://arxiv.org/abs/2010.12755) | Xinyu Zhao | [王亮](https://zhevent.github.io/2020/12/17/Distant-Supervision-for-temprel/) |
| 2020 | BioNLP | [A BERT-based One-Pass Multi-Task Model for Clinical Temporal Relation Extraction](https://www.aclweb.org/anthology/2020.bionlp-1.7/) | Chen Lin ||
| 2019 | ACL    | [Fine-Grained Temporal Relation Extraction](https://www.aclweb.org/anthology/P19-1280/) | Siddharth Vashishtha |[黄琮程](https://zhevent.github.io/2020/10/22/Fine-Grained-Temporal-Relation-Extraction/)|
| 2019 | EMNLP  | [Joint Event and Temporal Relation Extraction with Shared Representations and Structured Prediction](https://www.aclweb.org/anthology/D19-1041/) | Rujun Han |[黄琮程](https://zhevent.github.io/2020/10/15/joint-event-and-temporal/)|
| 2019 | EMNLP  | [An Improved Neural Baseline for Temporal Relation Extraction](https://www.aclweb.org/anthology/D19-1642/) | Qiang Ning |[李婧](https://zhevent.github.io/2020/10/31/An-improved-Neural-Baseline-for-Temporal-Relation-Extraction/)|
| 2019 | CoNLL  | [Deep Structured Neural Network for Event Temporal Relation Extraction](https://www.aclweb.org/anthology/K19-1062/) | Rujun Han | [王亮](https://zhevent.github.io/2020/11/17/deep-ssvm/) |
| 2019 | ClinicalNLP  | [A BERT-based Universal Model for Both Within- and Cross-sentence Clinical Temporal Relation Extraction](https://www.aclweb.org/anthology/W19-1908/) | Chen Lin ||
| 2019 | ClinicalNLP | [Attention Neural Model for Temporal Relation Extraction](https://www.aclweb.org/anthology/W19-1917/) | Sijia Liu ||
| 2018 | EMNLP | [CogCompTime-A tool for understanding Time in Natural Language](https://www.aclweb.org/anthology/D18-2013/) | Qiang Ning ||
| 2018 | ACL  | [Temporal Event Knowledge Acquisition via Identifying Narratives](https://www.aclweb.org/anthology/P18-1050//) | Wenlin Yao | [王亮](https://zhevent.github.io/2020/11/04/temporal-knowledge-via-Narratives/) |
| 2018 | ACL    | [Context-Aware Neural Model for Temporal Information Extraction](https://www.aclweb.org/anthology/P18-1049/) | Yuanliang Meng ||
| 2018 | ACL  | [A Multi-Axis Annotation Scheme for Event Temporal Relations](https://www.aclweb.org/anthology/P18-1122/) | Qiang Ning | [王亮](https://zhevent.github.io/2020/10/28/multi-axis-annotation-for-tempre/) |
| 2018 | ACL | [Joint Reasoning for Temporal and Causal Relations](https://www.aclweb.org/anthology/P18-1212/) | Qiang Ning ||
| 2018 | EMNLP | [Self-training improves Recurrent Neural Networks performance for Temporal Relation Extraction](https://www.aclweb.org/anthology/W18-5619/) | Chen Lin ||
| 2018 | NAACL  | [Improving Temporal Relation Extraction with a Globally Acquired Statistical Resource](https://www.aclweb.org/anthology/N18-1077/) | Qiang Ning | [王亮](https://zhevent.github.io/2020/10/21/improving-TempRel_with-statistical-resource/#improving-global-methods) |
| 2018 | NAACL  | [Inducing Temporal Relations from Time Anchor Annotation](https://www.aclweb.org/anthology/N18-1166/) | Fei Cheng | [王亮](https://zhevent.github.io/2020/11/12/temprel-from-time-anchor-annotation/) |
| 2018 | SEMEVAL | [Exploiting Partially Annotated Data in Temporal Relation Extraction](https://www.aclweb.org/anthology/S18-2018/) | Qiang Ning ||
| 2017 | ACL    | [Classifying Temporal Relations by Bidirectional LSTM over Dependency Paths](https://www.aclweb.org/anthology/P17-2001/) | Fei Cheng ||
| 2017 | ACL    | [Neural Architecture for Temporal Relation Extraction: A Bi-LSTM Approach for Detecting Narrative Containers](https://www.aclweb.org/anthology/P17-2035/) | Julien Tourille ||
| 2017 | EACL | [Structured Learning for Temporal Relation Extraction from Clinical Records](https://www.aclweb.org/anthology/E17-1108/) | Artuur Leeuwenberg ||
| 2017 | EACL | [Neural Temporal Relation Extraction](https://www.aclweb.org/anthology/E17-2118/) | Dmitriy Dligach ||
| 2017 | EMNLP  | [A Structured Learning Approach to Temporal Relation Extraction](https://www.aclweb.org/anthology/D17-1108/) | Qiang Ning ||
| 2017 | EMNLP  | [A Sequential Model for Classifying Temporal Relations between Intra-Sentence Events](https://www.aclweb.org/anthology/D17-1190/) | Prafulla K. Choubey ||
| 2017 | EMNLP  | [Temporal Information Extraction for Question Answering Using Syntactic Dependencies in an LSTM-based Architecture](https://www.aclweb.org/anthology/D17-1092/) | Yuanliang Meng ||
| 2017 | BioNLP | [Representations of Time Expressions for Temporal Relation Extraction with Convolutional Neural Networks](https://www.aclweb.org/anthology/W17-2341/) | Chen Lin ||
| 2017 | RANLP | [A Weakly Supervised Approach to Train Temporal Relation Classifiers and Acquire Regular Event Pairs Simultaneously](https://www.aclweb.org/anthology/R17-1103/) | Wenlin Yao ||
| 2016 | COLING | [CATENA: CAusal and TEmporal relation extraction from NAtural language texts](https://www.aclweb.org/anthology/C16-1007/) | Paramita Mirza ||
| 2016 | COLING | [Global Inference to Chinese Temporal Relation Extraction](https://www.aclweb.org/anthology/C16-1137/) | Peifeng Li ||
| 2016 | COLING | [On the contribution of word embeddings to temporal relation classiﬁcation](https://www.aclweb.org/anthology/C16-1265/) | Paramita Mirza ||
| 2014 | TACL | [Dense Event Ordering with a Multi-Pass Architecture](https://www.aclweb.org/anthology/Q14-1022/) | Nathanael Chambers ||

## Causal Relation Extraction

| 年份 | 来源 | 名称 | 作者 | 笔记 |
| :- | :-: | :- | :- | :- |
| 2021 | NAACL | [Graph Convolutional Networks for Event Causality Identification with Rich Document-level Structures](https://aclanthology.org/2021.naacl-main.273.pdf) | Minh Tran Phu ||
| 2020 | IJCAI | [Knowledge Enhanced Event Causality Identification with Mention Masking Generalizations](https://www.ijcai.org/proceedings/2020/0499.pdf) | Jian Liu ||
| 2020 | COLING | [KnowDis: Knowledge Enhanced Data Augmentation for Event Causality Detection via Distant Supervision](https://aclanthology.org/2020.coling-main.135.pdf) | Xinyu Zuo ||
| 2019 | NAACL | [Modeling Document-level Causal Structures for Event Causal Relation Identiﬁcation](https://aclanthology.org/N19-1179/) | Lei Gao ||

## Subevent Relation Extraction

| 年份 | 来源 | 名称 | 作者 | 笔记 |
| :- | :-: | :- | :- | :- |
| 2019 | ACL | [Detecting Subevents using Discourse and Narrative Features](https://www.aclweb.org/anthology/P19-1471/) | Mohammed Aldawsari ||
| 2019 |  NAACL-HLT | [Sub-Event Detection from Twitter Streams as a Sequence Labeling Problem](https://aclanthology.org/N19-1081/) | Giannis Bekoulis ||
| 2018 | COLING | [Graph Based Decoding for Event Sequencing and Coreference Resolution](https://www.aclweb.org/anthology/C18-1309/) | Zhengzhong Liu ||

## Spatial Relation Extraction

| 年份 | 来源 | 名称 | 作者 | 笔记 |
| :- | :-: | :- | :- | :- |
| 2018 | NAACL-HLT | [Visually Guided Spatial Relation Extraction from Text](https://aclanthology.org/N18-2124/) | Taher Rahgooy ||

## Other

| 年份 | 来源 | 名称 | 作者 | 笔记 |
| :- | :-: | :- | :- | :- |
| 2020 | EMNLP Findings | [Weakly-Supervised Modeling of Contextualized Event Embedding for Discourse Relations](https://aclanthology.org/2020.findings-emnlp.446/) | I-Ta Lee ||
| 2020 | ACL | [GAIA: A Fine-grained Multimedia Knowledge Extraction System](https://www.aclweb.org/anthology/2020.acl-demos.11/) | Manling Li ||
| 2020 | ACL | [Improving Multimodal Named Entity Recognition via Entity Span Detection with Unified Multimodal Transformer](https://www.aclweb.org/anthology/2020.acl-main.306/) | Jianfei Yu||
| 2020 | COLING | [Is Killed More Signiﬁcant than Fled? A Contextual Model for Salient Event Detection](https://aclanthology.org/2020.coling-main.10/) | Disha Jindal ||
| 2020 | LREC | [Towards Few-Shot Event Mention Retrieval: An Evaluation Framework and A Siamese Network Approach](https://www.aclweb.org/anthology/2020.lrec-1.216/) | Bonan Min ||
| 2020 | CODI | [Joint Modeling of Arguments for Event Understanding](https://aclanthology.org/2020.codi-1.10/) | Yunmo Chen ||
| 2019 | ACL | [Multi-Level Matching and Aggregation Network for Few-Shot Relation Classification](https://www.aclweb.org/anthology/P19-1277/) | Zhixiu Ye | [王亮](https://zhevent.github.io/2020/10/16/few-shot-relation-classification/) |
| 2019 | EMNLP | [A Regularization Approach for Incorporating Event Knowledge and Coreference Relations into Neural Discourse Parsing](https://aclanthology.org/D19-1295/) | Zeyu Dai ||
| 2019 | NACL | [Multilingual Entity, Relation, Event and Human Value Extraction](https://www.aclweb.org/anthology/N19-4019/) | Manling Li ||
| 2019 | NAACL | [Modeling Document-level Causal Structures for Event Causal Relation Identification](https://www.aclweb.org/anthology/N19-1179/) | Lei Gao |
| 2018 | LREC | [EventWiki: A Knowledge Base of Major Events](https://aclanthology.org/L18-1079/) | Tao Ge ||
| 2016 | EVENTS | [Event Nugget and Event Coreference Annotation](https://www.aclweb.org/anthology/W16-1005/) | Zhiyi Song |
