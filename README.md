# 事件论文阅读笔记

专注于事件以及事件关系抽取任务：

- [事件抽取（Event Detection）](EventDetection.md)
- [事件同指消解（Event Coreference Resolution）](EventCoreference.md)
- [事件时序关系（Temporal Relation Extraction）](EventTemporal.md)
- [事件因果关系（Causal Relation Extraction）](#causal-relation-extraction)
- [事件包含关系（Subevent Relation Extraction）](#subevent-relation-extraction)
- [事件空间关系（Spatial Relation Extraction）](#spatial-relation-extraction)
- [事件相关及应用（Other）](#other)

## Causal Relation Extraction

| 年份 | 来源 | 名称 | 作者 | 笔记 |
| :- | :-: | :- | :- | :- |
| 2022 | COLING | [Event Causality Identification via Derivative Prompt Joint Learning](https://aclanthology.org/2022.coling-1.200/) | Shirong Shen ||
| 2021 | ACL | [ExCAR: Event Graph Knowledge Enhanced Explainable Causal Reasoning](https://aclanthology.org/2021.acl-long.183/) | Li Du ||
| 2021 | ACL | [Knowledge-Enriched Event Causality Identification via Latent Structure Induction Networks](https://aclanthology.org/2021.acl-long.376/) | Pengfei Cao ||
| 2021 | NAACL | [Graph Convolutional Networks for Event Causality Identification with Rich Document-level Structures](https://aclanthology.org/2021.naacl-main.273.pdf) | Minh Tran Phu ||
| 2020 | AAAI | [Causal Knowledge Extraction through Large-Scale Text Mining](https://ojs.aaai.org/index.php/AAAI/article/view/7092) | Oktie Hassanzadeh ||
| 2020 | IJCAI | [Knowledge Enhanced Event Causality Identification with Mention Masking Generalizations](https://www.ijcai.org/proceedings/2020/0499.pdf) | Jian Liu | Note <br/> [`Code`](https://github.com/jianliu-ml/EventCausalityIdentification) |
| 2020 | COLING | [KnowDis: Knowledge Enhanced Data Augmentation for Event Causality Detection via Distant Supervision](https://aclanthology.org/2020.coling-main.135.pdf) | Xinyu Zuo ||
| 2019 | NAACL | [Modeling Document-level Causal Structures for Event Causal Relation Identiﬁcation](https://aclanthology.org/N19-1179/) | Lei Gao ||

## Subevent Relation Extraction

| 年份 | 来源 | 名称 | 作者 | 笔记 |
| :- | :-: | :- | :- | :- |
| 2021 | EMNLP | [Learning Constraints and Descriptive Segmentation for Subevent Detection](https://aclanthology.org/2021.emnlp-main.423/) | Haoyu Wang | Note <br/> [`Code`](https://github.com/CogComp/Subevent_EventSeg) |
| 2019 | ACL | [Detecting Subevents using Discourse and Narrative Features](https://www.aclweb.org/anthology/P19-1471/) | Mohammed Aldawsari ||
| 2019 |  NAACL-HLT | [Sub-Event Detection from Twitter Streams as a Sequence Labeling Problem](https://aclanthology.org/N19-1081/) | Giannis Bekoulis ||

## Few-shot Relation Extraction

| 年份 | 来源 | 名称 | 作者 | 笔记 |
| :- | :-: | :- | :- | :- |
| 2022 | ACL | [Continual Few-shot Relation Learning via Embedding Space Regularization and Data Augmentation](https://aclanthology.org/2022.acl-long.198/) | Chengwei Qin ||
| 2022 |  ACL | [Pre-training to Match for Unified Low-shot Relation Extraction](https://aclanthology.org/2022.acl-long.397/) | Fangchao Liu ||
| 2022 |  ACL | [Leveraging Prompts to Generate Synthetic Data for Zero-Shot Relation Triplet Extraction](https://aclanthology.org/2022.findings-acl.5.pdf) | Yew Ken Chia ||
| 2022 |  ACL | [A Hierarchical Contrastive Learning Framework for Distantly Supervised Relation Extraction](https://aclanthology.org/2022.findings-acl.202.pdf) | Dongyang Li ||
| 2022 |  ACL | [Improving Discriminative Learning for Zero-Shot Relation Extraction](https://aclanthology.org/2022.spanlp-1.1.pdf) | Van-Hien Tran ||


## Spatial Relation Extraction

| 年份 | 来源 | 名称 | 作者 | 笔记 |
| :- | :-: | :- | :- | :- |
| 2021 | MDAI | [Spatial Role Labeling System Capturing Both Characters and Word Information Using BiLSTM and CRF](http://www.mdai.cat/mdai2021/proceedings.MDAI2021.usb.pdf#page=63) | Alaeddine Moussa ||
| 2021 | IEEE | [Coarse-to-Fine Spatial-Temporal Relationship Inference for Temporal Sentence Grounding](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9475486) | SHANSHAN QI ||
| 2020 | EMNLP | [BERT-based Spatial Information Extraction](https://aclanthology.org/2020.splu-1.2.pdf) | Hyeong Jin Shin* ||
| 2020 | EMNLP | [Decoding Language Spatial Relations to 2D Spatial Arrangements](https://aclanthology.org/2020.findings-emnlp.408.pdf) | Gorjan Radevski ||
| 2018 | ICLR | [Interactive Grounded Language Acquisition and Generalization in a 2D World](https://arxiv.org/pdf/1802.01433.pdf) | Haonan Yu ||
| 2018 | NAACL-HLT | [Visually Guided Spatial Relation Extraction from Text](https://aclanthology.org/N18-2124/) | Taher Rahgooy ||
| 2016 | COLING | [Extracting Spatial Entities and Relations in Korean Text](https://aclanthology.org/C16-1225.pdf) | Bogyum Kim and Jae Sung Lee ||
| 2016 | ACM | [Extracting new spatial entities and relations from short messages](https://dl.acm.org/doi/pdf/10.1145/3012071.3012079) | Sarah Zenasni ||
| 2015 | SemEval | [A WordNet-based approach towards the Automatic Recognition of Spatial Information following the ISO-Space Annotation Scheme](https://aclanthology.org/S15-2145.pdf) | Haritz Salaberri, Olatz Arregi, Beñat Zapirain ||
| 2015 | ACL | [SemEval-2015 Task 8: SpaceEval](https://aclanthology.org/S15-2149.pdf) | James Pustejovsky||
| 2015 | SemEval | [Ensemble-Based Spatial Relation Extraction](https://aclanthology.org/S15-2146.pdf) | Jennifer D’Souz, Vincent Ng ||
| 2015 | SemEval | [SpRL-CWW: Spatial Relation Classification with Independent Multi-class Models](https://aclanthology.org/S15-2150.pdf) | Eric Nichols ||


## Other

| 年份 | 来源 | 名称 | 作者 | 笔记 |
| :- | :-: | :- | :- | :- |
| 2022 | AAAI | [Selecting Optimal Context Sentences for Event-Event Relation Extraction](https://ojs.aaai.org/index.php/AAAI/article/view/21354) | Hieu Man | Note <br> `Code` |
| 2022 | ACL | [Legal Judgment Prediction via Event Extraction with Constraints](https://aclanthology.org/2022.acl-long.48/) | Yi Feng | Note <br> [`Code`](https://github.com/wapay/epm) |
| 2022 | ACL | [Event-Event Relation Extraction using Probabilistic Box Embedding](https://aclanthology.org/2022.acl-short.26/) | EunJeong Hwang | Note <br/> [`Code`](https://github.com/iesl/CE2ERE) |
| 2021 | ACL-SRW | [Joint Detection and Coreference Resolution of Entities and Events with Document-level Context Aggregation](https://aclanthology.org/2021.acl-srw.18/) | Samuel Kriman | Note <br/> `Code` |
| 2021 | arXiv | [What is Event Knowledge Graph: A Survey](https://arxiv.org/abs/2112.15280) | Saiping Guan | Note <br/> [`Code`](https://github.com/sam1373/long_ie) |
| 2020 | EMNLP Findings | [Weakly-Supervised Modeling of Contextualized Event Embedding for Discourse Relations](https://aclanthology.org/2020.findings-emnlp.446/) | I-Ta Lee ||
| 2020 | ACL | [GAIA: A Fine-grained Multimedia Knowledge Extraction System](https://www.aclweb.org/anthology/2020.acl-demos.11/) | Manling Li ||
| 2020 | ACL | [Improving Multimodal Named Entity Recognition via Entity Span Detection with Unified Multimodal Transformer](https://www.aclweb.org/anthology/2020.acl-main.306/) | Jianfei Yu||
| 2020 | COLING | [Is Killed More Signiﬁcant than Fled? A Contextual Model for Salient Event Detection](https://aclanthology.org/2020.coling-main.10/) | Disha Jindal ||
| 2020 | LREC | [Towards Few-Shot Event Mention Retrieval: An Evaluation Framework and A Siamese Network Approach](https://www.aclweb.org/anthology/2020.lrec-1.216/) | Bonan Min ||
| 2020 | CODI | [Joint Modeling of Arguments for Event Understanding](https://aclanthology.org/2020.codi-1.10/) | Yunmo Chen ||
| 2019 | ACL | [Multi-Level Matching and Aggregation Network for Few-Shot Relation Classification](https://www.aclweb.org/anthology/P19-1277/) | Zhixiu Ye | [王亮](https://zhevent.github.io/2020/10/16/few-shot-relation-classification/) |
| 2019 | EMNLP | [A Regularization Approach for Incorporating Event Knowledge and Coreference Relations into Neural Discourse Parsing](https://aclanthology.org/D19-1295/) | Zeyu Dai ||
| 2019 | NACL | [Multilingual Entity, Relation, Event and Human Value Extraction](https://www.aclweb.org/anthology/N19-4019/) | Manling Li ||
| 2019 | NAACL | [Modeling Document-level Causal Structures for Event Causal Relation Identification](https://www.aclweb.org/anthology/N19-1179/) | Lei Gao ||
| 2018 | LREC | [EventWiki: A Knowledge Base of Major Events](https://aclanthology.org/L18-1079/) | Tao Ge ||
| 2016 | EVENTS | [Event Nugget and Event Coreference Annotation](https://www.aclweb.org/anthology/W16-1005/) | Zhiyi Song ||
