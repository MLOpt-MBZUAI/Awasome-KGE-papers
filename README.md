# Awasome Knowledge Graph Embedding (KGE) Papers (2021-2024)

## Table of Contents
1. [Introduction](#introduction)
2. [Paper Summary by Year](#paper-summary-by-year)
   - [Year 2024](#year-2024)
   - [Year 2023](#year-2023)
   - [Year 2022](#year-2022)
   - [Year 2021](#year-2021)

## Introduction
Knowledge Graph Embeddings (KGE) aim to map entities and relations from knowledge graphs into low-dimensional space while preserving the graph structure. Over recent years, KGE has played a significant role in various applications such as recommendation systems, question-answering systems, and healthcare.

This document aims to review the progress in KGE research from 2021 to 2024 in top-tier conferences such as NeurIPS, ICLR, ICML, AAAI, ACL, EMNLP, IJCAI, SIGIR, WWW and KDD. We will analyze the most influential papers of each year, explore the evolution of methods, and discuss future directions.

## Paper Summary by Year

### Year 2024
- AAAI 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|
| APST     |Zhixiang Su (NTU)      | Anchoring Path for Inductive Relation Prediction in Knowledge Graphs       |Inductive Relation Prediction |  Replacing the reliance on closed paths by introducing anchoring paths (APs)               | [Link](https://arxiv.org/pdf/2312.13596)   | [Code]   |
|ImgFact |Jingping Liu (ECUST) | Beyond Entities: A Large-Scale Multi-Modal Knowledge Graph with Triplet Fact Grounding |Multi-modal Knowledge Graph Construction |  The paper introduces ImgFact, a novel large-scale multi-modal knowledge graph that grounds triplet facts (entities and relations) on images to enhance the performance of NLP tasks like relation classification and link prediction.               | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/29828/31438)   | [Code](https://github.com/kleinercubs/ImgFact)  |
| KGEditor | Siyuan Cheng (Zhejiang University) | Editing Language Model-Based Knowledge Graph Embeddings | Editing and updating Knowledge Graph embeddings efficiently | The paper proposes KGEditor, a strong baseline that allows efficient updates and edits of knowledge graph embeddings without re-training, using additional parametric layers via a hypernetwork. | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/29737) | [Code](https://github.com/AnonymousForPapers/DeltaKG) |
| HGE | Jiaxin Pan (University of Stuttgart) | HGE: Embedding Temporal Knowledge Graphs in a Product Space of Heterogeneous Geometric Subspaces | Temporal Knowledge Graph Embedding | The paper introduces HGE, a method that embeds temporal knowledge graphs into a product space consisting of multiple heterogeneous geometric subspaces, using both temporal-relational and temporal-geometric attention mechanisms to model diverse temporal patterns. | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/28739) | [Code](https://github.com/NacyNiko/HGE) |
| KGDM | Xiao Long (USTC) | KGDM: A Diffusion Model to Capture Multiple Relation Semantics for Knowledge Graph Embedding | Knowledge Graph Embedding | The paper proposes KGDM, a novel diffusion model for embedding knowledge graphs, designed to capture multiple relation semantics using denoising diffusion probabilistic models to improve performance on knowledge graph completion tasks. | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/28732s) | [Code](https://github.com/key2long/KGDM)|
| LAFA | Bin Shang (Xi’an Jiaotong University) | LAFA: Multimodal Knowledge Graph Completion with Link Aware Fusion and Aggregation | Multimodal Knowledge Graph Completion | The paper presents LAFA, a model that improves multimodal knowledge graph completion by leveraging a link-aware fusion and aggregation mechanism, which selectively fuses visual and structural embeddings based on the importance of images and structural relationships in different link scenarios. | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/28744) | [Code] |
| MGTCA | Bin Shang (Xi’an Jiaotong University) | Mixed Geometry Message and Trainable Convolutional Attention Network for Knowledge Graph Completion | Knowledge Graph Completion | The paper presents MGTCA, a model that integrates mixed geometry spaces and trainable convolutional attention networks to enhance knowledge graph completion tasks by improving neighbor message aggregation and representation quality. | [Link](https://ojs.aaai.org/index.php/AAAI/article/download/28745/29434) | [Code] |
| MKG-FENN | Di Wu (Chongqing University of Posts and Telecommunications) | MKG-FENN: A Multimodal Knowledge Graph Fused End-to-End Neural Network for Accurate Drug–Drug Interaction Prediction | Drug–Drug Interaction Prediction | The paper introduces MKG-FENN, a model that uses multimodal knowledge graphs and a fused end-to-end neural network to predict drug-drug interactions with high accuracy by extracting and fusing drug-related features. | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/28887) | [Code](https://github.com/wudi1989/MKG-FENN) |
| CompilE | Wanyun Cui (Shanghai University of Finance and Economics) | Modeling Knowledge Graphs with Composite Reasoning | Knowledge Graph Completion | The paper presents CompilE, a framework that unifies different knowledge graph models by combining facts from various entities for better reasoning and improved performance on knowledge graph completion tasks. | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/28675) | [Code](https://github.com/zlq147/CompilE) |
|NestE |Bo Xiong (University of Stuttgart) |NestE: Modeling Nested Relational Structures for Knowledge Graph Reasoning | Knowledge Graph Reasoning | This paper presents NestE, a framework for embedding atomic and nested facts in knowledge graphs, leveraging hypercomplex number systems like quaternions to model relational and entity-based logical patterns. | [Link](https://ojs.aaai.org/index.php/AAAI/article/download/28772/29482)| [Code](https://github.com/xiongbo010/NestE) |
| PMD | Cunhang Fan (Anhui University) | Progressive Distillation Based on Masked Generation Feature Method for Knowledge Graph Completion | Knowledge Graph Completion | The paper introduces PMD, a progressive distillation method that reduces model parameters while maintaining performance in knowledge graph completion tasks using masked generation features. | [Link](https://ojs.aaai.org/index.php/AAAI/article/download/28680/29320)| [Code](https://github.com/cyjie429/PMD) |
| IncDE | Jiajun Liu (Southeast University) | Towards Continual Knowledge Graph Embedding via Incremental Distillation | Continual Knowledge Graph Embedding | The paper proposes IncDE, a novel method using incremental distillation to learn emerging knowledge while preserving old knowledge in knowledge graphs, featuring a hierarchical ordering strategy and a two-stage training paradigm. | [Link](https://ojs.aaai.org/index.php/AAAI/article/download/28722/29396)| [Code](https://github.com/seukgcode/IncDE) |


- ACL 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|
| TCRA | Jingtao Guo (Beijing Institute of Technology) | A Unified Joint Approach with Topological Context Learning and Rule Augmentation for Knowledge Graph Completion | Knowledge Graph Completion | The paper introduces TCRA, a model that improves knowledge graph completion by integrating topological context learning and rule augmentation to better capture logical semantics and global topological context for entities and relations. | [Link](https://aclanthology.org/2024.findings-acl.812.pdf) | [Code](https://github.com/starlet122/TCRA) |
| HyperCL | Yuhuan Lu (University of Macau) | HyperCL: A Contrastive Learning Framework for Hyper-Relational Knowledge Graph Embedding with Hierarchical Ontology | Hyper-Relational Knowledge Graph Embedding | The paper introduces HyperCL, a universal contrastive learning framework designed to integrate hyper-relational knowledge graph embedding methods and capture hierarchical ontology structures, which improves link prediction performance across datasets. | [Link](https://aclanthology.org/2024.findings-acl.171.pdf) | [Code](https://github.com/UM-Data-Intelligence-Lab/HyperCL_code) |
| KnowC | Guangqian Yang (University of Science and Technology of China) | Knowledge Context Modeling with Pre-trained Language Models for Contrastive Knowledge Graph Completion | Knowledge Graph Completion | The paper introduces KnowC, a framework that models the knowledge context as additional prompts for pre-trained language models, using several strategies like kNN and dynamic sampling to improve the effectiveness of knowledge graph completion tasks. | [Link](https://aclanthology.org/2024.findings-acl.509.pdf) | [code] |
| DuASE | Jiang Li (Inner Mongolia University) | Learning Low-dimensional Multi-domain Knowledge Graph Embedding via Dual Archimedean Spirals | Knowledge Graph Embedding | The paper introduces DuASE, a low-dimensional knowledge graph embedding model that maps entities with the same relation onto Archimedean spirals to avoid embedding overlaps across domains, improving link prediction on multi-domain knowledge graphs. | [Link](https://aclanthology.org/2024.findings-acl.198.pdf) | [Code](https://github.com/dellixx/DuASE) |
| DynaSemble | Ananjan Nandi (Indian Institute of Technology, Delhi) | DynaSemble: Dynamic Ensembling of Textual and Structure-Based Models for Knowledge Graph Completion | Knowledge Graph Completion | The paper proposes DynaSemble, a novel method that dynamically combines textual and structure-based models by learning query-dependent ensemble weights, achieving significant performance improvements on tasks such as link prediction and relation classification. | [Link](https://aclanthology.org/2024.acl-short.20.pdf) | [Code](https://github.com/dair-iitd/KGC-Ensemble) |
| TCompoundE | Rui Ying (Nankai University) | Simple but Effective Compound Geometric Operations for Temporal Knowledge Graph Completion | Temporal Knowledge Graph Completion | The paper introduces TCompoundE, a model utilizing time-specific and relation-specific geometric operations to effectively capture the complex temporal dynamics in temporal knowledge graphs, significantly improving performance on temporal knowledge graph completion tasks. | [Link](https://aclanthology.org/2024.acl-long.596.pdf) | [Code](https://github.com/nk-ruiying/TCompoundE) |

- EMNLP 2024

  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|



-ICLR 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|
| BioBridge | Zifeng Wang (University of Illinois Urbana-Champaign) | BioBridge: Bridging Biomedical Foundation Models via Knowledge Graphs | Multimodal Foundation Model Bridging | The paper introduces BioBridge, a framework that bridges independently trained unimodal foundation models through knowledge graphs, enabling multimodal behavior without fine-tuning the underlying models, significantly improving cross-modal prediction tasks. | [Link](https://openreview.net/forum?id=hGqaYvBHtv) | [Code](https://github.com/RyanWangZf/BioBridge) |
| FIT | Hang Yin (Tsinghua University) | Rethinking Complex Queries on Knowledge Graphs with Neural Link Predictions | Knowledge Graph Query Answering | This paper introduces FIT, a neural-symbolic approach combining fuzzy logic with neural link predictors to answer complex queries in knowledge graphs, significantly improving performance on new and existing datasets. | [Link](https://iclr.cc/virtual/2024/poster/19598) | [Code](https://github.com/HKUST-KnowComp/FIT) |
| ULTRA | Mikhail Galkin (Intel AI Lab) | Towards Foundation Models for Knowledge Graph Reasoning | Knowledge Graph Reasoning | The paper introduces ULTRA, a method for universal and transferable knowledge graph representations that can generalize across graphs with arbitrary entity and relation vocabularies, enabling zero-shot and fine-tuned inference across multiple KGs. | [Link](https://openreview.net/forum?id=JFG58vzFA2) | [Code](https://github.com/DeepGraphLearning/ULTRA) |

- ICML 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|
| GoldE | Rui Li (Renmin University of China) | Generalizing Knowledge Graph Embedding with Universal Orthogonal Parameterization | Knowledge Graph Embedding | The paper introduces GoldE, a framework for knowledge graph embedding that uses universal orthogonal parameterization to extend dimensions and unify geometric types, enabling improved logical pattern modeling and topology representation in knowledge graphs. | [Link](https://proceedings.mlr.press/v235/li24ah.html) | [Code](https://github.com/rui9812/GoldE) |
| KnowFormer | Junnan Liu (Zhongguancun Laboratory) | KnowFormer: Revisiting Transformers for Knowledge Graph Reasoning | Knowledge Graph Reasoning | The paper introduces KnowFormer, a transformer-based architecture designed to address limitations in path-based methods for knowledge graph reasoning. It incorporates a novel attention mechanism tailored to the unique structure of knowledge graphs, improving performance in both transductive and inductive tasks. | [Link](https://icml.cc/virtual/2024/poster/34564) | [Code](https://github.com/nju-websoft/Knowformer) |
| Intersection Features | Duy Le (Case Western Reserve University) | Knowledge Graphs Can be Learned with Just Intersection Features | Knowledge Graph Embedding | The paper introduces a novel method that utilizes intersection features between entities and relations within k-hop neighborhoods to efficiently model knowledge graph triples for link prediction, outperforming traditional KG embedding methods and GNNs. | [Link](https://proceedings.mlr.press/v235/le24c.html) | [Code](https://github.com/Escanord/Intersection_Features) |
|PAC-Bayesian Generalization Bounds for Knowledge Graph Representation Learning |Jaejun Lee | PAC-Bayesian Generalization Bounds for Knowledge Graph Representation Learning | Knowledge Graph Embedding and Generalization | The paper introduces ReED, a relation-aware encoder-decoder framework for analyzing knowledge graph representation learning models, and proves PAC-Bayesian generalization bounds for knowledge graph models, offering theoretical insights into model design for improved generalization. | [Link](https://proceedings.mlr.press/v235/lee24a.html) | [Code](https://github.com/bdi-lab/ReED) |


- IJCAI 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|
| OctaE | Victor Charpenay (Mines Saint-Etienne) | Knowledge Graph Embedding | The paper introduces OctaE, a novel framework that uses octagon-based embeddings to capture both knowledge graphs and rules, efficiently modeling relational composition and intersection with competitive performance on link prediction tasks. | [Link](https://www.ijcai.org/proceedings/2024/364) | [code]|
| CPa-WAC | Sudipta Modak (University of Windsor) | CPa-WAC: Constellation Partitioning-based Scalable Weighted Aggregation Composition for Knowledge Graph Embedding | Knowledge Graph Embedding | The paper introduces CPa-WAC, a novel knowledge graph embedding method that uses constellation partitioning and weighted aggregation composition to reduce memory costs and training time, while preserving original graph structures for effective link prediction. | [Link](https://www.ijcai.org/proceedings/2024/364) | [Code](https://github.com/ganzagun/CPa-WAC) |
| KG-MRI | Yuxing Lu (Peking University) | Enhancing Multimodal Knowledge Graph Representation Learning through Triple Contrastive Learning | Multimodal Knowledge Graph Embedding | The paper proposes KG-MRI, a novel method for multimodal knowledge graph representation learning using a triple contrastive learning module and a dual-phase training strategy, achieving superior performance in multimodal knowledge graph tasks such as link prediction. | [Link](https://www.ijcai.org/proceedings/2024/596) | [code] |
| FastKGE | Jiajun Liu (Southeast University) | Fast and Continual Knowledge Graph Embedding via Incremental LoRA | Continual Knowledge Graph Embedding | The paper introduces FastKGE, a novel framework using incremental low-rank adapters (IncLoRA) to efficiently learn new knowledge while preserving old knowledge in growing knowledge graphs, reducing training time significantly while maintaining competitive link prediction performance. | [Link](https://www.ijcai.org/proceedings/2024/364) | [Code](https://github.com/seukgcode/FastKGE) |
| MulGA | Ziyu Shang (Southeast University) | Learning Multi-Granularity and Adaptive Representation for Knowledge Graph Reasoning | Knowledge Graph Reasoning | The paper introduces MulGA, a framework designed for knowledge graph reasoning (KGR), leveraging multi-granularity representations of triples, relation paths, and subgraphs to enhance the accuracy and efficiency of KGR tasks. Extensive experiments show significant improvements over state-of-the-art methods on both transductive and inductive reasoning benchmarks. | [Link](https://www.ijcai.org/proceedings/2024/364) | [code] |


- KDD 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|
| AsyncET | Yun-Cheng Wang (University of Southern California) | AsyncET: Asynchronous Representation Learning for Knowledge Graph Entity Typing | Knowledge Graph Entity Typing | The paper presents AsyncET, a novel framework for knowledge graph entity typing (KGET) that leverages asynchronous representation learning. It introduces auxiliary relations to improve the modeling of diverse entity-type relationships and efficiently refines the embeddings. Experiments demonstrate its significant performance improvements on KGET tasks with reduced computational complexity. | [Link](https://doi.org/10.1145/3637528.3671832) | [Code](https://github.com/yunchengwang/AsyncET) |
| DiffusionE | Zongsheng Cao (University of Chinese Academy of Sciences)| DiffusionE: Reasoning on Knowledge Graphs via Diffusion-based Graph Neural Networks | Knowledge Graph Reasoning | The paper introduces DiffusionE, a novel framework for reasoning on knowledge graphs (KGs) by reformulating message propagation as a diffusion process, ensuring semantic consistency through wave-based propagation and semantics-dependent diffusion. It achieves state-of-the-art performance on both transductive and inductive reasoning benchmarks. | [Link](https://doi.org/10.1145/3637528.3671997) | [code] |
| CISS | Kyuhwan Yeom (Yonsei University)| Embedding Two-View Knowledge Graphs with Class Inheritance and Structural Similarity | Knowledge Graph Embedding | The paper introduces CISS, a model that enhances two-view knowledge graph (KG) embeddings by considering class inheritance and structural similarity between ontology-view and instance-view KGs. The approach models fine-grained class representations and reduces the structural gap between two views for better link prediction and entity typing tasks. | [Link](https://doi.org/10.1145/3637528.3671941) | [Code](https://github.com/Yonsei-ICL/CISS) |
| RobustFacts | Hanhua Xiao (Singapore Management University)| How to Avoid Jumping to Conclusions: Measuring the Robustness of Outstanding Facts in Knowledge Graphs | Knowledge Graph Robustness | The paper proposes RobustFacts, a methodology for assessing the robustness of outstanding facts (OFs) in knowledge graphs through perturbation analysis, addressing potential misinformation from unstable OFs by inspecting both entity and data perturbations. Extensive experiments show its effectiveness in detecting frail OFs generated by existing methods. | [Link](https://doi.org/10.1145/3637528.3671763) | [Code](https://github.com/xhh232018/RobustFacts) |
| NORAN | Qinggang Zhang (The Hong Kong Polytechnic University) | Logical Reasoning with Relation Network for Inductive Knowledge Graph Completion | Inductive Knowledge Graph Completion | The paper presents NORAN, a novel message-passing framework that mines latent relation semantics for inductive KG completion. By constructing a relation network, the model centers on relations to improve inductive inference. Experiments show that NORAN achieves state-of-the-art results on five KG benchmarks. | [Link](https://doi.org/10.1145/3637528.3671911) | [code] |
| Power-Link | Heng Chang (Huawei Technologies Co., Ltd.) | Path-based Explanation for Knowledge Graph Completion | Knowledge Graph Completion | The paper introduces Power-Link, a novel framework that provides path-based explanations for GNN-based models on knowledge graph completion tasks. It leverages a graph-powering technique for generating scalable and interpretable path explanations. Experiments demonstrate improved interpretability and efficiency over state-of-the-art methods. | [Link](https://doi.org/10.1145/3637528.3671683) | [Code](https://github.com/OUTHIM/power-link) |
| SimDiff | Ran Li (HKUST)| SimDiff: Simple Denoising Probabilistic Latent Diffusion Model for Data Augmentation on Multi-modal Knowledge Graph | Data Augmentation, Knowledge Graphs | The paper introduces SimDiff, a novel denoising probabilistic latent diffusion model for augmenting data on multi-modal knowledge graphs (MMKGs). SimDiff efficiently handles various modalities such as images, text, and graph structures in a unified latent space, enhancing downstream tasks like entity alignment. | [Link](https://doi.org/10.1145/3637528.3671769) | [Code](https://github.com/ranlislz/SimDiff) |




- NeurIPS 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- SIGIR 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|
| CMR | Yu Zhao (Nankai University) | Contrast then Memorize: Semantic Neighbor Retrieval-Enhanced Inductive Multimodal Knowledge Graph Completion | Multimodal Knowledge Graph Completion | The paper introduces CMR, a framework that enhances inductive multimodal knowledge graph completion (IMKGC) through semantic neighbor retrieval and cross-modal contrastive learning. CMR effectively generalizes to unseen entities by capturing textual-visual correlations and retrieving semantic neighbors for improved inference. | [Link](https://doi.org/10.1145/3626772.3657838) | [Code](https://github.com/OreOZhao/CMR) |
| NativE | Yichi Zhang (Zhejiang University) | NativE: Multi-modal Knowledge Graph Completion in the Wild | Multi-modal Knowledge Graph Completion | The paper introduces NativE, a comprehensive framework for multi-modal knowledge graph completion (MMKGC) that addresses the challenges of diversity and imbalanced modality distributions in real-world MMKGs. The framework uses a relation-guided dual adaptive fusion module and collaborative modality adversarial training to improve performance on MMKGC tasks across various datasets. | [Link](https://doi.org/10.1145/XXXXXX.XXXXXX) | [Code](https://github.com/zjukg/NATIVE) |
| SpherE | Zihao Li (University of Illinois at Urbana-Champaign) | SpherE: Expressive and Interpretable Knowledge Graph Embedding for Set Retrieval | Knowledge Graph Embedding | The paper introduces SpherE, a knowledge graph embedding model designed for set retrieval. Unlike traditional embedding models that represent entities as vectors, SpherE embeds entities as spheres, enhancing the ability to model complex many-to-many relations and improving interpretability. Extensive experiments demonstrate that SpherE achieves state-of-the-art performance on set retrieval tasks. | [Link](https://doi.org/10.1145/3626772.3657910) | [Code](https://github.com/Violet24K/SpherE) |
| UAA-KGE | Tianzhe Zhao (Xi’an Jiaotong University) | Untargeted Adversarial Attack on Knowledge Graph Embeddings | Knowledge Graph Embedding, Adversarial Attack | The paper introduces a novel approach to untargeted adversarial attacks on knowledge graph embeddings (KGE), which aims to diminish the overall performance of KGE methods by manipulating the knowledge graph structure without targeting specific triples. By leveraging logic rules for adversarial deletion and addition, this method explores the robustness of various KGE methods. | [Link](https://doi.org/10.1145/3637528.3671820) | [code] |




- WWW 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|
| IPAE | Narayanan Asuri Krishnan (Rochester Institute of Technology) | A Method for Assessing Inference Patterns Captured by Embedding Models in Knowledge Graphs | Knowledge Graph Embedding, Inference Patterns | The paper introduces IPAE, a model-agnostic method that quantifies how embedding models capture logical inference patterns in knowledge graphs. The method focuses on both positive and negative evidence, introducing a new empirical evaluation framework for assessing models' abilities to handle various inference patterns. | [Link](https://doi.org/10.1145/3589334.3645505) | [Code](https://github.com/nari97/WWW2024_Inference_Patterns) |
| UniGE | Yuhan Liu (Renmin University of China) | Bridging the Space Gap: Unifying Geometry Knowledge Graph Embedding with Optimal Transport | Knowledge Graph Embedding | The paper introduces UniGE, a novel framework that unifies knowledge graph embeddings in both Euclidean and hyperbolic spaces through optimal transport and Wasserstein barycenter techniques. This unified approach significantly improves link prediction performance, achieving state-of-the-art results on several benchmark datasets. | [Link](https://doi.org/10.1145/3589334.3645565) | [Code](https://github.com/LittlePrinceYu/UniGE) |
| EPR-KGQA | Wentao Ding (Nanjing University)| Enhancing Complex Question Answering over Knowledge Graphs through Evidence Pattern Retrieval | Knowledge Graph Question Answering | The paper introduces EPR-KGQA, a novel framework for improving complex question answering over knowledge graphs (KGs) by modeling structural dependencies among evidence facts during subgraph extraction. This framework achieves substantial performance improvements on datasets like ComplexWebQuestions and WebQuestionsSP. | [Link](https://doi.org/10.1145/3589334.3645563) | [Code](https://github.com/nju-websoft/EPR-KGQA) |
| FDM | Xiao Long (University of Science and Technology of China)| Fact Embedding through Diffusion Model for Knowledge Graph Completion | Knowledge Graph Completion | The paper introduces FDM, a novel framework leveraging diffusion models for knowledge graph completion by directly learning the distribution of plausible facts. Extensive experiments show that FDM significantly outperforms state-of-the-art methods on multiple benchmark datasets. | [Link](https://doi.org/10.1145/3589334.3645451) | [code] |
| HaSa | Honggen Zhang (University of Hawaii at Manoa) | HaSa: Hardness and Structure-Aware Contrastive Knowledge Graph Embedding | Knowledge Graph Embedding | The paper introduces HaSa, a contrastive learning method for knowledge graph embeddings that mitigates the impact of false negative triples while generating hard negative triples. It significantly improves InfoNCE-based KGE approaches and achieves state-of-the-art results on WN18RR and competitive results on FB15k-237. | [Link](https://doi.org/10.1145/3589334.3645564) | [Code](https://github.com/honggen-zhang/HaSa-CKGE) |
| IME | Jiapu Wang (Beijing University of Technology) | IME: Integrating Multi-curvature Shared and Specific Embedding for Temporal Knowledge Graph Completion | Temporal Knowledge Graph Completion | The paper introduces IME, a novel model designed for temporal knowledge graph completion (TKGC). It leverages multi-curvature spaces (hyperspherical, hyperbolic, and Euclidean) to capture complex geometric structures, incorporating both shared and specific embedding properties across these spaces. IME demonstrates state-of-the-art performance on multiple TKGC datasets. | [Link](https://doi.org/10.1145/3589334.3645361) | [code] |
| PAFKGE | Enyuan Zhou (Hong Kong Polytechnic University)| Poisoning Attack on Federated Knowledge Graph Embedding | Federated Learning, Knowledge Graph Embedding, Adversarial Attacks | The paper introduces a novel framework for poisoning attacks on federated knowledge graph embeddings (FKGE). It systematically explores the vulnerabilities in FKGE and presents dynamic and fixed poisoning schemes that force clients to learn false facts while maintaining overall task performance. | [Link](https://doi.org/10.1145/3589334.3645422) | [code] |
| Query2GMM | Yuhan Wu (East China Normal University) | Query2GMM: Learning Representation with Gaussian Mixture Model for Reasoning over Knowledge Graphs | Knowledge Graph Reasoning | The paper introduces Query2GMM, a novel approach leveraging Gaussian Mixture Models (GMM) to represent complex queries and their answers over knowledge graphs. Query2GMM significantly improves accuracy and precision in handling multi-modal query answers by incorporating cardinality, semantic centers, and dispersion degrees for multiple answer subsets. | [Link](https://doi.org/10.1145/3589334.3645569) | [code] |
| ReliK | Maximilian K. Egger (Aarhus University) | ReliK: A Reliability Measure for Knowledge Graph Embeddings | Knowledge Graph Embedding, Reliability | The paper introduces ReliK, a novel reliability measure for knowledge graph embeddings (KGEs). ReliK quantifies the performance reliability of KGEs for specific downstream tasks without retraining or task-specific knowledge. It proves effective for tasks such as relation prediction, rule mining, and question answering. | [Link](https://doi.org/10.1145/3589334.3645430) | [Code](https://github.com/AU-DIS/ReliK) |
| NYLON | Weijian Yu (University of Macau)| Robust Link Prediction over Noisy Hyper-Relational Knowledge Graphs via Active Learning | Knowledge Graph Link Prediction, Active Learning | The paper introduces NYLON, a noise-resistant hyper-relational link prediction method that leverages active crowd learning. It integrates element-wise confidence with fact-wise confidence to improve labeling efficiency and prediction robustness. NYLON achieves significant performance improvements on noisy hyper-relational knowledge graphs. | [Link](https://doi.org/10.1145/3589334.3645686) | [Code](https://github.com/UM-Data-Intelligence-Lab/NYLON_code) |
| SSTKG | Ruiyi Yang (University of New South Wales) | SSTKG: Simple Spatio-Temporal Knowledge Graph for Interpretable and Versatile Dynamic Information Embedding | Spatio-Temporal Knowledge Graph | The paper introduces SSTKG, a framework designed for integrating and embedding spatio-temporal data in knowledge graphs (KGs) to improve predictive tasks such as sales forecasting and traffic volume prediction. The model leverages a simplified 3-step embedding process to enhance accuracy, efficiency, and interpretability. | [Link](https://doi.org/10.1145/3589334.3645441) | [code] |
| LongTail-TKG | Mehrnoosh Mirtaheri (USC)| Tackling Long-Tail Entities for Temporal Knowledge Graph Completion | Temporal Knowledge Graph Completion | The paper introduces a model-agnostic enhancement layer for improving the performance of Temporal Knowledge Graph Completion (TKGC) methods, focusing on long-tail entities with sparse connections. The approach leverages global entity similarity to improve predictions and demonstrates a 10-15% improvement in Mean Reciprocal Rank (MRR) on benchmark datasets. | [Link](https://doi.org/10.1145/3589335.3651565) | [code] |
| Simple-HHEA | Xuhui Jiang (Chinese Academy of Sciences)| Toward Practical Entity Alignment Method Design: Insights from New Highly Heterogeneous Knowledge Graph Datasets | Entity Alignment, Knowledge Graphs | The paper introduces Simple-HHEA, a method designed for entity alignment (EA) in highly heterogeneous knowledge graphs (HHKGs). The authors propose two new datasets (ICEWS-WIKI, ICEWS-YAGO) and conduct extensive experiments demonstrating that GNN-based methods struggle with HHKGs. Simple-HHEA shows strong performance by leveraging name, structure, and temporal information. | [Link](https://doi.org/10.1145/3589334.3645720) | [Code](https://github.com/IDEA-FinAI/Simple-HHEA) |
| UniLP | Ben Liu (Wuhan University) | UniLP: Unified Topology-aware Generative Framework for Link Prediction in Knowledge Graph | Link Prediction, Knowledge Graph | The paper introduces UniLP, a unified framework designed to handle multiple link prediction subtasks in knowledge graphs using a topology-aware generative model. The approach integrates generative pre-trained language models with topology-aware soft prompts for efficient and accurate link prediction, achieving state-of-the-art performance across various benchmarks. | [Link](https://doi.org/10.1145/3589334.3645592) | [Code](https://github.com/LB0828/UniLP) |
| KGIL | Shuyao Wang (University of Science and Technology of China) | Unleashing the Power of Knowledge Graph for Recommendation via Invariant Learning | Knowledge Graph, Recommendation Systems | The paper introduces KGIL, a framework for knowledge-aware recommendations that uses invariant learning across noisy knowledge graph environments. By learning to distinguish task-relevant connections, the model significantly improves recommendation accuracy on multiple benchmarks. | [Link](https://doi.org/10.1145/3589334.3645576) | [code] |
| MC-KGE | Aishwarya Rao (Rochester Institute of Technology) | Using Model Calibration to Evaluate Link Prediction in Knowledge Graphs | Link Prediction, Knowledge Graphs | The paper proposes a novel model calibration protocol to evaluate link prediction in knowledge graphs more efficiently by leveraging posterior probabilities instead of ranks. This method significantly reduces computational costs while maintaining high correlation with traditional ranking metrics. | [Link](https://doi.org/10.1145/3589334.3645506) | [Code](https://github.com/nari97/WWW2024_Model_Calibration) |


### Year 2023

- AAAI 2023

| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|

- ACL 2023
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- EMNLP 2023
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|



-ICLR 2023
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- ICML 2023
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|

- IJCAI 2023
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- KDD 2023
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- NeurIPS 2023
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- SIGIR 2023
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|

- WWW 2023
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


### Year 2022

- AAAI 2022

| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|

- ACL 2022
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- EMNLP 2022
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|



-ICLR 2022
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- ICML 2022
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|

- IJCAI 2022
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- KDD 2022
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- NeurIPS 2022
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- SIGIR 2022
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|

- WWW 2022
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|



### Year 2021

- AAAI 2021

| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|

- ACL 2021
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- EMNLP 2021
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|



-ICLR 2021
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- ICML 2021
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|

- IJCAI 2021
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- KDD 2021
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- NeurIPS 2021
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- SIGIR 2021
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|

- WWW 2021
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|
