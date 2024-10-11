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
| KGDM | Xiao Long (USTC) | KGDM: A Diffusion Model to Capture Multiple Relation Semantics for Knowledge Graph Embedding | Knowledge Graph Embedding | The paper proposes KGDM, a novel diffusion model for embedding knowledge graphs, designed to capture multiple relation semantics using denoising diffusion probabilistic models to improve performance on knowledge graph completion tasks. | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/29828) | [Code]|
| LAFA | Bin Shang (Xi’an Jiaotong University) | LAFA: Multimodal Knowledge Graph Completion with Link Aware Fusion and Aggregation | Multimodal Knowledge Graph Completion | The paper presents LAFA, a model that improves multimodal knowledge graph completion by leveraging a link-aware fusion and aggregation mechanism, which selectively fuses visual and structural embeddings based on the importance of images and structural relationships in different link scenarios. | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/28744) | [Code] |
| MGTCA | Bin Shang (Xi’an Jiaotong University) | Mixed Geometry Message and Trainable Convolutional Attention Network for Knowledge Graph Completion | Knowledge Graph Completion | The paper presents MGTCA, a model that integrates mixed geometry spaces and trainable convolutional attention networks to enhance knowledge graph completion tasks by improving neighbor message aggregation and representation quality. | [Link](https://ojs.aaai.org/index.php/AAAI/article/download/28745/29434) | [Code] |
| MKG-FENN | Di Wu (Chongqing University of Posts and Telecommunications) | MKG-FENN: A Multimodal Knowledge Graph Fused End-to-End Neural Network for Accurate Drug–Drug Interaction Prediction | Drug–Drug Interaction Prediction | The paper introduces MKG-FENN, a model that uses multimodal knowledge graphs and a fused end-to-end neural network to predict drug-drug interactions with high accuracy by extracting and fusing drug-related features. | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/29888) | [Code](https://github.com/wudi1989/MKG-FENN) |
| CompilE | Wanyun Cui (Shanghai University of Finance and Economics) | Modeling Knowledge Graphs with Composite Reasoning | Knowledge Graph Completion | The paper presents CompilE, a framework that unifies different knowledge graph models by combining facts from various entities for better reasoning and improved performance on knowledge graph completion tasks. | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/28675) | [Code](https://github.com/zlq147/CompilE) |
|NestE |Bo Xiong (University of Stuttgart) |NestE: Modeling Nested Relational Structures for Knowledge Graph Reasoning | Knowledge Graph Reasoning | This paper presents NestE, a framework for embedding atomic and nested facts in knowledge graphs, leveraging hypercomplex number systems like quaternions to model relational and entity-based logical patterns. | [Link](https://ojs.aaai.org/index.php/AAAI/article/download/28772/29482)| [Code](https://github.com/xiongbo010/NestE) |
| PMD | Cunhang Fan (Anhui University) | Progressive Distillation Based on Masked Generation Feature Method for Knowledge Graph Completion | Knowledge Graph Completion | The paper introduces PMD, a progressive distillation method that reduces model parameters while maintaining performance in knowledge graph completion tasks using masked generation features. | [Link](https://ojs.aaai.org/index.php/AAAI/article/download/28680/29320)| [Code](https://github.com/cyjie429/PMD) |
| IncDE | Jiajun Liu (Southeast University) | Towards Continual Knowledge Graph Embedding via Incremental Distillation | Continual Knowledge Graph Embedding | The paper proposes IncDE, a novel method using incremental distillation to learn emerging knowledge while preserving old knowledge in knowledge graphs, featuring a hierarchical ordering strategy and a two-stage training paradigm. | [Link](https://ojs.aaai.org/index.php/AAAI/article/download/28722/29396)| [Code](https://github.com/seukgcode/IncDE) |


- ACL 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- EMNLP 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|



-ICLR 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- ICML 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|

- IJCAI 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- KDD 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- NeurIPS 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


- SIGIR 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|

- WWW 2024
  
| Name         | Author          |Title                                    |task      | Summarize                                                   | Paper Link                          | Code Link                          |
|--------------|--------------|------------------------------------------|----------------|-------------------------------------------------------------|-------------------------------------|------------------------------------|


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
