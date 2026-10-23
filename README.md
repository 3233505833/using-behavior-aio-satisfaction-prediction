# Incorporating User Behavior to Predict User Satisfaction with AI Overviews

Data and code accompanying the paper **Incorporating User Behavior to Predict User Satisfaction with AI Overviews**.

## Overview

This work studies user satisfaction with AI Overviews presented alongside traditional ranked search results. Our method, STaRS , uses an LLM to reason over AI Overview content, user behavior, and search context, and incorporates reinforcement learning with reward shaping.

The repository provides the user-study dataset that we collected for this paper, baseline implementations and code for the overall experiments, reinforcement learning code for our method, and prompt examples for the ablation study.

## Data

The **AI Overview User Study Dataset** was collected through a controlled user study on an experimental search platform that displayed an AI Overview together with traditional ranked results. We recruited **31 participants** to perform search tasks drawn from a set of **10 tasks** of varying difficulty, most selected from the information needs in the TREC 2002–2004 collection. Participants searched as they normally would with an AI-Overview-enabled search engine. A browser plugin recorded their interactions, and participants rated their satisfaction with the AI Overview on a **five-point scale after each query**.

The released dataset contains **243 task sessions and 554 query records**, including queries, AI Overviews and their references, traditional search results, behavioral information, and explicit satisfaction ratings.

For the study design, dataset statistics, and data organization, see the dataset documentation.

## Repository Structure

```text
.
├── README.md
├── data/
│   ├── data_readme.md                           # Dataset documentation
│   └── new ai overview user study dataset.json  # User-study dataset
└── code/
    ├── baselines/                              # Overall experiment baselines
    ├── our method/                             # Proposed method
    └── ablation study/                         # Reasoning-paradigm ablation
```

## Overall Experiments

We compare our method with three categories of baselines. For `category_1_baselines` and `category_2_baselines`, we use existing open-source code and packages, with the implementation sources listed below.

### category_1_baselines: AI Overview Quality Evaluation Methods

These methods assess AI Overview quality. We use their official code or package implementations. Following their order in Table 2 of the paper, the methods and corresponding sources are:

| Method | Measures reported in Table 2 | Official code / package |
| --- | --- | --- |
| ARES | context_relevance, faithfulness, answer_relevance | [their work](https://github.com/stanford-futuredata/ARES) |
| TruLens | context_relevance, groundedness, answer_relevance | [their work](https://github.com/truera/trulens) |
| RAGAS | faithfulness, answer_relevancy, context_relevancy | [their work](https://github.com/vibrantlabsai/ragas) |
| OpenEval | faithfulness | [their work](https://github.com/Razvanip13/Towards_Lighter_And_Robust_Evaluation) |

### category_2_baselines: AI Overview User-Centric Evaluation Methods

For the following three methods, we reuse the implementations provided in [their work](https://github.com/Academic-Hammer/G-CoS/blob/main/src/llm_baselines/llm_judge_baselines.py):

- **G-Eval** (G-Eval-Sat in the paper): chain-of-thought satisfaction scoring with a form-filling paradigm.
- **URS (User Response Satisfaction):** task-type-aware multi-dimensional scoring.
- **GAH:** five-dimension scoring covering coherence, coverage, consistency, correctness, and clarity.

### category_3_baselines: User-Centric Satisfaction Prediction Methods Incorporating User Behaviors

Our implementations are available in code/baselines/category_3_baselines/, organized into two groups:

**1. Heuristic methods**

- **Click-Reformulation Heuristic:** Rule-based satisfaction scoring from post-AIO clicks and subsequent query reformulation.
- **Dwell-time Heuristic:** Rule-based scoring from AI Overview dwell time, adjusted for clicks and query reformulation.

**2. Machine learning methods**

We include behavior-based satisfaction prediction methods commonly used in traditional IR evaluation: **Ridge, Random Forest, GBDT, MLP, LSTM, and Transformer**. These methods use user interaction features or sequences, with a single AI Overview response quality score as an auxiliary signal.

We also include **G-CoS**, adapted from [their work](https://github.com/Academic-Hammer/G-CoS/blob/main/src/models/gcos_model.py).

### Our Method

The our method/ directory contains the `verl`-based reinforcement learning code, including custom reward scoring and reference signal label generation.

## Ablation Study

The ablation study/ directory provides prompt examples for the full reasoning strategy and four ablation variants.
