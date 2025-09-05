# Sentiment-Driven ETF Decision

<!--This repository contains the implementation of "Sentiment-Driven ETF Decision: A Fully LLM-Centric Agent Framework with Dynamic Constraint Prompts", a novel approach for ETF price prediction that leverages LLMs and sentiment analysis. -->

## Overview

<!-- Our framework introduces an end-to-end architecture for ETF decision-making using LLMs, incorporating sentiment analysis and dynamic weighting mechanisms. The framework achieved superior performance in both directional prediction (accuracy: 0.80) and continuous price prediction (MAE: 0.011). -->

## Repository Structure
## Dataset
<!--
- **ETF Data**: Historical data (2024-2025) of the financial ETF (159931)  https://q.stock.sohu.com/cn/159931/lshq.shtml
- **Sentiment Data**: Textual data from: 东方财富、金融ETF吧、搜狐网、ETF最前线、QQ News等等
  -->
## Models & Implementation

### 1. Agent Framework
<!--
- Time-Series Agent
- Sentiment Agent
- Error Analysis Agent
- Dynamic Weighting Mechanism
 -->
### 2. Baseline Models
<!--
- LSTM
- LSTM + Sentiment
 -->
### 3. LLM Implementations
<!--
- DeepSeek
- ChatGPT 4o
- Doubao
- Kimi
- Claude
-->
## Experimental Results

### Implementation Details
Our evaluation metrics implementation is available in `experiments/metrics.py`

# prompts/prompt.txt
We provide ready-to-use prompts that can achieve similar performance to our agent-based approach without requiring API access. These prompts are available in the  `prompts/prompt.txt`

# Citation
This repository contains the code implementation for our research paper (under review). Citation information will be updated upon publication.
