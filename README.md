# Sentiment-Driven ETF Decision

This repository contains the implementation of "Sentiment-Driven ETF Decision: A Fully LLM-Centric Agent Framework with Dynamic Constraint Prompts", a novel approach for ETF price prediction that leverages LLMs and sentiment analysis.

## Overview

Our framework introduces an end-to-end architecture for ETF decision-making using LLMs, incorporating sentiment analysis and dynamic weighting mechanisms. The framework achieved superior performance in both directional prediction (accuracy: 0.80) and continuous price prediction (MAE: 0.011).

## Repository Structure
## Dataset

- **ETF Data**: Historical data (2024-2025) of the financial ETF (159931)
- **Sentiment Data**: Textual data from:
 
## Models & Implementation

### 1. Agent Framework
- Time-Series Agent
- Sentiment Agent
- Error Analysis Agent
- Dynamic Weighting Mechanism

### 2. Baseline Models
- LSTM
- LSTM + Sentiment

### 3. LLM Implementations
- DeepSeek
- ChatGPT 4o
- Doubao
- Kimi
- Claude

## Experimental Results

### Directional Prediction Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| Our Framework | 0.80 | 0.88 | 0.75 | 0.76 |
| LSTM | 0.60 | 0.30 | 0.50 | 0.37 |
| LSTM+Sentiment | 0.60 | 0.30 | 0.50 | 0.37 |

### Continuous Price Prediction Performance

| Model | MAE | RMSE | MAPE (%) |
|-------|-----|------|-----------|
| Our Framework | 0.011 | 0.014 | 0.62 |
| LSTM | 0.067 | 0.068 | 3.78 |
| LSTM+Sentiment | 0.064 | 0.066 | 3.66 |

## Evaluation Metrics

### Implementation Details
Our evaluation metrics implementation is available in `experiments/metrics.py`

# prompts/prompt.txt
We provide ready-to-use prompts that can achieve similar performance to our agent-based approach without requiring API access. These prompts are available in the  `prompts/prompt.txt`
