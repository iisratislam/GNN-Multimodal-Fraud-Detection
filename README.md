# Robust Financial Fraud Detection using Graph Neural Networks and Multimodal Fusion

**Author:** Israt Islam | Student Number: 23082056
**Institution:** University of Hertfordshire — MSc Data Science
**Dataset:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

---

## Project Overview

This project proposes and evaluates a novel fraud detection framework that combines Graph Neural Networks (GNNs) with Multimodal Data Fusion on the IEEE-CIS Vesta Corporation dataset of 590,540 real e-commerce transactions. Traditional machine learning models evaluate each transaction independently, making them blind to organised fraud rings where criminals reuse stolen credentials across multiple transactions. This project addresses that limitation by building a transaction graph and training a GraphSAGE model alongside tabular baselines, then combining both in a late-fusion architecture.

**Research Question:** To what extent can the integration of GNNs with Multimodal Data Fusion improve fraud detection and reduce false positives in high-velocity e-commerce environments?

**Key Results:**
- XGBoost achieved the best individual AUC-ROC of **0.9127** with FPR **0.0764**
- GraphSAGE improved AUC by **6.2%** over the tabular-only MLP baseline (0.855 vs 0.805)
- Multimodal Fusion reduced false positives by **23%** vs GraphSAGE alone (FPR 0.1635 vs 0.2127)
- Engineered feature `card1_txn_count` ranked **6th globally** out of 300+ features in SHAP analysis

---

## Repository Structure

```
GNN-Multimodal-Fraud-Detection/
│
├── NB01_EDA_and_Preprocessing.ipynb      # Data loading, EDA, feature engineering, temporal split
├── NB02_Baseline_Models.ipynb            # Logistic Regression, Random Forest, XGBoost, LightGBM
├── NB03_GNN_GraphSAGE.ipynb              # Graph construction, GraphSAGE training, embedding extraction
├── NB04_Multimodal_Fusion.ipynb          # Late-fusion model combining tabular + GNN embeddings
├── NB05_Evaluation_and_SHAP.ipynb        # Full evaluation, ablation study, SHAP explainability
│
├── figures/                              # All saved plots and graphs (300dpi PNG)
├── processed/                            # Preprocessed data files (not included — see Data Setup)
├── models/                               # Saved model files (not included — regenerate by running notebooks)
│
└── README.md
```

---

## Setup Instructions

### 1. Prerequisites

This project runs on **Google Colab** with a GPU runtime. All required libraries are installed within the notebooks.

Required libraries:
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `lightgbm`
- `torch`, `torch-geometric`
- `shap`

### 2. Data Setup

The raw dataset is not included in this repository due to file size and data handling constraints.

1. Download the dataset from Kaggle: https://www.kaggle.com/competitions/ieee-fraud-detection/data
2. You will need two files: `train_transaction.csv` and `train_identity.csv`
3. Upload both files to Google Drive at the following path:

```
MyDrive/FraudProject/data/train_transaction.csv
MyDrive/FraudProject/data/train_identity.csv
```

### 3. Running the Notebooks

Run the notebooks **in order** — each notebook saves outputs that the next one depends on:

```
NB01 → NB02 → NB03 → NB04 → NB05
```

| Notebook | Input | Output |
|---|---|---|
| NB01 | Raw CSV files | X_train.parquet, X_test.parquet, y_train.npy, y_test.npy |
| NB02 | Processed parquet files | Trained baseline models (.pkl), results_baseline.csv |
| NB03 | Processed parquet files | GraphSAGE model (.pt), node embeddings (.npy) |
| NB04 | Embeddings + parquet files | Fusion model (.pt), fusion predictions |
| NB05 | All models + test data | Evaluation metrics, SHAP plots, ablation results |

---

## Methodology Summary

### Data
- **590,540** real e-commerce transactions from Vesta Corporation
- **3.5% fraud** — severe class imbalance handled with sample weighting (48/52 ratio)
- Temporal train/test split (80/20) to prevent data leakage

### Feature Engineering (7 features)

| Feature | Description |
|---|---|
| `null_count` | Count of missing fields per transaction — synthetic identities leave more blank |
| `hour` | Hour of day extracted from TransactionDT — fraud peaks at midnight |
| `day_of_week` | Day of week — captures weekly fraud patterns |
| `email_mismatch` | Flag when purchaser and recipient email domains differ |
| `card1_txn_count` | Total transactions per card — high velocity signals stolen credentials |
| `amt_zscore` | Z-score of transaction amount relative to that card's history |
| `log_amt` | Log10 transformation of transaction amount — normalises skewed distribution |

### Models
- **Baselines:** Logistic Regression, Random Forest, XGBoost, LightGBM — all trained with 48/52 sample weights
- **GraphSAGE:** Transaction graph with 590,540 nodes and 258,808 edges (connected by shared card1, card2, addr1 values). Two SAGEConv layers producing 64-dimensional node embeddings
- **Fusion:** Late fusion combining 64-dim tabular MLP embeddings with 64-dim GNN embeddings, classification head produces final fraud logit

### Evaluation Metrics
AUC-ROC, F1-score, Precision, Recall, and False Positive Rate (FPR). Accuracy was excluded due to severe class imbalance.

---

## Results Summary

| Model | AUC-ROC | F1 | FPR |
|---|---|---|---|
| Logistic Regression | 0.8306 | 0.1703 | 0.2563 |
| Random Forest | 0.8715 | 0.3207 | 0.0811 |
| XGBoost | **0.9127** | **0.3683** | **0.0764** |
| LightGBM | 0.9099 | 0.3506 | 0.0844 |
| GraphSAGE (GNN) | 0.8551 | 0.1961 | 0.2127 |
| Fusion (GNN+Tabular) | 0.8551 | 0.2251 | **0.1635** |

---

## Key Findings

- GNNs and tabular models are **complementary** — each detects fraud through structurally distinct mechanisms
- GraphSAGE detects **fraud rings** through graph connectivity — invisible to tabular models
- Multimodal Fusion reduces false positives by **23%** compared to graph-only approach
- The engineered feature `card1_txn_count` (transaction velocity) ranked 6th out of 300+ features in SHAP — validating domain-informed feature engineering

---

## References

Hamilton, W.L., Ying, R. and Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. NeurIPS. https://arxiv.org/abs/1706.02216

Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. ACM SIGKDD. https://arxiv.org/abs/1603.02754

Lundberg, S.M. and Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS. https://arxiv.org/abs/1705.07874

Chen, H., et al. (2021). Fraud Detection with Graph Neural Networks. IEEE TNNLS. https://ieeexplore.ieee.org/document/9440733
