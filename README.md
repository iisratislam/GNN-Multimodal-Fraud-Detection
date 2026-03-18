# Robust Financial Fraud Detection using GNNs and Multimodal Fusion

**Author:** Israt Islam | 23082056 | University of Hertfordshire  
**Module:** PAM2002 Data Science Project  

## Research Question
To what extent can the integration of Graph Neural Networks (GNNs) 
with Multimodal Data Fusion techniques improve the detection of 
synthetic identity fraud and reduce false positives in high-velocity 
e-commerce environments?

## Dataset
IEEE-CIS Fraud Detection Dataset (Kaggle)  
590,540 transactions | Vesta Corporation | 3.5% fraud rate  
https://www.kaggle.com/competitions/ieee-fraud-detection

## Project Structure
```
FraudProject/
├── 01_EDA_and_Preprocessing.ipynb
├── 02_Baseline_Models.ipynb
├── 03_GNN_GraphSAGE.ipynb
├── 04_Multimodal_Fusion.ipynb
├── 05_Evaluation_and_SHAP.ipynb
└── requirements.txt
```

## Models Implemented
| Model | AUC-ROC | FPR |
|---|---|---|
| Logistic Regression | 0.8306 | 0.2563 |
| Random Forest | 0.8715 | 0.0811 |
| XGBoost | 0.9127 | 0.0764 |
| LightGBM | 0.9099 | 0.0844 |
| GraphSAGE (GNN) | 0.8551 | 0.2127 |
| Fusion (GNN+Tabular) | 0.8551 | 0.1635 |

## Key Findings
- XGBoost achieved best overall AUC of 0.9127
- GraphSAGE improved AUC by 6.2% over tabular-only approach
- Multimodal Fusion reduced false positives by 23% vs GNN alone
- Engineered feature card1_txn_count was top SHAP feature

## How to Run
1. Download dataset from Kaggle link above
2. Place CSVs in `FraudProject/data/` on Google Drive
3. Run notebooks in order: 01 → 02 → 03 → 04 → 05
4. All outputs saved automatically to Google Drive

## Requirements
See `requirements.txt`  
Platform: Google Colab (T4 GPU recommended for NB03 and NB04)
