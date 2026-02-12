# Robust Financial Fraud Detection using GNNs and Multimodal Fusion

## Project Overview
This research project focuses on improving the detection of **synthetic identity fraud** and **reducing false positives** in high-velocity e-commerce environments. By integrating **Graph Neural Networks (GNNs)** with **Multimodal Data Fusion**, the project aims to capture complex relational patterns between transactions and digital identities that traditional machine learning models often overlook.

## Research Question
> *To what extent can the integration of Graph Neural Networks (GNNs) with Multimodal Data Fusion techniques improve the detection of synthetic identity fraud and reduce false positives in high-velocity e-commerce environments?*

## Dataset
The project utilizes the **IEEE-CIS Fraud Detection dataset**, a large-scale, real-world e-commerce dataset provided by Vesta Corporation.
* **Scope**: 6 months of transaction history collected from real-world e-commerce transactions.
* **Multimodal Features**: Includes Transactional features (amounts, card types, addresses), Identity features (device info, network signatures), and temporal features.
* **Source**: [Kaggle - IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data).

## Project Structure
```text
├── data/                   # Raw CSV files from Kaggle (Local Only)
├── notebooks/              # Jupyter Notebooks for research stages
│   └── 01_Initial_EDA.ipynb # Initial Data Audit and Memory Optimization
├── src/                    # Future source code for GNN implementation
├── README.md               # Project documentation
└── .gitignore              # Configured to exclude large datasets
