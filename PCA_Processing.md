Financial Data PCA Feature Engineering Pipeline
📋 Overview
This Python script performs dimensionality reduction and feature engineering on financial data using Principal Component Analysis (PCA). It transforms raw financial features into meaningful principal components while preserving all original identifier and return columns.

🎯 Purpose
Reduce dimensionality of financial features while preserving information

Handle missing data intelligently using median imputation

Create interpretable feature groups based on financial concepts

Maintain 100% data integrity - no rows are lost during processing

📊 Input Data
File: train_2005_2014_us.parquet

Period: 2005-2014

Geography: US stocks

Format: Parquet file containing financial statement data and market information

🏗️ Feature Groups
The script organizes financial features into 7 logical groups:

1. Accruals & Earnings Quality
coa_grla, col_grla, fnl_grla, oaccruals_at, oaccruals_ni, etc.

Measures accounting quality and earnings manipulation

2. Financial Health
cash_at, ni_ar1, eq_dur, f_score, tangibility

Overall financial stability and health metrics

3. Financing Activities
be_grla, dbnetis_at, chcsho_12m, nfna_grla

Capital structure and financing decisions

4. Growth Metrics
capex_abn, at_grl, capx_grl, emp_grl, inv_gr1

Company growth and investment patterns

5. Leverage & Risk
at_be, debt_me, kz_index, o_score, z_score

Financial leverage and bankruptcy risk measures

6. Liquidity & Trading
ami_126d, dolvol_var_126d, turnover_126d, bidaskhl_21d

Market liquidity and trading activity

7. Profitability
at_turnover, cop_at, ebitda_mev, ocf_at, niq_at

Profitability and efficiency ratios

🔧 Processing Steps
1. Data Preservation
Preserves critical identifier and return columns:

id, date, ret_eom, gvkey, iid, stock_ret

Price and market cap: prc, me, market_equity

Temporal features: year, month, char_date

2. Missing Data Handling
Strategy: Median imputation

Benefit: Zero data loss - all observations preserved

Implementation: SimpleImputer(strategy='median')

3. Feature Standardization
Standardizes features to zero mean and unit variance

Uses StandardScaler for consistent scaling

4. PCA Application
Applies PCA separately to each feature group

Retains up to 5 principal components per group

Components explain maximum variance within each group

5. Model Storage
Saves complete PCA pipeline (imputer + scaler + PCA) for future use

📈 Output Features
Preserved Columns (Original)
Identifiers: id, gvkey, iid, date

Returns: ret_eom, stock_ret

Market data: prc, me, market_equity

Temporal: year, month, char_date

Generated PCA Features
Accruals_Earnings_PC1 to Accruals_Earnings_PC5

Financial_Health_PC1 to Financial_Health_PC5

Financing_PC1 to Financing_PC5

Growth_PC1 to Growth_PC5

Leverage_PC1 to Leverage_PC5

Liquidity_PC1 to Liquidity_PC5

Profitability_PC1 to Profitability_PC5

📁 Output Files
1. Main Dataset
train_financial_data_with_pca.parquet - Complete processed data

train_financial_data_with_pca.csv - CSV version

2. Model Artifacts
train_pca_models.pkl - Saved PCA pipelines for each feature group

🎨 Visualization
The script generates comprehensive plots for each feature group:

Individual explained variance per component

Cumulative explained variance across components

✅ Quality Assurance
Data Integrity Checks
100% row preservation - no observations lost

0 missing values in PCA features (due to imputation)

Feature validation - only existing features processed

Component validation - sufficient features for PCA

Verification Metrics
Original vs final dataset size comparison

Missing value analysis before/after imputation

Explained variance reporting for each component

Sample data inspection for validation

🚀 Usage
Prerequisites
python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
Execution
Simply run the script - it will:

Load the training data

Process all feature groups

Generate visualizations

Save the final dataset and models

💡 Key Benefits
Interpretability: PCA components within financial concept groups

Completeness: 100% data preservation through intelligent imputation

Scalability: Handles high-dimensional financial data efficiently

Reproducibility: Saved models enable consistent transformation

Analysis Ready: Clean dataset suitable for ML models and financial research

📊 Performance
Original features: 100+ raw financial metrics

Reduced features: ~35 PCA components (70%+ reduction)

Data preservation: 100% of observations retained

Information retention: Typically 80-95% variance explained per group

This pipeline creates a robust, analysis-ready dataset that maintains financial interpretability while enabling efficient machine learning modeling.
