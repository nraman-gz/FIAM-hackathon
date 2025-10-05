import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Read the TRAIN parquet file
file_path = "/teamspace/studios/this_studio/FIAM-hackathon/missing_data_anomalies/train_2005_2014_us.parquet"
df = pd.read_parquet(file_path)

print(f"TRAIN Data shape: {df.shape}")
print(f"Available columns: {df.columns.tolist()}")

# EXPLICITLY DEFINE COLUMNS TO PRESERVE (your identifiers, dates, returns)
columns_to_preserve = [
    'id', 'date', 'ret_eom', 'gvkey', 'iid', 'excntry', 'stock_ret', 
    'year', 'month', 'char_date', 'char_eom', 'me', 'prc', 'market_equity'
]

# Verify which of these actually exist in your data
existing_preserve_columns = [col for col in columns_to_preserve if col in df.columns]
print(f"\nColumns to preserve ({len(existing_preserve_columns)}): {existing_preserve_columns}")

# The remaining columns are features that will be reduced via PCA
feature_columns = [col for col in df.columns if col not in existing_preserve_columns]
print(f"\nFeature columns for PCA reduction ({len(feature_columns)}): {feature_columns[:10]}...")  # Show first 10

# Define feature groups for organized PCA
feature_groups = {
    'Accruals_Earnings': ['coa_grla','col_grla','fnl_grla','lnoa_grla','ncoa_grla','noa_grla','ncol_grla','oaccruals_at','oaccruals_ni','ni_su'],
    'Financial_Health': ['cash_at','ni_ar1','eq_dur','f_score','ni_inc8q','tangibility'],
    'Financing': ['be_grla','dbnetis_at','chcsho_12m','nfna_grla','eqnpo_12m','eqnetis_at','netis_at'],
    'Growth': ['capex_abn','at_grl','capx_grl','capx_gr2','capx_gr3','debt_gr3','emp_grl','lti_grla','inv_gr1'],
    'Leverage': ['at_be','debt_me','kz_index','netdebt_me','o_score','z_score'],
    'Liquidity': ['ami_126d','dolvol_var_126d','dolvol_126d','aliq_at','aliq_mat','bidaskhl_21d','turnover_var_126d','turnover_126d','zero_trades_252d','zero_trades_21d','zero_trades_126d'],
    'Profitability': ['at_turnover','cop_at','cop_atll','ebitda_mev','ebit_sale','ocf_at_chg','niq_at_chg','niq_be_chg','fcf_me','gp_at','gp_atll','ocf_at','niq_at','niq_be','ni_be']
}

# Start with preserved columns
preserved_data = df[existing_preserve_columns].copy()
print(f"\nPreserved data shape: {preserved_data.shape}")

# Create PCA features for each group WITH IMPUTATION
pca_results = pd.DataFrame(index=df.index)
pca_models = {}

# Set up plotting
n_groups = len(feature_groups)
fig, axes = plt.subplots(n_groups, 2, figsize=(15, 4 * n_groups))
if n_groups == 1:
    axes = axes.reshape(1, -1)

for idx, (group_name, features) in enumerate(feature_groups.items()):
    print(f"\n{'='*50}")
    print(f"Processing {group_name} WITH MEDIAN IMPUTATION")
    print(f"{'='*50}")
    
    # Get features that exist in the data AND are in our feature columns
    existing_features = [f for f in features if f in df.columns and f in feature_columns]
    
    if len(existing_features) < 2:
        print(f"Skipping {group_name} - need at least 2 features, only {len(existing_features)} available")
        continue
    
    # Extract data for this group
    group_data = df[existing_features].copy()
    
    # Check missing values BEFORE imputation
    print(f"Missing values BEFORE imputation:")
    missing_counts = group_data.isnull().sum()
    print(missing_counts)
    total_missing_before = group_data.isnull().sum().sum()
    print(f"Total missing values: {total_missing_before}")
    print(f"Rows before imputation: {len(group_data)}")
    
    # Use median imputation instead of dropping rows
    imputer = SimpleImputer(strategy='median')
    group_imputed = pd.DataFrame(
        imputer.fit_transform(group_data),
        columns=group_data.columns,
        index=group_data.index
    )
    
    # Check missing values AFTER imputation
    total_missing_after = group_imputed.isnull().sum().sum()
    print(f"Missing values AFTER imputation: {total_missing_after}")
    print(f"Rows preserved: {len(group_imputed)} (100% - no data loss!)")
    
    # Standardize features
    scaler = StandardScaler()
    group_scaled = scaler.fit_transform(group_imputed)
    
    # Determine number of components
    n_features = len(existing_features)
    n_samples = len(group_imputed)
    max_components = min(n_features, n_samples)
    n_components = min(max_components, 5)  # limit to 5 components per group
    
    print(f"Using {n_components} PCA components for {n_features} features")
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(group_scaled)
    
    # Store PCA model WITH IMPUTER
    pca_models[group_name] = {
        'pca': pca,
        'scaler': scaler,
        'imputer': imputer,  # Store the imputer for future use
        'features': existing_features
    }
    
    # Create column names for PCA features
    pca_columns = [f"{group_name}_PC{i+1}" for i in range(n_components)]
    
    # Add PCA results to ALL rows (no NaN values!)
    for i, col in enumerate(pca_columns):
        pca_results[col] = pca_features[:, i]  # All rows get values
    
    # Print explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    print(f"Explained variance ratio: {explained_var}")
    print(f"Cumulative explained variance: {cumulative_var}")
    
    # Plot results
    if idx < len(axes):  # Safety check
        ax1, ax2 = axes[idx]
        
        # Plot individual explained variance
        ax1.bar(range(1, len(explained_var) + 1), explained_var)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title(f'{group_name} - Individual Explained Variance')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative explained variance
        ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title(f'{group_name} - Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    
    # Print feature loadings for first 2 components
    if n_components >= 2:
        print(f"\nFeature loadings for first 2 components:")
        loadings_df = pd.DataFrame(
            pca.components_[:2].T,
            columns=['PC1', 'PC2'],
            index=existing_features
        )
        print(loadings_df.round(3))

plt.tight_layout()
plt.show()

# Combine preserved columns with PCA features - SINGLE COMPLETE DATABASE
final_dataset = pd.concat([preserved_data, pca_results], axis=1)

# Summary of created features
print(f"\n{'='*60}")
print("TRAIN FINAL DATASET SUMMARY WITH IMPUTATION")
print(f"{'='*60}")
print(f"Original TRAIN dataset shape: {df.shape}")
print(f"Final TRAIN dataset shape: {final_dataset.shape}")
print(f"Preserved identifier/return columns: {len(existing_preserve_columns)}")
print(f"PCA features created: {pca_results.shape[1]}")
print(f"Original feature columns reduced: {len(feature_columns)} → {pca_results.shape[1]}")
print(f"✅ DATA PRESERVATION: {len(final_dataset)}/{len(df)} rows (100% - no data loss!)")

print(f"\nPreserved columns: {existing_preserve_columns}")
print(f"PCA feature names: {pca_results.columns.tolist()}")

# Verify we can track specific stocks over time
print(f"\nSAMPLE - Tracking specific TRAIN stocks over time:")
sample_stocks = final_dataset.head(10)
print(sample_stocks[['id', 'date', 'ret_eom', 'stock_ret'] + pca_results.columns.tolist()[:3]])

# Check completeness - NOW SHOULD BE 100% FOR ALL PCA FEATURES!
print(f"\nTRAIN DATA COMPLETENESS WITH IMPUTATION:")
for col in pca_results.columns.tolist()[:5]:  # Show first 5 PCA features
    non_null = final_dataset[col].notna().sum()
    pct = (non_null / len(final_dataset)) * 100
    print(f"  {col}: {non_null}/{len(final_dataset)} ({pct:.1f}%)")

# Save results WITH "TRAIN_" PREFIX - ONLY ONE COMPLETE DATABASE
print(f"\nSaving TRAIN final dataset...")
final_dataset.to_csv("train_financial_data_with_pca.csv", index=False)
final_dataset.to_parquet("train_financial_data_with_pca.parquet", index=False)
print(f"TRAIN final dataset saved as 'train_financial_data_with_pca.parquet'")

# Save PCA models with "train_" prefix
import pickle
with open('train_pca_models.pkl', 'wb') as f:
    pickle.dump(pca_models, f)
print(f"TRAIN PCA models saved to 'train_pca_models.pkl'")

# Final verification
print(f"\n{'='*60}")
print("VERIFICATION - TRAIN DATA COMPLETE:")
print(f"{'='*60}")
print(f"✅ All {len(final_dataset)} TRAIN observations preserved")
print(f"✅ 0% data loss due to missing values") 
print(f"✅ All PCA features have 100% completeness")
print(f"✅ Single complete database with preserved columns + PCA features")
print(f"\nYour TRAIN dataset now contains:")
print(f"  • {len(existing_preserve_columns)} preserved identifier/return columns")
print(f"  • {len(pca_results.columns)} PCA feature columns") 
print(f"  • {len(final_dataset.columns)} total columns")
print(f"  • {len(final_dataset):,} total observations (2005-2014)")
print(f"\nFile saved: 'train_financial_data_with_pca.parquet'")
