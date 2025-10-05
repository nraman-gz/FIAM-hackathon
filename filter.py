import pandas as pd
import numpy as np
from pykalman import UnscentedKalmanFilter
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/nikhil/Documents/FIAM-hackathon/sample.csv")

# Define transition and observation functions for UKF
def transition_function(state, noise):
    return state + noise

def observation_function(state, noise):
    return state + noise

def apply_ukf(group):
    """Apply UKF to a single company's time series"""
    stock_ret = group['stock_ret'].values

    # Initialize UKF parameters
    initial_state_mean = stock_ret[0]
    initial_state_covariance = 1.0

    # Estimate noise parameters from data
    observation_covariance = np.var(np.diff(stock_ret)) if len(stock_ret) > 1 else 1.0
    transition_covariance = observation_covariance * 0.1

    # Create and apply Unscented Kalman Filter
    ukf = UnscentedKalmanFilter(
        transition_functions=transition_function,
        observation_functions=observation_function,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )

    # Smooth the series
    smoothed_state_means, _ = ukf.smooth(stock_ret)

    # Add smoothed values to group
    group['stock_ret_smoothed'] = smoothed_state_means.flatten()
    return group

# Select random sample of 5 companies
sample_companies = np.random.choice(data['id'].unique(), 5, replace=False)

# Apply UKF only to sampled companies
data['stock_ret_smoothed'] = np.nan
for company_id in sample_companies:
    mask = data['id'] == company_id
    company_data = data[mask].copy()
    smoothed_company = apply_ukf(company_data)
    data.loc[mask, 'stock_ret_smoothed'] = smoothed_company['stock_ret_smoothed'].values

# Sort by id and time to ensure proper order
if 'time' in data.columns:
    data = data.sort_values(['id', 'time']).reset_index(drop=True)
else:
    data = data.sort_values('id').reset_index(drop=True)

print(f"Total data shape: {data.shape}")
print(f"Number of companies: {data['id'].nunique()}")
print(f"Sampled companies for UKF: {list(sample_companies)}")
print(f"\nFirst 10 values comparison:")
print(data[['id', 'stock_ret', 'stock_ret_smoothed']].head(10))

# Plot raw vs smoothed data for the sampled companies

fig, axes = plt.subplots(len(sample_companies), 1, figsize=(12, 3*len(sample_companies)))
if len(sample_companies) == 1:
    axes = [axes]

for idx, company_id in enumerate(sample_companies):
    company_data = data[data['id'] == company_id]
    axes[idx].plot(company_data['stock_ret'].values, label='Raw Data', alpha=0.6, linewidth=1)
    axes[idx].plot(company_data['stock_ret_smoothed'].values, label='UKF Smoothed', linewidth=2)
    axes[idx].set_ylabel('Stock Returns')
    axes[idx].set_title(f'Company ID: {company_id}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time Index')
plt.tight_layout()
plt.show()

# Save to CSV
# data.to_csv("/Users/nikhil/Documents/FIAM-hackathon/sample_smoothed.csv", index=False)
# print("\nSmoothed data saved to sample_smoothed.csv")





