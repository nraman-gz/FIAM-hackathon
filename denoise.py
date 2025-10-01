import pandas as pd
import numpy as np
import pywt as pywt
from get_company_returns import get_company_returns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# compute mean abs deviation
def mad(d, axis = None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

# Get data for a company
# data = get_company_returns(
#     company_id='comp_001186_01C', plot = False, return_type='stock_ret')

# data.to_csv('comp_001186_01C_data.csv')
data = pd.read_csv('comp_001186_01C_data.csv')

# Split data into 80%  training, 20% testing
split_index = int(len(data) * 0.8)
train_data = data[:split_index].copy()
test_data = data[split_index:].copy()

# Discrete Wavelet Transform
def DiscreteWaveletTransform(series, level, wavelet_type, mode):
   
    # decomposition/DWT
    level = int(level)
    wavelet_type = wavelet_type
    coeff = pywt.wavedec(series, wavelet_type, level = level)

    #thresholding
    sigma = mad(coeff[-level])/0.6745
    threshold = sigma * np.sqrt(2*np.log(len(series)))
    coeff[1:] = (pywt.threshold(i, value=threshold, mode = mode) for i in coeff[1:])

    #reconstruction/Inverse Discrete Wavelet Transform
    denoised = pywt.waverec(coeff, wavelet_type)

    if len(series)!=len(denoised):
        denoised = denoised[:-1]

    return denoised

train_data_returns = train_data['stock_ret']
returns_denoised = DiscreteWaveletTransform(train_data_returns, 
                                            level = 4, wavelet_type='db2', mode ='soft')

# Estimate ARMA(2,2) to benchmark 
noisy_ARMA = ARIMA(
    train_data_returns,
    order = (2,0,2)
)
denoised_ARMA = ARIMA(
    returns_denoised,
    order = (2,0,2)
)

# Fit the models
noisy_ARMA_fitted = noisy_ARMA.fit()
denoised_ARMA_fitted = denoised_ARMA.fit()

# Generate forecasts for test period
test_steps = len(test_data)

# Forecast with original data model
noisy_forecast = noisy_ARMA_fitted.forecast(steps=test_steps)

# Forecast with denoised data model
denoised_forecast = denoised_ARMA_fitted.forecast(steps=test_steps)

# Add forecasts to test data
test_data['noisy_forecast'] = noisy_forecast
test_data['denoised_forecast'] = denoised_forecast

# Calculate forecast errors
test_data['noisy_error'] = test_data['stock_ret'] - test_data['noisy_forecast']
test_data['denoised_error'] = test_data['stock_ret'] - test_data['denoised_forecast']

# Calculate accuracy metrics
noisy_mse = mean_squared_error(test_data['stock_ret'], test_data['noisy_forecast'])
denoised_mse = mean_squared_error(test_data['stock_ret'], test_data['denoised_forecast'])

print(f"Noisy Model MSE: {noisy_mse:.6f}")
print(f"Denoised Model MSE: {denoised_mse:.6f}")
print(f"MSE Improvement: {((noisy_mse - denoised_mse) / noisy_mse * 100):.2f}%")

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(test_data.index, test_data['stock_ret'], label='Actual', color='black')
plt.plot(test_data.index, test_data['noisy_forecast'], label='Noisy Model Forecast', alpha=0.8)
plt.plot(test_data.index, test_data['denoised_forecast'], label='Denoised Model Forecast', alpha=0.8)
plt.title('Forecasts vs Actual Test Data')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(test_data.index, test_data['noisy_error'], label='Noisy Model Error', alpha=0.8)
plt.plot(test_data.index, test_data['denoised_error'], label='Denoised Model Error', alpha=0.8)
plt.title('Forecast Errors')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

