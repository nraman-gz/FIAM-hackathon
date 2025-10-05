"""
STOCK TRADING STRATEGY BACKTEST
Combines Machine Learning Predictions with Sentiment Analysis
Period: 2015-2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("STOCK STRATEGY BACKTEST: 2015-2025")
print("Machine Learning + Sentiment Analysis")
print("=" * 80 + "\n")


# =============================================================================
# SETTINGS
# =============================================================================

class Config:
    # File names
    PREDICTIONS_FILE = "xgb_output.csv"  # ML model predictions
    SENTIMENT_FILE = "processed_sentiment(1).csv"  # News sentiment scores
    MARKET_FILE = "mkt_ind.csv"  # Market benchmark data

    # Output folder
    OUTPUT_DIR = "./backtest_results"

    # Strategy settings
    N_PORTFOLIOS = 10  # Create 10 portfolios each month
    PRED_WEIGHT = 0.7  # 70% weight to ML predictions
    SENT_WEIGHT = 0.3  # 30% weight to sentiment
    TC_BPS = 50  # 0.50% transaction costs

    # Time period
    START_YEAR = 2015
    END_YEAR = 2025


config = Config()

# =============================================================================
# LOAD AND CLEAN DATA
# =============================================================================

print("STEP 1: Loading data files")
print("-" * 80)

# Load ML predictions
pred = pd.read_csv(config.PREDICTIONS_FILE)

# Fix date format
if 'ret_eom' in pred.columns:
    pred['date'] = pd.to_datetime(pred['ret_eom'].astype(str), format='%Y%m%d')
else:
    pred['date'] = pd.to_datetime(pred['date'])

# Extract year and month
pred['year'] = pred['date'].dt.year
pred['month'] = pred['date'].dt.month

# Standardize column names
if 'gvkey' in pred.columns:
    pred = pred.rename(columns={'gvkey': 'id'})

# Find which column has the predictions
pred_col = next((col for col in ['pred_ret', 'xgb', 'prediction', 'predicted_return']
                 if col in pred.columns), None)
if pred_col is None:
    print("Error: No prediction column found!")
    exit()

# Keep only needed columns and clean data
pred_clean = pred[['date', 'year', 'month', 'id', 'stock_ret', pred_col]].copy()
pred_clean = pred_clean.rename(columns={pred_col: 'pred_ret'})
pred_clean = pred_clean.drop_duplicates(subset=['date', 'id']).dropna()

print(f"Loaded predictions: {len(pred_clean):,} observations, {pred_clean['id'].nunique():,} stocks")

# Load sentiment data
sent = pd.read_csv(config.SENTIMENT_FILE)
sent['date'] = pd.to_datetime(sent['date'], infer_datetime_format=True)
sent['year'] = sent['date'].dt.year
sent['month'] = sent['date'].dt.month

if 'gvkey' in sent.columns:
    sent = sent.rename(columns={'gvkey': 'id'})

# Keep only sentiment score column
sent_clean = sent[['year', 'month', 'id', 'delta_score']].copy()
sent_clean = sent_clean.drop_duplicates(subset=['year', 'month', 'id'], keep='last')

print(f"Loaded sentiment: {len(sent_clean):,} observations")

# Load market data for benchmarking
market = pd.read_csv(config.MARKET_FILE)
market['mkt_rf'] = market['ret'] - market['rf']  # Market excess return
market['mkt_total'] = market['mkt_rf'] + market['rf']  # Total market return

print(f"Loaded market data: {len(market)} periods")

# Filter to our target time period
target_years = [y for y in range(config.START_YEAR, config.END_YEAR + 1)
                if y in pred_clean['year'].unique() and
                y in sent_clean['year'].unique() and
                y in market['year'].unique()]

pred_clean = pred_clean[pred_clean['year'].isin(target_years)].copy()
sent_clean = sent_clean[sent_clean['year'].isin(target_years)].copy()
market = market[market['year'].isin(target_years)].copy()

print(f"Analysis period: {min(target_years)}-{max(target_years)} ({len(target_years)} years)")

# =============================================================================
# BASELINE STRATEGY (ML PREDICTIONS ONLY)
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: Baseline Strategy - ML Predictions Only")
print("=" * 80 + "\n")


def create_portfolios(data, rank_col, strategy_name):
    """Create portfolios by ranking stocks each month"""

    # Calculate average returns for each portfolio
    portfolio_rets = data.groupby(['year', 'month', rank_col]).agg({
        'stock_ret': ['mean', 'count']
    }).reset_index()
    portfolio_rets.columns = ['year', 'month', 'rank', 'return', 'n_stocks']

    # Reshape to have portfolios as columns
    monthly_port = portfolio_rets.pivot(
        index=['year', 'month'],
        columns='rank',
        values='return'
    ).reset_index()

    # Rename portfolio columns
    n_portfolios = config.N_PORTFOLIOS
    monthly_port.columns = ['year', 'month'] + [f'port_{i + 1}' for i in range(n_portfolios)]

    # Create long-short portfolio (buy best, sell worst)
    monthly_port['port_ls'] = monthly_port[f'port_{n_portfolios}'] - monthly_port['port_1']

    # Add market data for comparison
    monthly_port = monthly_port.merge(market[['year', 'month', 'mkt_rf', 'mkt_total']],
                                      on=['year', 'month'], how='inner')

    return monthly_port


# Rank stocks each month based on ML predictions
grouped = pred_clean.groupby(['year', 'month'])['pred_ret']
pred_clean['rank_baseline'] = np.floor(
    grouped.transform(lambda s: s.rank(method='first'))
    * config.N_PORTFOLIOS
    / grouped.transform(lambda s: len(s) + 1)
)

# Create baseline portfolios
baseline_port = create_portfolios(pred_clean, 'rank_baseline', 'Baseline')

print(f"Created baseline strategy: {len(baseline_port)} months of returns")

# =============================================================================
# COMBINED STRATEGY (ML + SENTIMENT)
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: Combined Strategy - ML + Sentiment")
print("=" * 80 + "\n")

# Merge predictions with sentiment data
combined = pred_clean.merge(
    sent_clean[['year', 'month', 'id', 'delta_score']],
    on=['year', 'month', 'id'],
    how='left'
)

# Fill missing sentiment with neutral (0)
combined['delta_score'] = combined['delta_score'].fillna(0)

# Calculate how many stocks have sentiment data
coverage = (combined['delta_score'] != 0).sum() / len(combined)
print(f"Sentiment coverage: {coverage:.1%} of observations")

# Standardize both signals within each month (z-scores)
for col in ['pred_ret', 'delta_score']:
    grouped = combined.groupby(['year', 'month'])[col]
    combined[f'{col}_zscore'] = (
            (combined[col] - grouped.transform('mean')) /
            grouped.transform('std')
    ).fillna(0)

# Combine ML predictions and sentiment
print(f"Combining signals: {config.PRED_WEIGHT:.0%} predictions + {config.SENT_WEIGHT:.0%} sentiment")
combined['combined_signal'] = (
        config.PRED_WEIGHT * combined['pred_ret_zscore'] +
        config.SENT_WEIGHT * combined['delta_score_zscore']
)

# Rank stocks based on combined signal
grouped = combined.groupby(['year', 'month'])['combined_signal']
combined['rank_combined'] = np.floor(
    grouped.transform(lambda s: s.rank(method='first'))
    * config.N_PORTFOLIOS
    / grouped.transform(lambda s: len(s) + 1)
)

# Create combined strategy portfolios
combined_port = create_portfolios(combined, 'rank_combined', 'Combined')

print(f"Created combined strategy: {len(combined_port)} months of returns")

# =============================================================================
# CALCULATE PERFORMANCE METRICS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4: Performance Analysis")
print("=" * 80 + "\n")


def calculate_metrics(portfolio_returns, strategy_name):
    """Calculate performance statistics for a strategy"""

    metrics = {'strategy': strategy_name}
    ls = portfolio_returns['port_ls']  # Long-short returns
    mkt = portfolio_returns['mkt_total']  # Market returns
    long = portfolio_returns['port_10']  # Top portfolio returns
    short = portfolio_returns['port_1']  # Bottom portfolio returns

    # Basic return statistics
    metrics['mean_monthly'] = ls.mean()
    metrics['annual_return'] = ls.mean() * 12
    metrics['volatility'] = ls.std() * np.sqrt(12)
    metrics['sharpe'] = ls.mean() / ls.std() * np.sqrt(12) if ls.std() > 0 else np.nan
    metrics['cumulative'] = (1 + ls).cumprod().iloc[-1] - 1

    # Risk metrics
    metrics['max_1m_loss'] = ls.min()
    metrics['max_1m_gain'] = ls.max()
    metrics['win_rate'] = (ls > 0).mean()
    metrics['skewness'] = ls.skew()

    # Drawdown calculation
    cumulative = (1 + ls).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdowns.min()

    # Comparison to market
    metrics['mkt_annual_return'] = mkt.mean() * 12
    metrics['outperformance'] = metrics['annual_return'] - metrics['mkt_annual_return']
    metrics['win_vs_market'] = (ls > mkt).mean()
    metrics['correlation'] = ls.corr(mkt)

    # CAPM Alpha calculation
    try:
        rf_rate = portfolio_returns['mkt_total'] - portfolio_returns['mkt_rf']
        mkt_rf = portfolio_returns['mkt_rf'].values
        port_rf = ls.values - rf_rate.values

        valid_mask = ~(np.isnan(mkt_rf) | np.isnan(port_rf))
        if valid_mask.sum() >= 10:
            x = mkt_rf[valid_mask]
            y = port_rf[valid_mask]

            beta = np.cov(y, x)[0, 1] / np.var(x) if np.var(x) > 0 else np.nan
            alpha_monthly = np.mean(y) - beta * np.mean(x)

            metrics['alpha_monthly'] = alpha_monthly
            metrics['alpha_annual'] = alpha_monthly * 12
            metrics['beta'] = beta
    except:
        metrics['alpha_annual'] = np.nan
        metrics['beta'] = np.nan

    # Long and short side performance
    metrics['long_annual_return'] = long.mean() * 12
    metrics['short_annual_return'] = short.mean() * 12

    # Sample information
    metrics['n_periods'] = len(ls)
    metrics['n_years'] = portfolio_returns['year'].nunique()

    return metrics


# Calculate metrics for both strategies
baseline_metrics = calculate_metrics(baseline_port, 'Baseline (Predictions Only)')
combined_metrics = calculate_metrics(combined_port, 'Combined (Pred + Sentiment)')

# Display performance comparison
print("OVERALL PERFORMANCE COMPARISON")
print("-" * 80)
print(f"{'Metric':<30} {'Baseline':>12} {'Combined':>12} {'Difference':>12}")
print("-" * 80)

# List of metrics to display
comparison_metrics = [
    ('Annual Return', 'annual_return', '.2%'),
    ('Annual Volatility', 'volatility', '.2%'),
    ('Sharpe Ratio', 'sharpe', '.3f'),
    ('Max Drawdown', 'max_drawdown', '.2%'),
    ('Win Rate', 'win_rate', '.2%'),
    ('Alpha (Annual)', 'alpha_annual', '.2%'),
    ('Beta', 'beta', '.3f'),
    ('Market Return', 'mkt_annual_return', '.2%'),
    ('Outperformance', 'outperformance', '.2%'),
]

for label, key, fmt in comparison_metrics:
    base_val = baseline_metrics.get(key, np.nan)
    comb_val = combined_metrics.get(key, np.nan)

    if not np.isnan(base_val) and not np.isnan(comb_val):
        diff = comb_val - base_val
        print(f"{label:<30} {base_val:>12{fmt}} {comb_val:>12{fmt}} {diff:>12{fmt}}")

# =============================================================================
# YEAR-BY-YEAR PERFORMANCE
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5: Year-by-Year Performance")
print("=" * 80 + "\n")

yearly_comparison = []

for year in sorted(target_years):
    base_year = baseline_port[baseline_port['year'] == year]['port_ls']
    comb_year = combined_port[combined_port['year'] == year]['port_ls']
    mkt_year = baseline_port[baseline_port['year'] == year]['mkt_total']

    if len(base_year) > 0 and len(comb_year) > 0:
        yearly_comparison.append({
            'year': year,
            'baseline_return': base_year.mean() * 12,
            'combined_return': comb_year.mean() * 12,
            'market_return': mkt_year.mean() * 12,
            'n_months': len(base_year)
        })

yearly_df = pd.DataFrame(yearly_comparison)

print(f"{'Year':>6} {'Baseline':>10} {'Combined':>10} {'Improvement':>12} {'Market':>10}")
print("-" * 80)
for _, row in yearly_df.iterrows():
    improvement = row['combined_return'] - row['baseline_return']
    print(f"{int(row['year']):>6} {row['baseline_return']:>9.1%} {row['combined_return']:>9.1%} "
          f"{improvement:>11.1%} {row['market_return']:>9.1%}")

# =============================================================================
# PREDICTION QUALITY ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 6: Prediction Quality Metrics")
print("=" * 80 + "\n")

# Calculate prediction accuracy
actual = pred_clean['stock_ret'].values
predicted = pred_clean['pred_ret'].values
valid = ~(np.isnan(actual) | np.isnan(predicted))

# Information Coefficient (rank correlation)
ic, ic_pval = stats.spearmanr(actual[valid], predicted[valid])

# Monthly IC to check consistency
monthly_ics = []
for (year, month), group in pred_clean.groupby(['year', 'month']):
    if len(group) > 10:
        g_actual = group['stock_ret'].values
        g_pred = group['pred_ret'].values
        valid_g = ~(np.isnan(g_actual) | np.isnan(g_pred))
        if valid_g.sum() > 10:
            m_ic, _ = stats.spearmanr(g_actual[valid_g], g_pred[valid_g])
            if not np.isnan(m_ic):
                monthly_ics.append(m_ic)

# Statistical significance
ic_tstat = np.mean(monthly_ics) / (np.std(monthly_ics) / np.sqrt(len(monthly_ics)))

print("PREDICTION ACCURACY STATISTICS")
print("-" * 80)
print(f"Information Coefficient (IC):          {ic:>8.4f}")
print(f"IC t-statistic:                        {ic_tstat:>8.2f}")
print(f"Statistically Significant (|t|>2):     {'YES' if abs(ic_tstat) > 2 else 'NO':>8}")
print(f"Monthly IC Mean:                       {np.mean(monthly_ics):>8.4f}")
print(f"Monthly IC Std:                        {np.std(monthly_ics):>8.4f}")
print(f"Positive Months:                       {(np.array(monthly_ics) > 0).mean():>8.1%}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 7: Saving Results")
print("=" * 80 + "\n")

output_dir = Path(config.OUTPUT_DIR)
output_dir.mkdir(exist_ok=True, parents=True)

# Save portfolio returns
baseline_port.to_csv(output_dir / "baseline_returns.csv", index=False)
combined_port.to_csv(output_dir / "combined_returns.csv", index=False)

# Save yearly performance
yearly_df.to_csv(output_dir / "yearly_performance.csv", index=False)

# Save strategy metrics
metrics_comparison = pd.DataFrame([baseline_metrics, combined_metrics])
metrics_comparison.to_csv(output_dir / "strategy_metrics.csv", index=False)

print(f"Results saved to: {output_dir}/")
print(f"  - baseline_returns.csv")
print(f"  - combined_returns.csv")
print(f"  - yearly_performance.csv")
print(f"  - strategy_metrics.csv")

# =============================================================================
# FINAL CONCLUSIONS
# =============================================================================

print("\n" + "=" * 80)
print("FINAL CONCLUSIONS")
print("=" * 80 + "\n")

print(f"ANALYSIS PERIOD: {min(target_years)}-{max(target_years)} ({len(target_years)} years)")
print(f"DATA: {len(pred_clean):,} predictions across {pred_clean['id'].nunique():,} stocks")
print(f"SENTIMENT COVERAGE: {coverage:.1%}\n")

print(f"PREDICTION QUALITY:")
print(f"  IC: {ic:.4f} (t-stat: {ic_tstat:.2f}) {'SIGNIFICANT' if abs(ic_tstat) > 2 else 'NOT SIGNIFICANT'}\n")

print(f"BASELINE STRATEGY (ML Only):")
print(f"  Annual Return: {baseline_metrics['annual_return']:.1%}")
print(f"  Sharpe Ratio: {baseline_metrics['sharpe']:.2f}")
print(f"  Alpha: {baseline_metrics['alpha_annual']:.1%}\n")

print(f"COMBINED STRATEGY (ML + Sentiment):")
print(f"  Annual Return: {combined_metrics['annual_return']:.1%}")
print(f"  Sharpe Ratio: {combined_metrics['sharpe']:.2f}")
print(f"  Alpha: {combined_metrics['alpha_annual']:.1%}\n")

improvement = combined_metrics['annual_return'] - baseline_metrics['annual_return']
sharpe_imp = combined_metrics['sharpe'] - baseline_metrics['sharpe']

print(f"SENTIMENT VALUE ADD:")
print(f"  Return Improvement: {improvement:.1%}")
print(f"  Sharpe Improvement: {sharpe_imp:.2f}")

if sharpe_imp > 0.1:
    verdict = "SENTIMENT SIGNIFICANTLY IMPROVES PERFORMANCE"
    recommendation = "USE SENTIMENT-ENHANCED STRATEGY"
elif sharpe_imp > 0:
    verdict = "SENTIMENT MARGINALLY IMPROVES PERFORMANCE"
    recommendation = "CONSIDER USING SENTIMENT"
else:
    verdict = "SENTIMENT DOES NOT IMPROVE PERFORMANCE"
    recommendation = "USE BASELINE STRATEGY ONLY"

print(f"\nCONCLUSION: {verdict}")
print(f"RECOMMENDATION: {recommendation}")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80 + "\n")