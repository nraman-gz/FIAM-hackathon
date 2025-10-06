"""
COMPREHENSIVE PORTFOLIO EVALUATION PLOTS
All requested visualizations for strategy analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PortfolioVisualizer:
    """
    Comprehensive portfolio visualization toolkit
    """

    def __init__(self, results_dir="./complete_backtest_results_2015_2025"):
        self.results_dir = Path(results_dir)
        self.data_loaded = False

    def load_data(self):
        """Load all necessary data files"""
        try:
            # Load portfolio returns
            self.baseline_returns = pd.read_csv(self.results_dir / "portfolio_returns_baseline.csv")
            self.combined_returns = pd.read_csv(self.results_dir / "portfolio_returns_combined.csv")

            # Load prediction data
            self.baseline_pred = pd.read_csv(self.results_dir / "predictions_baseline_ranks.csv")
            self.combined_pred = pd.read_csv(self.results_dir / "predictions_combined_ranks.csv")

            # Load yearly comparison
            self.yearly_df = pd.read_csv(self.results_dir / "yearly_comparison.csv")

            # Create date index for time series plots
            for df in [self.baseline_returns, self.combined_returns]:
                df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
                df.set_index('date', inplace=True)

            self.data_loaded = True
            print("✓ Data loaded successfully")

        except Exception as e:
            print(f"Error loading data: {e}")
            self.data_loaded = False

    # =========================================================================
    # 1. RETURN ANALYSIS
    # =========================================================================

    def plot_cumulative_returns_covid(self, save_path=None):
        """Cumulative returns with COVID recession shading"""
        if not self.data_loaded:
            self.load_data()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Calculate cumulative returns
        baseline_cumulative = (1 + self.baseline_returns['port_ls']).cumprod()
        combined_cumulative = (1 + self.combined_returns['port_ls']).cumprod()
        market_cumulative = (1 + self.baseline_returns['mkt_total']).cumprod()

        # Plot Long-Short Portfolio
        ax1.plot(baseline_cumulative.index, baseline_cumulative,
                 label='Baseline (ML Only)', linewidth=2.5)
        ax1.plot(combined_cumulative.index, combined_cumulative,
                 label='Combined (ML + Sentiment)', linewidth=2.5)
        ax1.plot(market_cumulative.index, market_cumulative,
                 label='Market', linewidth=2, linestyle='--', alpha=0.8)

        # Shade COVID recession period (Feb-Apr 2020 per NBER)
        covid_start = pd.Timestamp('2020-02-01')
        covid_end = pd.Timestamp('2020-04-30')
        ax1.axvspan(covid_start, covid_end, alpha=0.3, color='red', label='COVID Recession (NBER)')

        ax1.set_title('Long-Short Portfolio: Evolution of $1 Investment', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Add final values annotation
        final_baseline = baseline_cumulative.iloc[-1]
        final_combined = combined_cumulative.iloc[-1]
        final_market = market_cumulative.iloc[-1]

        ax1.annotate(f'Baseline: ${final_baseline:.2f}',
                     xy=(baseline_cumulative.index[-1], final_baseline),
                     xytext=(10, 10), textcoords='offset points', fontsize=9, fontweight='bold')
        ax1.annotate(f'Combined: ${final_combined:.2f}',
                     xy=(combined_cumulative.index[-1], final_combined),
                     xytext=(10, 25), textcoords='offset points', fontsize=9, fontweight='bold')
        ax1.annotate(f'Market: ${final_market:.2f}',
                     xy=(market_cumulative.index[-1], final_market),
                     xytext=(10, 40), textcoords='offset points', fontsize=9, fontweight='bold')

        # Plot Long-Only Portfolio (Top Decile)
        baseline_long = (1 + self.baseline_returns['port_10']).cumprod()
        combined_long = (1 + self.combined_returns['port_10']).cumprod()

        ax2.plot(baseline_long.index, baseline_long,
                 label='Baseline Long (Top Decile)', linewidth=2.5)
        ax2.plot(combined_long.index, combined_long,
                 label='Combined Long (Top Decile)', linewidth=2.5)
        ax2.plot(market_cumulative.index, market_cumulative,
                 label='Market', linewidth=2, linestyle='--', alpha=0.8)

        # Shade COVID period
        ax2.axvspan(covid_start, covid_end, alpha=0.3, color='red', label='COVID Recession (NBER)')

        ax2.set_title('Long-Only Portfolio (Top Decile): Evolution of $1 Investment',
                      fontsize=14, fontweight='bold')
        ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(fontsize=10, loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved cumulative returns plot to {save_path}")

        plt.show()

    # =========================================================================
    # 2. RISK ANALYSIS
    # =========================================================================

    def plot_drawdown_analysis(self, save_path=None):
        """Maximum drawdowns over time"""
        if not self.data_loaded:
            self.load_data()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Calculate drawdowns
        def calculate_drawdown(returns):
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            return drawdown

        baseline_drawdown = calculate_drawdown(self.baseline_returns['port_ls'])
        combined_drawdown = calculate_drawdown(self.combined_returns['port_ls'])
        market_drawdown = calculate_drawdown(self.baseline_returns['mkt_total'])

        # Plot drawdowns
        ax1.fill_between(baseline_drawdown.index, baseline_drawdown, 0,
                         alpha=0.7, label='Baseline Drawdown', color='blue')
        ax1.fill_between(combined_drawdown.index, combined_drawdown, 0,
                         alpha=0.7, label='Combined Drawdown', color='orange')
        ax1.fill_between(market_drawdown.index, market_drawdown, 0,
                         alpha=0.3, label='Market Drawdown', color='red')

        ax1.set_title('Portfolio Drawdowns Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Drawdown', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=min(baseline_drawdown.min(), combined_drawdown.min(), market_drawdown.min()) - 0.05)

        # Add max drawdown annotations
        max_baseline_dd = baseline_drawdown.min()
        max_combined_dd = combined_drawdown.min()
        max_market_dd = market_drawdown.min()

        ax1.axhline(y=max_baseline_dd, color='blue', linestyle=':', alpha=0.7,
                    label=f'Max Baseline: {max_baseline_dd:.1%}')
        ax1.axhline(y=max_combined_dd, color='orange', linestyle=':', alpha=0.7,
                    label=f'Max Combined: {max_combined_dd:.1%}')
        ax1.axhline(y=max_market_dd, color='red', linestyle=':', alpha=0.7,
                    label=f'Max Market: {max_market_dd:.1%}')
        ax1.legend()

        # Plot rolling volatility (6-month)
        baseline_vol = self.baseline_returns['port_ls'].rolling(6).std() * np.sqrt(12)
        combined_vol = self.combined_returns['port_ls'].rolling(6).std() * np.sqrt(12)
        market_vol = self.baseline_returns['mkt_total'].rolling(6).std() * np.sqrt(12)

        ax2.plot(baseline_vol.index, baseline_vol, label='Baseline Volatility', linewidth=2)
        ax2.plot(combined_vol.index, combined_vol, label='Combined Volatility', linewidth=2)
        ax2.plot(market_vol.index, market_vol, label='Market Volatility', linewidth=2, linestyle='--')

        ax2.set_title('6-Month Rolling Volatility (Annualized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Annualized Volatility', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved drawdown analysis plot to {save_path}")

        plt.show()

    def plot_var_analysis(self, save_path=None):
        """Value at Risk analysis with distribution"""
        if not self.data_loaded:
            self.load_data()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Calculate VaR metrics
        baseline_returns = self.baseline_returns['port_ls']
        combined_returns = self.combined_returns['port_ls']
        market_returns = self.baseline_returns['mkt_total']

        # Plot return distributions
        ax1.hist(baseline_returns, bins=50, alpha=0.7, density=True,
                 label='Baseline', color='blue')
        ax1.hist(combined_returns, bins=50, alpha=0.7, density=True,
                 label='Combined', color='orange')

        # Add VaR lines
        var_95_baseline = np.percentile(baseline_returns, 5)
        var_95_combined = np.percentile(combined_returns, 5)
        var_99_baseline = np.percentile(baseline_returns, 1)
        var_99_combined = np.percentile(combined_returns, 1)

        ax1.axvline(var_95_baseline, color='blue', linestyle='--',
                    label=f'Baseline VaR 95%: {var_95_baseline:.2%}')
        ax1.axvline(var_95_combined, color='orange', linestyle='--',
                    label=f'Combined VaR 95%: {var_95_combined:.2%}')
        ax1.axvline(var_99_baseline, color='blue', linestyle=':',
                    label=f'Baseline VaR 99%: {var_99_baseline:.2%}')
        ax1.axvline(var_99_combined, color='orange', linestyle=':',
                    label=f'Combined VaR 99%: {var_99_combined:.2%}')

        ax1.set_xlabel('Monthly Return')
        ax1.set_ylabel('Density')
        ax1.set_title('Return Distribution with VaR Thresholds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Create VaR comparison table
        var_data = {
            'Strategy': ['Baseline', 'Combined'],
            'VaR 95%': [var_95_baseline, var_95_combined],
            'VaR 99%': [var_99_baseline, var_99_combined],
            'CVaR 95%': [
                baseline_returns[baseline_returns <= var_95_baseline].mean(),
                combined_returns[combined_returns <= var_95_combined].mean()
            ]
        }

        var_df = pd.DataFrame(var_data)

        # Plot VaR comparison as bar chart
        x = np.arange(len(var_df))
        width = 0.25

        ax2.bar(x - width, var_df['VaR 95%'], width, label='VaR 95%', alpha=0.8)
        ax2.bar(x, var_df['VaR 99%'], width, label='VaR 99%', alpha=0.8)
        ax2.bar(x + width, var_df['CVaR 95%'], width, label='CVaR 95%', alpha=0.8)

        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Return')
        ax2.set_title('Value at Risk Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(var_df['Strategy'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, (var95, var99, cvar) in enumerate(zip(var_df['VaR 95%'], var_df['VaR 99%'], var_df['CVaR 95%'])):
            ax2.text(i - width, var95 - 0.01, f'{var95:.2%}', ha='center', va='top', fontsize=8)
            ax2.text(i, var99 - 0.01, f'{var99:.2%}', ha='center', va='top', fontsize=8)
            ax2.text(i + width, cvar - 0.01, f'{cvar:.2%}', ha='center', va='top', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved VaR analysis plot to {save_path}")

        plt.show()

    # =========================================================================
    # 3. PREDICTION QUALITY
    # =========================================================================

    def plot_prediction_vs_actual(self, save_path=None):
        """Scatter plot of predicted vs actual returns"""
        if not self.data_loaded:
            self.load_data()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Sample data for clarity
        sample_size = min(5000, len(self.baseline_pred))
        baseline_sample = self.baseline_pred.sample(sample_size, random_state=42)
        combined_sample = self.combined_pred.sample(sample_size, random_state=42)

        # Baseline predictions
        ax1.scatter(baseline_sample['pred_ret'], baseline_sample['stock_ret'],
                    alpha=0.6, s=2, color='blue')

        # Perfect prediction line
        min_val = min(baseline_sample['pred_ret'].min(), baseline_sample['stock_ret'].min())
        max_val = max(baseline_sample['pred_ret'].max(), baseline_sample['stock_ret'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2,
                 label='Perfect Prediction')

        # Regression line
        z = np.polyfit(baseline_sample['pred_ret'], baseline_sample['stock_ret'], 1)
        p = np.poly1d(z)
        ax1.plot(baseline_sample['pred_ret'], p(baseline_sample['pred_ret']),
                 "g--", alpha=0.8, linewidth=2, label='Regression Line')

        ax1.set_xlabel('Predicted Returns')
        ax1.set_ylabel('Actual Returns')
        ax1.set_title('Baseline: Predicted vs Actual Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Combined predictions
        ax2.scatter(combined_sample['pred_ret'], combined_sample['stock_ret'],
                    alpha=0.6, s=2, color='orange')

        # Perfect prediction line
        min_val = min(combined_sample['pred_ret'].min(), combined_sample['stock_ret'].min())
        max_val = max(combined_sample['pred_ret'].max(), combined_sample['stock_ret'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2,
                 label='Perfect Prediction')

        # Regression line
        z = np.polyfit(combined_sample['pred_ret'], combined_sample['stock_ret'], 1)
        p = np.poly1d(z)
        ax2.plot(combined_sample['pred_ret'], p(combined_sample['pred_ret']),
                 "g--", alpha=0.8, linewidth=2, label='Regression Line')

        ax2.set_xlabel('Predicted Returns')
        ax2.set_ylabel('Actual Returns')
        ax2.set_title('Combined: Predicted vs Actual Returns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Calculate and display statistics
        baseline_corr = baseline_sample['pred_ret'].corr(baseline_sample['stock_ret'])
        combined_corr = combined_sample['pred_ret'].corr(combined_sample['stock_ret'])

        fig.suptitle(f'Prediction Accuracy: Baseline R={baseline_corr:.3f}, Combined R={combined_corr:.3f}',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved prediction vs actual plot to {save_path}")

        plt.show()

    def plot_hit_rate_over_time(self, save_path=None):
        """Rolling directional accuracy over time"""
        if not self.data_loaded:
            self.load_data()

        # Calculate monthly hit rates
        baseline_hit_rates = []
        combined_hit_rates = []
        dates = []

        for (year, month), group in self.baseline_pred.groupby(['year', 'month']):
            hit_rate = (np.sign(group['stock_ret']) == np.sign(group['pred_ret'])).mean()
            if not np.isnan(hit_rate):
                baseline_hit_rates.append(hit_rate)
                dates.append(pd.Timestamp(f'{year}-{month}-01'))

        for (year, month), group in self.combined_pred.groupby(['year', 'month']):
            hit_rate = (np.sign(group['stock_ret']) == np.sign(group['pred_ret'])).mean()
            if not np.isnan(hit_rate):
                combined_hit_rates.append(hit_rate)

        # Create series for rolling calculations
        baseline_series = pd.Series(baseline_hit_rates, index=dates[:len(baseline_hit_rates)])
        combined_series = pd.Series(combined_hit_rates, index=dates[:len(combined_hit_rates)])

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot rolling hit rates (6-month)
        ax.plot(baseline_series.rolling(6).mean(),
                label='Baseline (6M Avg)', linewidth=2.5)
        ax.plot(combined_series.rolling(6).mean(),
                label='Combined (6M Avg)', linewidth=2.5)

        # Add random benchmark (50%) and skilled benchmark (55%)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7,
                   label='Random (50%)')
        ax.axhline(y=0.55, color='green', linestyle='--', alpha=0.7,
                   label='Skilled (55%)')

        ax.set_xlabel('Date')
        ax.set_ylabel('Hit Rate (Directional Accuracy)')
        ax.set_title('6-Month Rolling Directional Accuracy', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add average hit rate annotations
        avg_baseline = baseline_series.mean()
        avg_combined = combined_series.mean()

        ax.annotate(f'Avg Baseline: {avg_baseline:.1%}',
                    xy=(baseline_series.index[-1], avg_baseline),
                    xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold')
        ax.annotate(f'Avg Combined: {avg_combined:.1%}',
                    xy=(combined_series.index[-1], avg_combined),
                    xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved hit rate plot to {save_path}")

        plt.show()

    # =========================================================================
    # 4. PORTFOLIO COMPOSITION
    # =========================================================================

    def plot_decile_performance(self, save_path=None):
        """Returns across all 10 portfolios"""
        if not self.data_loaded:
            self.load_data()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Calculate annual returns for each decile
        baseline_decile_returns = []
        combined_decile_returns = []

        for i in range(1, 11):
            baseline_ret = self.baseline_returns[f'port_{i}'].mean() * 12
            combined_ret = self.combined_returns[f'port_{i}'].mean() * 12
            baseline_decile_returns.append(baseline_ret)
            combined_decile_returns.append(combined_ret)

        # Plot decile returns
        x = np.arange(1, 11)

        ax1.bar(x - 0.2, baseline_decile_returns, 0.4,
                label='Baseline (ML Only)', alpha=0.8, color='blue')
        ax1.bar(x + 0.2, combined_decile_returns, 0.4,
                label='Combined (ML + Sentiment)', alpha=0.8, color='orange')

        ax1.set_xlabel('Portfolio Decile (1 = Worst, 10 = Best)', fontsize=12)
        ax1.set_ylabel('Annual Return', fontsize=12)
        ax1.set_title('Performance Across All Deciles', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add market return line for reference
        market_return = self.baseline_returns['mkt_total'].mean() * 12
        ax1.axhline(y=market_return, color='red', linestyle='--', alpha=0.7,
                    label=f'Market Return ({market_return:.1%})')
        ax1.legend()

        # Add value labels
        for i, (b, c) in enumerate(zip(baseline_decile_returns, combined_decile_returns)):
            ax1.text(i + 0.8, b + 0.01, f'{b:.1%}', ha='center', va='bottom', fontsize=8)
            ax1.text(i + 1.2, c + 0.01, f'{c:.1%}', ha='center', va='bottom', fontsize=8)

        # Plot monotonicity check (returns should increase)
        ax2.plot(x, baseline_decile_returns, 'o-', label='Baseline', linewidth=2, markersize=8)
        ax2.plot(x, combined_decile_returns, 'o-', label='Combined', linewidth=2, markersize=8)

        ax2.set_xlabel('Portfolio Decile', fontsize=12)
        ax2.set_ylabel('Annual Return', fontsize=12)
        ax2.set_title('Monotonicity Check: Returns Should Increase', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Calculate monotonicity scores
        def monotonicity_score(returns):
            increasing = sum(1 for i in range(len(returns) - 1) if returns[i + 1] > returns[i])
            return increasing / (len(returns) - 1)

        baseline_mono = monotonicity_score(baseline_decile_returns)
        combined_mono = monotonicity_score(combined_decile_returns)

        ax2.text(0.5, 0.95, f'Baseline Monotonicity: {baseline_mono:.1%}\nCombined Monotonicity: {combined_mono:.1%}',
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved decile performance plot to {save_path}")

        plt.show()

    def plot_top_holdings(self, save_path=None):
        """Most frequently longed/shorted stocks"""
        if not self.data_loaded:
            self.load_data()

        # Analyze long positions (rank 9 = top decile)
        long_positions = self.combined_pred[self.combined_pred['rank_combined'] == 9]
        short_positions = self.combined_pred[self.combined_pred['rank_combined'] == 0]

        # Count frequency
        long_counts = long_positions['id'].value_counts().head(10)
        short_counts = short_positions['id'].value_counts().head(10)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot top long holdings
        ax1.barh(range(len(long_counts)), long_counts.values, alpha=0.7, color='green')
        ax1.set_yticks(range(len(long_counts)))
        ax1.set_yticklabels([f'Stock {idx}' for idx in long_counts.index])
        ax1.set_xlabel('Number of Months in Portfolio')
        ax1.set_title('Top 10 Most Frequently Longed Stocks', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(long_counts.values):
            ax1.text(v + 0.1, i, str(v), va='center', fontsize=10)

        # Plot top short holdings
        ax2.barh(range(len(short_counts)), short_counts.values, alpha=0.7, color='red')
        ax2.set_yticks(range(len(short_counts)))
        ax2.set_yticklabels([f'Stock {idx}' for idx in short_counts.index])
        ax2.set_xlabel('Number of Months in Portfolio')
        ax2.set_title('Top 10 Most Frequently Shorted Stocks', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(short_counts.values):
            ax2.text(v + 0.1, i, str(v), va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved top holdings plot to {save_path}")

        plt.show()

    def plot_turnover_analysis(self, save_path=None):
        """Portfolio turnover over time"""
        if not self.data_loaded:
            self.load_data()

        # Simplified turnover calculation (approximate)
        def calculate_turnover(predictions, rank_col, top_rank=9, bottom_rank=0):
            """Calculate approximate turnover for long and short portfolios"""
            turnover_data = []

            for rank, name in [(top_rank, 'long'), (bottom_rank, 'short')]:
                portfolio = predictions[predictions[rank_col] == rank].copy()
                if len(portfolio) == 0:
                    continue

                # Group by date and count unique stocks
                monthly_counts = portfolio.groupby(['year', 'month'])['id'].nunique()

                # Calculate turnover as change in composition (simplified)
                turnover = monthly_counts.pct_change().fillna(0)
                for date, t in turnover.items():
                    turnover_data.append({
                        'year': date[0],
                        'month': date[1],
                        'turnover': t,
                        'portfolio': name
                    })

            return pd.DataFrame(turnover_data)

        # Calculate turnover for both strategies
        baseline_turnover = calculate_turnover(self.baseline_pred, 'rank_baseline')
        combined_turnover = calculate_turnover(self.combined_pred, 'rank_combined')

        # Create date index
        def add_date(df):
            df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
            return df

        baseline_turnover = add_date(baseline_turnover)
        combined_turnover = add_date(combined_turnover)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot long portfolio turnover
        baseline_long = baseline_turnover[baseline_turnover['portfolio'] == 'long']
        combined_long = combined_turnover[combined_turnover['portfolio'] == 'long']

        ax1.plot(baseline_long['date'], baseline_long['turnover'].rolling(6).mean(),
                 label='Baseline Long', linewidth=2)
        ax1.plot(combined_long['date'], combined_long['turnover'].rolling(6).mean(),
                 label='Combined Long', linewidth=2)

        ax1.set_title('6-Month Rolling Turnover: Long Portfolio', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Turnover Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot short portfolio turnover
        baseline_short = baseline_turnover[baseline_turnover['portfolio'] == 'short']
        combined_short = combined_turnover[combined_turnover['portfolio'] == 'short']

        ax2.plot(baseline_short['date'], baseline_short['turnover'].rolling(6).mean(),
                 label='Baseline Short', linewidth=2)
        ax2.plot(combined_short['date'], combined_short['turnover'].rolling(6).mean(),
                 label='Combined Short', linewidth=2)

        ax2.set_title('6-Month Rolling Turnover: Short Portfolio', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Turnover Rate')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add average turnover annotations
        avg_baseline_long = baseline_long['turnover'].mean()
        avg_combined_long = combined_long['turnover'].mean()
        avg_baseline_short = baseline_short['turnover'].mean()
        avg_combined_short = combined_short['turnover'].mean()

        ax1.text(0.02, 0.98, f'Avg Baseline: {avg_baseline_long:.1%}\nAvg Combined: {avg_combined_long:.1%}',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.text(0.02, 0.98, f'Avg Baseline: {avg_baseline_short:.1%}\nAvg Combined: {avg_combined_short:.1%}',
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved turnover analysis plot to {save_path}")

        plt.show()

    # =========================================================================
    # 5. STATISTICAL SIGNIFICANCE
    # =========================================================================

    def plot_capm_analysis(self, save_path=None):
        """CAPM regression visualization"""
        if not self.data_loaded:
            self.load_data()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # CAPM analysis for baseline
        baseline_ls = self.baseline_returns['port_ls']
        mkt_rf = self.baseline_returns['mkt_rf']

        # Run CAPM regression
        X_baseline = sm.add_constant(mkt_rf)
        model_baseline = sm.OLS(baseline_ls, X_baseline).fit()

        # Plot baseline CAPM
        ax1.scatter(mkt_rf, baseline_ls, alpha=0.6, s=30, label='Monthly Returns')

        # Regression line
        x_range = np.linspace(mkt_rf.min(), mkt_rf.max(), 100)
        y_pred = model_baseline.params[0] + model_baseline.params[1] * x_range
        ax1.plot(x_range, y_pred, 'r-', linewidth=2, label='CAPM Regression')

        ax1.set_xlabel('Market Excess Return')
        ax1.set_ylabel('Portfolio Excess Return')
        ax1.set_title(f'Baseline CAPM: Alpha={model_baseline.params[0]:.4f}, Beta={model_baseline.params[1]:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add regression statistics
        ax1.text(0.05, 0.95, f'R² = {model_baseline.rsquared:.3f}\nAlpha t-stat = {model_baseline.tvalues[0]:.2f}',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # CAPM analysis for combined
        combined_ls = self.combined_returns['port_ls']

        X_combined = sm.add_constant(mkt_rf)
        model_combined = sm.OLS(combined_ls, X_combined).fit()

        # Plot combined CAPM
        ax2.scatter(mkt_rf, combined_ls, alpha=0.6, s=30, label='Monthly Returns')

        # Regression line
        y_pred_combined = model_combined.params[0] + model_combined.params[1] * x_range
        ax2.plot(x_range, y_pred_combined, 'r-', linewidth=2, label='CAPM Regression')

        ax2.set_xlabel('Market Excess Return')
        ax2.set_ylabel('Portfolio Excess Return')
        ax2.set_title(f'Combined CAPM: Alpha={model_combined.params[0]:.4f}, Beta={model_combined.params[1]:.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add regression statistics
        ax2.text(0.05, 0.95, f'R² = {model_combined.rsquared:.3f}\nAlpha t-stat = {model_combined.tvalues[0]:.2f}',
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved CAPM analysis plot to {save_path}")

        plt.show()

    # =========================================================================
    # 6. STRATEGY COMPARISON
    # =========================================================================

    def plot_strategy_comparison(self, save_path=None):
        """Side-by-side performance metrics comparison"""
        if not self.data_loaded:
            self.load_data()

        # Calculate key metrics
        metrics = ['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Volatility']

        baseline_values = [
            self.baseline_returns['port_ls'].mean() * 12,
            self.baseline_returns['port_ls'].mean() / self.baseline_returns['port_ls'].std() * np.sqrt(12),
            self.calculate_max_drawdown(self.baseline_returns['port_ls']),
            (self.baseline_returns['port_ls'] > 0).mean(),
            self.baseline_returns['port_ls'].std() * np.sqrt(12)
        ]

        combined_values = [
            self.combined_returns['port_ls'].mean() * 12,
            self.combined_returns['port_ls'].mean() / self.combined_returns['port_ls'].std() * np.sqrt(12),
            self.calculate_max_drawdown(self.combined_returns['port_ls']),
            (self.combined_returns['port_ls'] > 0).mean(),
            self.combined_returns['port_ls'].std() * np.sqrt(12)
        ]

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width / 2, baseline_values, width, label='Baseline', alpha=0.8)
        bars2 = ax.bar(x + width / 2, combined_values, width, label='Combined', alpha=0.8)

        ax.set_xlabel('Performance Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels
        def add_value_labels(bars, values):
            for bar, value in zip(bars, values):
                if bar.get_height() >= 0:
                    va = 'bottom'
                    y_offset = 0.01
                else:
                    va = 'top'
                    y_offset = -0.01

                if 'Return' in metrics[bar.get_x() + width / 2] or 'Win Rate' in metrics[bar.get_x() + width / 2]:
                    label = f'{value:.2%}'
                elif 'Sharpe' in metrics[bar.get_x() + width / 2]:
                    label = f'{value:.3f}'
                else:
                    label = f'{value:.2%}' if value < 0 else f'{value:.3f}'

                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                        label, ha='center', va=va, fontsize=9, fontweight='bold')

        add_value_labels(bars1, baseline_values)
        add_value_labels(bars2, combined_values)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved strategy comparison plot to {save_path}")

        plt.show()

    def plot_transaction_cost_impact(self, save_path=None):
        """Net returns at different transaction cost levels"""
        if not self.data_loaded:
            self.load_data()

        # Calculate gross returns
        baseline_gross = self.baseline_returns['port_ls'].mean() * 12
        combined_gross = self.combined_returns['port_ls'].mean() * 12

        # Estimate turnover (simplified)
        def estimate_turnover(predictions, rank_col):
            """Estimate monthly turnover rate"""
            portfolio = predictions[predictions[rank_col].isin([0, 9])]  # Long and short
            monthly_changes = portfolio.groupby(['year', 'month'])['id'].nunique().pct_change().fillna(0)
            return monthly_changes.mean()

        baseline_turnover = estimate_turnover(self.baseline_pred, 'rank_baseline')
        combined_turnover = estimate_turnover(self.combined_pred, 'rank_combined')

        # Calculate net returns for different cost scenarios
        cost_levels = [0, 5, 10, 20, 50]  # basis points
        baseline_net_returns = []
        combined_net_returns = []

        for cost_bps in cost_levels:
            # Annual cost = monthly turnover * 12 * cost in decimal
            baseline_cost = baseline_turnover * 12 * (cost_bps / 10000)
            combined_cost = combined_turnover * 12 * (cost_bps / 10000)

            baseline_net_returns.append(baseline_gross - baseline_cost)
            combined_net_returns.append(combined_gross - combined_cost)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot net returns vs transaction costs
        ax1.plot(cost_levels, baseline_net_returns, 'o-', label='Baseline', linewidth=2, markersize=8)
        ax1.plot(cost_levels, combined_net_returns, 'o-', label='Combined', linewidth=2, markersize=8)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')

        ax1.set_xlabel('Transaction Cost (Basis Points)')
        ax1.set_ylabel('Net Annual Return')
        ax1.set_title('Transaction Cost Impact on Net Returns', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for i, (cost, base_net, comb_net) in enumerate(zip(cost_levels, baseline_net_returns, combined_net_returns)):
            ax1.annotate(f'{base_net:.2%}', (cost, base_net), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8)
            ax1.annotate(f'{comb_net:.2%}', (cost, comb_net), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8)

        # Plot breakeven analysis
        ax2.bar(['Baseline', 'Combined'], [baseline_turnover, combined_turnover],
                alpha=0.7, color=['blue', 'orange'])
        ax2.set_ylabel('Estimated Monthly Turnover')
        ax2.set_title('Portfolio Turnover Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate([baseline_turnover, combined_turnover]):
            ax2.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved transaction cost analysis plot to {save_path}")

        plt.show()

    def calculate_max_drawdown(self, returns):
        """Helper function to calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    # =========================================================================
    # MAIN EXECUTION METHOD
    # =========================================================================

    def generate_all_plots(self, output_dir="./portfolio_plots"):
        """Generate all requested plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        print("Generating all portfolio evaluation plots...")

        # 1. Return Analysis
        self.plot_cumulative_returns_covid(output_path / "1_cumulative_returns_covid.png")

        # 2. Risk Analysis
        self.plot_drawdown_analysis(output_path / "2_drawdown_analysis.png")
        self.plot_var_analysis(output_path / "3_var_analysis.png")

        # 3. Prediction Quality
        self.plot_prediction_vs_actual(output_path / "4_prediction_vs_actual.png")
        self.plot_hit_rate_over_time(output_path / "5_hit_rate_over_time.png")

        # 4. Portfolio Composition
        self.plot_decile_performance(output_path / "6_decile_performance.png")
        self.plot_top_holdings(output_path / "7_top_holdings.png")
        self.plot_turnover_analysis(output_path / "8_turnover_analysis.png")

        # 5. Statistical Significance
        self.plot_capm_analysis(output_path / "9_capm_analysis.png")

        # 6. Strategy Comparison
        self.plot_strategy_comparison(output_path / "10_strategy_comparison.png")
        self.plot_transaction_cost_impact(output_path / "11_transaction_cost_impact.png")

        print(f"✓ All plots saved to: {output_path}")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Initialize visualizer
    visualizer = PortfolioVisualizer("./complete_backtest_results_2015_2025")

    # Generate all plots
    visualizer.generate_all_plots("./portfolio_evaluation_plots")

    # Or generate individual plots
    # visualizer.plot_cumulative_returns_covid()
    # visualizer.plot_drawdown_analysis()
    # visualizer.plot_prediction_vs_actual()