"""
CORPORATE PORTFOLIO EVALUATION - EXECUTIVE DASHBOARD
4 Executive Diagrams - Pure Purple Corporate Theme
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set corporate purple theme
plt.style.use('seaborn-v0_8')
PURPLE_PALETTE = ["#4B0082", "#6A0DAD", "#8A2BE2", "#9370DB", "#B19CD9"]


class ExecutivePortfolioVisualizer:

    def __init__(self, results_dir="./complete_backtest_results_2015_2025"):
        self.results_dir = Path(results_dir)
        self.data_loaded = False

    def load_data(self):
        """Load portfolio returns data"""
        try:
            self.baseline_returns = pd.read_csv(self.results_dir / "portfolio_returns_baseline.csv")
            self.combined_returns = pd.read_csv(self.results_dir / "portfolio_returns_combined.csv")
            self.yearly_df = pd.read_csv(self.results_dir / "yearly_comparison.csv")

            for df in [self.baseline_returns, self.combined_returns]:
                df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
                df.set_index('date', inplace=True)

            self.data_loaded = True
            print("✓ Data loaded successfully")

        except Exception as e:
            print(f"Error loading data: {e}")
            self.data_loaded = False

    # =========================================================================
    # DIAGRAM 1: Cumulative Returns (Clean Version - No Numerical Box)
    # =========================================================================

    def plot_cumulative_returns_executive(self, save_path=None):
        """Executive cumulative returns - clean version without numerical box"""
        if not self.data_loaded:
            self.load_data()

        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate cumulative returns
        combined_cumulative = (1 + self.combined_returns['port_ls']).cumprod()
        market_cumulative = (1 + self.baseline_returns['mkt_total']).cumprod()

        # Plot with purple palette
        ax.plot(combined_cumulative.index, combined_cumulative,
                label='AI + Sentiment Strategy', linewidth=4, color=PURPLE_PALETTE[0])
        ax.plot(market_cumulative.index, market_cumulative,
                label='Market Benchmark', linewidth=3, color=PURPLE_PALETTE[3], linestyle='--')

        # COVID shading in subtle purple
        covid_start = pd.Timestamp('2020-02-01')
        covid_end = pd.Timestamp('2020-04-30')
        ax.axvspan(covid_start, covid_end, alpha=0.1, color=PURPLE_PALETTE[4], label='COVID Period')

        ax.set_title('Investment Growth: Strategy vs Market', fontsize=18, fontweight='bold', color=PURPLE_PALETTE[0],
                     pad=20)
        ax.set_ylabel('Portfolio Value ($)', fontsize=14)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.2)

        # Clean professional styling - NO NUMERICAL BOX
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved cumulative returns plot to {save_path}")
        plt.show()

    # =========================================================================
    # DIAGRAM 2: Yearly Outperformance (STOP AT 2024)
    # =========================================================================

    def plot_yearly_outperformance_executive(self, save_path=None):
        """Yearly outperformance with purple gradient - STOP AT 2024"""
        if not self.data_loaded:
            self.load_data()

        # Filter data to stop at 2024
        filtered_yearly = self.yearly_df[self.yearly_df['year'] <= 2024]

        # Calculate yearly outperformance
        years = filtered_yearly['year']
        outperformance = filtered_yearly['combined_return'] - filtered_yearly['market_return']

        fig, ax = plt.subplots(figsize=(12, 6))

        # Use purple colors based on positive/negative
        bar_colors = [PURPLE_PALETTE[0] if x > 0 else PURPLE_PALETTE[3] for x in outperformance]

        # Create bars with consistent alpha
        bars = ax.bar(years.astype(str), outperformance, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)

        ax.set_title('Annual Strategy Outperformance vs Market (2015-2024)', fontsize=18, fontweight='bold',
                     color=PURPLE_PALETTE[0], pad=20)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Outperformance (%)', fontsize=14)
        ax.grid(True, alpha=0.2, axis='y')

        # Value labels (keep percentages on bars)
        for i, (bar, perf) in enumerate(zip(bars, outperformance)):
            ax.text(bar.get_x() + bar.get_width() / 2, perf + (0.003 if perf >= 0 else -0.005),
                    f'{perf:.1%}', ha='center', va='bottom' if perf >= 0 else 'top',
                    fontweight='bold', fontsize=10, color=PURPLE_PALETTE[0] if perf >= 0 else PURPLE_PALETTE[3])

        # Clean styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add performance summary - NO PERCENTAGES
        positive_years = (outperformance > 0).sum()
        total_years = len(outperformance)

        summary_text = f'Performance Summary:\nPositive Years: {positive_years} of {total_years}'
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=PURPLE_PALETTE[4], alpha=0.1))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved yearly outperformance plot to {save_path}")
        plt.show()

    # =========================================================================
    # DIAGRAM 3: Drawdown Comparison (NO PERCENTAGES IN BOX)
    # =========================================================================

    def plot_drawdown_comparison(self, save_path=None):
        """Drawdown analysis - shows risk management during stress periods"""
        if not self.data_loaded:
            self.load_data()

        def calculate_drawdown(returns):
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            return (cumulative - running_max) / running_max

        combined_drawdown = calculate_drawdown(self.combined_returns['port_ls'])
        market_drawdown = calculate_drawdown(self.baseline_returns['mkt_total'])

        fig, ax = plt.subplots(figsize=(12, 6))

        # Fill between for visual impact
        ax.fill_between(combined_drawdown.index, combined_drawdown, 0,
                        alpha=0.6, color=PURPLE_PALETTE[0], label='Strategy Drawdown')
        ax.fill_between(market_drawdown.index, market_drawdown, 0,
                        alpha=0.4, color=PURPLE_PALETTE[3], label='Market Drawdown')

        # COVID period
        covid_start = pd.Timestamp('2020-02-01')
        covid_end = pd.Timestamp('2020-04-30')
        ax.axvspan(covid_start, covid_end, alpha=0.1, color=PURPLE_PALETTE[4], label='COVID Period')

        ax.set_title('Portfolio Drawdown Analysis', fontsize=18, fontweight='bold', color=PURPLE_PALETTE[0], pad=20)
        ax.set_ylabel('Drawdown (%)', fontsize=14)
        ax.set_xlabel('Date', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.2)

        # Add max drawdown annotations - NO PERCENTAGES
        ax.text(0.02, 0.15, f'Maximum Drawdown Comparison:\nStrategy vs Market',
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor=PURPLE_PALETTE[4], alpha=0.1))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved drawdown comparison plot to {save_path}")
        plt.show()

    # =========================================================================
    # DIAGRAM 4: Rolling Performance (NO PERCENTAGES IN BOX)
    # =========================================================================

    def plot_rolling_performance(self, save_path=None):
        """Rolling 6-month performance - shows strategy consistency"""
        if not self.data_loaded:
            self.load_data()

        # Calculate rolling 6-month returns (annualized)
        window = 6
        combined_rolling = self.combined_returns['port_ls'].rolling(window).mean() * 12
        market_rolling = self.baseline_returns['mkt_total'].rolling(window).mean() * 12

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(combined_rolling.index, combined_rolling,
                label='Strategy (6-Month Rolling)', linewidth=3, color=PURPLE_PALETTE[0])
        ax.plot(market_rolling.index, market_rolling,
                label='Market (6-Month Rolling)', linewidth=2, color=PURPLE_PALETTE[3], linestyle='--')
        ax.axhline(y=0, color='black', linewidth=1, alpha=0.3)

        # COVID period
        covid_start = pd.Timestamp('2020-02-01')
        covid_end = pd.Timestamp('2020-04-30')
        ax.axvspan(covid_start, covid_end, alpha=0.1, color=PURPLE_PALETTE[4], label='COVID Period')

        ax.set_title('6-Month Rolling Performance', fontsize=18, fontweight='bold', color=PURPLE_PALETTE[0], pad=20)
        ax.set_ylabel('Annualized Return (%)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.2)

        # Add performance stats - NO PERCENTAGES
        stats_text = f'Performance Consistency:\nStrategy vs Market'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=PURPLE_PALETTE[4], alpha=0.1))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved rolling performance plot to {save_path}")
        plt.show()

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    def generate_executive_deck(self, output_dir="./executive_deck"):
        """Generate your 4 chosen diagrams for the executive deck"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        print("Generating Executive Deck Diagrams...")
        print("=" * 50)

        # Your 2 favorites
        print("1. Generating Cumulative Returns (Clean Version)...")
        self.plot_cumulative_returns_executive(output_path / "1_cumulative_returns.png")

        print("2. Generating Yearly Outperformance (2015-2024)...")
        self.plot_yearly_outperformance_executive(output_path / "2_yearly_outperformance.png")

        # My top 2 suggestions
        print("3. Generating Drawdown Comparison...")
        self.plot_drawdown_comparison(output_path / "3_drawdown_comparison.png")

        print("4. Generating Rolling Performance...")
        self.plot_rolling_performance(output_path / "4_rolling_performance.png")

        print("=" * 50)
        print(f"✓ All 4 executive diagrams saved to: {output_path}")


# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    visualizer = ExecutivePortfolioVisualizer("./complete_backtest_results_2015_2025")
    visualizer.generate_executive_deck("./executive_presentation")