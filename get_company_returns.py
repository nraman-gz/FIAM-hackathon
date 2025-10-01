import pandas as pd
import matplotlib.pyplot as plt

def get_company_returns(company_id, plot=True, return_type='stock_ret'):
    """
    Extract and optionally plot stock returns for a specific company over time

    Parameters:
    company_id (str): Company identifier (e.g., 'comp_001081_01C')
    plot (bool): Whether to create a plot of returns over time
    return_type (str): Type of return to analyze ('stock_ret', 'ret_1_0', 'ret_3_1', etc.)

    Returns:
    pandas.DataFrame: Filtered data for the specified company
    """

    # Load the data
    print(f"Loading data for company: {company_id}")
    data = pd.read_parquet("ret_sample.parquet")

    # Filter for specific company
    company_data = data[data['id'] == company_id].copy()

    if company_data.empty:
        print(f"No data found for company ID: {company_id}")
        return None

    # Sort by date
    company_data = company_data.sort_values('date')

    # Convert date to datetime for better plotting
    company_data['datetime'] = pd.to_datetime(company_data['date'], format='%Y%m%d')

    print(f"Found {len(company_data)} time periods for {company_id}")
    print(f"Date range: {company_data['date'].min()} to {company_data['date'].max()}")

    # Show summary statistics
    if return_type in company_data.columns:
        returns = company_data[return_type].dropna()
        print(f"\n{return_type} summary statistics:")
        print(f"  Mean: {returns.mean():.4f}")
        print(f"  Std: {returns.std():.4f}")
        print(f"  Min: {returns.min():.4f}")
        print(f"  Max: {returns.max():.4f}")

    # Create plot if requested
    if plot and return_type in company_data.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(company_data['datetime'], company_data[return_type],
                linewidth=1, markersize=3)
        plt.title(f'{return_type} Over Time for {company_id}')
        plt.xlabel('Date')
        plt.ylabel(f'{return_type}')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return company_data

def list_available_companies(n=20):
    """List first n unique company IDs in the dataset"""
    data = pd.read_parquet("ret_sample.parquet")
    unique_companies = data['id'].unique()[:n]
    print(f"First {len(unique_companies)} company IDs:")
    for i, company in enumerate(unique_companies, 1):
        print(f"{i:2d}. {company}")
    return unique_companies

def compare_companies(company_ids, return_type='stock_ret'):
    """Compare returns across multiple companies"""
    data = pd.read_parquet("ret_sample.parquet")

    plt.figure(figsize=(14, 8))

    for company_id in company_ids:
        company_data = data[data['id'] == company_id].sort_values('date')
        if not company_data.empty:
            company_data['datetime'] = pd.to_datetime(company_data['date'], format='%Y%m%d')
            plt.plot(company_data['datetime'], company_data[return_type],
                    label=company_id, linewidth=1, marker='o', markersize=2)

    plt.title(f'{return_type} Comparison Across Companies')
    plt.xlabel('Date')
    plt.ylabel(f'{return_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    get_company_returns(company_id='comp_001186_01C', plot = True, return_type='stock_ret')