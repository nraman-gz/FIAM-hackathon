import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict


class DataLoader:
    """
    Loads and prepares your specific CSV format for the missing data pipeline.

    Your format:
    - Column A: Index (no header)
    - Columns B-L: Identifiers and dates
    - Columns M+: Features
    """

    def __init__(self, filepath: str = 'ret_sample.parquet'):
        """
        Initialize loader with your parquet file.

        Args:
            filepath: Path to your ret_sample.parquet
        """
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None

    def load_raw_data(self, verbose: bool = True) -> pd.DataFrame:
        """
        Load the raw CSV file.

        Handles the column A (no header) issue.
        """
        if verbose:
            print(f"Loading data from: {self.filepath}")

        # Read file - handle both parquet and CSV formats
        import os
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        file_size_gb = os.path.getsize(self.filepath) / (1024**3)
        if verbose:
            print(f"File size: {file_size_gb:.2f} GB")

        # Determine file type and read accordingly
        if self.filepath.lower().endswith('.csv'):
            if verbose:
                print("Reading CSV file...")
            self.raw_data = pd.read_csv(self.filepath)

        elif self.filepath.lower().endswith('.parquet'):
            if verbose:
                print(f"Attempting to read parquet file...")

            # Try reading with timeout protection for large parquet files
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("File reading timed out - file may be corrupted or too large")

            # Set timeout for file operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout

            try:
                # Try reading with specific columns first to see the structure
                import pyarrow.parquet as pq

                if verbose:
                    print("Reading parquet metadata...")

                parquet_file = pq.ParquetFile(self.filepath)

                if verbose:
                    print(f"Parquet file info:")
                    print(f"  Rows: {parquet_file.metadata.num_rows:,}")
                    print(f"  Columns: {len(parquet_file.schema)}")

                # Read the parquet file
                if verbose:
                    print("Reading parquet data...")
                self.raw_data = pd.read_parquet(self.filepath)

            except (TimeoutError, OSError) as e:
                signal.alarm(0)  # Cancel timeout
                if verbose:
                    print(f"File access error: {e}")
                    print("\n" + "="*50)
                    print("FILE ACCESS ISSUE DETECTED")
                    print("="*50)
                    print("The parquet file appears to be:")
                    print("1. Corrupted")
                    print("2. Too large for available memory")
                    print("3. Located on slow/network storage")
                    print("4. Locked by another process")
                    print("\nSuggested solutions:")
                    print("1. Copy file to local SSD storage")
                    print("2. Use a smaller sample of the data")
                    print("3. Convert to CSV format first")
                    print("4. Check file integrity")
                    print("5. Run 'python3.13 run_analysis_demo.py' to test with synthetic data")
                    print("="*50)

                raise FileNotFoundError(
                    f"Cannot access parquet file: {self.filepath}. "
                    f"File may be corrupted, too large, or on slow storage. "
                    f"Try running 'python3.13 run_analysis_demo.py' to test with synthetic data."
                )

            except Exception as e:
                signal.alarm(0)  # Cancel timeout
                if verbose:
                    print(f"Error reading parquet: {e}")
                    print("Trying alternative reading method...")

                # Fallback: try reading in chunks if the file is too large
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    parquet_file = pq.ParquetFile(self.filepath)
                    # Read in batches if file is very large
                    if parquet_file.metadata.num_rows > 1000000:  # More than 1M rows
                        if verbose:
                            print("Large file detected, reading in batches...")

                        # Read first batch to get column structure
                        batch_iter = parquet_file.iter_batches(batch_size=100000)
                        first_batch = next(batch_iter)
                        self.raw_data = first_batch.to_pandas()

                        # Read remaining batches
                        for batch in batch_iter:
                            batch_df = batch.to_pandas()
                            self.raw_data = pd.concat([self.raw_data, batch_df], ignore_index=True)

                            if verbose and len(self.raw_data) % 500000 == 0:
                                print(f"  Read {len(self.raw_data):,} rows so far...")
                    else:
                        self.raw_data = pd.read_parquet(self.filepath)

                except Exception as e2:
                    if verbose:
                        print(f"All reading methods failed: {e2}")
                        print("\nTROUBLESHOOTING STEPS:")
                        print("1. Check if file is corrupted")
                        print("2. Try converting to CSV format")
                        print("3. Use a smaller sample")
                        print("4. Check available memory")
                        print("5. Run 'python3.13 run_analysis_demo.py' for demo")
                    raise e2

            finally:
                signal.alarm(0)  # Cancel timeout

        else:
            raise ValueError(f"Unsupported file format: {self.filepath}. Use .csv or .parquet files.")

        if verbose:
            print(f"✓ Loaded {len(self.raw_data):,} rows")
            print(f"✓ Columns: {len(self.raw_data.columns)}")
            print(f"\nFirst few column names:")
            print(list(self.raw_data.columns[:15]))

        return self.raw_data

    def process_for_pipeline(self, verbose: bool = True) -> pd.DataFrame:
        """
        Process raw data into format needed for the missing data pipeline.

        Pipeline expects:
        - ticker: Stock identifier
        - date: Trading date
        - price: Stock price (we'll derive from returns)
        - volume: Trading volume
        - All your features

        Returns:
            DataFrame ready for CompleteMissingDataPipeline
        """
        if self.raw_data is None:
            self.load_raw_data(verbose=verbose)

        if verbose:
            print("\nProcessing data for pipeline...")

        df = self.raw_data.copy()

        # Create ticker from gvkey + iid (unique identifier)
        df['ticker'] = df['gvkey'].astype(str) + '_' + df['iid'].astype(str)

        # Use date column (already in YYYYMMDD format)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        # Create price from returns (cumulative product)
        # Start at 100 for each stock and apply monthly returns
        if verbose:
            print("Computing prices from returns...")

        price_data = []
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].sort_values('date').copy()

            # Cumulative returns to create price series starting at 100
            ticker_data['price'] = 100 * (1 + ticker_data['stock_ret']).cumprod()

            price_data.append(ticker_data)

        df = pd.concat(price_data, ignore_index=True)

        # Add volume placeholder (you don't have volume data, so use a proxy)
        # We can use the variability in returns as a proxy for activity
        df['volume'] = np.abs(df['stock_ret']) * 1000000  # Rough proxy

        # Rename features to match expected names (if needed)
        # Your features are already named, but let's standardize some key ones
        feature_mapping = {
            # Add mappings here if your column names differ
            # Example: 'your_column_name': 'expected_column_name'
        }

        df = df.rename(columns=feature_mapping)

        # Select essential columns + all features
        essential_cols = ['ticker', 'date', 'price', 'volume', 'stock_ret',
                          'gvkey', 'iid', 'excntry', 'year', 'month']

        # Get all feature columns (everything not in essential cols)
        feature_cols = [col for col in df.columns if col not in essential_cols]

        # Final dataframe
        self.processed_data = df[essential_cols + feature_cols]

        if verbose:
            print(f"✓ Processed data ready!")
            print(f"  Unique tickers: {self.processed_data['ticker'].nunique()}")
            print(f"  Date range: {self.processed_data['date'].min()} to {self.processed_data['date'].max()}")
            print(f"  Total features: {len(feature_cols)}")
            print(f"\nFeature columns available:")
            print(feature_cols[:20], "..." if len(feature_cols) > 20 else "")

        return self.processed_data

    def check_feature_availability(self) -> pd.DataFrame:
        """
        Check which critical features are available in your data.

        Returns:
            DataFrame showing which features exist and their coverage
        """
        if self.processed_data is None:
            self.process_for_pipeline(verbose=False)

        # Critical features for detection (using actual column names from your dataset)
        critical_features = {
            'Bankruptcy Detection': [
                'z_score',  # Altman Z-score
                'o_score',  # Ohlson O-score
                'kz_index',  # Kaplan-Zingales index
                'ni_be',  # Return on equity
                'ocf_at',  # Operating cash flow to assets
                'zero_trades_252d',  # Number of zero trades (12 months)
                'ami_126d',  # Amihud Measure
                'at_be'  # Book leverage
            ],
            'Merger Detection': [
                'dolvol_126d',  # Dollar trading volume
                'bidaskhl_21d',  # The high-low bid-ask spread
                'ni_be',  # Return on equity
                'eqnpo_me'  # Net payout yield
            ],
            'Liquidity Metrics': [
                'zero_trades_21d',  # Number of zero trades (1 month)
                'zero_trades_126d',  # Number of zero trades (6 months)
                'zero_trades_252d',  # Number of zero trades (12 months)
                'turnover_126d',  # Share turnover
                'ami_126d',  # Amihud Measure
                'dolvol_var_126d'  # Coefficient of variation for dollar trading volume
            ],
            'Quality Metrics': [
                'f_score',  # Pitroski F-score
                'ni_ar1',  # Earnings persistence
                'ni_ivol'  # Earnings volatility
            ]
        }

        results = []

        for category, features in critical_features.items():
            for feature in features:
                # Try different possible column names
                possible_names = [
                    feature,
                    feature.replace('_', ' ').title().replace(' ', ''),
                    feature.lower(),
                    feature.upper()
                ]

                found = False
                actual_name = None
                coverage = 0

                for name in possible_names:
                    if name in self.processed_data.columns:
                        found = True
                        actual_name = name
                        coverage = (self.processed_data[name].notna().sum() /
                                    len(self.processed_data) * 100)
                        break

                results.append({
                    'category': category,
                    'feature': feature,
                    'available': '✓' if found else '✗',
                    'column_name': actual_name if found else 'NOT FOUND',
                    'coverage_pct': f"{coverage:.1f}%" if found else "N/A"
                })

        return pd.DataFrame(results)

    def get_sample_statistics(self) -> Dict:
        """
        Get basic statistics about your dataset.
        """
        if self.processed_data is None:
            self.process_for_pipeline(verbose=False)

        df = self.processed_data

        stats = {
            'total_rows': len(df),
            'unique_tickers': df['ticker'].nunique(),
            'unique_stocks': df['gvkey'].nunique(),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'years_covered': df['year'].nunique(),
            'countries': df['excntry'].unique().tolist() if 'excntry' in df.columns else [],
            'avg_observations_per_stock': len(df) / df['ticker'].nunique(),
            'total_features': len([c for c in df.columns if c not in
                                   ['ticker', 'date', 'price', 'volume', 'stock_ret',
                                    'gvkey', 'iid', 'excntry', 'year', 'month']])
        }

        return stats

    def detect_missing_data_patterns(self) -> pd.DataFrame:
        """
        Analyze missing data patterns before running detection.

        This helps you understand what you're dealing with.
        """
        if self.processed_data is None:
            self.process_for_pipeline(verbose=False)

        df = self.processed_data

        patterns = []

        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].sort_values('date')

            # Basic info
            first_date = ticker_data['date'].min()
            last_date = ticker_data['date'].max()
            n_observations = len(ticker_data)

            # Date gaps
            ticker_data['date_diff'] = ticker_data['date'].diff().dt.days
            gaps = ticker_data[ticker_data['date_diff'] > 35]  # More than ~1 month
            n_gaps = len(gaps)
            max_gap = ticker_data['date_diff'].max() if len(ticker_data) > 1 else 0

            # Ending status
            days_since_last = (pd.Timestamp.now() - last_date).days

            if days_since_last > 365:
                status = 'likely_delisted'
            elif days_since_last > 90:
                status = 'possibly_delisted'
            elif n_gaps > 5:
                status = 'many_gaps'
            else:
                status = 'active'

            patterns.append({
                'ticker': ticker,
                'first_date': first_date,
                'last_date': last_date,
                'n_observations': n_observations,
                'n_gaps': n_gaps,
                'max_gap_days': max_gap,
                'days_since_last': days_since_last,
                'status': status
            })

        return pd.DataFrame(patterns)


# ============================================================================
# COMPLETE WORKFLOW FOR YOUR DATA
# ============================================================================

def run_complete_workflow_on_your_data(
        csv_filepath: str = 'ret_sample.parquet',
        claude_api_key: str = None,
        verify_batch_size: int = 100,
        use_haiku: bool = True
):
    """
    Complete end-to-end workflow customized for your data format.

    Steps:
    1. Load your ret_sample.parquet
    2. Process into pipeline format
    3. Check feature availability
    4. Run detection rules
    5. Verify with Claude (when you have API key)
    6. Create labeled dataset
    7. Export results

    Args:
        csv_filepath: Path to your ret_sample.parquet
        claude_api_key: Your Claude API key (skip verification if None)
        verify_batch_size: How many events to verify
        use_haiku: Use cheaper Haiku model (recommended)
    """

    print("=" * 70)
    print("COMPLETE MISSING DATA PIPELINE - YOUR DATA FORMAT")
    print("=" * 70)

    # STEP 1: Load your data
    print("\n" + "=" * 70)
    print("STEP 1: LOAD YOUR DATA")
    print("=" * 70)

    loader = DataLoader(csv_filepath)
    processed_data = loader.process_for_pipeline(verbose=True)

    # STEP 2: Check features
    print("\n" + "=" * 70)
    print("STEP 2: CHECK FEATURE AVAILABILITY")
    print("=" * 70)

    feature_check = loader.check_feature_availability()
    print("\nCritical Features Status:")
    print(feature_check.to_string(index=False))

    # STEP 3: Analyze missing data patterns
    print("\n" + "=" * 70)
    print("STEP 3: ANALYZE MISSING DATA PATTERNS")
    print("=" * 70)

    patterns = loader.detect_missing_data_patterns()
    print("\nMissing Data Summary:")
    print(f"Total stocks: {len(patterns)}")
    print(f"\nBy status:")
    print(patterns['status'].value_counts())
    print(f"\nStocks with gaps: {(patterns['n_gaps'] > 0).sum()}")
    print(f"Stocks likely delisted: {(patterns['status'] == 'likely_delisted').sum()}")

    # STEP 4: Get dataset statistics
    print("\n" + "=" * 70)
    print("STEP 4: DATASET STATISTICS")
    print("=" * 70)

    stats = loader.get_sample_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # STEP 5: Run detection (if user wants to continue)
    print("\n" + "=" * 70)
    print("STEP 5: RUN DETECTION RULES")
    print("=" * 70)

    proceed = input("\nProceed with event detection? (yes/no): ")

    if proceed.lower() != 'yes':
        print("\nStopping here. Your processed data is ready in loader.processed_data")
        return loader

    # Import the pipeline
    from CompleteMissingDataPipeline import CompleteMissingDataPipeline

    pipeline = CompleteMissingDataPipeline(
        price_data=processed_data,
        claude_api_key=claude_api_key or "dummy_key",  # Placeholder if no key yet
        dataset_start_date=processed_data['date'].min(),
        company_names_file="cik_gvkey_linktable_USA_only.csv"  # Add company names mapping
    )

    # Run detection only (skip verification if no API key)
    detected_events = pipeline.step1_detect_all_events(verbose=True)

    # STEP 6: Verify with Claude (if API key provided)
    if claude_api_key:
        print("\n" + "=" * 70)
        print("STEP 6: VERIFY WITH CLAUDE")
        print("=" * 70)

        verified_events = pipeline.step2_verify_with_llm(
            batch_size=verify_batch_size,
            use_haiku=use_haiku,
            verbose=True
        )

        # Create labeled dataset
        labeled_data = pipeline.step3_create_labeled_dataset(verbose=True)

        # Export everything
        pipeline.step4_export_results(output_dir='results', verbose=True)

    else:
        print("\n" + "=" * 70)
        print("NO API KEY PROVIDED")
        print("=" * 70)
        print("\nSkipping Claude verification.")
        print("Detected events saved for later verification.")

        # Save detected events
        detected_events.to_csv('detected_events_to_verify.csv', index=False)
        print(f"\n✓ Saved {len(detected_events)} detected events to: detected_events_to_verify.csv")
        print("\nNext steps:")
        print("1. Get Claude API key from: console.anthropic.com")
        print("2. Run again with: run_complete_workflow_on_your_data(claude_api_key='your-key')")

    return loader, pipeline


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  MISSING DATA EXPLANATION PIPELINE                ║
    ║                      Customized for Your Data                     ║
    ╚══════════════════════════════════════════════════════════════════╝

    This will:
    1. Load your ret_sample.parquet with the specific format
    2. Check which features are available
    3. Analyze missing data patterns
    4. Run aggressive detection rules
    5. (Optional) Verify with Claude when you have API key

    """)

    # Run workflow
    # WITHOUT API key (just detection):
    loader, pipeline = run_complete_workflow_on_your_data(
        csv_filepath='ret_sample.parquet',
        claude_api_key=None  # Set to your key when you have it
    )

    # WITH API key (full pipeline):
    # loader, pipeline = run_complete_workflow_on_your_data(
    #     csv_filepath='ret_sample.parquet',
    #     claude_api_key='sk-ant-...',  # Your actual key
    #     verify_batch_size=50,
    #     use_haiku=True  # Cheaper model
    # )
