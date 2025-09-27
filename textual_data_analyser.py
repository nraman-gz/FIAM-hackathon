import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Dict, List, Optional
import logging
import time
from functools import wraps


class Config:
    """Configuration class for FIAM Data Processor"""
    EXPECTED_COLUMNS = ['cik', 'date', 'mgmt', 'rf']
    TEXT_COLUMNS = ['mgmt', 'rf']
    REQUIRED_COLUMNS = ['cik', 'date']
    PORTFOLIO_THRESHOLDS = {'short': 0.2, 'long': 0.8}
    RISK_WORDS = [
        'risk', 'uncertainty', 'challenge', 'volatility', 'exposure',
        'hazard', 'threat', 'litigation', 'credit', 'operational',
        'default', 'bankruptcy', 'recession', 'competition', 'regulatory'
    ]
    MIN_DATA_FOR_PORTFOLIOS = 100
    DATE_FORMAT = '%Y%m%d'


def timing_decorator(func):
    """Decorator to time function execution"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        self.logger.info(f"Starting {func.__name__}...")

        result = func(self, *args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time

        # Format time nicely
        if execution_time < 60:
            time_str = f"{execution_time:.2f} seconds"
        elif execution_time < 3600:
            time_str = f"{execution_time / 60:.2f} minutes"
        else:
            time_str = f"{execution_time / 3600:.2f} hours"

        self.logger.info(f"Completed {func.__name__} in {time_str}")

        # Store timing info
        if not hasattr(self, 'execution_times'):
            self.execution_times = {}
        self.execution_times[func.__name__] = execution_time

        return result

    return wrapper


class FIAMDataProcessor:
    """
    MAIN PROCESSOR CLASS: Handles loading, cleaning, and analyzing SEC filing text data
    for the FIAM competition. This replicates the research paper's methodology with
    improved error handling and validation.
    """

    def __init__(self, base_path: str = None, text_file_name: str = "text_us_2005.pkl"):
        """
        INITIALIZATION: Set up file paths and create directory structure

        Args:
            base_path: Base directory for the project
            text_file_name: Name of the text data file
        """
        # Set default path if none provided
        if base_path is None:
            base_path = "/Users/krishna_dewan/Desktop/FIAM/"

        self.base_path = Path(base_path)
        # File is directly in the base path, not in a subfolder
        self.text_file_path = Path("/Users/krishna_dewan/Desktop/FIAM/text_us_2005.pkl")
        self.execution_times = {}  # Initialize timing storage

        # Validate and setup (order matters - logging first!)
        self._validate_initialization()
        self._setup_logging()
        self.setup_directories()

    def _validate_initialization(self):
        """Validate initialization parameters and create base path if needed"""
        # Create base path if it doesn't exist
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)
            print(f"Created base directory: {self.base_path}")

    def _setup_logging(self):
        """Setup logging for the processor"""
        # Ensure the base directory exists before creating log file
        log_file = self.base_path / 'fiam_processor.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")  # Unique logger name

    def setup_directories(self):
        """
        CREATE FOLDER STRUCTURE: Organizes project files into categories
        """
        directories = ['raw_data', 'cleaned_data', 'text_embeddings', 'models', 'visualizations']

        for dir_name in directories:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(exist_ok=True)
            self.logger.info(f"Created/verified directory: {dir_path}")

    @timing_decorator
    def load_text_data(self) -> Optional[Union[pd.DataFrame, Dict, List]]:
        """
        LOAD RAW DATA: Reads the pickle file containing SEC filings data
        Returns: Loaded data or None if error occurs
        """
        try:
            if not self.text_file_path.exists():
                raise FileNotFoundError(f"Text file not found: {self.text_file_path}")

            with open(self.text_file_path, 'rb') as f:
                text_data = pickle.load(f)

            self.logger.info("=== Text Data Summary ===")
            self.logger.info(f"Data type: {type(text_data)}")
            self.logger.info(f"File loaded successfully: {self.text_file_path}")

            if isinstance(text_data, pd.DataFrame):
                self.logger.info(f"DataFrame shape: {text_data.shape}")
                self.logger.info(f"Columns: {text_data.columns.tolist()}")

                if not text_data.empty:
                    self.logger.info("First few rows preview available")
                    print(text_data.head())
                else:
                    self.logger.warning("DataFrame is empty")

            return text_data

        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading pickle file: {e}")
            return None

    @timing_decorator
    def process_text_data(self, text_data) -> Optional[pd.DataFrame]:
        """
        MAIN PROCESSING ROUTER: Directs data to appropriate cleaning method based on type
        """
        if text_data is None:
            self.logger.warning("No text data to process")
            return None

        if isinstance(text_data, pd.DataFrame):
            return self._process_dataframe_text(text_data)
        elif isinstance(text_data, dict):
            return self._process_dict_text(text_data)
        else:
            return self._process_unknown_text(text_data)

    def _process_dataframe_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and prepares raw filing data stored in a DataFrame with improved validation.
        """
        self.logger.info("Processing DataFrame text data...")

        # Validation checks
        if df.empty:
            self.logger.warning("Empty DataFrame provided")
            return df

        # Check for required columns
        missing_cols = [col for col in Config.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return df

        available_text_cols = [col for col in Config.TEXT_COLUMNS if col in df.columns]
        self.logger.info(f"Available text columns: {available_text_cols}")

        # Create copy to avoid modifying original data
        df_clean = df.copy()

        # CLEAN CIK (Company Identifier)
        if 'cik' in df.columns:
            initial_count = len(df_clean)
            df_clean = df_clean[df_clean['cik'].notna()]
            df_clean['cik'] = df_clean['cik'].astype(int)

            dropped_count = initial_count - len(df_clean)
            if dropped_count > 0:
                self.logger.info(f"Dropped {dropped_count} rows with missing CIK")

        # CONVERT DATE with better error handling
        if 'date' in df.columns:
            try:
                df_clean['filing_date'] = pd.to_datetime(
                    df_clean['date'],
                    format=Config.DATE_FORMAT,
                    errors='coerce'
                )

                # Check for failed date conversions
                failed_dates = df_clean['filing_date'].isna().sum()
                if failed_dates > 0:
                    self.logger.warning(f"Failed to convert {failed_dates} dates")

                valid_dates = df_clean['filing_date'].notna()
                if valid_dates.any():
                    date_range = f"{df_clean.loc[valid_dates, 'filing_date'].min()} to {df_clean.loc[valid_dates, 'filing_date'].max()}"
                    self.logger.info(f"Date range: {date_range}")

            except Exception as e:
                self.logger.error(f"Error converting dates: {e}")

        # IMPROVED TEXT CLEANING
        for col in Config.TEXT_COLUMNS:
            if col in df_clean.columns:
                # Fill missing values
                df_clean[col] = df_clean[col].fillna('')

                # Comprehensive text cleaning
                df_clean[col] = (df_clean[col]
                                 .str.replace('\n', ' ')
                                 .str.replace('\r', ' ')
                                 .str.replace('\t', ' ')
                                 .str.replace(r'\s+', ' ', regex=True)
                                 .str.strip())

                # Validation and reporting
                if len(df_clean[col]) > 0:
                    avg_length = df_clean[col].str.len().mean()
                    non_empty = (df_clean[col].str.len() > 0).sum()
                    self.logger.info(f"{col} - Average length: {avg_length:.0f} chars, Non-empty: {non_empty}")

                    # Show sample
                    first_non_empty = df_clean[df_clean[col].str.len() > 0][col].iloc[0] if non_empty > 0 else ""
                    sample_text = first_non_empty[:200] if first_non_empty else "No text available"
                    print(f"Sample {col} text: {sample_text}...")

        self.logger.info(f"Processed {len(df_clean)} text records")
        return df_clean

    def _process_dict_text(self, text_dict: Dict) -> Optional[pd.DataFrame]:
        """
        Converts a dictionary of text data into a DataFrame with better error handling.
        """
        self.logger.info("Processing dictionary text data...")
        try:
            df = pd.DataFrame.from_dict(text_dict, orient='index')
            df.reset_index(inplace=True)
            self.logger.info(f"Converted dict to DataFrame with shape: {df.shape}")
            return self._process_dataframe_text(df)
        except Exception as e:
            self.logger.error(f"Could not convert dict to DataFrame: {e}")
            return None

    def _process_unknown_text(self, text_data) -> any:
        """
        Fallback processor for unrecognized data types with logging.
        """
        self.logger.warning(f"Processing unknown data type: {type(text_data)}")
        return text_data

    @timing_decorator
    def add_text_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FEATURE ENGINEERING: Create numerical features from text with improved risk detection
        """
        if 'mgmt' not in df.columns:
            self.logger.warning("No 'mgmt' column found for text statistics")
            return df

        # FEATURE 1: Length metrics
        df['mda_length'] = df['mgmt'].str.len()
        df['mda_word_count'] = df['mgmt'].str.split().str.len()

        # FEATURE 2: Improved risk word detection
        risk_pattern = '|'.join([f'\\b{word}\\b' for word in Config.RISK_WORDS])
        df['mda_has_risk_words'] = df['mgmt'].str.contains(
            risk_pattern, case=False, na=False, regex=True
        ).astype(int)

        # FEATURE 3: Additional text features
        df['mda_sentence_count'] = df['mgmt'].str.count(r'[.!?]+')
        df['mda_avg_word_length'] = df['mgmt'].str.replace(r'[^\w\s]', '', regex=True).str.split().apply(
            lambda x: np.mean([len(word) for word in x]) if x and len(x) > 0 else 0
        )

        # Risk Factors column if available
        if 'rf' in df.columns:
            df['rf_length'] = df['rf'].str.len()
            df['rf_has_risk_words'] = df['rf'].str.contains(
                risk_pattern, case=False, na=False, regex=True
            ).astype(int)

        # REPORT STATISTICS
        stats = {
            'avg_length': df['mda_length'].mean(),
            'avg_word_count': df['mda_word_count'].mean(),
            'risk_word_pct': df['mda_has_risk_words'].mean() * 100,
            'avg_sentences': df['mda_sentence_count'].mean()
        }

        self.logger.info("MD&A Statistics:")
        for stat_name, value in stats.items():
            self.logger.info(f"  {stat_name}: {value:.1f}")

        return df

    def validate_processed_data(self, df: pd.DataFrame) -> bool:
        """Validate processed data meets expectations"""
        if df is None or df.empty:
            self.logger.error("Data validation failed: No data")
            return False

        checks = {
            'has_data': len(df) > 0,
            'has_cik': 'cik' in df.columns and df['cik'].notna().all(),
            'valid_dates': 'filing_date' in df.columns and df['filing_date'].notna().any(),
            'text_not_empty': 'mgmt' in df.columns and df['mgmt'].str.len().mean() > 50
        }

        failed_checks = [check for check, passed in checks.items() if not passed]
        if failed_checks:
            self.logger.error(f"Data validation failed: {failed_checks}")
            return False

        self.logger.info("Data validation passed")
        return True

    @timing_decorator
    def test_mda_length_strategy(self, processed_data: pd.DataFrame) -> Optional[Dict]:
        """
        REPLICATE PAPER'S FINDING: Test if MD&A length predicts stock performance
        Enhanced with better portfolio construction and validation
        """
        if not self.validate_processed_data(processed_data) or 'mgmt' not in processed_data.columns:
            self.logger.error("Invalid data for MD&A length strategy")
            return None

        if len(processed_data) < Config.MIN_DATA_FOR_PORTFOLIOS:
            self.logger.error(
                f"Insufficient data for portfolio construction. Need at least {Config.MIN_DATA_FOR_PORTFOLIOS} records")
            return None

        # Create working copy
        data = processed_data.copy()
        data['mda_length'] = data['mgmt'].str.len()

        self.logger.info("=== MD&A Length Strategy Analysis ===")
        length_stats = {
            'count': len(data),
            'mean': data['mda_length'].mean(),
            'median': data['mda_length'].median(),
            'min': data['mda_length'].min(),
            'max': data['mda_length'].max(),
            'std': data['mda_length'].std()
        }

        for stat_name, value in length_stats.items():
            self.logger.info(f"{stat_name.capitalize()}: {value:.0f}")

        # PORTFOLIO CONSTRUCTION with time-based ranking
        if 'filing_date' in data.columns:
            # Group by time period (quarter or year) for ranking
            data['year_quarter'] = data['filing_date'].dt.to_period('Q')

            # Rank within each time period
            data['mda_length_rank'] = (
                data.groupby('year_quarter')['mda_length']
                .rank(pct=True, method='min')
            )

            # Create portfolios
            short_threshold = Config.PORTFOLIO_THRESHOLDS['short']
            long_threshold = Config.PORTFOLIO_THRESHOLDS['long']

            short_mda = data[data['mda_length_rank'] <= short_threshold]
            long_mda = data[data['mda_length_rank'] >= long_threshold]

            results = {
                'total_companies': len(data),
                'short_mda_count': len(short_mda),
                'long_mda_count': len(long_mda),
                'avg_short_length': short_mda['mda_length'].mean() if len(short_mda) > 0 else 0,
                'avg_long_length': long_mda['mda_length'].mean() if len(long_mda) > 0 else 0,
                'short_portfolio_pct': len(short_mda) / len(data) * 100,
                'long_portfolio_pct': len(long_mda) / len(data) * 100,
                'time_periods': data['year_quarter'].nunique()
            }

            self.logger.info("Portfolio Results:")
            for key, value in results.items():
                if 'pct' in key or 'avg' in key:
                    self.logger.info(f"  {key}: {value:.2f}")
                else:
                    self.logger.info(f"  {key}: {value}")

            return results
        else:
            self.logger.error("No filing_date column - cannot create time-based portfolios")
            return None

    @timing_decorator
    def create_visualizations(self, processed_data: pd.DataFrame):
        """
        DATA VISUALIZATION: Create comprehensive charts to understand data distribution
        """
        if not self.validate_processed_data(processed_data):
            self.logger.error("Cannot create visualizations: invalid data")
            return

        viz_path = self.base_path / 'visualizations'

        # Set style
        plt.style.use('default')

        # CHART 1: Distribution of MD&A lengths
        if 'mda_length' in processed_data.columns:
            plt.figure(figsize=(12, 6))
            plt.hist(processed_data['mda_length'], bins=50, edgecolor='black', alpha=0.7)
            plt.title('Distribution of MD&A Section Lengths', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Characters')
            plt.ylabel('Number of Companies')
            plt.grid(True, alpha=0.3)

            # Add statistics text
            mean_length = processed_data['mda_length'].mean()
            median_length = processed_data['mda_length'].median()
            plt.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.0f}')
            plt.axvline(median_length, color='blue', linestyle='--', label=f'Median: {median_length:.0f}')
            plt.legend()

            plt.tight_layout()
            plt.savefig(viz_path / 'mda_length_distribution.png', dpi=300)
            plt.show()

        # CHART 2: Time trends
        if 'filing_date' in processed_data.columns and 'mda_length' in processed_data.columns:
            # Create year column
            processed_data['year'] = processed_data['filing_date'].dt.year
            yearly_stats = processed_data.groupby('year').agg({
                'mda_length': ['mean', 'median', 'count']
            }).round(0)

            # Flatten column names
            yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns]

            if len(yearly_stats) > 1:  # Only create if multiple years
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                # Average length over time
                ax1.plot(yearly_stats.index, yearly_stats['mda_length_mean'],
                         marker='o', linewidth=2, label='Mean')
                ax1.plot(yearly_stats.index, yearly_stats['mda_length_median'],
                         marker='s', linewidth=2, label='Median')
                ax1.set_title('Average MD&A Length Over Time', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Year')
                ax1.set_ylabel('Average Characters')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Number of filings over time
                ax2.bar(yearly_stats.index, yearly_stats['mda_length_count'], alpha=0.7)
                ax2.set_title('Number of Filings Over Time', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Year')
                ax2.set_ylabel('Number of Filings')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(viz_path / 'mda_time_trends.png', dpi=300)
                plt.show()

        # CHART 3: Risk word analysis
        if 'mda_has_risk_words' in processed_data.columns:
            risk_summary = processed_data.groupby('mda_has_risk_words').size()

            plt.figure(figsize=(8, 6))
            labels = ['No Risk Words', 'Contains Risk Words']
            colors = ['lightcoral', 'lightblue']
            plt.pie(risk_summary.values, labels=labels, autopct='%1.1f%%',
                    colors=colors, startangle=90)
            plt.title('Distribution of Risk Word Usage in MD&A', fontsize=14, fontweight='bold')
            plt.axis('equal')

            plt.tight_layout()
            plt.savefig(viz_path / 'risk_words_distribution.png', dpi=300)
            plt.show()

        self.logger.info(f"Visualizations saved to: {viz_path}")

    @timing_decorator
    def explore_text_data_structure(self):
        """
        DATA EXPLORATION: Comprehensive analysis of raw data structure
        """
        text_data = self.load_text_data()

        if text_data is not None:
            self.logger.info("\n=== Detailed Text Data Exploration ===")

            if isinstance(text_data, pd.DataFrame):
                # Comprehensive DataFrame analysis
                self.logger.info("DataFrame Info:")
                print("\nDataFrame Info:")
                print(text_data.info())

                print(f"\nShape: {text_data.shape}")
                print(f"Memory usage: {text_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

                # Column analysis
                print("\nColumn Analysis:")
                for col in text_data.columns:
                    if not text_data.empty:
                        col_info = {
                            'dtype': str(text_data[col].dtype),
                            'null_count': text_data[col].isnull().sum(),
                            'null_pct': text_data[col].isnull().mean() * 100,
                        }

                        # Add sample for text columns
                        if col in Config.TEXT_COLUMNS:
                            non_null_series = text_data[col].dropna()
                            if not non_null_series.empty:
                                sample = str(non_null_series.iloc[0])[:100]
                                col_info['sample'] = f"{sample}..."
                                col_info['avg_length'] = non_null_series.astype(str).str.len().mean()
                        else:
                            sample = text_data[col].iloc[0] if not text_data.empty else 'Empty'
                            col_info['sample'] = str(sample)[:100]

                        print(f"  {col}: {col_info}")

        return text_data

    def save_data(self, data, filename: str, description: Optional[str] = None):
        """
        SAVE PROCESSED DATA: Export data with better error handling and metadata
        """
        output_path = self.base_path / 'cleaned_data' / filename

        try:
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path, index=False)
                self.logger.info(f"DataFrame saved to: {output_path}")

                # Save metadata
                metadata = {
                    'shape': data.shape,
                    'columns': data.columns.tolist(),
                    'dtypes': data.dtypes.astype(str).to_dict(),
                    'description': description or 'Processed FIAM data'
                }

                metadata_path = output_path.with_suffix('.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            elif isinstance(data, (dict, list)):
                pkl_path = output_path.with_suffix('.pkl')
                with open(pkl_path, 'wb') as f:
                    pickle.dump(data, f)
                self.logger.info(f"Object saved to: {pkl_path}")

            else:
                txt_path = output_path.with_suffix('.txt')
                with open(txt_path, 'w') as f:
                    f.write(str(data))
                self.logger.info(f"Text saved to: {txt_path}")

        except Exception as e:
            self.logger.error(f"Error saving data: {e}")

    def print_execution_summary(self):
        """Print a summary of execution times for all operations"""
        if not self.execution_times:
            self.logger.info("No execution times recorded")
            return

        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXECUTION TIME SUMMARY")
        self.logger.info("=" * 60)

        total_time = sum(self.execution_times.values())

        # Sort by execution time (longest first)
        sorted_times = sorted(self.execution_times.items(), key=lambda x: x[1], reverse=True)

        for func_name, exec_time in sorted_times:
            percentage = (exec_time / total_time) * 100 if total_time > 0 else 0

            # Format time nicely
            if exec_time < 60:
                time_str = f"{exec_time:.2f}s"
            elif exec_time < 3600:
                time_str = f"{exec_time / 60:.2f}m"
            else:
                time_str = f"{exec_time / 3600:.2f}h"

            self.logger.info(f"{func_name:<25} {time_str:>8} ({percentage:5.1f}%)")

        # Format total time
        if total_time < 60:
            total_str = f"{total_time:.2f} seconds"
        elif total_time < 3600:
            total_str = f"{total_time / 60:.2f} minutes"
        else:
            total_str = f"{total_time / 3600:.2f} hours"

        self.logger.info("-" * 60)
        self.logger.info(f"{'TOTAL TIME':<25} {total_str:>15}")
        self.logger.info("=" * 60)

    def run_full_pipeline(self) -> Optional[pd.DataFrame]:
        """
        Execute the complete data processing pipeline with timing
        """
        pipeline_start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("STARTING FULL FIAM DATA PROCESSING PIPELINE")
        self.logger.info("=" * 60)

        # Step 1: Explore raw data
        self.logger.info("Step 1: Exploring raw data structure...")
        text_data = self.explore_text_data_structure()

        if text_data is None:
            self.logger.error("No data loaded. Pipeline terminated.")
            return None

        # Step 2: Process data
        self.logger.info("Step 2: Processing text data...")
        processed_data = self.process_text_data(text_data)

        if processed_data is None:
            self.logger.error("Data processing failed. Pipeline terminated.")
            return None

        # Step 3: Add features
        self.logger.info("Step 3: Adding text statistics...")
        processed_data = self.add_text_statistics(processed_data)

        # Step 4: Validate
        if not self.validate_processed_data(processed_data):
            self.logger.error("Data validation failed. Pipeline terminated.")
            return None

        # Step 5: Test strategy
        self.logger.info("Step 4: Testing MD&A length strategy...")
        strategy_results = self.test_mda_length_strategy(processed_data)

        # Step 6: Create visualizations
        self.logger.info("Step 5: Creating visualizations...")
        self.create_visualizations(processed_data)

        # Step 7: Save results
        self.logger.info("Step 6: Saving processed data...")
        save_start = time.time()
        self.save_data(
            processed_data,
            'processed_text_data.csv',
            'Cleaned and processed SEC filing text data with features'
        )

        if strategy_results:
            self.save_data(
                strategy_results,
                'strategy_results.json',
                'MD&A length strategy portfolio construction results'
            )

        save_time = time.time() - save_start
        self.execution_times['save_data'] = save_time
        self.logger.info(f"Completed save_data in {save_time:.2f} seconds")

        # Calculate total pipeline time
        total_pipeline_time = time.time() - pipeline_start_time

        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info(f"Total Pipeline Time: {total_pipeline_time / 60:.2f} minutes")
        self.logger.info("=" * 60)

        # Print detailed timing summary
        self.print_execution_summary()

        return processed_data


# ===========================
# MAIN EXECUTION SCRIPT
# ===========================
if __name__ == "__main__":
    """
    MAIN PIPELINE: Execute the complete FIAM data processing workflow
    """
    overall_start_time = time.time()

    try:
        print("🚀 Starting FIAM Data Processing Pipeline...")
        print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)

        # Initialize processor - file path is now correctly set
        processor = FIAMDataProcessor()

        # Run the complete pipeline
        final_data = processor.run_full_pipeline()

        # Calculate total execution time
        total_execution_time = time.time() - overall_start_time

        if final_data is not None:
            print(f"\n{'=' * 60}")
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"📊 Final dataset shape: {final_data.shape}")
            print(f"💾 Data saved to: {processor.base_path / 'cleaned_data'}")
            print(f"📈 Visualizations saved to: {processor.base_path / 'visualizations'}")

            # Format and display total time
            if total_execution_time < 60:
                time_str = f"{total_execution_time:.2f} seconds"
            elif total_execution_time < 3600:
                time_str = f"{total_execution_time / 60:.2f} minutes"
            else:
                time_str = f"{total_execution_time / 3600:.2f} hours"

            print(f"⏱️  Total execution time: {time_str}")
            print(f"🕐 End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 60}")
        else:
            print("❌ Pipeline failed. Check logs for details.")
            print(f"⏱️  Failed after: {total_execution_time:.2f} seconds")

    except Exception as e:
        total_execution_time = time.time() - overall_start_time
        print(f"💥 Critical error in main execution: {e}")
        print(f"⏱️  Failed after: {total_execution_time:.2f} seconds")
        raise
