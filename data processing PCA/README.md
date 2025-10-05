# Missing Data Anomalies Detection System

A comprehensive pipeline for detecting, verifying, and labeling missing data anomalies in financial time series data using advanced algorithms and LLM verification.

## Overview

This system provides an end-to-end solution for identifying corporate events and anomalies that cause missing data in financial datasets. It combines aggressive detection rules with Claude LLM verification to create a labeled dataset explaining ALL missing data patterns.

**Cost-effective**: ~$5 to explain all missing data in 1,000 stocks over 20 years.

## Core Components

### 1. `data_loader_helper.py`
- **DataLoader class**: Handles loading and preprocessing of financial data
- **Supported formats**: CSV and Parquet files
- **Features**:
  - Automatic file format detection
  - Memory-efficient loading with timeout protection
  - Data validation and preprocessing
  - Feature availability checking
  - Missing data pattern analysis

### 2. `liquidity_enhanced_rules.py`
- **LiquidityEnhancedDetector class**: Advanced corporate event detection
- **Detection methods**:
  - Bankruptcy/delisting detection
  - Merger & acquisition identification
  - Trading halt detection
  - IPO/listing date analysis
  - Volume and liquidity anomalies
- **StockStatus enum**: Categorizes different stock lifecycle states

### 3. `CompleteMissingDataPipeline.py`
- **Main orchestration class**: Coordinates the entire detection and verification workflow
- **Pipeline stages**:
  1. Aggressive detection (casts wide net)
  2. LLM verification with web search
  3. Labeled dataset creation
  4. Export functionality
- **Output**: ~1,500-2,000 verified flagged events

### 4. `run_analysis.py`
- **Entry point**: Main script to execute the analysis
- **Two modes**:
  - **Quick Start**: Automatic workflow execution
  - **Step-by-Step**: Manual control for advanced users

## Installation

### Prerequisites
```bash
pip install pandas numpy anthropic pyarrow
```

### Required Files

#### Core Python Files
Ensure you have these 4 core files in your project directory:
- `data_loader_helper.py`
- `liquidity_enhanced_rules.py`
- `CompleteMissingDataPipeline.py`
- `run_analysis.py`

#### GVKEY-to-Company Name Mapping Files (Optional but Recommended)
For complete company name resolution, include these files in your project directory:
- `cik_gvkey_linktable_USA_only.csv` - USA companies mapping (37,744+ companies)
- `GlobalGVKEY.csv` - International companies mapping (1,922+ additional companies)

**Total Coverage**: 39,666+ company names when both files are present.

**Without these files**: The system will still work but will show "Unknown" for company names in detection results.

## Quick Start

### Option 1: Automatic Workflow
```python
python run_analysis.py
```

This runs the complete pipeline automatically using your data file (`sample.parquet`).

### Option 2: Programmatic Usage
```python
from data_loader_helper import run_complete_workflow_on_your_data

# Run complete analysis
loader, pipeline = run_complete_workflow_on_your_data(
    csv_filepath='your_data.parquet',
    claude_api_key='sk-ant-...',  # Optional for initial detection
    verify_batch_size=50,
    use_haiku=True
)
```

## Step-by-Step Usage

### 1. Load and Process Data
```python
from data_loader_helper import DataLoader

# Initialize loader
loader = DataLoader('your_data.parquet')

# Load raw data
raw_data = loader.load_raw_data()

# Process for pipeline
processed_data = loader.process_for_pipeline()

# Check available features
feature_check = loader.check_feature_availability()
print(feature_check)
```

### 2. Analyze Missing Data Patterns
```python
# Detect patterns
patterns = loader.detect_missing_data_patterns()
print(patterns)

# Get statistics
stats = loader.get_sample_statistics()
print(stats)
```

### 3. Run Detection Pipeline
```python
from CompleteMissingDataPipeline import CompleteMissingDataPipeline

# Initialize pipeline
pipeline = CompleteMissingDataPipeline(
    price_data=processed_data,
    claude_api_key='sk-ant-...',  # Add your Claude API key
    dataset_start_date=processed_data['date'].min()
)

# Run complete pipeline
results = pipeline.run_complete_pipeline(verify_batch_size=100)
```

### 4. Step-by-Step Detection (Advanced)
```python
# Step 1: Detect all events
detected = pipeline.step1_detect_all_events()
detected.to_csv('detected_events.csv', index=False)

# Step 2: Verify with LLM (requires API key)
verified = pipeline.step2_verify_with_llm(detected, batch_size=50)
verified.to_csv('verified_events.csv', index=False)

# Step 3: Create labeled dataset
labeled = pipeline.step3_create_labeled_dataset(verified)
labeled.to_csv('labeled_dataset.csv', index=False)
```

## Data Format Requirements

### Input Data Structure
Your data should have the following structure:
- **Column A**: Index (no header)
- **Columns B-L**: Identifiers and dates
- **Columns M+**: Financial features (price, volume, etc.)

### Required Columns
- `date`: Trading date
- `ticker`: Stock symbol
- `price`: Stock price
- `volume`: Trading volume (optional but recommended)

### Supported File Formats
- **CSV**: Standard comma-separated values
- **Parquet**: Compressed columnar format (recommended for large datasets)

## Output Files

### Detection Results
- `detected_events.csv`: Raw detection results
- `detected_events_to_verify.csv`: Events ready for LLM verification

### Verified Results
- `verified_events.csv`: LLM-verified events with explanations
- `labeled_dataset.csv`: Final labeled dataset for ML training

### Export Formats
- CSV for data analysis
- JSON for API integration
- Parquet for efficient storage

## Configuration

### Claude API Setup
```python
# Set your Claude API key
claude_api_key = 'sk-ant-...'

# Choose model (recommended: Haiku for cost efficiency)
use_haiku = True
```

### Detection Parameters
```python
# Batch size for LLM verification
verify_batch_size = 50  # Adjust based on API limits

# Detection sensitivity
aggressive_detection = True  # Cast wider net for initial detection
```

## Performance & Costs

### Processing Capacity
- **Small datasets** (<1M rows): Near real-time processing
- **Large datasets** (>10M rows): Optimized chunked processing
- **Memory usage**: Efficient with large parquet files

### API Costs (Claude)
- **Haiku model**: ~$0.005 per event verification
- **Complete analysis**: ~$5 for 1,000 stocks over 20 years
- **Batch processing**: Optimized for cost efficiency

## Troubleshooting

### Common Issues

#### File Loading Problems
```
FileNotFoundError: File not found
```
**Solution**: Verify file path and format. Use absolute paths when possible.

#### Memory Issues
```
MemoryError: Unable to load large parquet file
```
**Solutions**:
1. Copy file to local SSD storage
2. Use a smaller sample of the data
3. Convert to CSV format first

#### API Rate Limits
```
Rate limit exceeded
```
**Solution**: Reduce `verify_batch_size` parameter or add delays between requests.

### Data Quality Issues
- **Missing required columns**: Check column names match expected format
- **Date format problems**: Ensure dates are in consistent format
- **Ticker symbol inconsistencies**: Standardize ticker symbols

## Advanced Features

### Company Name Resolution
The system automatically resolves GVKEY identifiers to company names using both files:

```python
# Test company name coverage
from data_loader_helper import create_combined_company_mapping

company_map = create_combined_company_mapping()
print(f"Total companies mapped: {len(company_map)}")

# Check specific GVKEY
gvkey = "001004"
company_name = company_map.get(gvkey, "Unknown")
print(f"GVKEY {gvkey}: {company_name}")
```

**Coverage Statistics**:
- USA companies: 37,744+ names
- International companies: 1,922+ additional names
- Total coverage: 39,666+ company names

**Enhanced Output**: Detection results will include human-readable company names instead of just "Unknown" for GVKEYs.

### Custom Detection Rules
Extend the `LiquidityEnhancedDetector` class:
```python
class CustomDetector(LiquidityEnhancedDetector):
    def detect_custom_event(self, data):
        # Your custom detection logic
        return detected_events
```

### Custom Data Processing
Extend the `DataLoader` class:
```python
class CustomDataLoader(DataLoader):
    def custom_preprocessing(self, data):
        # Your custom preprocessing
        return processed_data
```

## Contributing

When modifying the codebase:
1. Follow existing code patterns and conventions
2. Test with small datasets first
3. Update documentation for new features
4. Ensure backward compatibility

## License

[Add your license information here]

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the example usage patterns
3. Test with the provided sample data
4. Check file paths and permissions

## Example Workflow

```python
# Complete example workflow
from data_loader_helper import run_complete_workflow_on_your_data

print("Starting missing data analysis...")

# Run everything automatically
loader, pipeline = run_complete_workflow_on_your_data(
    csv_filepath='sample.parquet',
    claude_api_key=None,  # Add API key for verification
    verify_batch_size=50,
    use_haiku=True
)

print("✅ Analysis complete!")
print("Check: detected_events_to_verify.csv")
```

This will detect all anomalies and prepare them for LLM verification. Add your Claude API key to enable automatic verification and labeling.