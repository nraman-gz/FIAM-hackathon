# FIAM Data Processor - SEC Filings Analysis Pipeline

## Overview

This is a comprehensive data processing pipeline for analyzing SEC filing text data (specifically MD&A and Risk Factors sections) to predict stock returns. The code replicates academic research methodologies with production-grade error handling and validation.

## Purpose

Process SEC filings to extract textual features that can predict future stock performance, specifically testing the hypothesis that MD&A section length correlates with stock returns.

## Project Structure

```
FIAM/
├── raw_data/           # Raw data files
├── cleaned_data/       # Processed data output
├── text_embeddings/    # Text vector representations
├── models/            # Trained ML models
├── visualizations/    # Analysis charts and graphs
├── fiam_processor.log # Processing logs
└── text_us_2005.pkl   # YOUR DATA FILE GOES HERE
```

## Setup Requirements

### Required Python Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn pathlib typing logging pickle
```

### File Replacement Needed

**CRITICAL: You MUST replace the default data file path**

**File to replace:** `text_us_2005.pkl`

**Current hardcoded path:**
```python
self.text_file_path = Path("/Users/krishna_dewan/Desktop/FIAM/text_us_2005.pkl")
```

**Replace with your actual data file:**
```python
# Option 1: Update the path in the __init__ method
self.text_file_path = Path("/path/to/your/actual/data_file.pkl")

# Option 2: Pass custom path when initializing
processor = FIAMDataProcessor(base_path="/your/custom/path/", text_file_name="your_file.pkl")
```

## Expected Data Format

Your data file should be a pickle file containing one of these formats:

### Preferred Format (DataFrame):
```python
# Columns expected:
cik        date        mgmt        rf
12345      20201231    "MD&A text here..."  "Risk factors text..."
67890      20210331    "Management discussion..." "Risk disclosures..."
```

### Alternative Formats:
- **Dictionary**: `{cik: {date: "", mgmt: "", rf: ""}}`
- **List of dictionaries**: `[{"cik": 12345, "date": "20201231", "mgmt": "...", "rf": "..."}]`

## Quick Start

### Basic Usage
```python
from fiam_processor import FIAMDataProcessor

# Initialize with your data path
processor = FIAMDataProcessor(base_path="/your/project/path/")

# Run complete pipeline
results = processor.run_full_pipeline()
```

### Advanced Usage
```python
# Custom initialization
processor = FIAMDataProcessor(
    base_path="/your/custom/path/",
    text_file_name="your_sec_data.pkl"
)

# Step-by-step execution
raw_data = processor.load_text_data()
processed_data = processor.process_text_data(raw_data)
enhanced_data = processor.add_text_statistics(processed_data)
strategy_results = processor.test_mda_length_strategy(enhanced_data)
processor.create_visualizations(enhanced_data)
```

## Configuration

The `Config` class contains key parameters:

```python
class Config:
    EXPECTED_COLUMNS = ['cik', 'date', 'mgmt', 'rf']  # Your data must have these
    RISK_WORDS = ['risk', 'uncertainty', 'challenge', ...]  # Customizable
    PORTFOLIO_THRESHOLDS = {'short': 0.2, 'long': 0.8}  # Strategy parameters
```

## Features Extracted

### Text Statistics
- **MD&A Length**: Character count of Management Discussion
- **Word Count**: Number of words in MD&A
- **Risk Word Detection**: Presence of risk-related vocabulary
- **Sentence Count**: Number of sentences
- **Average Word Length**: Linguistic complexity measure

### Portfolio Strategy
- **Time-based ranking** of MD&A length within quarters
- **Portfolio construction** (short vs long MD&A)
- **Performance validation** with academic methodology

## Outputs Generated

### Data Files
- `processed_text_data.csv` - Cleaned data with features
- `strategy_results.json` - Portfolio construction results

### Visualizations
- `mda_length_distribution.png` - Histogram of MD&A lengths
- `mda_time_trends.png` - Length trends over time
- `risk_words_distribution.png` - Risk word usage analysis

### Logs
- `fiam_processor.log` - Detailed processing timeline and errors

## Common Issues & Solutions

### File Not Found Error
```python
# Wrong: File doesn't exist
processor = FIAMDataProcessor()  # Uses default path

# Correct: Specify your actual file path
processor = FIAMDataProcessor(base_path="/your/actual/data/directory/")
```

### Data Format Issues
```python
# If your columns have different names, update Config:
Config.EXPECTED_COLUMNS = ['company_id', 'filing_date', 'management_text', 'risk_text']
```

### Memory Errors
- The processor handles large files with chunking
- Check available RAM for very large datasets (>1GB)

## Validation Checks

The pipeline includes comprehensive validation:
- Required column presence
- Data type consistency
- Date format conversion
- Text content quality
- Minimum data thresholds

## Performance Metrics

The processor provides detailed timing:
- Function-level execution times
- Memory usage optimization
- Progress tracking with logging



