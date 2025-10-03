"""
Main script to run missing data analysis on sample.csv
"""
import pandas as pd
from data_loader_helper import DataLoader, run_complete_workflow_on_your_data

# ============================================================================
# OPTION 1: Quick Start (Automatic)
# ============================================================================

print("Starting missing data analysis...")

# This does everything automatically!
loader, pipeline = run_complete_workflow_on_your_data(
    csv_filepath='sample.parquet',        # ← Your data file
    claude_api_key=None,              # ← No API key yet
    verify_batch_size=50,
    use_haiku=True
)

print("\n✅ Analysis complete!")
print("Check: detected_events_to_verify.csv")


# ============================================================================
# OPTION 2: Step-by-Step (Manual Control)
# ============================================================================

# Uncomment below if you want more control:

"""
# Step 1: Load your data
loader = DataLoader('sample.csv')
raw_data = loader.load_raw_data()

# Step 2: Process it
processed_data = loader.process_for_pipeline()

# Step 3: Check what features you have
feature_check = loader.check_feature_availability()
print(feature_check)

# Step 4: Analyze missing data patterns
patterns = loader.detect_missing_data_patterns()
print(patterns)

# Step 5: Get statistics
stats = loader.get_sample_statistics()
print(stats)

# Step 6: Run detection (when ready)
from complete_pipeline import CompleteMissingDataPipeline

pipeline = CompleteMissingDataPipeline(
    price_data=processed_data,
    claude_api_key=None,  # Add later
    dataset_start_date=processed_data['date'].min()
)

detected = pipeline.step1_detect_all_events()
detected.to_csv('detected_events.csv', index=False)
"""

