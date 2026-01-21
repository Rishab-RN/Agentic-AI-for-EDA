"""
Debug script to trace exactly what happens to Size column
"""
import pandas as pd
import sys
import os

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from outlier_agent_node import (
    normalize_numeric, 
    analyze_numeric_columns,
    classify_column_intents,
    detect_outliers,
    apply_treatment,
    generate_report,
    OutlierState
)

# Load data
csv_path = r"c:\Users\Rishab Nayak\Downloads\EL_sem3\EL_sem3\missing_value_detector\googleplaystore.csv"
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

print(f"\n=== ORIGINAL DATA ===")
print(f"Size column sample: {df['Size'].head(10).tolist()}")
print(f"Installs column sample: {df['Installs'].head(10).tolist()}")

# Test normalize_numeric directly
print(f"\n=== TESTING normalize_numeric() ===")
test_values = ["19M", "5.6M", "8.7M", "Varies with device", "100,000+", "10M+"]
for val in test_values:
    result = normalize_numeric(val)
    print(f"  '{val}' -> {result}")

# Run pipeline step by step
print(f"\n=== RUNNING PIPELINE ===")

state: OutlierState = {
    "df": df,
    "original_shape": df.shape,
    "numeric_analysis": {},
    "column_intents": {},
    "outlier_report": {},
    "treatment_log": [],
    "errors": [],
    "config": {"use_llm": True, "aggressive_mode": True}
}

# Step 1: Analyze
print("\n--- Step 1: analyze_numeric_columns ---")
state = analyze_numeric_columns(state)
print(f"Columns in numeric_analysis: {list(state['numeric_analysis'].keys())}")

if 'Size' in state['numeric_analysis']:
    analysis = state['numeric_analysis']['Size']
    print(f"Size analysis:")
    print(f"  parsed_count: {analysis['parsed_count']}")
    print(f"  nan_count: {analysis['nan_count']}")
    print(f"  failed_parse_ratio: {analysis['failed_parse_ratio']:.2%}")
    print(f"  samples: {analysis['samples'][:5]}")
    print(f"  mixed_samples: {analysis['mixed_samples'][:5]}")
    
    # Check parsed_series
    parsed = analysis['parsed_series']
    print(f"  parsed_series sample: {parsed.head(10).tolist()}")
else:
    print("  WARNING: Size NOT in numeric_analysis!")

# Step 2: Classify
print("\n--- Step 2: classify_column_intents ---")
state = classify_column_intents(state)
print(f"Column intents: {state['column_intents']}")

if 'Size' in state['column_intents']:
    print(f"Size intent: {state['column_intents']['Size']}")
else:
    print("  WARNING: Size NOT in column_intents!")

# Step 3 & 4: Detect and Apply
print("\n--- Step 3: detect_outliers ---")
state = detect_outliers(state)

print("\n--- Step 4: apply_treatment ---")
state = apply_treatment(state)

# Check final result
print(f"\n=== FINAL RESULT ===")
final_df = state['df']
print(f"Size column sample: {final_df['Size'].head(10).tolist()}")
print(f"Installs column sample: {final_df['Installs'].head(10).tolist()}")

# Save to test file
output_path = r"c:\Users\Rishab Nayak\Downloads\EL_sem3\EL_sem3\Outlier_detector\test_output.csv"
final_df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
