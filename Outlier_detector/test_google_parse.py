
import pandas as pd
import numpy as np
import re
import sys
import os

# Add path to import outlier_agent_node
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from outlier_agent_node import normalize_numeric, classify_mixed_type_llm, MISSING_TOKENS

# Load Data
csv_path = r"c:\Users\Rishab Nayak\Downloads\EL_sem3\EL_sem3\missing_value_detector\googleplaystore.csv"
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

# Test Columns
cols_to_test = ["Size", "Installs"]

for col in cols_to_test:
    print(f"\nAnalyzing '{col}'...")
    raw = df[col]
    
    # 1. Calc failed_parse_ratio logic manually
    nan_before = raw.isna().sum()
    
    # Parse
    parsed = raw.apply(normalize_numeric)
    nan_after = parsed.isna().sum()
    
    failed_parse_ratio = (nan_after - nan_before) / len(df)
    
    print(f"  Length: {len(df)}")
    print(f"  NaN Before: {nan_before}")
    print(f"  NaN After: {nan_after}")
    print(f"  Failed Parse Ratio: {failed_parse_ratio:.4f}")
    
    # Samples
    samples = raw.dropna().head(10).astype(str).tolist()
    mixed_samples = raw[parsed.isna() & raw.notna()].head(10).astype(str).tolist()
    
    print(f"  Samples: {samples}")
    print(f"  Mixed/Failed Samples: {mixed_samples}")
    
    stats = {
        "failed_parse_ratio": failed_parse_ratio,
        "samples": samples
    }
    
    # 2. Simulate LLM Check if ratio > 0.10
    if failed_parse_ratio > 0.10:
        print("  -> Triggering LLM Check...")
        try:
            decision = classify_mixed_type_llm(col, samples, mixed_samples, stats)
            print(f"  -> LLM Decision: {decision}")
        except Exception as e:
            print(f"  -> LLM Failed: {e}")
    else:
        print("  -> LLM Check NOT triggered.")
