
import pandas as pd
import sys
import os

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from outlier_agent_node import outlier_agent_node

# Load data
input_csv = r"c:\Users\Rishab Nayak\Downloads\EL_sem3\EL_sem3\missing_value_detector\car_price.csv"
print(f"Loading {input_csv}...")
df = pd.read_csv(input_csv)

print(f"Original 'Model' sample:\n{df['Model'].head(10)}")
print(f"Original Shape: {df.shape}")

# Run Agent
state = {
    "data": df,
    "config": {
        "use_llm": True,
        "aggressive_mode": True
    }
}

print("\n--- Running Outlier Agent ---")
result = outlier_agent_node(state)

print("\n--- Result Analysis ---")
final_df = result["data"]
print(f"Final Shape: {final_df.shape}")

# Check Model column
print(f"Final 'Model' sample:\n{final_df['Model'].head(10)}")

# Check logs for Model
logs = result.get("log", [])
model_log = [l for l in logs if l.get("column") == "Model"]
print("\nLog for 'Model':")
for l in model_log:
    print(l)

# Check errors
if result.get("errors"):
    print("\nErrors encountered:")
    for e in result["errors"]:
        print(e)
