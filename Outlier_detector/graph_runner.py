# =====================================================
# Graph Runner for Outlier Treatment Pipeline
# =====================================================

import pandas as pd
import json
import os
import sys
from datetime import datetime

# Import the outlier agent
from outlier_agent_node import outlier_agent_node, build_outlier_graph, LANGGRAPH_AVAILABLE


def run_pipeline_simple(input_csv_path: str, output_dir: str = ".") -> dict:
    """
    Run the outlier pipeline using the simple function interface.
    Works without LangGraph dependency.
    """
    print(f"[INFO] Loading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"[INFO] Original shape: {df.shape}")
    
    # Initial state
    state = {"data": df}
    
    # Run outlier detection agent
    print("[INFO] Running outlier treatment...")
    state = outlier_agent_node(state)
    
    # Generate output paths
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_csv = os.path.join(output_dir, f"{base_name}_cleaned_{timestamp}.csv")
    output_log = os.path.join(output_dir, f"{base_name}_cleaning_log_{timestamp}.json")
    
    # Save cleaned data
    state["data"].to_csv(output_csv, index=False)
    print(f"[INFO] Cleaned data saved to: {output_csv}")
    
    # Prepare log for JSON serialization (remove non-serializable items)
    serializable_log = []
    for entry in state.get("log", []):
        clean_entry = {}
        for k, v in entry.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                clean_entry[k] = v
        serializable_log.append(clean_entry)
    
    # Save log
    log_data = {
        "input_file": input_csv_path,
        "timestamp": timestamp,
        "original_shape": list(df.shape),
        "final_shape": list(state["data"].shape),
        "rows_removed": df.shape[0] - state["data"].shape[0],
        "treatment_log": serializable_log,
        "errors": state.get("errors", [])
    }
    
    with open(output_log, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print(f"[INFO] Cleaning log saved to: {output_log}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("OUTLIER TREATMENT SUMMARY")
    print("=" * 60)
    print(f"Rows before: {df.shape[0]}")
    print(f"Rows after:  {state['data'].shape[0]}")
    print(f"Rows removed: {df.shape[0] - state['data'].shape[0]}")
    
    if state.get("errors"):
        print(f"\n[WARN] {len(state['errors'])} errors encountered:")
        for err in state["errors"][:5]:
            print(f"  - {err.get('column', 'N/A')}: {err.get('error', 'Unknown')}")
    
    print("\nPipeline completed successfully.")
    return state


def run_pipeline_langgraph(input_csv_path: str, output_dir: str = ".") -> dict:
    """
    Run the outlier pipeline using LangGraph (if available).
    """
    if not LANGGRAPH_AVAILABLE:
        print("[WARN] LangGraph not available, falling back to simple pipeline")
        return run_pipeline_simple(input_csv_path, output_dir)
    
    print(f"[INFO] Loading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"[INFO] Original shape: {df.shape}")
    
    # Build and run graph
    graph = build_outlier_graph()
    
    initial_state = {
        "df": df,
        "original_shape": df.shape,
        "numeric_analysis": {},
        "column_intents": {},
        "outlier_report": {},
        "treatment_log": [],
        "errors": [],
        "config": {}
    }
    
    print("[INFO] Running outlier treatment via LangGraph...")
    final_state = graph.invoke(initial_state)
    
    # Generate output paths
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_csv = os.path.join(output_dir, f"{base_name}_cleaned_{timestamp}.csv")
    output_log = os.path.join(output_dir, f"{base_name}_cleaning_log_{timestamp}.json")
    
    # Save outputs
    final_state["df"].to_csv(output_csv, index=False)
    
    log_data = {
        "input_file": input_csv_path,
        "timestamp": timestamp,
        "original_shape": list(final_state["original_shape"]),
        "final_shape": list(final_state["df"].shape),
        "treatment_log": final_state["treatment_log"],
        "errors": final_state["errors"]
    }
    
    with open(output_log, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)
    
    print(f"[INFO] Cleaned data saved to: {output_csv}")
    print(f"[INFO] Cleaning log saved to: {output_log}")
    print(f"\nPipeline completed. Rows: {final_state['original_shape'][0]} -> {final_state['df'].shape[0]}")
    
    return final_state


def run_pipeline(input_csv_path: str, output_dir: str = ".", use_langgraph: bool = False) -> dict:
    """
    Main entry point. Choose between simple or LangGraph pipeline.
    """
    if use_langgraph and LANGGRAPH_AVAILABLE:
        return run_pipeline_langgraph(input_csv_path, output_dir)
    else:
        return run_pipeline_simple(input_csv_path, output_dir)


# =====================================================
# CLI ENTRY POINT
# =====================================================

if __name__ == "__main__":
    # Default test file
    default_file = r"C:\Users\rohan\Antigravity\EL_sem3\Outlier_detector\AmesHousing_cleaned.csv"
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_file
    output_directory = sys.argv[2] if len(sys.argv) > 2 else "."
    
    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        sys.exit(1)
    
    run_pipeline(input_file, output_directory)
