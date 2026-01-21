# =====================================================
# Graph Runner for Intelligent Visualization Pipeline
# =====================================================
# Main entry point for running the visualization agent
# Can be used standalone or integrated with other EDA agents
# =====================================================

import pandas as pd
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Import the visualization agent
from viz_agent_node import visualization_agent_node, build_visualization_graph, LANGGRAPH_AVAILABLE

# Import plot generator
from plot_generator import generate_all_selected_plots


# =====================================================
# 1. SIMPLE PIPELINE (No LangGraph Required)
# =====================================================

def run_pipeline_simple(input_csv_path: str, output_dir: str = ".") -> dict:
    """
    Run the visualization pipeline using the simple function interface.
    Works without LangGraph dependency.
    
    Args:
        input_csv_path: Path to input CSV file
        output_dir: Directory to save outputs (plots and reports)
    
    Returns:
        Final state with selected plots and decision trace
    """
    print(f"[INFO] Loading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Total columns: {len(df.columns)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initial state
    state = {"data": df}
    
    # Run visualization selection agent
    print("\n[INFO] Running intelligent visualization selection...")
    state = visualization_agent_node(state)
    
    # Generate plots if any were selected
    selected_plots = state.get("selected_plots", [])
    if selected_plots:
        plots_dir = os.path.join(output_dir, "plots")
        generation_results = generate_all_selected_plots(df, selected_plots, plots_dir)
        state["generation_results"] = generation_results
    else:
        print("[WARN] No plots were selected for generation.")
    
    # Generate output paths
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save decision report
    report_path = os.path.join(output_dir, f"{base_name}_viz_report_{timestamp}.json")
    
    # Prepare serializable report
    serializable_report = {
        "input_file": input_csv_path,
        "timestamp": timestamp,
        "dataset_shape": list(df.shape),
        "total_columns": len(df.columns),
        "selected_plots_count": len(selected_plots),
        "selected_plots": selected_plots,
        "decision_trace": state.get("decision_trace", []),
        "priority_scores": {k: v for k, v in state.get("priority_scores", {}).items()},
        "top_correlations": state.get("top_correlations", []),
        "errors": state.get("errors", [])
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable_report, f, indent=2, default=str)
    print(f"\n[INFO] Visualization report saved to: {report_path}")
    
    # Print summary
    print_summary(state)
    
    return state


# =====================================================
# 2. LANGGRAPH PIPELINE
# =====================================================

def run_pipeline_langgraph(input_csv_path: str, output_dir: str = ".") -> dict:
    """
    Run the visualization pipeline using LangGraph (if available).
    
    Args:
        input_csv_path: Path to input CSV file
        output_dir: Directory to save outputs
    
    Returns:
        Final state with selected plots and decision trace
    """
    if not LANGGRAPH_AVAILABLE:
        print("[WARN] LangGraph not available, falling back to simple pipeline")
        return run_pipeline_simple(input_csv_path, output_dir)
    
    print(f"[INFO] Loading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"[INFO] Dataset shape: {df.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build and run graph
    graph = build_visualization_graph()
    
    initial_state = {
        "df": df,
        "column_metadata": {},
        "priority_scores": {},
        "eligible_plots": {},
        "selected_plots": [],
        "decision_trace": [],
        "correlation_matrix": None,
        "top_correlations": [],
        "config": {},
        "errors": []
    }
    
    print("[INFO] Running visualization selection via LangGraph...")
    final_state = graph.invoke(initial_state)
    
    # Generate plots
    selected_plots = final_state.get("selected_plots", [])
    if selected_plots:
        plots_dir = os.path.join(output_dir, "plots")
        generation_results = generate_all_selected_plots(df, selected_plots, plots_dir)
        final_state["generation_results"] = generation_results
    
    # Save report
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"{base_name}_viz_report_{timestamp}.json")
    
    serializable_report = {
        "input_file": input_csv_path,
        "timestamp": timestamp,
        "dataset_shape": list(df.shape),
        "selected_plots_count": len(selected_plots),
        "selected_plots": selected_plots,
        "decision_trace": final_state.get("decision_trace", []),
        "errors": final_state.get("errors", [])
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable_report, f, indent=2, default=str)
    
    print(f"\n[INFO] Report saved to: {report_path}")
    print_summary(final_state)
    
    return final_state


# =====================================================
# 3. MAIN ENTRY POINT
# =====================================================

def run_pipeline(
    input_csv_path: str, 
    output_dir: str = ".", 
    use_langgraph: bool = False
) -> dict:
    """
    Main entry point. Choose between simple or LangGraph pipeline.
    
    Args:
        input_csv_path: Path to input CSV file
        output_dir: Directory to save outputs
        use_langgraph: Whether to use LangGraph (if available)
    
    Returns:
        Final state with results
    """
    if use_langgraph and LANGGRAPH_AVAILABLE:
        return run_pipeline_langgraph(input_csv_path, output_dir)
    else:
        return run_pipeline_simple(input_csv_path, output_dir)


# =====================================================
# 4. SUMMARY PRINTER
# =====================================================

def print_summary(state: dict):
    """Print a formatted summary of the visualization selection."""
    
    print("\n" + "=" * 70)
    print("INTELLIGENT VISUALIZATION SELECTION SUMMARY")
    print("=" * 70)
    
    # Get summary from decision trace
    decision_trace = state.get("decision_trace", [])
    summary = {}
    if decision_trace and isinstance(decision_trace[0], dict):
        summary = decision_trace[0].get("report_summary", {})
    
    # Dataset overview
    print(f"\n[DATA] Dataset Overview:")
    print(f"   Total columns analyzed: {summary.get('total_columns_analyzed', 'N/A')}")
    print(f"   Numeric columns: {summary.get('numeric_columns', 'N/A')}")
    print(f"   Categorical columns: {summary.get('categorical_columns', 'N/A')}")
    
    # Selection results
    print(f"\n[RESULTS] Selection Results:")
    print(f"   Plots selected: {summary.get('plots_selected', len(state.get('selected_plots', [])))}")
    print(f"   Columns skipped: {summary.get('columns_skipped', 'N/A')}")
    
    # Top features
    top_features = summary.get("top_features", [])
    if top_features:
        print(f"\n[TOP] Top Scoring Features:")
        for i, feat in enumerate(top_features[:5], 1):
            print(f"   {i}. {feat['column']} (score: {feat['score']})")
    
    # Selected plots breakdown
    selected_plots = state.get("selected_plots", [])
    if selected_plots:
        print(f"\n[PLOTS] Selected Plots ({len(selected_plots)} total):")
        
        # Group by category
        univariate_numeric = [p for p in selected_plots if p.get("category") == "univariate_numeric"]
        univariate_categorical = [p for p in selected_plots if p.get("category") == "univariate_categorical"]
        bivariate = [p for p in selected_plots if p.get("category") == "bivariate"]
        other = [p for p in selected_plots if p.get("category") not in ["univariate_numeric", "univariate_categorical", "bivariate"]]
        
        if univariate_numeric:
            print(f"\n   [NUMERIC] Univariate Numeric ({len(univariate_numeric)}):")
            for p in univariate_numeric[:5]:
                print(f"      - {p['plot_type'].upper()}: {p['column']}")
            if len(univariate_numeric) > 5:
                print(f"      ... and {len(univariate_numeric) - 5} more")
        
        if univariate_categorical:
            print(f"\n   [CATEGORICAL] Univariate Categorical ({len(univariate_categorical)}):")
            for p in univariate_categorical[:5]:
                print(f"      - {p['plot_type'].upper()}: {p['column']}")
        
        if bivariate:
            print(f"\n   [BIVARIATE] Bivariate ({len(bivariate)}):")
            for p in bivariate[:5]:
                print(f"      - SCATTER: {p['column1']} vs {p['column2']} (r={p['correlation']:.2f})")
        
        if other:
            print(f"\n   [OTHER] Other ({len(other)}):")
            for p in other:
                print(f"      - {p['plot_type'].upper()}: {p.get('reason', 'N/A')}")
    
    # Skip decisions (sample)
    skip_count = 0
    skip_samples = []
    for trace in decision_trace:
        if isinstance(trace, dict) and trace.get("action") == "skipped":
            skip_samples.append(trace)
            skip_count += 1
    
    if skip_samples:
        print(f"\n[SKIPPED] Sample Skip Decisions ({skip_count} total):")
        for trace in skip_samples[:5]:
            reasons = trace.get("reasons", [])
            print(f"   - {trace['column']}: {', '.join(reasons)}")
        if skip_count > 5:
            print(f"   ... and {skip_count - 5} more")
    
    # Errors
    errors = state.get("errors", [])
    if errors:
        print(f"\n[ERRORS] Errors ({len(errors)}):")
        for err in errors[:3]:
            print(f"   - {err.get('column', 'N/A')}: {err.get('error', 'Unknown')}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Pipeline completed successfully!")
    print("=" * 70)


# =====================================================
# 5. CLI ENTRY POINT
# =====================================================

def get_dataset_path() -> str:
    """Interactively prompts the user for a dataset path."""
    while True:
        print("\n" + "=" * 60)
        print("INTELLIGENT VISUALIZATION AGENT")
        print("=" * 60)
        
        dataset_path = input("\nEnter the path to your CSV dataset: ").strip()
        
        # Remove quotes if user copied path with quotes
        dataset_path = dataset_path.strip('"').strip("'")
        
        if not dataset_path:
            print("[ERROR] Please provide a valid file path.")
            continue
        
        if not os.path.exists(dataset_path):
            print(f"[ERROR] File not found at '{dataset_path}'")
            print("   Please check the path and try again.")
            continue
        
        if not dataset_path.lower().endswith('.csv'):
            print("[WARN] The file does not have a .csv extension.")
            confirm = input("   Do you want to continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        return dataset_path


if __name__ == "__main__":
    # Default test file
    default_file = r"C:\Users\rohan\Antigravity\EL_sem3\visualization_agent\Titanic-Dataset_cleaned_cleaned_20260118_164603.csv"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"[ERROR] File not found: {input_file}")
            sys.exit(1)
    else:
        # Interactive mode or use default
        if os.path.exists(default_file):
            print(f"[INFO] Using default test file: {default_file}")
            input_file = default_file
        else:
            input_file = get_dataset_path()
    
    # Output directory
    output_directory = sys.argv[2] if len(sys.argv) > 2 else "."
    
    # Run pipeline
    result = run_pipeline(input_file, output_directory)
    
    # Print final message
    plots_dir = os.path.join(output_directory, "plots")
    if os.path.exists(plots_dir):
        plot_count = len([f for f in os.listdir(plots_dir) if f.endswith('.png')])
        print(f"\n[OUTPUT] {plot_count} plots saved to: {plots_dir}")
