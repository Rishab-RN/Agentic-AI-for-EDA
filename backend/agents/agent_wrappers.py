# =====================================================
# Agent Wrappers - Bridge between Flask and existing agents
# =====================================================

import pandas as pd
import sys
import os

# Load environment variables (for GROQ_API_KEY etc.)
from dotenv import load_dotenv

# Try multiple locations for .env file
env_paths = [
    os.path.join(os.path.dirname(__file__), '..', '.env'),  # backend/.env
    os.path.join(os.path.dirname(__file__), '..', '..', '.env'),  # EL_sem3/.env
    os.path.join(os.path.dirname(__file__), '..', '..', 'missing_value_detector', '.env'),  # missing_value_detector/.env
]
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break
else:
    load_dotenv()  # Try default locations

# Add agent directories to path
AGENTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(AGENTS_DIR, "missing_value_detector"))
sys.path.insert(0, os.path.join(AGENTS_DIR, "Outlier_detector"))
sys.path.insert(0, os.path.join(AGENTS_DIR, "visualization_agent"))
sys.path.insert(0, os.path.join(AGENTS_DIR, "correlation_agent"))


def run_missing_value_agent(input_csv: str, output_csv: str) -> dict:
    """
    Run the missing value agent on a CSV file.
    
    Args:
        input_csv: Path to input CSV
        output_csv: Path where cleaned CSV should be saved
    
    Returns:
        Report dictionary with cleaning results
    """
    try:
        from missingvalue_update import run_agent
        
        # Run the agent
        cleaned_df, report = run_agent(input_csv)
        
        # Save cleaned data
        cleaned_df.to_csv(output_csv, index=False)
        
        return {
            "status": "success",
            "input_file": input_csv,
            "output_file": output_csv,
            "original_shape": report.get("original_shape"),
            "final_shape": report.get("final_shape"),
            "rows_dropped": report.get("rows_dropped", 0),
            "columns_dropped": report.get("dropped_columns", []),
            "imputed_values": report.get("imputed_values", {}),
            "column_actions": report.get("column_actions", {}),
            "global_row_loss_pct": report.get("global_row_loss_pct", 0),
            "llm_guided_columns": report.get("llm_guided_columns", []),
            "llm_used": report.get("llm_was_called", False)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "input_file": input_csv
        }


def run_outlier_agent(input_csv: str, output_csv: str) -> dict:
    """
    Run the outlier agent on a CSV file.
    
    Args:
        input_csv: Path to input CSV
        output_csv: Path where cleaned CSV should be saved
    
    Returns:
        Report dictionary with outlier treatment results
    """
    try:
        from outlier_agent_node import outlier_agent_node
        
        # Load data
        df = pd.read_csv(input_csv)
        original_shape = df.shape
        
        # Run agent
        state = {"data": df}
        result = outlier_agent_node(state)
        
        # Save cleaned data
        result["data"].to_csv(output_csv, index=False)
        
        # Extract summary from log
        log = result.get("log", [])
        summary = next((entry for entry in log if "report_summary" in entry), {})
        
        return {
            "status": "success",
            "input_file": input_csv,
            "output_file": output_csv,
            "original_shape": list(original_shape),
            "final_shape": list(result["data"].shape),
            "rows_removed": original_shape[0] - result["data"].shape[0],
            "treatment_log": log,
            "errors": result.get("errors", [])
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "input_file": input_csv
        }


def run_visualization_agent(input_csv: str, output_dir: str) -> dict:
    """
    Run the visualization agent on a CSV file.
    
    Args:
        input_csv: Path to input CSV
        output_dir: Directory to save plots
    
    Returns:
        Report dictionary with selected plots
    """
    try:
        from viz_agent_node import visualization_agent_node
        from plot_generator import generate_all_selected_plots
        
        # Load data
        df = pd.read_csv(input_csv)
        
        # Run visualization selection
        state = {"data": df}
        result = visualization_agent_node(state)
        
        # Generate plots
        selected_plots = result.get("selected_plots", [])
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        generation_results = generate_all_selected_plots(df, selected_plots, plots_dir)
        
        return {
            "status": "success",
            "input_file": input_csv,
            "plots_dir": plots_dir,
            "plots_selected": len(selected_plots),
            "plots_generated": len(generation_results.get("plots_generated", [])),
            "plots_failed": len(generation_results.get("plots_failed", [])),
            "selected_plots": selected_plots,
            "generated_plots_details": generation_results.get("plots_generated", []),  # Includes inference
            "decision_trace": result.get("decision_trace", []),
            "top_correlations": result.get("top_correlations", [])
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "input_file": input_csv
        }


def run_correlation_agent(input_csv: str, output_csv: str, model_type: str, 
                          target_col: str, output_dir: str) -> dict:
    """
    Run the correlation agent on a CSV file.
    
    Args:
        input_csv: Path to input CSV (output from visualization or outlier agent)
        output_csv: Path where refined CSV should be saved
        model_type: ML model type (linear, logistic, ridge, lasso, tree, forest, xgboost, nn)
        target_col: Target column name (can be empty string or None)
        output_dir: Directory for output files and plots
    
    Returns:
        Report dictionary with correlation analysis results
    """
    try:
        from correlation_agent_final import run_correlation_agent as run_corr
        
        # Handle empty target column
        target = target_col if target_col else None
        
        # Run the agent
        result = run_corr(input_csv, output_csv, model_type, target, output_dir)
        
        return result
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "input_file": input_csv
        }

