# =====================================================
# Correlation Agent - Feature Redundancy & Multicollinearity
# =====================================================
# Callable module for the EDA pipeline
# =====================================================

import pandas as pd
import numpy as np
import json
import os
import re
import shutil
from typing import TypedDict, Dict, Any, Optional
from itertools import combinations
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Optional: LangGraph and LLM
try:
    from langgraph.graph import StateGraph, END
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# =========================
# CONSTANTS
# =========================

LINEAR_MODELS = ["linear", "logistic"]
CORR_THRESHOLD = 0.85
PLOT_THRESHOLD = 0.6
VIF_THRESHOLD = 10


# =========================
# STATE DEFINITION
# =========================

class FeatureState(TypedDict):
    df: pd.DataFrame
    numeric_cols: list
    categorical_cols: list
    redundant_pairs: list
    chi_square_results: list
    anova_results: list
    vif_results: list
    refined_df: pd.DataFrame
    metadata: dict
    llm_summary: str
    # Config passed in
    model_type: str
    target_col: Optional[str]
    plots_dir: str


# =========================
# UTILITIES
# =========================

def should_enforce_vif(model_type: Optional[str]) -> bool:
    """Check if VIF-based removal should be applied (only for linear/logistic models)."""
    if not model_type:
        return False
    return model_type.lower() in LINEAR_MODELS


def parse_numeric_like(val):
    try:
        if isinstance(val, str):
            v = val.lower().strip().replace(",", "").replace("+", "")
            if v.endswith("k"):
                return float(v[:-1]) * 1_000
            if v.endswith("m"):
                return float(v[:-1]) * 1_000_000
            if v.endswith("%"):
                return float(v[:-1])
        return float(val)
    except:
        return np.nan


def safe_float(val):
    """Safely convert value to float, handling NaNs and Infinity."""
    try:
        if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    except:
        return None


def safe_filename(name: str) -> str:
    """Convert column names to safe filenames."""
    name = name.strip()
    name = re.sub(r"[^\w\-]+", "_", name)
    return name


def scatter_plot(df, x, y, plots_dir: str):
    """Generate scatter plot for correlated features."""
    r = df[[x, y]].corr().iloc[0, 1]
    n = len(df)

    if abs(r) >= 0.85:
        strength = "strong"
    elif abs(r) >= 0.5:
        strength = "moderate"
    else:
        strength = "weak"

    plt.figure(figsize=(6, 5))
    plt.scatter(df[x], df[y], alpha=0.4)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y}")

    textstr = f"r = {r:.3f} ({strength})\nN = {n}"
    plt.gca().text(
        0.05, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    plt.tight_layout()
    safe_x = safe_filename(x)
    safe_y = safe_filename(y)
    filepath = os.path.join(plots_dir, f"correlation_{safe_x}_vs_{safe_y}.png")
    plt.savefig(filepath)
    plt.close()
    return filepath


def column_missing_ratio(df, col):
    return df[col].isna().mean()


# =========================
# PIPELINE FUNCTIONS
# =========================

def infer_types(df: pd.DataFrame):
    """Infer numeric and categorical column types."""
    numeric_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    for col in categorical_cols[:]:
        parsed = df[col].apply(parse_numeric_like)
        if parsed.notna().mean() > 0.7:
            new_col = f"{col}_numeric"
            df[new_col] = parsed
            numeric_cols.append(new_col)
            categorical_cols.remove(col)

    return df, numeric_cols, categorical_cols


def detect_redundancy(df: pd.DataFrame, numeric_cols: list, target_col: Optional[str], plots_dir: str):
    """Detect highly correlated feature pairs."""
    redundant_pairs = []
    generated_plots = []

    for c1, c2 in combinations(numeric_cols, 2):
        if target_col and (c1 == target_col or c2 == target_col):
            continue

        sub = df[[c1, c2]].dropna()
        if len(sub) < 10:
            continue

        r = sub.corr().iloc[0, 1]

        if abs(r) >= PLOT_THRESHOLD:
            plot_path = scatter_plot(sub, c1, c2, plots_dir)
            generated_plots.append(plot_path)

        if abs(r) >= CORR_THRESHOLD:
            redundant_pairs.append({
                "col1": c1,
                "col2": c2,
                "correlation": safe_float(r),
                "plot": os.path.basename(plot_path) if abs(r) >= PLOT_THRESHOLD else None
            })

    return redundant_pairs, generated_plots


def chi_square_analysis(df: pd.DataFrame, categorical_cols: list):
    """Chi-square test for categorical pairs."""
    results = []

    for c1, c2 in combinations(categorical_cols, 2):
        table = pd.crosstab(df[c1], df[c2])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        chi2, p, _, _ = chi2_contingency(table)
        n = table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))

        if p < 0.05:
            results.append({
                "col1": c1,
                "col2": c2,
                "p_value": safe_float(p),
                "cramers_v": safe_float(cramers_v)
            })

    return results


def anova_analysis(df: pd.DataFrame, numeric_cols: list, categorical_cols: list, target_col: Optional[str]):
    """ANOVA for numeric-categorical relationships."""
    results = []

    for num in numeric_cols:
        if target_col and num == target_col:
            continue

        for cat in categorical_cols:
            groups = [
                df[df[cat] == level][num].dropna()
                for level in df[cat].unique()
                if len(df[df[cat] == level][num].dropna()) > 5
            ]

            if len(groups) < 2:
                continue

            _, p = f_oneway(*groups)
            if p < 0.05:
                results.append({
                    "numeric": num,
                    "categorical": cat,
                    "p_value": safe_float(p)
                })

    return results


def compute_vif(df: pd.DataFrame, features: list):
    """Calculate Variance Inflation Factor."""
    X = df[features].dropna()
    if X.shape[1] < 2:
        return []

    vif_results = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_results.append({"feature": col, "vif": safe_float(vif)})
        except:
            pass
    return vif_results


def encode_categorical_features(df: pd.DataFrame, target_col: Optional[str]):
    """Encode categorical features using One-Hot Encoding and Label Encoding."""
    df = df.copy()
    encoded_info = []
    
    categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    
    # Encode target column if categorical
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        encoded_info.append({"column": target_col, "method": "Label Encoding (Target)"})
        
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= 10:
            # One-Hot Encode for low cardinality
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            encoded_info.append({"column": col, "method": "One-Hot Encoding", "unique_values": int(unique_count)})
        else:
            # Label Encode for high cardinality
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoded_info.append({"column": col, "method": "Label Encoding", "unique_values": int(unique_count)})
            
    # Convert bool columns from get_dummies to int
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    return df, encoded_info


def apply_removal(df: pd.DataFrame, redundant_pairs: list, vif_results: list, 
                  model_type: str, target_col: Optional[str]):
    """Remove redundant features based on analysis."""
    df = df.copy()
    removed = []

    # Pearson-based removal (ALWAYS)
    for pair in redundant_pairs:
        c1, c2 = pair["col1"], pair["col2"]

        if c1 not in df.columns or c2 not in df.columns:
            continue

        drop_col = c1 if column_missing_ratio(df, c1) > column_missing_ratio(df, c2) else c2
        df.drop(columns=drop_col, inplace=True)

        removed.append({
            "removed": drop_col,
            "reason": "high_pairwise_correlation",
            "metric": pair["correlation"]
        })

    # VIF-based removal (MODEL DEPENDENT)
    if should_enforce_vif(model_type):
        for entry in vif_results:
            if entry["vif"] is not None and entry["vif"] > VIF_THRESHOLD and entry["feature"] in df.columns:
                df.drop(columns=entry["feature"], inplace=True)
                removed.append({
                    "removed": entry["feature"],
                    "reason": "high_multicollinearity_vif",
                    "metric": entry["vif"]
                })

    return df, removed


def generate_llm_summary(model_type: str, target_col: Optional[str], 
                         removed_columns: list, vif_results: list, encoded_info: list = None) -> str:
    """Generate LLM explanation (optional)."""
    if not LANGGRAPH_AVAILABLE:
        return "LLM summary not available (langgraph not installed)"
    
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        prompt = f"""
Generate a professional EDA explanation.

Model type: {model_type}
Target column: {target_col}

Rules applied:
- Pearson correlation enforced for redundancy
- VIF enforced only for linear/logistic models
- VIF does not require a target variable
- Categorical features numerically encoded for ML readiness

Removed features:
{removed_columns}

VIF results:
{vif_results}

Encoded features:
{encoded_info}
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"LLM summary generation failed: {str(e)}"


# =========================
# MAIN CALLABLE FUNCTION
# =========================

def run_correlation_agent(
    input_csv: str,
    output_csv: str,
    model_type: str,
    target_col: Optional[str],
    output_dir: str
) -> Dict[str, Any]:
    """
    Run the correlation agent on a CSV file.
    
    Args:
        input_csv: Path to input CSV (output from outlier agent)
        output_csv: Path where refined CSV should be saved
        model_type: ML model type (linear, logistic, ridge, lasso, tree, forest, xgboost, nn)
        target_col: Target column name (optional)
        output_dir: Directory for output files and plots
    
    Returns:
        Report dictionary with correlation analysis results
    """
    try:
        # Setup
        plots_dir = os.path.join(output_dir, "correlation_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Load data
        df = pd.read_csv(input_csv)
        original_shape = df.shape
        
        # Infer types
        df, numeric_cols, categorical_cols = infer_types(df)
        
        # Detect redundancy
        redundant_pairs, generated_plots = detect_redundancy(
            df, numeric_cols, target_col, plots_dir
        )
        
        # Chi-square analysis
        chi_results = chi_square_analysis(df, categorical_cols)
        
        # ANOVA analysis
        anova_results = anova_analysis(df, numeric_cols, categorical_cols, target_col)
        
        # VIF analysis
        vif_features = [
            col for col in numeric_cols
            if col in df.columns and (target_col is None or col != target_col)
        ]
        vif_results = compute_vif(df, vif_features)
        
        # Apply removal
        refined_df, removed_columns = apply_removal(
            df, redundant_pairs, vif_results, model_type, target_col
        )
        
        # Encode categorical features
        refined_df, encoded_info = encode_categorical_features(refined_df, target_col)
        
        # Generate LLM summary
        llm_summary = generate_llm_summary(model_type, target_col, removed_columns, vif_results, encoded_info)
        
        # Save outputs
        refined_df.to_csv(output_csv, index=False)
        
        # Save metadata
        metadata = {
            "model_type": model_type,
            "target_column": target_col,
            "original_shape": list(original_shape),
            "final_shape": list(refined_df.shape),
            "removed_columns": removed_columns,
            "redundant_pairs": redundant_pairs,
            "chi_square": chi_results,
            "anova": anova_results,
            "vif": vif_results,
            "encoded_features": encoded_info,
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(output_dir, "correlation_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Save LLM summary
        summary_path = os.path.join(output_dir, "correlation_llm_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(llm_summary)
        
        return {
            "status": "success",
            "input_file": input_csv,
            "output_file": output_csv,
            "original_shape": list(original_shape),
            "final_shape": list(refined_df.shape),
            "columns_removed": len(removed_columns),
            "removed_columns": removed_columns,
            "redundant_pairs_found": len(redundant_pairs),
            "high_vif_features": len([v for v in vif_results if v["vif"] is not None and v["vif"] > VIF_THRESHOLD]),
            "plots_generated": len(generated_plots),
            "plots_dir": plots_dir,
            "llm_summary": llm_summary[:500] + "..." if len(llm_summary) > 500 else llm_summary
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "input_file": input_csv
        }


# =========================
# CLI ENTRY POINT
# =========================

if __name__ == "__main__":
    import sys
    
    # CLI mode with prompts
    model_type = input(
        "Enter ML model type (linear, logistic, ridge, lasso, tree, forest, xgboost, nn): "
    ).strip().lower()
    
    target_col = input(
        "Enter target column name (press Enter if none): "
    ).strip()
    target_col = target_col if target_col else None
    
    # Default paths
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "AmesHousing.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    output_csv = os.path.join(output_dir, "refined_output.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    result = run_correlation_agent(input_csv, output_csv, model_type, target_col, output_dir)
    
    print("\n" + "=" * 60)
    print("CORRELATION AGENT RESULTS")
    print("=" * 60)
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Original shape: {result['original_shape']}")
        print(f"Final shape: {result['final_shape']}")
        print(f"Columns removed: {result['columns_removed']}")
        print(f"Plots generated: {result['plots_generated']}")
    else:
        print(f"Error: {result.get('error')}")
