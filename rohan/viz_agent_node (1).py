# =====================================================
# Intelligent Visualization Agent (LangGraph Compatible)
# =====================================================
# Agentic EDA System - Visualization Selection Framework
# 
# This agent intelligently selects a limited set of meaningful
# plots based on data characteristics such as variance, outliers,
# missingness, and correlation, ensuring scalability to 
# high-dimensional datasets (100+ columns).
#
# Enhanced with LLM edge case handling (Groq Compound)
# =====================================================

import pandas as pd
import numpy as np
from typing import TypedDict, Dict, List, Optional, Any, Tuple
from scipy import stats

# Try to import LangGraph (optional for standalone use)
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# Import edge case handler
try:
    from edge_case_handler import (
        EdgeCaseHandler, 
        EdgeCaseConfig, 
        EdgeCaseDetector,
        handle_zero_selection_fallback,
        sanitize_column_name
    )
    EDGE_CASE_HANDLER_AVAILABLE = True
except ImportError:
    EDGE_CASE_HANDLER_AVAILABLE = False
    print("[WARN] edge_case_handler not found. Edge case handling disabled.")


# =====================================================
# 1. STATE DEFINITION
# =====================================================

class VisualizationState(TypedDict):
    df: pd.DataFrame
    column_metadata: Dict[str, Dict]      # Per-column analysis
    priority_scores: Dict[str, float]     # Feature importance scores
    eligible_plots: Dict[str, List[str]]  # What plots are allowed per column
    selected_plots: List[Dict]            # Final selected plots
    decision_trace: List[Dict]            # Skip explanations (AGENTIC)
    correlation_matrix: Optional[pd.DataFrame]  # For bivariate decisions
    top_correlations: List[Dict]          # Top correlated pairs
    config: Dict[str, Any]                # Budget and thresholds
    errors: List[Dict]                    # Any errors encountered
    # Edge case handling fields (new)
    edge_case_results: Optional[Dict]     # Results from edge case handler
    columns_to_skip: List[str]            # Columns flagged by edge case handler
    column_type_overrides: Dict[str, Dict]  # Type corrections from LLM
    target_columns: List[str]             # Identified target variables
    datetime_columns: List[str]           # Identified datetime columns


# =====================================================
# 2. CONFIGURATION DEFAULTS
# =====================================================

DEFAULT_CONFIG = {
    # Plot Budgets (Hard Constraints)
    "max_univariate_plots": 15,
    "max_bivariate_plots": 8,
    "max_categorical_plots": 5,
    
    # Thresholds
    "min_variance_threshold": 0.01,      # Minimum variance to consider
    "high_correlation_threshold": 0.7,   # Threshold for bivariate plots
    "max_cardinality": 20,               # Max unique values for categorical
    "outlier_iqr_multiplier": 1.5,       # IQR multiplier for outlier detection
    "skewness_threshold": 1.0,           # Above this = highly skewed
    
    # Feature Scoring Weights
    "missing_weight": 1,
    "outlier_weight": 2,
    "variance_weight": 2,
    "correlation_weight": 3,
    "cardinality_weight": 2,
    "target_weight": 5,                  # Boost for target columns (new)
    
    # Minimum Score for Selection
    "min_priority_score": 2,
    
    # Edge Case Handling (new)
    "enable_edge_case_handling": True,   # Enable edge case detection
    "groq_api_key": None,                # Groq API key (or use env var)
    "groq_model": "compound-beta",       # Groq Compound model
    "min_valid_count": 5,                # Min data points per column
    "max_missing_pct": 95.0,             # Skip columns above this
    "min_plots_fallback": 5,             # Min plots if nothing selected
}


# =====================================================
# 3. METADATA EXTRACTION (Observation Phase)
# =====================================================

def extract_column_metadata(state: VisualizationState) -> VisualizationState:
    """
    Node 1: Extract metadata for each column.
    
    For each column, extracts:
    - Data type (continuous/categorical)
    - Missing percentage
    - Variance / IQR (information content)
    - Outlier presence
    - Cardinality (for categorical)
    - Skewness
    """
    df = state["df"]
    config = state.get("config", DEFAULT_CONFIG)
    column_metadata = {}
    errors = state.get("errors", [])
    
    for col in df.columns:
        try:
            meta = {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
                "is_categorical": pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype),
                "total_count": len(df[col]),
                "missing_count": df[col].isna().sum(),
                "missing_pct": round(df[col].isna().mean() * 100, 2),
            }
            
            # Get non-null values
            valid_values = df[col].dropna()
            meta["valid_count"] = len(valid_values)
            
            if meta["is_numeric"] and len(valid_values) > 0:
                # Numeric column analysis
                meta["min"] = float(valid_values.min())
                meta["max"] = float(valid_values.max())
                meta["mean"] = float(valid_values.mean())
                meta["median"] = float(valid_values.median())
                meta["std"] = float(valid_values.std())
                meta["variance"] = float(valid_values.var())
                
                # IQR and Outliers
                Q1 = valid_values.quantile(0.25)
                Q3 = valid_values.quantile(0.75)
                IQR = Q3 - Q1
                meta["Q1"] = float(Q1)
                meta["Q3"] = float(Q3)
                meta["IQR"] = float(IQR)
                
                # Outlier detection using IQR
                multiplier = config.get("outlier_iqr_multiplier", 1.5)
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                outlier_mask = (valid_values < lower_bound) | (valid_values > upper_bound)
                meta["outlier_count"] = int(outlier_mask.sum())
                meta["outlier_pct"] = round(outlier_mask.mean() * 100, 2)
                meta["has_outliers"] = meta["outlier_count"] > 0
                
                # Skewness
                meta["skewness"] = float(valid_values.skew())
                meta["is_skewed"] = abs(meta["skewness"]) > config.get("skewness_threshold", 1.0)
                
                # Coefficient of Variation (normalized variance)
                if meta["mean"] != 0:
                    meta["cv"] = abs(meta["std"] / meta["mean"])
                else:
                    meta["cv"] = 0
                    
            elif not meta["is_numeric"]:
                # Categorical column analysis
                meta["cardinality"] = valid_values.nunique()
                meta["unique_ratio"] = meta["cardinality"] / len(valid_values) if len(valid_values) > 0 else 0
                meta["top_values"] = valid_values.value_counts().head(5).to_dict()
                meta["is_high_cardinality"] = meta["cardinality"] > config.get("max_cardinality", 20)
                
                # Check if might be an ID column
                meta["is_potential_id"] = meta["cardinality"] / len(valid_values) > 0.9 if len(valid_values) > 0 else False
                
            column_metadata[col] = meta
            
        except Exception as e:
            errors.append({
                "column": col,
                "stage": "metadata_extraction",
                "error": str(e),
                "type": type(e).__name__
            })
    
    state["column_metadata"] = column_metadata
    state["errors"] = errors
    return state


# =====================================================
# 3.5 EDGE CASE PROCESSING (NEW - LLM Enhanced)
# =====================================================

def process_edge_cases(state: VisualizationState) -> VisualizationState:
    """
    Node 1.5: Process edge cases using rules first, then LLM for semantic cases.
    
    This node:
    1. Detects all edge cases (rule-based + semantic)
    2. Applies rule-based fixes immediately
    3. Calls Groq LLM for semantic understanding (datetime, ID, target detection)
    4. Updates state with columns to skip, type overrides, etc.
    """
    if not EDGE_CASE_HANDLER_AVAILABLE:
        # Skip edge case handling if module not available
        state["edge_case_results"] = None
        state["columns_to_skip"] = []
        state["column_type_overrides"] = {}
        state["target_columns"] = []
        state["datetime_columns"] = []
        return state
    
    config = state.get("config", DEFAULT_CONFIG)
    
    # Check if edge case handling is enabled
    if not config.get("enable_edge_case_handling", True):
        state["edge_case_results"] = None
        state["columns_to_skip"] = []
        state["column_type_overrides"] = {}
        state["target_columns"] = []
        state["datetime_columns"] = []
        return state
    
    df = state["df"]
    column_metadata = state["column_metadata"]
    decision_trace = state.get("decision_trace", [])
    errors = state.get("errors", [])
    
    try:
        # Create edge case config from visualization config
        edge_config = EdgeCaseConfig(
            groq_api_key=config.get("groq_api_key"),
            groq_model=config.get("groq_model", "compound-beta"),
            min_valid_count=config.get("min_valid_count", 5),
            max_missing_pct=config.get("max_missing_pct", 95.0),
            min_plots_fallback=config.get("min_plots_fallback", 5)
        )
        
        # Create handler and process
        handler = EdgeCaseHandler(edge_config)
        results = handler.process(df, column_metadata)
        
        # Check for abort condition
        if results.get("should_abort"):
            decision_trace.append({
                "action": "edge_case_abort",
                "reason": results.get("abort_reason", "Critical edge case detected"),
                "stage": "edge_case_processing"
            })
            state["edge_case_results"] = results
            state["columns_to_skip"] = []
            state["column_type_overrides"] = {}
            state["target_columns"] = []
            state["datetime_columns"] = []
            state["errors"] = errors
            state["decision_trace"] = decision_trace
            return state
        
        # Store results
        state["edge_case_results"] = results
        state["columns_to_skip"] = results.get("columns_to_skip", [])
        state["column_type_overrides"] = results.get("column_type_overrides", {})
        state["target_columns"] = results.get("target_columns", [])
        state["datetime_columns"] = results.get("datetime_columns", [])
        
        # Log edge case decisions to trace
        for ec in results.get("edge_cases", []):
            if hasattr(ec, 'case_type'):
                decision_trace.append({
                    "action": "edge_case_detected",
                    "type": ec.case_type.value if hasattr(ec.case_type, 'value') else str(ec.case_type),
                    "columns": ec.columns if hasattr(ec, 'columns') else [],
                    "requires_llm": ec.requires_llm if hasattr(ec, 'requires_llm') else False,
                    "resolved": ec.is_resolved if hasattr(ec, 'is_resolved') else False
                })
        
        # Log summary
        decision_trace.append({
            "action": "edge_case_summary",
            "total_edge_cases": len(results.get("edge_cases", [])),
            "columns_to_skip": len(results.get("columns_to_skip", [])),
            "type_overrides": len(results.get("column_type_overrides", {})),
            "target_columns": results.get("target_columns", []),
            "datetime_columns": results.get("datetime_columns", [])
        })
        
    except Exception as e:
        errors.append({
            "stage": "edge_case_processing",
            "error": str(e),
            "type": type(e).__name__
        })
        # Set defaults on error
        state["edge_case_results"] = None
        state["columns_to_skip"] = []
        state["column_type_overrides"] = {}
        state["target_columns"] = []
        state["datetime_columns"] = []
    
    state["decision_trace"] = decision_trace
    state["errors"] = errors
    return state


# =====================================================
# 4. CORRELATION ANALYSIS
# =====================================================

def compute_correlation_matrix(state: VisualizationState) -> VisualizationState:
    """
    Node 2: Compute correlation matrix for numeric columns.
    Identifies top correlated pairs for bivariate plots.
    """
    df = state["df"]
    config = state.get("config", DEFAULT_CONFIG)
    errors = state.get("errors", [])
    
    try:
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            state["correlation_matrix"] = None
            state["top_correlations"] = []
            return state
        
        # Compute correlation matrix
        corr_matrix = df[numeric_cols].corr()
        state["correlation_matrix"] = corr_matrix
        
        # Find top correlated pairs (above threshold)
        threshold = config.get("high_correlation_threshold", 0.7)
        max_pairs = config.get("max_bivariate_plots", 8)
        
        correlations = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_value = corr_matrix.loc[col1, col2]
                if not pd.isna(corr_value):
                    correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": abs(corr_value),
                        "correlation_raw": corr_value
                    })
        
        # Sort by absolute correlation and take top pairs above threshold
        correlations.sort(key=lambda x: x["correlation"], reverse=True)
        top_correlations = [c for c in correlations if c["correlation"] >= threshold][:max_pairs]
        
        state["top_correlations"] = top_correlations
        
    except Exception as e:
        errors.append({
            "stage": "correlation_analysis",
            "error": str(e),
            "type": type(e).__name__
        })
        state["correlation_matrix"] = None
        state["top_correlations"] = []
    
    state["errors"] = errors
    return state


# =====================================================
# 5. FEATURE PRIORITY SCORING
# =====================================================

def compute_priority_scores(state: VisualizationState) -> VisualizationState:
    """
    Node 3: Compute priority scores for each feature.
    
    Scoring Logic:
    - IF missing % > 0 → +1 (worth visualizing missingness impact)
    - IF outliers present → +2 (boxplot candidate)
    - IF variance > threshold → +2 (informative feature)
    - IF correlated with other features > 0.7 → +3 (bivariate candidate)
    - IF categorical AND cardinality < 20 → +2 (clean bar plot)
    - IF target column (from LLM) → +5 (priority visualization)
    """
    config = state.get("config", DEFAULT_CONFIG)
    column_metadata = state["column_metadata"]
    top_correlations = state.get("top_correlations", [])
    priority_scores = {}
    decision_trace = state.get("decision_trace", [])
    
    # Get edge case results
    columns_to_skip = set(state.get("columns_to_skip", []))
    column_type_overrides = state.get("column_type_overrides", {})
    target_columns = set(state.get("target_columns", []))
    
    # Build set of columns with high correlations
    correlated_columns = set()
    for corr in top_correlations:
        correlated_columns.add(corr["column1"])
        correlated_columns.add(corr["column2"])
    
    for col, meta in column_metadata.items():
        # Skip columns flagged by edge case handler
        if col in columns_to_skip:
            priority_scores[col] = {
                "score": -1,  # Negative score = skip
                "breakdown": ["SKIPPED by edge case handler"]
            }
            decision_trace.append({
                "column": col,
                "action": "skipped_edge_case",
                "score": -1,
                "reason": "Flagged by edge case handler (ID, empty, or invalid)"
            })
            continue
        
        score = 0
        score_breakdown = []
        
        # Target column boost (from LLM)
        if col in target_columns:
            score += config.get("target_weight", 5)
            score_breakdown.append(f"+{config.get('target_weight', 5)} (target column detected by LLM)")
        
        # Check for type overrides from LLM
        override = column_type_overrides.get(col, {})
        if override.get("priority_boost"):
            boost = override["priority_boost"]
            score += boost
            score_breakdown.append(f"+{boost} (LLM priority boost)")
        
        # Missing value contribution
        if meta.get("missing_pct", 0) > 0:
            score += config.get("missing_weight", 1)
            score_breakdown.append(f"+{config.get('missing_weight', 1)} (has missing values)")
        
        if meta.get("is_numeric"):
            # Outlier contribution
            if meta.get("has_outliers", False):
                score += config.get("outlier_weight", 2)
                score_breakdown.append(f"+{config.get('outlier_weight', 2)} (has outliers)")
            
            # Variance contribution
            if meta.get("variance", 0) > config.get("min_variance_threshold", 0.01):
                score += config.get("variance_weight", 2)
                score_breakdown.append(f"+{config.get('variance_weight', 2)} (high variance)")
            
            # Correlation contribution
            if col in correlated_columns:
                score += config.get("correlation_weight", 3)
                score_breakdown.append(f"+{config.get('correlation_weight', 3)} (strongly correlated)")
                
        elif meta.get("is_categorical"):
            # Cardinality contribution
            if not meta.get("is_high_cardinality", True) and not meta.get("is_potential_id", False):
                score += config.get("cardinality_weight", 2)
                score_breakdown.append(f"+{config.get('cardinality_weight', 2)} (low cardinality categorical)")
        
        priority_scores[col] = {
            "score": score,
            "breakdown": score_breakdown
        }
        
        # Log decision trace
        decision_trace.append({
            "column": col,
            "action": "scored",
            "score": score,
            "breakdown": score_breakdown
        })
    
    state["priority_scores"] = priority_scores
    state["decision_trace"] = decision_trace
    return state


# =====================================================
# 6. PLOT ELIGIBILITY RULES (Intelligence Layer)
# =====================================================

def determine_plot_eligibility(state: VisualizationState) -> VisualizationState:
    """
    Node 4: Determine which plots are eligible for each column.
    
    Univariate Plot Rules:
    - Continuous + outliers → Box plot
    - Continuous + no outliers → Histogram / KDE
    - Highly skewed → Violin plot
    - Categorical + low cardinality → Bar / Count plot
    - Categorical + high cardinality → Skip
    
    Bivariate Plot Rules:
    - corr > 0.7 → Scatter plot
    - continuous vs categorical → Box / Violin
    - weak correlation → Skip
    """
    config = state.get("config", DEFAULT_CONFIG)
    column_metadata = state["column_metadata"]
    priority_scores = state["priority_scores"]
    decision_trace = state.get("decision_trace", [])
    
    eligible_plots = {}
    min_score = config.get("min_priority_score", 2)
    
    for col, meta in column_metadata.items():
        col_score = priority_scores.get(col, {}).get("score", 0)
        plots = []
        skip_reasons = []
        
        # Check minimum score threshold
        if col_score < min_score:
            skip_reasons.append(f"Low priority score ({col_score} < {min_score})")
        
        # Skip potential ID columns
        if meta.get("is_potential_id", False):
            skip_reasons.append("Potential ID column (high uniqueness)")
        
        if not skip_reasons:
            if meta.get("is_numeric"):
                # Numeric column plot selection
                if meta.get("has_outliers", False):
                    plots.append("boxplot")
                    plots.append("histogram")  # Still useful to see distribution
                elif meta.get("is_skewed", False):
                    plots.append("violin")
                    plots.append("histogram")
                else:
                    plots.append("histogram")
                    plots.append("kde")
                    
            elif meta.get("is_categorical"):
                if meta.get("is_high_cardinality", True):
                    skip_reasons.append(f"High cardinality ({meta.get('cardinality', 'N/A')} unique values)")
                else:
                    plots.append("barplot")
                    plots.append("countplot")
        
        eligible_plots[col] = {
            "plots": plots,
            "skip_reasons": skip_reasons,
            "is_eligible": len(plots) > 0
        }
        
        # Log decision trace
        if skip_reasons:
            decision_trace.append({
                "column": col,
                "action": "skipped",
                "reasons": skip_reasons
            })
        else:
            decision_trace.append({
                "column": col,
                "action": "eligible",
                "plots": plots
            })
    
    state["eligible_plots"] = eligible_plots
    state["decision_trace"] = decision_trace
    return state


# =====================================================
# 7. PLOT SELECTION (with Budget Constraints)
# =====================================================

def select_plots_with_budget(state: VisualizationState) -> VisualizationState:
    """
    Node 5: Select final plots respecting budget constraints.
    
    Budgets:
    - MAX_UNIVARIATE_PLOTS = 15
    - MAX_BIVARIATE_PLOTS = 8
    - MAX_CATEGORICAL_PLOTS = 5
    """
    config = state.get("config", DEFAULT_CONFIG)
    column_metadata = state["column_metadata"]
    priority_scores = state["priority_scores"]
    eligible_plots = state["eligible_plots"]
    top_correlations = state.get("top_correlations", [])
    decision_trace = state.get("decision_trace", [])
    
    # Budget limits
    max_univariate = config.get("max_univariate_plots", 15)
    max_bivariate = config.get("max_bivariate_plots", 8)
    max_categorical = config.get("max_categorical_plots", 5)
    
    selected_plots = []
    univariate_count = 0
    bivariate_count = 0
    categorical_count = 0
    
    # Sort columns by priority score (descending)
    sorted_columns = sorted(
        [(col, priority_scores.get(col, {}).get("score", 0)) for col in eligible_plots.keys()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # === UNIVARIATE PLOT SELECTION ===
    for col, score in sorted_columns:
        eligibility = eligible_plots.get(col, {})
        if not eligibility.get("is_eligible", False):
            continue
        
        meta = column_metadata.get(col, {})
        plots = eligibility.get("plots", [])
        
        if meta.get("is_categorical"):
            # Categorical budget
            if categorical_count >= max_categorical:
                decision_trace.append({
                    "column": col,
                    "action": "budget_exceeded",
                    "reason": f"Categorical plot budget exceeded ({max_categorical} max)",
                    "plots_skipped": plots
                })
                continue
            
            # Select only one plot type per categorical column
            selected_plot = plots[0] if plots else None
            if selected_plot:
                selected_plots.append({
                    "column": col,
                    "plot_type": selected_plot,
                    "category": "univariate_categorical",
                    "priority_score": score,
                    "reason": f"Selected based on priority score ({score})"
                })
                categorical_count += 1
                
        else:
            # Numeric univariate budget
            if univariate_count >= max_univariate:
                decision_trace.append({
                    "column": col,
                    "action": "budget_exceeded",
                    "reason": f"Univariate plot budget exceeded ({max_univariate} max)",
                    "plots_skipped": plots
                })
                continue
            
            # Select primary plot type
            selected_plot = plots[0] if plots else None
            if selected_plot:
                selected_plots.append({
                    "column": col,
                    "plot_type": selected_plot,
                    "category": "univariate_numeric",
                    "priority_score": score,
                    "reason": f"Selected based on priority score ({score})"
                })
                univariate_count += 1
    
    # === BIVARIATE PLOT SELECTION ===
    for corr in top_correlations[:max_bivariate]:
        if bivariate_count >= max_bivariate:
            decision_trace.append({
                "action": "budget_exceeded",
                "reason": f"Bivariate plot budget exceeded ({max_bivariate} max)",
                "skipped_pair": f"{corr['column1']} vs {corr['column2']}"
            })
            continue
        
        selected_plots.append({
            "column1": corr["column1"],
            "column2": corr["column2"],
            "plot_type": "scatter",
            "category": "bivariate",
            "correlation": corr["correlation"],
            "reason": f"Strong correlation ({corr['correlation']:.2f})"
        })
        bivariate_count += 1
    
    # Add correlation heatmap if we have correlations
    if len(top_correlations) > 0:
        selected_plots.append({
            "plot_type": "heatmap",
            "category": "correlation_overview",
            "reason": "Overview of feature correlations"
        })
    
    # === ZERO-SELECTION FALLBACK ===
    # If no univariate or categorical plots were selected, use LLM fallback
    if univariate_count == 0 and categorical_count == 0:
        decision_trace.append({
            "action": "zero_selection_detected",
            "reason": "No plots selected by standard pipeline, invoking fallback"
        })
        
        if EDGE_CASE_HANDLER_AVAILABLE:
            try:
                # Create edge case config
                edge_config = EdgeCaseConfig(
                    groq_api_key=config.get("groq_api_key"),
                    groq_model=config.get("groq_model", "compound-beta"),
                    min_plots_fallback=config.get("min_plots_fallback", 5)
                )
                
                # Call fallback handler
                fallback_plots = handle_zero_selection_fallback(
                    df, column_metadata, priority_scores, edge_config
                )
                
                if fallback_plots:
                    selected_plots.extend(fallback_plots)
                    univariate_count = len([p for p in fallback_plots if "numeric" in p.get("category", "")])
                    categorical_count = len([p for p in fallback_plots if "categorical" in p.get("category", "")])
                    
                    decision_trace.append({
                        "action": "fallback_selection_applied",
                        "plots_added": len(fallback_plots),
                        "method": "llm_fallback"
                    })
            except Exception as e:
                decision_trace.append({
                    "action": "fallback_error",
                    "error": str(e)
                })
    
    # === FINAL SUMMARY ===
    decision_trace.append({
        "action": "selection_complete",
        "summary": {
            "total_columns": len(column_metadata),
            "univariate_selected": univariate_count,
            "categorical_selected": categorical_count,
            "bivariate_selected": bivariate_count,
            "total_plots": len(selected_plots)
        }
    })
    
    state["selected_plots"] = selected_plots
    state["decision_trace"] = decision_trace
    return state


# =====================================================
# 8. GENERATE REPORT
# =====================================================

def generate_visualization_report(state: VisualizationState) -> VisualizationState:
    """
    Node 6: Generate final summary report.
    """
    column_metadata = state["column_metadata"]
    priority_scores = state["priority_scores"]
    selected_plots = state["selected_plots"]
    decision_trace = state["decision_trace"]
    
    # Build summary
    summary = {
        "total_columns_analyzed": len(column_metadata),
        "numeric_columns": sum(1 for m in column_metadata.values() if m.get("is_numeric")),
        "categorical_columns": sum(1 for m in column_metadata.values() if m.get("is_categorical")),
        "plots_selected": len(selected_plots),
        "columns_skipped": sum(1 for e in state["eligible_plots"].values() if not e.get("is_eligible")),
    }
    
    # Top scoring features
    top_features = sorted(
        [(col, data.get("score", 0)) for col, data in priority_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    summary["top_features"] = [{"column": col, "score": score} for col, score in top_features]
    
    # Add summary to decision trace
    decision_trace.insert(0, {"report_summary": summary})
    
    state["decision_trace"] = decision_trace
    return state


# =====================================================
# 9. BUILD LANGGRAPH (if available)
# =====================================================

def build_visualization_graph():
    """Build the LangGraph for visualization selection pipeline."""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph is not installed. Install with: pip install langgraph")
    
    graph = StateGraph(VisualizationState)
    
    graph.add_node("extract_metadata", extract_column_metadata)
    graph.add_node("process_edge_cases", process_edge_cases)  # NEW: LLM edge case handling
    graph.add_node("compute_correlations", compute_correlation_matrix)
    graph.add_node("compute_scores", compute_priority_scores)
    graph.add_node("determine_eligibility", determine_plot_eligibility)
    graph.add_node("select_plots", select_plots_with_budget)
    graph.add_node("generate_report", generate_visualization_report)
    
    graph.set_entry_point("extract_metadata")
    
    graph.add_edge("extract_metadata", "process_edge_cases")  # NEW edge
    graph.add_edge("process_edge_cases", "compute_correlations")  # NEW edge
    graph.add_edge("compute_correlations", "compute_scores")
    graph.add_edge("compute_scores", "determine_eligibility")
    graph.add_edge("determine_eligibility", "select_plots")
    graph.add_edge("select_plots", "generate_report")
    graph.add_edge("generate_report", END)
    
    return graph.compile()


# =====================================================
# 10. STANDALONE NODE FUNCTION (for simple integration)
# =====================================================

def visualization_agent_node(state: dict) -> dict:
    """
    Main entry point for visualization selection.
    Compatible with both simple dict state and full VisualizationState.
    
    Args:
        state: dict with at least {"data": pd.DataFrame} or {"df": pd.DataFrame}
    
    Returns:
        Updated state with selected plots and decision trace
    """
    # Handle both "data" and "df" keys
    if "data" in state:
        df = state["data"].copy()
    elif "df" in state:
        df = state["df"].copy()
    else:
        raise ValueError("State must contain 'data' or 'df' key with DataFrame")
    
    # Initialize VisualizationState
    viz_state: VisualizationState = {
        "df": df,
        "column_metadata": {},
        "priority_scores": {},
        "eligible_plots": {},
        "selected_plots": [],
        "decision_trace": [],
        "correlation_matrix": None,
        "top_correlations": [],
        "config": state.get("config", DEFAULT_CONFIG),
        "errors": [],
        # Edge case fields (new)
        "edge_case_results": None,
        "columns_to_skip": [],
        "column_type_overrides": {},
        "target_columns": [],
        "datetime_columns": []
    }
    
    # Run pipeline steps (with edge case processing)
    viz_state = extract_column_metadata(viz_state)
    viz_state = process_edge_cases(viz_state)  # NEW: Edge case handling
    viz_state = compute_correlation_matrix(viz_state)
    viz_state = compute_priority_scores(viz_state)
    viz_state = determine_plot_eligibility(viz_state)
    viz_state = select_plots_with_budget(viz_state)
    viz_state = generate_visualization_report(viz_state)
    
    # Update original state
    state["selected_plots"] = viz_state["selected_plots"]
    state["decision_trace"] = viz_state["decision_trace"]
    state["column_metadata"] = viz_state["column_metadata"]
    state["priority_scores"] = viz_state["priority_scores"]
    state["eligible_plots"] = viz_state["eligible_plots"]
    state["correlation_matrix"] = viz_state["correlation_matrix"]
    state["top_correlations"] = viz_state["top_correlations"]
    state["errors"] = viz_state["errors"]
    # Edge case results (new)
    state["edge_case_results"] = viz_state.get("edge_case_results")
    state["columns_to_skip"] = viz_state.get("columns_to_skip", [])
    state["column_type_overrides"] = viz_state.get("column_type_overrides", {})
    state["target_columns"] = viz_state.get("target_columns", [])
    state["datetime_columns"] = viz_state.get("datetime_columns", [])
    
    return state


# =====================================================
# 11. ENTRY POINT FOR TESTING
# =====================================================

if __name__ == "__main__":
    import sys
    import io
    
    # Fix Windows console encoding
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    # Test with a sample file
    test_file = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\rohan\Antigravity\EL_sem3\Outlier_detector\AmesHousing_cleaned.csv"
    
    print(f"Loading: {test_file}")
    df = pd.read_csv(test_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    state = {"data": df}
    result = visualization_agent_node(state)
    
    print("\n" + "=" * 70)
    print("INTELLIGENT VISUALIZATION SELECTION REPORT")
    print("=" * 70)
    
    # Summary
    summary = result["decision_trace"][0].get("report_summary", {})
    print(f"\n[DATA] Dataset Overview:")
    print(f"   Total columns: {summary.get('total_columns_analyzed', 'N/A')}")
    print(f"   Numeric columns: {summary.get('numeric_columns', 'N/A')}")
    print(f"   Categorical columns: {summary.get('categorical_columns', 'N/A')}")
    
    print(f"\n[PLOTS] Plot Selection:")
    print(f"   Plots selected: {summary.get('plots_selected', 'N/A')}")
    print(f"   Columns skipped: {summary.get('columns_skipped', 'N/A')}")
    
    # Top features
    print(f"\n[TOP] Top Scoring Features:")
    for i, feat in enumerate(summary.get("top_features", [])[:5], 1):
        print(f"   {i}. {feat['column']} (score: {feat['score']})")
    
    # Selected plots
    print(f"\n[SELECTED] Selected Plots:")
    for i, plot in enumerate(result["selected_plots"][:10], 1):
        if plot["category"] == "bivariate":
            print(f"   {i}. {plot['plot_type'].upper()}: {plot['column1']} vs {plot['column2']} ({plot['reason']})")
        elif plot["category"] == "correlation_overview":
            print(f"   {i}. {plot['plot_type'].upper()}: {plot['reason']}")
        else:
            print(f"   {i}. {plot['plot_type'].upper()}: {plot.get('column', 'N/A')} ({plot['reason']})")
    
    # Skip reasons (sample)
    print(f"\n[SKIPPED] Sample Skip Decisions:")
    skip_count = 0
    for trace in result["decision_trace"]:
        if trace.get("action") == "skipped" and skip_count < 5:
            reasons = trace.get("reasons", [])
            print(f"   - {trace['column']}: {', '.join(reasons)}")
            skip_count += 1
    
    if result.get("errors"):
        print(f"\n[ERRORS] Errors ({len(result['errors'])}):")
        for err in result["errors"][:3]:
            print(f"   - {err.get('column', 'N/A')}: {err.get('error', 'Unknown')}")
    
    print("\n" + "=" * 70)
    print("Pipeline completed successfully.")
