# =====================================================
# Intelligent Visualization Agent (LangGraph Compatible)
# =====================================================
# Agentic EDA System - Visualization Selection Framework
# 
# This agent intelligently selects a limited set of meaningful
# plots based on data characteristics such as variance, outliers,
# missingness, and correlation, ensuring scalability to 
# high-dimensional datasets (100+ columns).
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

# Import agentic visualization agent
try:
    from agentic_viz_agent import (
        AgenticVisualizationAgent,
        get_agentic_plot_recommendations,
        AgentDecisionResult
    )
    AGENTIC_AGENT_AVAILABLE = True
except ImportError:
    AGENTIC_AGENT_AVAILABLE = False
    print("[WARN] agentic_viz_agent not found. Using rule-based mode.")


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
    # Edge case handling fields
    edge_case_results: Optional[Dict]     # Results from edge case handler
    columns_to_skip: List[str]            # Columns flagged by edge case handler
    column_type_overrides: Dict[str, Dict]  # Type corrections from LLM
    target_columns: List[str]             # Identified target variables
    datetime_columns: List[str]           # Identified datetime columns


# =====================================================
# 2. CONFIGURATION DEFAULTS
# =====================================================

DEFAULT_CONFIG = {
    # Plot Budgets - COMPREHENSIVE like Kaggle Gold notebooks
    "max_univariate_plots": 25,    # Histograms, barplots for all meaningful columns
    "max_bivariate_plots": 15,     # Scatter, grouped boxplots, target vs features
    "max_categorical_plots": 12,   # Barplots, pies for categorical
    
    # Thresholds
    "min_variance_threshold": 0.01,      # Minimum variance to consider
    "high_correlation_threshold": 0.5,   # Lowered: Show more correlations
    "max_cardinality": 20,               # Allow more categories
    "outlier_iqr_multiplier": 1.5,       # IQR multiplier for outlier detection
    "skewness_threshold": 1.0,           # Above this = highly skewed
    
    # Feature Scoring Weights
    "missing_weight": 1,
    "outlier_weight": 2,
    "variance_weight": 2,
    "correlation_weight": 3,
    "cardinality_weight": 2,
    "target_weight": 5,                  # Boost for target columns
    
    # Minimum Score for Selection
    "min_priority_score": 2,
    
    # Edge Case Handling
    "enable_edge_case_handling": True,   # Enable edge case detection
    "groq_api_key": None,                # Groq API key (or use env var)
    "groq_model": "llama-3.3-70b-versatile",  # Groq model
    "min_valid_count": 5,                # Min data points per column
    "max_missing_pct": 95.0,             # Skip columns above this
    "min_plots_fallback": 5,             # Min plots if nothing selected
    
    # Agentic Mode (LLM-driven plot selection)
    "use_agentic_mode": True,            # Enable LLM-driven plot selection
    "dataset_name": "dataset",           # Name for LLM context
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
                
                # Zero-inflation check (Crucial for Industry-Grade EDA)
                zero_count = (valid_values == 0).sum()
                meta["zero_pct"] = round((zero_count / len(valid_values)) * 100, 2)
                meta["is_zero_inflated"] = meta["zero_pct"] > 50.0  # Flag if >50% zeros
                
                # === FIX #1: SMART CATEGORICAL DETECTION ===
                # Numeric columns with few unique values should be treated as discrete/ordinal
                unique_count = valid_values.nunique()
                meta["unique_count"] = unique_count
                
                # Detect ordinal: 1-10 ratings, years binned, etc.
                value_range = meta["max"] - meta["min"]
                is_ordinal_rating = (unique_count <= 10 and unique_count == int(value_range) + 1)
                is_discrete_few_values = (unique_count <= 12)  # OverallQual, MonthSold, etc.
                
                if is_ordinal_rating or is_discrete_few_values:
                    meta["is_discrete_categorical"] = True
                    meta["cardinality"] = unique_count
                    meta["should_use_barplot"] = True  # Signal to use barplot not histogram
                else:
                    meta["is_discrete_categorical"] = False
                    meta["should_use_barplot"] = False
                    
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
# 3.5 EDGE CASE PROCESSING (LLM Enhanced)
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
            groq_model=config.get("groq_model", "llama-3.3-70b-versatile"),
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

def select_plots_agentic(state: VisualizationState) -> VisualizationState:
    """
    AGENTIC MODE: LLM decides which plots to generate.
    
    The LLM acts as a data analyst and decides:
    - Which columns to visualize
    - What plot types to use
    - Why each plot is useful
    
    This matches traditional EDA practices.
    """
    config = state.get("config", DEFAULT_CONFIG)
    df = state["df"]
    column_metadata = state["column_metadata"]
    top_correlations = state.get("top_correlations", [])
    decision_trace = state.get("decision_trace", [])
    
    dataset_name = config.get("dataset_name", "dataset")
    groq_api_key = config.get("groq_api_key")
    
    # Extract target info for Hybrid Architecture
    target_col = None
    if state.get("target_columns") and len(state["target_columns"]) > 0:
        target_col = state["target_columns"][0]
    
    try:
        # Call the agentic agent
        result = get_agentic_plot_recommendations(
            df=df,
            column_metadata=column_metadata,
            top_correlations=top_correlations,
            groq_api_key=groq_api_key,
            dataset_name=dataset_name,
            target_col=target_col
        )
        
        # Log agentic decision
        decision_trace.append({
            "action": "agentic_selection",
            "llm_used": result.llm_used,
            "fallback_used": result.fallback_used,
            "total_plots": result.total_plots,
            "strategy": result.overall_strategy
        })
        
        # Add reasoning trace
        for trace in result.reasoning_trace:
            decision_trace.append({
                "action": "agentic_reasoning",
                **trace
            })
        
        # Log skipped columns
        for skip in result.skipped_columns:
            decision_trace.append({
                "action": "agentic_skipped",
                "column": skip.get("column"),
                "reason": skip.get("reason")
            })
        
        state["selected_plots"] = result.selected_plots
        state["decision_trace"] = decision_trace
        
        # === DATA-DRIVEN VALIDATION ===
        # Filter scatter plots that have no actual correlation (LLM hallucinations)
        correlation_matrix = state.get("correlation_matrix")
        if correlation_matrix is not None:
            validated_plots = []
            for plot in state["selected_plots"]:
                if plot.get("plot_type") in ["scatter", "scatter_plot", "scatterplot"]:
                    col1 = plot.get("column1")
                    col2 = plot.get("column2")
                    
                    # Check if both columns exist in correlation matrix
                    if col1 in correlation_matrix.columns and col2 in correlation_matrix.columns:
                        corr_value = abs(correlation_matrix.loc[col1, col2])
                        
                        # Only keep scatter if correlation is meaningful (|r| > 0.8)
                        if corr_value >= 0.8:
                            plot["reason"] = f"Strong correlation r={corr_value:.2f}"
                            validated_plots.append(plot)
                            print(f"[DEBUG] Scatter {col1} vs {col2}: KEPT (r={corr_value:.2f})")
                        else:
                            print(f"[DEBUG] Scatter {col1} vs {col2}: FILTERED (r={corr_value:.2f} < 0.8)")
                    else:
                        # Can't validate, skip this scatter
                        print(f"[DEBUG] Scatter {col1} vs {col2}: FILTERED (not in correlation matrix)")
                else:
                    # Not a scatter plot, keep it
                    validated_plots.append(plot)
            
            state["selected_plots"] = validated_plots
            print(f"[DEBUG] After scatter validation: {len(validated_plots)} plots")
        
        # === AGGRESSIVE PLOT AUGMENTATION ===
        # If agentic agent returned too few plots, augment to reach target
        MIN_PLOTS = 20  # Target minimum for comprehensive EDA
        current_plots = len(state["selected_plots"])
        
        print(f"[DEBUG] Agentic returned {current_plots} plots, target: {MIN_PLOTS}")
        
        if current_plots < MIN_PLOTS:
            print(f"[DEBUG] Augmenting: need {MIN_PLOTS - current_plots} more plots")
            
            # Get already plotted columns
            plotted = set()
            for p in state["selected_plots"]:
                if "column" in p: plotted.add(p["column"])
                if "column1" in p: plotted.add(p["column1"])
                if "column2" in p: plotted.add(p["column2"])
            
            # Add plots for eligible columns not yet plotted
            eligible = state.get("eligible_plots", {})
            added = 0
            needed = MIN_PLOTS - current_plots
            
            for col, info in column_metadata.items():
                if added >= needed:
                    break
                if col in plotted:
                    continue
                    
                # Skip ID-like columns
                col_lower = col.lower()
                if any(x in col_lower for x in ["id", "index", "pid", "order"]):
                    continue
                    
                # Add appropriate plot type
                if info.get("is_numeric"):
                    # Check if discrete categorical (few unique values)
                    if info.get("is_discrete_categorical") or info.get("unique_count", 100) <= 12:
                        state["selected_plots"].append({
                            "column": col,
                            "plot_type": "barplot",
                            "category": "univariate_numeric",
                            "priority_score": 4,
                            "reason": f"Augmented: discrete numeric distribution"
                        })
                    else:
                        state["selected_plots"].append({
                            "column": col,
                            "plot_type": "histogram",
                            "category": "univariate_numeric",
                            "priority_score": 4,
                            "reason": f"Augmented: numeric distribution"
                        })
                    plotted.add(col)
                    added += 1
                    
                    # Also add boxplot if still need more
                    if added < needed:
                        state["selected_plots"].append({
                            "column": col,
                            "plot_type": "boxplot",
                            "category": "univariate_numeric",
                            "priority_score": 3,
                            "reason": f"Augmented: outlier visualization"
                        })
                        added += 1
                        
                elif info.get("is_categorical"):
                    state["selected_plots"].append({
                        "column": col,
                        "plot_type": "barplot",
                        "category": "univariate_categorical",
                        "priority_score": 4,
                        "reason": f"Augmented: categorical distribution"
                    })
                    plotted.add(col)
                    added += 1
            
            print(f"[DEBUG] Added {added} augmented plots. Total now: {len(state['selected_plots'])}")
        
        # Store agentic mode info
        state["agentic_result"] = {
            "llm_used": result.llm_used,
            "overall_strategy": result.overall_strategy,
            "skipped_columns": result.skipped_columns,
            # Captured Senior Analyst Insights
            "feature_ranking": getattr(result, "feature_ranking", []),
            "feature_engineering": getattr(result, "feature_engineering", []),
            "modeling_implications": getattr(result, "modeling_implications", []),
            "key_insights": getattr(result, "key_insights", [])
        }
        
        return state
        
    except Exception as e:
        # Fall back to rule-based on error
        decision_trace.append({
            "action": "agentic_error",
            "error": str(e),
            "fallback": "Using rule-based selection"
        })
        state["decision_trace"] = decision_trace
        return select_plots_rule_based(state)


def select_plots_rule_based(state: VisualizationState) -> VisualizationState:
    """
    RULE-BASED MODE: Original logic for plot selection.
    Used when agentic mode is disabled or fails.
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


def select_plots_with_budget(state: VisualizationState) -> VisualizationState:
    """
    Node 5: Main plot selection function.
    
    Routes to either:
    - AGENTIC MODE: LLM decides plots (like a data analyst)
    - RULE-BASED MODE: Fixed logic based on data characteristics
    
    Config option: use_agentic_mode (default: True)
    """
    config = state.get("config", DEFAULT_CONFIG)
    use_agentic = config.get("use_agentic_mode", True)
    decision_trace = state.get("decision_trace", [])
    
    # DEBUG: Which mode are we using?
    print(f"[DEBUG] AGENTIC_AGENT_AVAILABLE: {AGENTIC_AGENT_AVAILABLE}")
    print(f"[DEBUG] use_agentic config: {use_agentic}")
    
    # Check if agentic mode is available and enabled
    if use_agentic and AGENTIC_AGENT_AVAILABLE:
        print("[DEBUG] Using AGENTIC mode")
        decision_trace.append({
            "action": "mode_selection",
            "mode": "AGENTIC",
            "reason": "LLM-driven plot selection enabled"
        })
        state["decision_trace"] = decision_trace
        return select_plots_agentic(state)
    else:
        reason = "Agentic agent not available" if not AGENTIC_AGENT_AVAILABLE else "Agentic mode disabled in config"
        print(f"[DEBUG] Using RULE_BASED mode: {reason}")
        decision_trace.append({
            "action": "mode_selection", 
            "mode": "RULE_BASED",
            "reason": reason
        })
        state["decision_trace"] = decision_trace
        return select_plots_rule_based(state)


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
    graph.add_node("process_edge_cases", process_edge_cases)  # LLM edge case handling
    graph.add_node("compute_correlations", compute_correlation_matrix)
    graph.add_node("compute_scores", compute_priority_scores)
    graph.add_node("determine_eligibility", determine_plot_eligibility)
    graph.add_node("select_plots", select_plots_with_budget)
    graph.add_node("generate_report", generate_visualization_report)
    
    graph.set_entry_point("extract_metadata")
    
    graph.add_edge("extract_metadata", "process_edge_cases")  # New edge
    graph.add_edge("process_edge_cases", "compute_correlations")  # New edge
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
        # Edge case fields
        "edge_case_results": None,
        "columns_to_skip": [],
        "column_type_overrides": {},
        "target_columns": [],
        "datetime_columns": []
    }
    
    # Run pipeline steps (with edge case processing)
    viz_state = extract_column_metadata(viz_state)
    print(f"[DEBUG] After metadata: {len(viz_state['column_metadata'])} columns")
    
    viz_state = process_edge_cases(viz_state)  # Edge case handling
    viz_state = compute_correlation_matrix(viz_state)
    viz_state = compute_priority_scores(viz_state)
    viz_state = determine_plot_eligibility(viz_state)
    
    print(f"[DEBUG] Eligible plots: {len([e for e in viz_state['eligible_plots'].values() if e.get('is_eligible')])}")
    
    viz_state = select_plots_with_budget(viz_state)
    print(f"[DEBUG] Selected plots after budget: {len(viz_state['selected_plots'])}")
    
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
    # Edge case results
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
