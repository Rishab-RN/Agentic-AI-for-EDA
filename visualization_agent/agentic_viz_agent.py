# =====================================================
# Agentic Visualization Agent
# =====================================================
# LLM-driven plot selection that matches traditional EDA
# The agent DECIDES which plots to generate based on data
# =====================================================

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .local_intel import LocalSemanticEngine

# ... (imports)


# =====================================================
# 1. AGENTIC PROMPT TEMPLATES (Senior Manager Persona)
# =====================================================

AGENTIC_SYSTEM_PROMPT = """You are a SENIOR DATA SCIENCE MANAGER. You don't just generate plots - you provide STRATEGIC INTELLIGENCE.

## 🧠 MANAGER PERSONA
- **Strategic**: You care about *why* a feature matters to the business/domain.
- **Decisive**: You give clear recommendations (Keep/Drop/Transform).
- **Communication**: You speak in bullet points, utilizing "Business Impact" language.
- **Grounded**: You NEVER guess. You rely on the STATISTICAL FACTS provided in the context.

## 🎯 AGENTIC WORKFLOW

### PHASE 1: OBSERVE
- Check Domain & Target.
- Identify Data Quality Risks (Zero-Inflation, Missingness).

### PHASE 2: THINK
- What is the *story* here? 
- If this is Housing, is it about "Size vs Quality"?
- If this is Churn, is it about "Tenure vs Contract"?

### PHASE 3: PLAN
- Prioritize the TOP 20% of features that drive 80% of value.
- Ensure the Target Variable is the "Star" of the show.

### PHASE 4: ACT
- Generate a Visualization Plan that guides the junior analysts (the code).
- Write "Key Insights" that I can copy-paste into an executive slide deck.

## ⚠️ BEHAVIORAL RULES
1. **No "Maybe"**: Don't say "It might be useful". Say "High Priority".
2. **Context First**: Use the "Statistical Facts" provided to you.
3. **Business Value**: A plot of "ID vs Row Index" has ZERO business value. Block it.
"""

AGENTIC_DECISION_PROMPT = """Analyze this dataset as a SENIOR DATA SCIENCE MANAGER.

## Dataset Context:
- Name: {dataset_name}
- Dimensions: {row_count} rows x {col_count} cols

## Statistical Profile (Verified Facts):
{column_metadata}

## Ground Truth (From Symbolic Engine):
{correlations}

## Proposed Feature Engineering (Heuristics):
{feature_engineering_ideas}

## MANAGER OUTPUT REQUIREMENTS:

Your output must be **EXECUTIVE GRADE**. 

1. **Feature Ranking**: Rank top 10 drivers. Explain *Why* in business terms.
2. **Feature Engineering**: Evaluate the "Proposed" ideas above. If good, recommend them. Add your own.
3. **Data Strategy**: Firm handling of messy data (Zero-inflation, Outliers).
4. **Modeling Implications**: Explicitly provide "Next Steps" for the modeling team (e.g. "Tree-based models preferred due to non-linearity").

## JSON Response Format:
{{
    "observe": {{
        "domain": "survival|housing|finance|...",
        "target_variable": "column_name or null",
        "dataset_size": "SMALL|MEDIUM|LARGE",
        "numeric_columns": N,
        "categorical_columns": N,
        "zero_inflated_columns": ["col1"],
        "data_quality_issues": ["issue1"]
    }},
    "think": {{
        "key_questions": ["What drives prices?"],
        "executive_summary": "The data suggests..."
    }},
    "plan": {{
        "target_plot_count": 25,
        "reasoning": "Focus on high-value drivers",
        "priority_features": ["col1"],
        "target_analysis_percentage": 40
    }},
    "feature_ranking": [
        {{"feature": "OverallQual", "rank": 1, "reason": "Primary driver of asset value"}}
    ],
    "feature_engineering": [
        "Create 'TotalLuxuryScore'..."
    ],
    "modeling_implications": [
        "Log-transform 'SalePrice'..."
    ],
    "column_roles": {{ "col": "ROLE" }},
    "visualization_plan": [
        {{
            "category": "target_analysis",
            "plot_type": "grouped_boxplot",
            "columns": ["col1", "target"],
            "analytical_question": "Does X drive Y?",
            "priority": 10
        }}
    ],
    "key_insights": [
        "**Strategic Driver**: Construction Quality outweighs sheer size.",
        "**Risk Alert**: 15% of properties are outliers.",
        "**Opportunity**: Remodeled homes perform like new."
    ],
    "skipped_columns": [],
    "total_plots": N
}}
"""

# =====================================================
# 2. METADATA FORMATTER
# =====================================================

def format_column_metadata_for_llm(column_metadata: Dict[str, Dict], max_columns: int = 50) -> str:
    formatted = []
    for i, (col, meta) in enumerate(list(column_metadata.items())[:max_columns]):
        col_info = f"**{col}**\n"
        col_info += f"  - Type: {'Numeric' if meta.get('is_numeric') else 'Categorical'}\n"
        col_info += f"  - Missing: {meta.get('missing_pct', 0):.1f}%\n"
        if meta.get('is_numeric'):
            col_info += f"  - Range: {meta.get('min', 'N/A')} to {meta.get('max', 'N/A')}\n"
            if meta.get('zero_pct', 0) > 0:
                col_info += f"  - Zeros: {meta.get('zero_pct'):.1f}% (Zero-Inflation Risk)\n"
        else:
            col_info += f"  - Unique: {meta.get('cardinality', 'N/A')}\n"
        formatted.append(col_info)
    return "\n".join(formatted)

def format_correlations_for_llm(top_correlations: List[Dict], max_pairs: int = 10) -> str:
    if not top_correlations: return "No significant correlations."
    lines = []
    for corr in top_correlations[:max_pairs]:
        lines.append(f"- {corr['column1']} vs {corr['column2']}: r = {corr['correlation']:.3f}")
    return "\n".join(lines)

# =====================================================
# 3. DATA STRUCTURES
# =====================================================

@dataclass
class PlotDecision:
    plot_type: str
    priority: int
    reason: str
    column: Optional[str] = None
    column1: Optional[str] = None
    column2: Optional[str] = None
    columns: Optional[List[str]] = None

@dataclass 
class AgentDecisionResult:
    selected_plots: List[Dict]
    skipped_columns: List[Dict]
    overall_strategy: str
    reasoning_trace: List[Dict]
    total_plots: int
    llm_used: bool = True
    fallback_used: bool = False
    feature_ranking: List[Dict] = field(default_factory=list)
    feature_engineering: List[str] = field(default_factory=list)
    modeling_implications: List[str] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)

class AgenticVisualizationAgent:
    """
    LLM-driven visualization agent that decides which plots to generate.
    
    The agent analyzes dataset metadata and decides what a professional
    data analyst would typically generate for EDA.
    """
    
    def __init__(self, groq_api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.client = None
        self.reasoning_trace = []
        self.slm = LocalSemanticEngine()  # Hybrid Architecture
        
        # Initialize Groq client
        if GROQ_AVAILABLE:
            api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
            if api_key:
                self.client = Groq(api_key=api_key)
    
    # ...
    
    def decide_plots(
        self,
        df: pd.DataFrame,
        column_metadata: Dict[str, Dict],
        top_correlations: List[Dict],
        dataset_name: str = "uploaded_dataset",
        target_col: Optional[str] = None
    ) -> AgentDecisionResult:
        """
        Use LLM to decide which plots to generate.
        """
        self.reasoning_trace = []
        
        # Log start
        self._log_reasoning("analysis_start", {
            "dataset": dataset_name,
            "rows": len(df),
            "columns": len(df.columns)
        })
        
        # Try LLM decision
        if self.is_available():
            try:
                result = self._llm_decide(df, column_metadata, top_correlations, dataset_name, target_col)
                self._log_reasoning("llm_decision_complete", {
                    "total_plots": result.total_plots
                })
                return result
            except Exception as e:
                self._log_reasoning("llm_error", {"error": str(e)})
                print(f"[WARN] LLM decision failed: {e}. Using fallback.")
        
        # Fallback to rule-based
        return self._fallback_decide(df, column_metadata, top_correlations)
    
    def _llm_decide(
        self,
        df: pd.DataFrame,
        column_metadata: Dict[str, Dict],
        top_correlations: List[Dict],
        dataset_name: str,
        target_col: Optional[str] = None
    ) -> AgentDecisionResult:
        """Use LLM to make plot decisions."""
        
        # 1. Run SLM first (The "Symbolic" Step)
        # This provides the GROUND TRUTH for the LLM
        slm_result = self.slm.analyze_dataset(df, target_col=target_col)
        
        # Format metadata for LLM
        col_meta_str = format_column_metadata_for_llm(column_metadata)
        corr_str = format_correlations_for_llm(top_correlations)
        
        # Get FE ideas from SLM
        fe_ideas = slm_result.get("feature_engineering", [])
        fe_str = json.dumps(fe_ideas, indent=2) if fe_ideas else "None suggested by rules."

        # Inject SLM facts into prompt
        slm_context = f"""
## 🧠 STATISTICAL FACTS (Ground Truth - DO NOT HALLUCINATE):
- **Detected Target**: {target_col if target_col else "None provided"}
- **Top 10 Drivers**: {json.dumps([x['feature'] for x in slm_result.get('feature_ranking', [])])}
- **Zero-Inflated Columns**: {json.dumps(slm_result.get('strategy', {}).get('zero_inflated_cols', []))}
- **Recommended Plot Count**: {slm_result.get('strategy', {}).get('target_plot_count', 20)}
"""

        # Build prompt
        prompt = AGENTIC_DECISION_PROMPT.format(
            dataset_name=dataset_name,
            row_count=len(df),
            col_count=len(df.columns),
            column_metadata=col_meta_str,
            correlations=corr_str,
            feature_engineering_ideas=fe_str
        ) + "\n" + slm_context
        
        self._log_reasoning("llm_prompt_sent", {
            "columns_included": len(column_metadata),
            "slm_ranking": len(slm_result.get('feature_ranking', []))
        })
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low for consistency
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        result_json = json.loads(response.choices[0].message.content)
        
        self._log_reasoning("llm_response_received", {
            "strategy": result_json.get("overall_strategy", ""),
            "total_plots": result_json.get("total_plots", 0)
        })
        
        # Convert to standardized format
        return self._parse_llm_response(result_json, column_metadata)
    
    def _parse_llm_response(self, result_json: Dict, column_metadata: Dict = {}) -> AgentDecisionResult:
        """Parse LLM response into standardized format."""
        
        selected_plots = []
        
        # New unified "visualization_plan" format
        if "visualization_plan" in result_json:
            for plot in result_json["visualization_plan"]:
                # Normalize keys
                p = {
                    "plot_type": plot.get("plot_type"),
                    "category": plot.get("category", "general"),
                    "priority_score": plot.get("priority", 5),
                    "reason": plot.get("reason", "LLM recommendation")
                }
                
                # Handle columns
                cols = plot.get("columns", [])
                if isinstance(cols, str): cols = [cols]
                
                if p["plot_type"] in ["scatter", "grouped_boxplot"]:
                    if len(cols) >= 2:
                        p["column1"] = cols[0]
                        p["column2"] = cols[1]
                    elif "column1" in plot:
                        p["column1"] = plot["column1"]
                        p["column2"] = plot.get("column2")
                elif p["plot_type"] in ["heatmap", "pairplot"]:
                    p["columns"] = cols
                else: 
                    # Univariate
                    if cols:
                        p["column"] = cols[0]
                    elif "column" in plot:
                        p["column"] = plot["column"]
                        
                selected_plots.append(p)
                
        else:
            # Fallback for legacy format (just in case LLM hallucinates old schema)
            # Univariate plots
            for plot in result_json.get("univariate_plots", []):
                selected_plots.append({
                    "column": plot.get("column"),
                    "plot_type": plot.get("plot_type"),
                    "category": "univariate_numeric" if plot.get("plot_type") in ["histogram", "boxplot", "violin", "kde"] else "univariate_categorical",
                    "priority_score": plot.get("priority", 5),
                    "reason": plot.get("reason", "LLM recommendation")
                })
            
            # Bivariate plots
            for plot in result_json.get("bivariate_plots", []):
                selected_plots.append({
                    "column1": plot.get("column1"),
                    "column2": plot.get("column2"),
                    "plot_type": plot.get("plot_type", "scatter"),
                    "category": "bivariate",
                    "priority_score": plot.get("priority", 5),
                    "reason": plot.get("reason", "LLM recommendation")
                })
            
            # Multivariate plots
            for plot in result_json.get("multivariate_plots", []):
                selected_plots.append({
                    "plot_type": plot.get("plot_type", "heatmap"),
                    "columns": plot.get("columns", []),
                    "category": "correlation_overview",
                    "reason": plot.get("reason", "LLM recommendation")
                })
        
        # Apply Senior Analyst Filters (Dedup & Zero-Inflation Check)
        selected_plots = self._post_process_plots(selected_plots, column_metadata)
        
        # Sort by priority
        selected_plots.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        
        # === PLOT AUGMENTATION: Enforce minimum plot count ===
        # LLMs often return fewer plots than requested. Augment if needed.
        min_plots = 15  # Minimum acceptable plots
        plots_from_llm = len(selected_plots)
        
        print(f"[DEBUG] LLM returned {plots_from_llm} plots")
        
        if plots_from_llm < min_plots:
            print(f"[DEBUG] Augmenting plots: LLM gave {plots_from_llm}, need at least {min_plots}")
            
            # Get columns already plotted
            plotted_cols = set()
            for p in selected_plots:
                if "column" in p: plotted_cols.add(p["column"])
                if "column1" in p: plotted_cols.add(p["column1"])
                if "column2" in p: plotted_cols.add(p["column2"])
                if "columns" in p: plotted_cols.update(p["columns"])
            
            # Get column roles from LLM response
            column_roles = result_json.get("column_roles", {})
            target_var = result_json.get("target_variable")
            
            # Augment with more univariate plots
            plots_needed = min_plots - plots_from_llm
            added = 0
            
            for col, role in column_roles.items():
                if added >= plots_needed:
                    break
                if col in plotted_cols or col == target_var:
                    continue
                if role in ["ID", "HIGH_MISSING"]:
                    continue
                
                # Add appropriate plot based on role
                if role in ["CATEGORICAL", "COUNT"]:
                    selected_plots.append({
                        "column": col,
                        "plot_type": "barplot",
                        "category": "univariate_categorical",
                        "priority_score": 4,
                        "reason": f"Augmented: {role} column distribution"
                    })
                    added += 1
                elif role in ["CONTINUOUS", "BOUNDED", "NUMERIC"]:
                    selected_plots.append({
                        "column": col,
                        "plot_type": "histogram",
                        "category": "univariate_numeric",
                        "priority_score": 4,
                        "reason": f"Augmented: {role} column distribution"
                    })
                    added += 1
                    
                    # Also add boxplot for outlier visualization
                    if added < plots_needed:
                        selected_plots.append({
                            "column": col,
                            "plot_type": "boxplot",
                            "category": "univariate_numeric",
                            "priority_score": 3,
                            "reason": f"Augmented: Outlier visualization for {col}"
                        })
                        added += 1
            
            print(f"[DEBUG] Added {added} augmented plots. Total now: {len(selected_plots)}")
        
        return AgentDecisionResult(
            selected_plots=selected_plots,
            skipped_columns=result_json.get("skipped_columns", []),
            overall_strategy=result_json.get("analytical_strategy", result_json.get("overall_strategy", "")),
            reasoning_trace=self.reasoning_trace,
            total_plots=len(selected_plots),
            llm_used=True,
            fallback_used=False,
            feature_ranking=result_json.get("feature_ranking", []),
            feature_engineering=result_json.get("feature_engineering", []),
            modeling_implications=result_json.get("modeling_implications", []),
            key_insights=result_json.get("key_insights", [])
        )
    
    def _post_process_plots(self, plots: List[Dict], column_metadata: Dict) -> List[Dict]:
        """Apply Senior Analyst rules: Dedup & Check Zero Inflation.
        
        Rules:
        1. If feature involves target (in another plot), drop its univariate plot.
        2. If feature is zero-inflated (>40%), drop standard histogram.
        """
        final_plots = []
        target_features = set()
        
        # 1. Identify features plotted against target
        for p in plots:
             if p.get("category") == "target_analysis":
                 if "column" in p: target_features.add(p["column"])
                 if "column1" in p: target_features.add(p["column1"]) 
                 if "column2" in p: target_features.add(p["column2"])
        
        dropped_zero = 0
        dropped_dup = 0

        for p in plots:
            p_type = p.get("plot_type", "")
            col = p.get("column")
            
            # RULE: Eliminate Zero-Inflated Histograms per User Feedback
            if p_type == "histogram" and col and col in column_metadata:
                zero_pct = column_metadata[col].get("zero_pct", 0)
                # User specifically complained about >40% zeros being histogrammed
                if zero_pct > 40: 
                    # Drop high zero-inflated histograms to "avoid histogram spam"
                    dropped_zero += 1
                    continue

            # RULE: Dedup univariate if target plot exists (and priority isn't super high)
            if p.get("category") == "univariate_numeric" and col in target_features:
                 # Keep only if explicitly high priority (e.g. 10)
                 if p.get("priority_score", 0) < 9:
                    dropped_dup += 1
                    continue
            
            final_plots.append(p)
            
        if dropped_zero + dropped_dup > 0:
            print(f"[AGENT] Post-process: Dropped {dropped_zero} zero-inflated, {dropped_dup} duplicate plots.")


    def _fallback_decide(
        self,
        df: pd.DataFrame,
        column_metadata: Dict[str, Dict],
        top_correlations: List[Dict]
    ) -> AgentDecisionResult:
        """
        4-PHASE INTELLIGENT EDA SYSTEM
        Acts like a SENIOR DATA SCIENTIST with adaptive decision-making.
        """
        
        # =============================================================
        # PHASE 1: PERCEPTION (Quick Dataset Profiling) [< 1 second]
        # =============================================================
        total_cols = len(column_metadata)
        n_rows = len(df)
        
        # Classify columns
        numeric_cols = []
        categorical_cols = []
        id_cols = []
        high_missing_cols = []
        
        for col, meta in column_metadata.items():
            if meta.get("is_potential_id"):
                id_cols.append(col)
            elif meta.get("missing_pct", 0) > 80:
                high_missing_cols.append(col)
            elif meta.get("is_numeric"):
                numeric_cols.append(col)
            elif meta.get("is_categorical"):
                categorical_cols.append(col)
        
        n_numeric = len(numeric_cols)
        n_categorical = len(categorical_cols)
        n_usable = n_numeric + n_categorical
        
        # Detect target variable
        target_keywords = ['target', 'label', 'class', 'outcome', 'y', 'survived', 
                          'price', 'saleprice', 'revenue', 'churn', 'fraud', 
                          'diagnosis', 'default', 'diabetes']
        target_col = None
        for col in column_metadata.keys():
            if col.lower().replace('_', '') in [k.replace('_', '') for k in target_keywords]:
                target_col = col
                break
        
        # Detect feature groups (for engineered datasets like test_x.csv)
        feature_groups = {}
        for col in numeric_cols:
            # Extract prefix pattern (FFT_Mag_, rolling_mean_, etc.)
            parts = col.split('_')
            if len(parts) >= 2:
                prefix = '_'.join(parts[:2])  # e.g., "FFT_Mag"
                if prefix not in feature_groups:
                    feature_groups[prefix] = []
                feature_groups[prefix].append(col)
        
        is_engineered_dataset = len(feature_groups) > 0 and max(len(g) for g in feature_groups.values()) > 10
        
        self._log_reasoning("phase1_perception", {
            "total_columns": total_cols,
            "numeric": n_numeric,
            "categorical": n_categorical,
            "usable": n_usable,
            "target_variable": target_col,
            "is_engineered_dataset": is_engineered_dataset,
            "feature_groups_detected": len(feature_groups)
        })
        
        # =============================================================
        # PHASE 2: FEATURE INTELLIGENCE (Importance Ranking) [< 5 sec]
        # =============================================================
        
        # Compute feature importance scores
        feature_importance = {}
        
        # Use sampling for large datasets (5000 rows is statistically sufficient)
        if n_rows > 5000:
            sample_df = df.sample(n=5000, random_state=42)
        else:
            sample_df = df
        
        for col in numeric_cols:
            meta = column_metadata.get(col, {})
            score = 0
            
            # Variance contribution (normalized)
            variance = meta.get("variance", 0)
            if variance > 0.01:
                score += 1
            if variance > 1:
                score += 1
            
            # Has outliers = interesting
            if meta.get("has_outliers"):
                score += 1
            
            # Correlation with target (if exists)
            if target_col and col != target_col:
                for corr in top_correlations:
                    if (corr.get("column1") == col and corr.get("column2") == target_col) or \
                       (corr.get("column2") == col and corr.get("column1") == target_col):
                        corr_val = abs(corr.get("correlation", 0))
                        score += corr_val * 5  # Strong weight for target correlation
                        break
            
            # Skewness = interesting distribution
            if abs(meta.get("skewness", 0)) > 1:
                score += 1
            
            feature_importance[col] = score
        
        # Add categorical columns with their cardinality as score proxy
        for col in categorical_cols:
            meta = column_metadata.get(col, {})
            cardinality = meta.get("cardinality", 0)
            # Lower cardinality = more useful for visualization
            if 2 <= cardinality <= 10:
                feature_importance[col] = 3
            elif cardinality <= 20:
                feature_importance[col] = 2
            elif cardinality <= 50:
                feature_importance[col] = 1
            else:
                feature_importance[col] = 0  # Too many categories
        
        # Sort by importance
        ranked_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        self._log_reasoning("phase2_feature_intelligence", {
            "features_ranked": len(ranked_features),
            "top_5_features": [f[0] for f in ranked_features[:5]],
            "sampling_used": n_rows > 5000
        })
        
        # =============================================================
        # PHASE 3: PLANNING (Adaptive Strategy) 
        # =============================================================
        
        # Determine dataset size category and plot strategy
        if n_usable <= 15:
            size_category = "SMALL"
            # Thorough EDA: histogram + boxplot for all numerics, barplot for categoricals
            max_univariate = n_numeric * 2 + n_categorical
            max_bivariate = 8 if target_col else 4
            target_plot_count = max_univariate + max_bivariate + 2  # +2 for heatmap, missing
        elif n_usable <= 50:
            size_category = "MEDIUM"
            # Comprehensive: top 20 features, good coverage
            max_features_to_plot = 20
            max_bivariate = 10 if target_col else 5
            target_plot_count = max_features_to_plot + max_bivariate + 3
        else:
            size_category = "LARGE"
            # Strategic: top 25-30 important features
            max_features_to_plot = 25
            max_bivariate = 12 if target_col else 6
            target_plot_count = max_features_to_plot + max_bivariate + 3
        
        # For engineered datasets, select representatives from each group
        if is_engineered_dataset:
            representative_features = []
            for prefix, group_cols in feature_groups.items():
                if len(group_cols) > 1:
                    # Pick the one with highest importance
                    best_in_group = max(group_cols, key=lambda c: feature_importance.get(c, 0))
                    representative_features.append(best_in_group)
                else:
                    representative_features.extend(group_cols)
            # Use representatives instead of all
            features_to_plot = representative_features[:max_features_to_plot] if size_category != "SMALL" else representative_features
        else:
            # Use top features by importance
            features_to_plot = [f[0] for f in ranked_features[:target_plot_count]]
        
        # Ensure minimum plots
        target_plot_count = max(target_plot_count, 15)
        
        self._log_reasoning("phase3_planning", {
            "size_category": size_category,
            "target_plot_count": target_plot_count,
            "features_to_visualize": len(features_to_plot),
            "is_engineered": is_engineered_dataset,
            "strategy": f"{size_category} dataset: plotting top {len(features_to_plot)} features"
        })
        
        # =============================================================
        # PHASE 4: EXECUTION (Generate plots for TOP features only)
        # =============================================================
        selected_plots = []
        skipped_columns = []
        
        # ID patterns for filtering
        id_patterns = ['id', 'index', 'unnamed', 'order', 'pid', 'customerid', 
                      'userid', 'orderid', 'transactionid', 'rowid']
        
        # Generate plots ONLY for features_to_plot (not all columns)
        for col in features_to_plot:
            meta = column_metadata.get(col, {})
            
            is_target = (col == target_col)
            importance = feature_importance.get(col, 0)
            base_priority = 10 if is_target else min(importance + 3, 9)
            
            if meta.get("is_numeric"):
                # === FIX #2: Check if discrete numeric should use barplot ===
                if meta.get("should_use_barplot") or meta.get("is_discrete_categorical"):
                    # Discrete/ordinal: use barplot (e.g., OverallQual 1-10)
                    selected_plots.append({
                        "column": col,
                        "plot_type": "barplot",
                        "category": "univariate_categorical",
                        "priority_score": base_priority + 1,
                        "reason": f"Discrete values for {'TARGET: ' if is_target else ''}{col} ({meta.get('unique_count', 'N/A')} unique)"
                    })
                else:
                    # Continuous: use histogram
                    selected_plots.append({
                        "column": col,
                        "plot_type": "histogram",
                        "category": "univariate_numeric",
                        "priority_score": base_priority + 1,
                        "reason": f"Distribution of {'TARGET: ' if is_target else ''}{col}"
                    })
                
                # Boxplot for numeric with outliers (always useful)
                if meta.get("has_outliers") or is_target:
                    selected_plots.append({
                        "column": col,
                        "plot_type": "boxplot",
                        "category": "univariate_numeric",
                        "priority_score": base_priority,
                        "reason": f"Boxplot for {col}"
                    })
            
            elif meta.get("is_categorical"):
                cardinality = meta.get("cardinality", 0)
                if cardinality <= 50:
                    plot_type = "pie" if cardinality == 2 else "barplot"
                    selected_plots.append({
                        "column": col,
                        "plot_type": plot_type,
                        "category": "univariate_categorical",
                        "priority_score": base_priority,
                        "reason": f"Categories for {col} ({cardinality} values)"
                    })
        
        # Target vs TOP features (bivariate analysis)
        if target_col:
            top_numeric_features = [col for col in features_to_plot 
                                   if column_metadata.get(col, {}).get("is_numeric") 
                                   and col != target_col][:8]  # Max 8 bivariate plots
            
            for col in top_numeric_features:
                selected_plots.append({
                    "columns": [col, target_col],
                    "plot_type": "grouped_boxplot",
                    "category": "target_analysis",
                    "priority_score": 8,
                    "reason": f"How {col} relates to {target_col}"
                })
        
        # Correlation heatmap (for top numeric features only, not all 800+)
        top_numeric_for_heatmap = [col for col in features_to_plot 
                                   if column_metadata.get(col, {}).get("is_numeric")][:20]
        if len(top_numeric_for_heatmap) >= 3:
            selected_plots.append({
                "plot_type": "heatmap",
                "columns": top_numeric_for_heatmap,
                "category": "correlation_overview",
                "priority_score": 9,
                "reason": f"Correlation heatmap of top {len(top_numeric_for_heatmap)} features"
            })
        
        # Scatter plots for strong correlations (only from selected features)
        correlation_threshold = 0.5 if len(top_correlations) > 10 else 0.3
        valid_correlations = [
            c for c in top_correlations 
            if c.get("correlation") is not None 
            and not pd.isna(c.get("correlation"))
            and abs(c.get("correlation", 0)) >= correlation_threshold
            and c.get("column1") in features_to_plot 
            and c.get("column2") in features_to_plot
        ][:5]
        
        for corr in valid_correlations:
            selected_plots.append({
                "column1": corr["column1"],
                "column2": corr["column2"],
                "plot_type": "scatter",
                "category": "bivariate",
                "priority_score": 7,
                "reason": f"Correlation r={corr.get('correlation', 0):.2f}"
            })
        
        # Missing value visualization if applicable
        if high_missing_cols:
            selected_plots.append({
                "plot_type": "missing_value",
                "category": "data_quality",
                "priority_score": 6,
                "reason": f"{len(high_missing_cols)} columns have >80% missing"
            })
        
        # === FIX #4: PROPER PLOT COUNT HANDLING WITH DEBUG ===
        # Sort by priority
        selected_plots.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        
        # DEBUG: Log plot counts at each stage
        print(f"[DEBUG] PHASE 4 - Plots before limit: {len(selected_plots)}")
        print(f"[DEBUG] Target plot count: {target_plot_count}")
        print(f"[DEBUG] Features processed: {len(features_to_plot)}")
        
        # Ensure we generate at least 20 plots for any meaningful dataset
        min_plots = 20 if n_usable >= 10 else 12
        final_plot_count = max(len(selected_plots), min_plots)
        
        # Cap at target but never below minimum
        final_plot_count = min(final_plot_count, target_plot_count) if target_plot_count > min_plots else min_plots
        selected_plots = selected_plots[:final_plot_count]
        
        print(f"[DEBUG] Final plots generated: {len(selected_plots)}")
        
        self._log_reasoning("phase4_execution", {
            "plots_before_limit": len(selected_plots),
            "plots_generated": len(selected_plots),
            "target_was": target_plot_count,
            "min_plots": min_plots,
            "features_processed": len(features_to_plot)
        })
        
        strategy = (f"INTELLIGENT 4-PHASE EDA: "
                   f"{size_category} dataset ({total_cols} columns, {n_usable} usable). "
                   f"Selected top {len(features_to_plot)} features by importance. "
                   f"Generated {len(selected_plots)} visualizations.")
        if target_col:
            strategy = f"Target: {target_col}. " + strategy
        if is_engineered_dataset:
            strategy += f" Detected {len(feature_groups)} feature groups (engineered dataset)."
        
        return AgentDecisionResult(
            selected_plots=selected_plots,
            skipped_columns=skipped_columns,
            overall_strategy=strategy,
            reasoning_trace=self.reasoning_trace,
            total_plots=len(selected_plots),
            llm_used=False,
            fallback_used=True
        )
    
    def _log_reasoning(self, action: str, details: Dict):
        """Log reasoning step."""
        self.reasoning_trace.append({
            "action": action,
            "details": details
        })


# =====================================================
# 4. MAIN ENTRY POINT
# =====================================================

def get_agentic_plot_recommendations(
    df: pd.DataFrame,
    column_metadata: Dict[str, Dict],
    top_correlations: List[Dict],
    groq_api_key: Optional[str] = None,
    dataset_name: str = "dataset",
    target_col: Optional[str] = None
) -> AgentDecisionResult:
    """
    Main entry point for agentic plot recommendations.
    
    Args:
        df: DataFrame to analyze
        column_metadata: Pre-computed column metadata
        top_correlations: Top correlated pairs
        groq_api_key: Optional Groq API key
        dataset_name: Name of the dataset
        target_col: Target variable name (optional)
        
    Returns:
        AgentDecisionResult with plot recommendations
    """
    agent = AgenticVisualizationAgent(groq_api_key=groq_api_key)
    return agent.decide_plots(df, column_metadata, top_correlations, dataset_name, target_col)


# =====================================================
# 5. TESTING
# =====================================================

if __name__ == "__main__":
    import sys
    
    # Test with a sample dataframe
    print("=" * 60)
    print("AGENTIC VISUALIZATION AGENT TEST")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    test_df = pd.DataFrame({
        "id": range(1, 101),
        "age": np.random.randint(18, 80, 100),
        "income": np.random.lognormal(10, 1, 100),
        "score": np.random.normal(75, 15, 100),
        "category": np.random.choice(["A", "B", "C", "D"], 100),
        "rating": np.random.choice([1, 2, 3, 4, 5], 100),
        "city": np.random.choice(["NYC", "LA", "Chicago", "Houston", "Phoenix"], 100),
    })
    
    # Simulate metadata
    column_metadata = {}
    for col in test_df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(test_df[col])
        meta = {
            "is_numeric": is_numeric,
            "is_categorical": not is_numeric,
            "missing_pct": 0,
            "cardinality": test_df[col].nunique(),
            "is_high_cardinality": test_df[col].nunique() > 20,
            "is_potential_id": col.lower() == "id",
        }
        if is_numeric:
            meta["min"] = float(test_df[col].min())
            meta["max"] = float(test_df[col].max())
            meta["mean"] = float(test_df[col].mean())
            meta["std"] = float(test_df[col].std())
            meta["has_outliers"] = col == "income"  # income has outliers
            meta["outlier_count"] = 5 if col == "income" else 0
            meta["is_skewed"] = col == "income"
            meta["skewness"] = 1.5 if col == "income" else 0.1
        column_metadata[col] = meta
    
    # Simulate correlations
    top_correlations = [
        {"column1": "age", "column2": "income", "correlation": 0.72},
        {"column1": "score", "column2": "rating", "correlation": 0.65}
    ]
    
    # Run agent
    print("\n[TEST] Running agentic visualization agent...")
    result = get_agentic_plot_recommendations(
        test_df, 
        column_metadata, 
        top_correlations,
        dataset_name="test_dataset"
    )
    
    print(f"\n[RESULT] LLM Used: {result.llm_used}")
    print(f"[RESULT] Fallback Used: {result.fallback_used}")
    print(f"[RESULT] Total Plots: {result.total_plots}")
    print(f"[RESULT] Strategy: {result.overall_strategy}")
    
    print("\n[PLOTS] Selected Plots:")
    for i, plot in enumerate(result.selected_plots, 1):
        ptype = plot.get("plot_type")
        col = plot.get("column") or f"{plot.get('column1')} vs {plot.get('column2')}"
        reason = plot.get("reason", "")
        print(f"  {i}. {ptype.upper()}: {col}")
        print(f"     Reason: {reason}")
    
    print("\n[SKIPPED] Skipped Columns:")
    for skip in result.skipped_columns:
        print(f"  - {skip['column']}: {skip['reason']}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
