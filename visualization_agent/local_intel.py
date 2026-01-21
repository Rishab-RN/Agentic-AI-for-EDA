
# =====================================================
# Local Semantic Intelligence Engine ("The SLM")
# =====================================================
# A specialized "Small Language Model" architecture designed 
# for EDA reasoning without external API dependencies.
#
# ARCHITECTURE LEVEL: 2 (Hybrid Rule-Based + Statistical)
# =====================================================

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional

class LocalSemanticEngine:
    """
    The 'Brain' of the EDA system. 
    Replaces external LLMs for column understanding, feature ranking, and insight generation.
    """
    
    def __init__(self):
        # 1. SEMANTIC DICTIONARIES (The "Weights" of our SLM)
        self.semantic_patterns = {
            "ID": [r"id$", r"^id", r"_id", r"code$", r"key$", r"pk_"],
            "DATE": [r"date", r"time", r"year", r"month", r"day", r"timestamp", r"created_at"],
            "PRICE": [r"price", r"cost", r"amount", r"salary", r"revenue", r"total", r"fare"],
            "GEO": [r"city", r"state", r"zip", r"country", r"lat", r"long", r"address"],
            "CATEGORY": [r"status", r"type", r"class", r"mode", r"gender", r"sex", r"category"],
            "TEXT": [r"desc", r"detail", r"comment", r"text", r"subjec", r"review"]
        }
        
    def analyze_dataset(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """
        Run the full local intelligence pipeline on a dataset.
        """
        # Phase 1: Semantic Column Tagging
        col_tags = {col: self._detect_semantic_type(col, df[col]) for col in df.columns}
        
        # Phase 2: Feature Ranking (if target exists)
        ranking = []
        if target_col and target_col in df.columns:
            ranking = self._rank_features(df, target_col, col_tags)
            
        # Phase 3: Strategic Planning
        strategy = self._generate_strategy(df, col_tags, target_col)
        
        # Phase 3.5: Feature Engineering Proposals (Senior Analyst Level)
        fe_ideas = self._propose_feature_engineering(df, col_tags)
        
        # Phase 4: Insight Generation (Narrative)
        insights = self._generate_narrative(ranking, strategy, fe_ideas)
        
        return {
            "column_tags": col_tags,
            "feature_ranking": ranking,
            "strategy": strategy,
            "feature_engineering": fe_ideas,
            "key_insights": insights
        }

    def _detect_semantic_type(self, col_name: str, series: pd.Series) -> str:
        """
        Classifies a column into a rich semantic type using Rules + Stats.
        Input: Column Data -> Output: Semantic Label
        """
        # 1. Structural Checks
        n_unique = series.nunique()
        n_rows = len(series)
        dtype = str(series.dtype)
        
        # ID Check: High uniqueness, monotonic
        if n_unique == n_rows or (n_unique > 0.98 * n_rows and n_rows > 100):
            # Check name for ID patterns
            if any(re.search(p, col_name.lower()) for p in self.semantic_patterns["ID"]):
                return "ID"
            if dtype.startswith('int') or dtype.startswith('obj'):
                return "POTENTIAL_ID"

        # 2. Semantic Name Matching
        clean_name = col_name.lower()
        for tag, patterns in self.semantic_patterns.items():
            if any(re.search(p, clean_name) for p in patterns):
                # Validation
                if tag == "DATE" and not np.issubdtype(series.dtype, np.number):
                    return "DATE"
                if tag == "PRICE" and np.issubdtype(series.dtype, np.number):
                    return "MONEY"
                    
        # 3. Statistical Profiling
        if np.issubdtype(series.dtype, np.number):
            # Zero Inflation Check (Senior Analyst Threshold)
            # User Feedback: "Mishandled Zero-Inflated Features"
            zero_pct = (series == 0).mean()
            if zero_pct > 0.40:  # STRICTER threshold (was 0.90)
                return "ZERO_INFLATED_BINARY"  # Treat as flag (e.g. HasPool, HasDeck)
                
            # Discrete Number Check
            if n_unique < 15 and n_unique / n_rows < 0.05:
                return "NUMERIC_DISCRETE" # Treat as categorical
                
            return "NUMERIC_CONTINUOUS"
            
        elif n_unique < 50:
            return "CATEGORICAL_LOW_CARDINALITY"
            
        return "TEXT_OR_HIGH_CARDINALITY"

    def _propose_feature_engineering(self, df: pd.DataFrame, tags: Dict) -> List[str]:
        """
        Generates concrete, formula-based feature engineering ideas.
        "Patent Grade" addition: Heuristic-based interaction detection.
        """
        ideas = []
        cols = df.columns
        
        # 1. Date Differences (e.g. YearSold - YearBuilt)
        date_cols = [c for c, t in tags.items() if t == "DATE" or "year" in c.lower()]
        if len(date_cols) >= 2:
            sorted_dates = sorted(date_cols) # Heuristic: usually earlier dates first alphabetically? Not reliable.
            # Compare every pair
            found_age = False
            for i in range(len(date_cols)):
                for j in range(i+1, len(date_cols)):
                    c1, c2 = date_cols[i], date_cols[j]
                    # Check for "Year" in both
                    if "year" in c1.lower() and "year" in c2.lower():
                        ideas.append(f"Calculate Age/Duration: {c1} - {c2}")
                        found_age = True
            if not found_age and len(date_cols) > 0:
                 ideas.append(f"Extract Components (Year, Month) from {date_cols[0]}")

        # 2. Summing Related Metrics (e.g. FullBath + HalfBath)
        # Look for shared prefixes/suffixes
        suffixes = ["sf", "area", "bath", "porch", "score"]
        for suffix in suffixes:
            matches = [c for c in cols if c.lower().endswith(suffix) and np.issubdtype(df[c].dtype, np.number)]
            if len(matches) > 1:
                formula = " + ".join(matches)
                ideas.append(f"Aggregated {suffix.upper()}: {formula}")
                
        # 3. Zero-Inflation Flags
        zero_cols = [c for c, t in tags.items() if t == "ZERO_INFLATED_BINARY"]
        for c in zero_cols[:3]: # Limit to top 3
            ideas.append(f"Create Binary Flag: Has_{c} = ({c} > 0)")
            
        return ideas

    def _rank_features(self, df: pd.DataFrame, target: str, tags: Dict) -> List[Dict]:
        """
        Ranks features by predictive power (Correlation + Variance + Mutual Info).
        """
        scores = []
        target_data = df[target]
        is_target_numeric = np.issubdtype(target_data.dtype, np.number)
        
        for col in df.columns:
            if col == target or tags[col] in ["ID", "POTENTIAL_ID"]:
                continue
                
            score = 0
            reason = []
            
            # Numeric Feature Scoring
            if np.issubdtype(df[col].dtype, np.number):
                # 1. Variance Score (Normalize)
                var = df[col].var()
                if var > 0: score += 1
                
                # 2. Correlation Score (if target numeric)
                if is_target_numeric:
                    corr = df[col].corr(target_data)
                    abs_corr = abs(corr)
                    if abs_corr > 0.1: score += abs_corr * 5
                    if abs_corr > 0.5: reason.append(f"Strong correlation ({corr:.2f})")
                    elif abs_corr > 0.3: reason.append(f"Moderate correlation ({corr:.2f})")
                    
            # Categorical Feature Scoring
            else:
                # Simple cardinality check
                n_unique = df[col].nunique()
                if 1 < n_unique < 50: score += 2  # Good categorical
                
            if score > 0:
                scores.append({
                    "feature": col,
                    "rank_score": score,
                    "reason": "; ".join(reason) if reason else "Statistical importance"
                })
                
        return sorted(scores, key=lambda x: x["rank_score"], reverse=True)[:10]

    def _generate_strategy(self, df: pd.DataFrame, tags: Dict, target: str) -> Dict:
        """
        Decides the 'Plan' phase: How many plots, what types.
        """
        n_cols = len(df.columns)
        
        # Sizing
        size = "LARGE" if n_cols > 50 else "MEDIUM" if n_cols > 15 else "SMALL"
        plot_budget = 30 if size == "LARGE" else 20
        
        plan = {
            "dataset_size": size,
            "target_plot_count": plot_budget,
            "priority_features": [],
            "zero_inflated_cols": [c for c, t in tags.items() if t == "ZERO_INFLATED_BINARY"]
        }
        return plan

    def _generate_narrative(self, ranking: List[Dict], strategy: Dict, fe_ideas: List[str] = []) -> List[str]:
        """
        Generates textual 'Key Insights' using templates.
        """
        insights = []
        
        # 1. Driver Insight
        if ranking:
            top_feat = ranking[0]["feature"]
            insights.append(f"**Primary Driver**: '{top_feat}' shows the strongest statistical relationship with the target.")
            
        # 2. Data Structure Insight
        n_zero = len(strategy["zero_inflated_cols"])
        if n_zero > 0:
            examples = ", ".join(strategy['zero_inflated_cols'][:2])
            insights.append(f"**Data Quality**: {n_zero} columns (e.g., {examples}) are dominated by zeros. Recommend converting to Binary Flags (HasFeature).")
            
        # 3. Model Readiness / FE
        if fe_ideas:
            insights.append(f"**Optimization**: Value can be unlocked by combining features: {fe_ideas[0]}")
             
        return insights

# Example Usage
if __name__ == "__main__":
    # Mock Data
    df = pd.DataFrame({
        "PassengerId": range(1, 101),
        "Survived": np.random.randint(0, 2, 100),
        "Age": np.random.normal(30, 10, 100),
        "Fare": np.random.exponential(50, 100),
        "YearBuilt":  [1990] * 50 + [2000] * 50,
        "YearSold": [2010] * 100,
        "BsmtFullBath": [0] * 80 + [1] * 20, # Zero-inflated
        "BsmtHalfBath": [0] * 90 + [1] * 10 
    })
    
    engine = LocalSemanticEngine()
    result = engine.analyze_dataset(df, target_col="Survived")
    
    print("SLM Analysis Result:")
    print("Tags:", result["column_tags"])
    print("FE Ideas:", result["feature_engineering"])
    print("Insights:", result["key_insights"])
