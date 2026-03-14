import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MasterAgent:
    """
    Intelligent Master Agent that orchestrates the EDA pipeline.
    Analyzes data and decides which sub-agents need to run.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            "missing_threshold": 0,           
            "outlier_iqr_multiplier": 1.5,    
            "min_outlier_columns": 1,         
        }
        self.decision_trace = []
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze dataset to determine which agents should run.
        
        Returns:
            Analysis results with agents_to_run and skip_reasons
        """
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "dataset_shape": list(df.shape),  # Convert tuple to list
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "numeric_columns": int(len(df.select_dtypes(include=[np.number]).columns)),
            "categorical_columns": int(len(df.select_dtypes(include=['object']).columns)),
            "missing_analysis": {},
            "outlier_analysis": {},
            "agents_to_run": [],
            "agents_skipped": [],
            "skip_reasons": [],
            "decision_trace": []
        }
        
        # === MISSING VALUE ANALYSIS ===
        missing_analysis = self._analyze_missing_values(df)
        analysis["missing_analysis"] = missing_analysis
        
        if missing_analysis["has_missing"]:
            analysis["agents_to_run"].append("missing_value_agent")
            analysis["decision_trace"].append({
                "agent": "missing_value_agent",
                "decision": "RUN",
                "reason": f"Found {missing_analysis['total_missing']} missing values ({missing_analysis['missing_pct']:.2f}%)"
            })
        else:
            analysis["agents_skipped"].append("missing_value_agent")
            analysis["skip_reasons"].append("No missing values detected in dataset")
            analysis["decision_trace"].append({
                "agent": "missing_value_agent",
                "decision": "SKIP",
                "reason": "No missing values detected in dataset"
            })
        
        # === OUTLIER ANALYSIS ===
        outlier_analysis = self._analyze_outliers(df)
        analysis["outlier_analysis"] = outlier_analysis
        
        if outlier_analysis["has_outliers"]:
            analysis["agents_to_run"].append("outlier_agent")
            analysis["decision_trace"].append({
                "agent": "outlier_agent",
                "decision": "RUN",
                "reason": f"Found outliers in {outlier_analysis['columns_with_outliers']} columns"
            })
        else:
            analysis["agents_skipped"].append("outlier_agent")
            analysis["skip_reasons"].append("No outliers detected in numeric columns")
            analysis["decision_trace"].append({
                "agent": "outlier_agent",
                "decision": "SKIP",
                "reason": "No outliers detected in numeric columns"
            })
        
        # === VISUALIZATION AGENT (Always runs) ===
        analysis["agents_to_run"].append("visualization_agent")
        analysis["decision_trace"].append({
            "agent": "visualization_agent",
            "decision": "RUN",
            "reason": "Visualization always runs to provide insights"
        })
        
        self.decision_trace = analysis["decision_trace"]
        return analysis
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in the dataset."""
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        columns_with_missing = df.columns[df.isna().any()].tolist()
        rows_with_missing = df.isna().any(axis=1).sum()
        
        return {
            "has_missing": bool(missing_cells > 0),  # Ensure Python bool
            "total_missing": int(missing_cells),
            "missing_pct": float(round(missing_pct, 2)),
            "columns_affected": int(len(columns_with_missing)),
            "columns_with_missing": columns_with_missing[:10],  # First 10
            "rows_affected": int(rows_with_missing),
            "row_loss_pct": float(round((rows_with_missing / len(df)) * 100, 2)) if len(df) > 0 else 0.0
        }
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Quick outlier scan using IQR method on numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {
                "has_outliers": False,
                "columns_with_outliers": 0,
                "outlier_columns": [],
                "reason": "No numeric columns found"
            }
        
        outlier_columns = []
        multiplier = self.config.get("outlier_iqr_multiplier", 1.5)
        
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) < 4:  # Need enough data for IQR
                continue
            
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # No spread = no outliers
                continue
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = ((values < lower_bound) | (values > upper_bound)).sum()
            if outliers > 0:
                outlier_columns.append({
                    "column": col,
                    "outlier_count": int(outliers),
                    "outlier_pct": round((outliers / len(values)) * 100, 2)
                })
        
        return {
            "has_outliers": bool(len(outlier_columns) >= self.config.get("min_outlier_columns", 1)),
            "columns_with_outliers": int(len(outlier_columns)),
            "outlier_columns": outlier_columns[:10],  # First 10
            "total_numeric_columns": int(len(numeric_cols))
        }
    
    def get_decision_summary(self) -> str:
        """Get human-readable decision summary."""
        if not self.decision_trace:
            return "No analysis performed yet"
        
        lines = ["Master Agent Decision Summary:", "-" * 40]
        for decision in self.decision_trace:
            status = "[RUN]" if decision["decision"] == "RUN" else "[SKIP]"
            lines.append(f"{status} {decision['agent']}: {decision['reason']}")
        
        return "\n".join(lines)


# =====================================================
# STANDALONE TEST
# =====================================================

if __name__ == "__main__":
    # Test with a sample dataset
    test_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if test_file and os.path.exists(test_file):
        df = pd.read_csv(test_file)
        print(f"Loaded: {test_file}")
        print(f"Shape: {df.shape}")
    else:
        # Create test data
        print("Creating test dataset...")
        df = pd.DataFrame({
            "A": [1, 2, None, 4, 5, 100],  # Missing + outlier
            "B": [10, 20, 30, 40, 50, 60],  # Clean
            "C": ["a", "b", None, "d", "e", "f"],  # Missing
        })
    
    # Run master agent
    master = MasterAgent()
    analysis = master.analyze_data(df)
    
    print("\n" + "=" * 60)
    print("MASTER AGENT ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nDataset: {analysis['total_rows']} rows x {analysis['total_columns']} columns")
    print(f"Numeric columns: {analysis['numeric_columns']}")
    print(f"Categorical columns: {analysis['categorical_columns']}")
    
    print(f"\n[MISSING VALUES]")
    ma = analysis["missing_analysis"]
    print(f"  Has missing: {ma['has_missing']}")
    print(f"  Total missing: {ma['total_missing']} ({ma['missing_pct']}%)")
    print(f"  Columns affected: {ma['columns_affected']}")
    
    print(f"\n[OUTLIERS]")
    oa = analysis["outlier_analysis"]
    print(f"  Has outliers: {oa['has_outliers']}")
    print(f"  Columns with outliers: {oa['columns_with_outliers']}")
    
    print(f"\n[DECISIONS]")
    print(f"  Agents to run: {analysis['agents_to_run']}")
    print(f"  Agents skipped: {analysis['agents_skipped']}")
    
    print(f"\n[DECISION TRACE]")
    for trace in analysis["decision_trace"]:
        status = "[RUN]" if trace["decision"] == "RUN" else "[SKIP]"
        print(f"  {status} {trace['agent']}: {trace['reason']}")
    
    print("\n" + "=" * 60)
