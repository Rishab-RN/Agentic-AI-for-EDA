# =====================================================
# LLM Utilities for Outlier Module
# =====================================================

import numpy as np
from typing import Dict, List, Optional

# Try to import LLM dependencies
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


# =====================================================
# SIMULATED LLM STRATEGIES (Fallback)
# =====================================================

def simulate_llm_column_strategy(column_name: str, stats: Dict) -> str:
    """
    Simulated LLM planner for outlier detection method selection.
    Returns: 'zscore', 'iqr', 'ml_model', or 'ignore'
    """
    skew = stats.get("skew", 0)
    unique_vals = stats.get("unique", 0)
    outlier_pct = stats.get("outlier_pct", 0)

    # Skip identifier columns
    if unique_vals <= 2 or "id" in column_name.lower():
        return "ignore"

    # Normal distribution -> Z-score
    if abs(skew) < 0.5 and outlier_pct < 0.1:
        return "zscore"

    # Skewed distribution -> IQR
    if abs(skew) >= 0.5 and outlier_pct < 0.2:
        return "iqr"

    # High outlier percentage -> ML model
    if outlier_pct >= 0.2:
        return "ml_model"

    return "iqr"


# =====================================================
# COLUMN SEMANTIC CLASSIFICATION
# =====================================================

def classify_column_semantics(column_name: str, sample_values: List[str], stats: Dict) -> str:
    """
    Classify column semantic type based on name, samples, and stats.
    Returns: VERSION, DATE, ID_CODE, NUMERIC_BOUNDED, NUMERIC_MEASURE
    """
    skew = stats.get("skew", 0)
    unique_ratio = stats.get("unique_ratio", 0)

    # Version-like (many dots)
    if any("." in str(v) for v in sample_values):
        dot_count = sum(str(v).count(".") for v in sample_values) / max(len(sample_values), 1)
        if dot_count > 0.5:
            return "VERSION"

    # Date-like
    if any("-" in str(v) or "/" in str(v) for v in sample_values):
        return "DATE"

    # ID-like (very unique)
    if unique_ratio > 0.9:
        return "ID_CODE"

    # Numeric bounded (ratings, percentages)
    min_val = stats.get("min", 0)
    max_val = stats.get("max", float('inf'))
    
    if min_val >= 0 and max_val <= 10:
        return "NUMERIC_BOUNDED"
    
    if min_val >= 0 and max_val <= 100:
        return "NUMERIC_PERCENTAGE"

    return "NUMERIC_MEASURE"


# =====================================================
# MAIN INTENT CLASSIFIER
# =====================================================

def classify_intent(column_name: str, samples: List[str], stats: Dict) -> str:
    """
    Main intent classification function.
    Uses LLM if available, otherwise falls back to heuristics.
    
    Returns: TIME, VERSION, ID, BOUNDED, PERCENTAGE, CURRENCY, COUNT, MEASURE
    """
    col_lower = column_name.lower()
    
    # High-priority column name patterns
    if any(kw in col_lower for kw in ["_id", "id", "code", "sku", "uuid", "key"]):
        return "ID"
    if any(kw in col_lower for kw in ["date", "time", "timestamp", "created", "updated"]):
        return "TIME"
    if any(kw in col_lower for kw in ["version", "ver", "release"]):
        return "VERSION"
    if any(kw in col_lower for kw in ["percent", "pct", "rate", "ratio"]):
        return "PERCENTAGE"
    if any(kw in col_lower for kw in ["price", "cost", "amount", "revenue", "salary"]):
        return "CURRENCY"
    if any(kw in col_lower for kw in ["rating", "score", "grade", "stars"]):
        return "BOUNDED"
    if any(kw in col_lower for kw in ["count", "num", "total", "quantity", "installs", "downloads"]):
        return "COUNT"
    
    # Sample content detection
    
    # Date patterns
    date_keywords = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    if any("-" in str(v) or "/" in str(v) for v in samples):
        if any(any(m in str(v).lower() for m in date_keywords) for v in samples):
            return "TIME"
    
    # Version patterns (many dots)
    dot_ratio = sum(str(v).count(".") for v in samples) / max(len(samples), 1)
    if dot_ratio > 0.8:
        return "VERSION"
    
    # High uniqueness -> ID
    if stats.get("unique_ratio", 0) > 0.9:
        return "ID"
    
    # Range-based
    min_val = stats.get("min", 0)
    max_val = stats.get("max", float('inf'))
    
    # Rating range (0-5 or 0-10)
    if min_val >= 0 and max_val <= 10:
        return "BOUNDED"
    
    # Percentage range
    if min_val >= 0 and max_val <= 100:
        return "PERCENTAGE"
    
    return "MEASURE"


# =====================================================
# LLM-BASED CLASSIFICATION
# =====================================================

def classify_intent_with_llm(column_name: str, samples: List[str], stats: Dict) -> Optional[str]:
    """
    Use LLM for intent classification if available.
    Returns None if LLM not available or fails.
    """
    if not LLM_AVAILABLE:
        return None
    
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        
        prompt = f"""Classify this data column for outlier treatment.

Column: {column_name}
Samples: {samples[:8]}
Stats: min={stats.get('min')}, max={stats.get('max')}, unique_ratio={stats.get('unique_ratio', 0):.2f}

Return ONLY one label:
TIME, VERSION, ID, BOUNDED, PERCENTAGE, CURRENCY, COUNT, MEASURE"""

        response = llm.invoke(prompt)
        result = response.content.strip().upper()
        
        valid = ["TIME", "VERSION", "ID", "BOUNDED", "PERCENTAGE", "CURRENCY", "COUNT", "MEASURE"]
        return result if result in valid else None
        
    except Exception:
        return None


def classify_intent_smart(column_name: str, samples: List[str], stats: Dict, use_llm: bool = False) -> str:
    """
    Smart intent classifier - uses LLM if enabled and available, otherwise heuristics.
    """
    if use_llm:
        llm_result = classify_intent_with_llm(column_name, samples, stats)
        if llm_result:
            return llm_result
    
    return classify_intent(column_name, samples, stats)
