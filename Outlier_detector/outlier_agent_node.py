# =====================================================
# Agentic Outlier Treatment Module (LangGraph Compatible)
# =====================================================

import pandas as pd
import numpy as np
import re
import json
from typing import TypedDict, Dict, List, Optional, Any
from sklearn.ensemble import IsolationForest

# Try to import LangGraph (optional for standalone use)
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# Try to import LLM (optional for standalone use)
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


# =====================================================
# 1. STATE DEFINITION
# =====================================================

class OutlierState(TypedDict):
    df: pd.DataFrame
    original_shape: tuple
    numeric_analysis: Dict[str, Dict]      # Column stats and analysis
    column_intents: Dict[str, str]         # LLM-classified intents per column
    outlier_report: Dict[str, Dict]        # Detected outliers info
    treatment_log: List[Dict]              # Actions taken
    errors: List[Dict]                     # Any errors encountered
    config: Dict[str, Any]                 # Configuration options


# =====================================================
# 2. CONFIGURATION DEFAULTS
# =====================================================

DEFAULT_CONFIG = {
    "min_valid_samples": 20,      # Lowered for smaller datasets
    "iqr_multiplier": 1.5,        # Standard IQR multiplier
    "zscore_threshold": 2.5,      # More aggressive z-score (was 3.0)
    "digit_threshold": 0.4,       # Lower threshold to catch more numeric columns
    "use_llm": True,              # Enable LLM by default if available
    "aggressive_mode": True,      # Remove outliers from all types (not just cap)
    "process_native_numeric": True,  # Also process already-numeric columns
}


# =====================================================
# 3. ROBUST NUMERIC NORMALIZATION
# =====================================================

# Known non-numeric placeholders
MISSING_TOKENS = {
    "", "nan", "n/a", "null", "--", "##", "unknown", 
    "varies with device", "free", "none", "nil", "na",
    "varies", "varies with", "-", "...", "undefined",
    "not available", "tbd", "n.a.", "n.a"
}

# Suffix multipliers
SUFFIX_MULTIPLIERS = {
    # Standard K/M/B/T (thousands, millions, billions, trillions)
    'k': 1_000, 
    'm': 1_000_000, 
    'b': 1_000_000_000,
    't': 1_000_000_000_000,
    # Byte suffixes
    'kb': 1_024,
    'mb': 1_024 ** 2,
    'gb': 1_024 ** 3,
    'tb': 1_024 ** 4,
    # Weight units (convert to base unit, e.g., kg)
    'kg': 1.0,
    'g': 0.001,
    'lb': 0.453592,
    'lbs': 0.453592,
    'oz': 0.0283495,
    # Length units (convert to base unit, e.g., cm)
    'cm': 1.0,
    'm': 100.0,  # meters to cm - BUT this conflicts with "M" for millions!
    'mm': 0.1,
    'in': 2.54,
    'ft': 30.48,
    # For app stores, "M" usually means millions, not meters
    # We'll prioritize millions unless context says otherwise
}

def normalize_numeric(x) -> Optional[float]:
    """
    Robust numeric parser that handles various real-world formats:
    - K/M/B suffixes (1.5K, 10M, 1B, 50,000+)
    - Byte suffixes (19MB, 1.2GB, 500KB)
    - Percentages (50%)
    - Currency symbols ($500, ₹1000, €50)
    - Commas (1,000,000)
    - Plus signs (1000+, 10M+)
    - Negative numbers (-500)
    - Units (50kg, 100lb, 6ft)
    """
    if pd.isna(x):
        return np.nan
    
    original = str(x)
    x = original.lower().strip()
    
    # Check for known missing tokens
    if x in MISSING_TOKENS:
        return np.nan
    
    # Remove currency symbols
    x = re.sub(r"[$€£¥₹]", "", x)
    
    # Remove commas (thousands separator)
    x = x.replace(",", "")
    
    # Remove plus signs (e.g., "10,000+")
    x = x.replace("+", "")
    
    # Remove parentheses
    x = x.replace("(", "").replace(")", "")
    
    # Remove extra whitespace
    x = x.strip()
    
    # Handle percentages: "50%" -> 50.0
    if x.endswith("%"):
        try:
            return float(x[:-1])
        except ValueError:
            return np.nan
    
    # Priority 1: Handle standalone K/M/B/T suffix (common in app stores)
    # Pattern: number followed by k/m/b/t (case insensitive, already lowered)
    match = re.fullmatch(r"(-?\d+\.?\d*)\s*(k|m|b|t)$", x)
    if match:
        try:
            val = float(match.group(1))
            suffix = match.group(2)
            multiplier = {'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000, 't': 1_000_000_000_000}
            return val * multiplier.get(suffix, 1)
        except ValueError:
            pass
    
    # Priority 2: Handle byte suffixes (KB, MB, GB, TB)
    match = re.fullmatch(r"(-?\d+\.?\d*)\s*(kb|mb|gb|tb|bytes?|b)$", x)
    if match:
        try:
            val = float(match.group(1))
            unit = match.group(2)
            multipliers = {
                "b": 1, "byte": 1, "bytes": 1,
                "kb": 1024, 
                "mb": 1024**2, 
                "gb": 1024**3, 
                "tb": 1024**4
            }
            return val * multipliers.get(unit, 1)
        except ValueError:
            pass
    
    # Priority 3: Handle weight units (kg, lb, g, oz)
    match = re.fullmatch(r"(-?\d+\.?\d*)\s*(kg|g|lb|lbs|oz)$", x)
    if match:
        try:
            val = float(match.group(1))
            # Just return the number, unit conversion is optional
            return val
        except ValueError:
            pass
    
    # Priority 4: Handle length units (cm, m, mm, in, ft)
    match = re.fullmatch(r"(-?\d+\.?\d*)\s*(cm|mm|in|ft|m)$", x)
    if match:
        try:
            val = float(match.group(1))
            return val
        except ValueError:
            pass
    
    # Priority 5: Plain number (with or without decimals, negative support)
    if re.fullmatch(r"-?\d+\.?\d*", x):
        try:
            return float(x)
        except ValueError:
            return np.nan
    
    # NO FALLBACK HERE! 
    # We do NOT want to extract "320" from "BMW 320" - that would corrupt categorical data.
    # If a value doesn't match any known numeric pattern, it should return NaN.
    # The aggressive extraction (extract_aggressive_numeric) is only used AFTER 
    # the LLM confirms a column is DIRTY_NUMERIC, not CATEGORICAL.
    
    return np.nan


def extract_aggressive_numeric(x) -> Optional[float]:
    """
    Fallback parser: Extracts the first valid number from ANY string.
    Useful for 'Weight: 50kg', 'USD 100', 'approx 200'.
    Regex matches: Optional negative sign, digits, optional decimal point, more digits.
    """
    if pd.isna(x):
        return np.nan
    
    x = str(x).lower().strip()
    
    # regex for float/int
    match = re.search(r"(-?\d+(?:\.\d+)?)", x)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
            
    return np.nan
# =====================================================
# 4. COLUMN INTENT CLASSIFICATION
# =====================================================

def classify_intent_heuristic(column_name: str, samples: List[str], stats: Dict) -> str:
    """
    Simplified heuristic-based column intent classification.
    Defaults to MEASURE for aggressive outlier removal.
    Only excludes truly non-measurement columns like IDs, dates, versions.
    """
    col_lower = column_name.lower()
    
    # HIGH PRIORITY: Column name patterns that should be SKIPPED
    # Only skip true identifiers
    if any(kw in col_lower for kw in ["_id", "id", "code", "sku", "uuid", "key", "index", "order"]) and not any(kw in col_lower for kw in ["qual", "cond", "area", "sf", "price"]):
        # Make sure it's actually an ID (very high uniqueness)
        if stats.get("unique_ratio", 0) > 0.95:
            return "ID"
    
    # Time/Date columns - skip
    if any(kw in col_lower for kw in ["date", "time", "timestamp", "created", "updated", "yr", "mo", "day"]) and "year" not in col_lower:
        return "TIME"
    
    # Version columns - skip
    if any(kw in col_lower for kw in ["version", "ver", "release", "build"]):
        dot_ratio = sum(str(v).count(".") for v in samples) / max(len(samples), 1)
        if dot_ratio > 0.5:
            return "VERSION"
    
    # Content-based: check for date patterns
    if any("-" in str(v) or "/" in str(v) for v in samples):
        date_keywords = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        if any(any(m in str(v).lower() for m in date_keywords) for v in samples):
            return "TIME"
    
    # EVERYTHING ELSE is a numeric measure - process for outliers
    # Special hints for treatment style (but all will get outlier removal)
    
    # Currency columns
    if any(kw in col_lower for kw in ["price", "cost", "amount", "revenue", "salary", "income", "fee", "tax", "saleprice"]):
        return "CURRENCY"
    
    # Count columns - DISCRETE BOUNDED, rare values are NOT outliers
    # SibSp=5 is rare but valid, not an error
    if any(kw in col_lower for kw in ["count", "num", "total", "quantity", "installs", "downloads", "views", "bsmt", "sf", "sibsp", "parch", "children", "family"]):
        return "COUNT"
    
    # Bounded/Rating columns (but still remove outliers)
    if any(kw in col_lower for kw in ["rating", "score", "grade", "stars", "qual", "cond"]):
        return "BOUNDED"
    
    # Default: MEASURE - will have outliers removed
    return "MEASURE"


def classify_intent_llm(column_name: str, samples: List[str], stats: Dict) -> str:
    """
    LLM-based column intent classification (if available).
    Falls back to heuristic if LLM not available.
    """
    if not LLM_AVAILABLE:
        return classify_intent_heuristic(column_name, samples, stats)
    
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        
        prompt = f"""Classify this data column's semantic type.

Column name: {column_name}
Sample values: {samples[:10]}
Stats: min={stats.get('min')}, max={stats.get('max')}, unique_ratio={stats.get('unique_ratio'):.2f}

Return ONLY ONE of these labels:
- TIME: dates, timestamps, years
- VERSION: software versions (e.g., 1.0.0)
- ID: unique identifiers, codes, keys
- BOUNDED: ratings, scores with fixed range (e.g., 1-5 stars)
- PERCENTAGE: percentages, ratios (0-100)
- CURRENCY: prices, costs, monetary values
- COUNT: counts, quantities, totals
- MEASURE: general numeric measurements

Return ONLY the label, nothing else."""

        response = llm.invoke(prompt)
        intent = response.content.strip().upper()
        
        valid_intents = ["TIME", "VERSION", "ID", "BOUNDED", "PERCENTAGE", "CURRENCY", "COUNT", "MEASURE"]
        if intent in valid_intents:
            return intent
        
    except Exception:
        pass
    
    return classify_intent_heuristic(column_name, samples, stats)


def classify_mixed_type_llm(column_name: str, numeric_samples: List[Any], text_samples: List[str], stats: Dict) -> str:
    """
    Specific LLM check for suspicious mixed-type columns (high parse failure).
    Distinguishes between dirty numbers (e.g. '1,200', '$50') and categorical (e.g. '320', 'Rav4').
    """
    if not LLM_AVAILABLE:
        return "CATEGORICAL" # Fail safe: if looks mixed and no LLM, assume categorical to protect data
        
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        
        prompt = f"""Analyze this mixed-type column data.
Column: '{column_name}'
Numeric Samples: {numeric_samples[:5]}
Text Samples: {text_samples[:5]}
Data Stats: {stats}

Determine the true semantic type:
1. CATEGORICAL: Values are identifiers/names (e.g. "BMW 320", "Model 3", "A4", "Ford"). 
   - Rule: If values are distinct names/labels, choose CATEGORICAL.
   
2. DIRTY_NUMERIC: Values are measurements but formatted poorly or contain placeholders.
   - Rule: If values contain "M", "k", "+", "$", "," (e.g. "19M", "10,000+", "$500").
   - Rule: If text samples are just missing indicators like "Varies with device", "Unknown", "NaN", ignore them and choose DIRTY_NUMERIC so we can extract the valid numbers.

3. ID: Unique keys/UUIDs.

Return ONLY the label: CATEGORICAL, DIRTY_NUMERIC, or ID.
"""
        response = llm.invoke(prompt)
        intent = response.content.strip().upper()
        if "CATEGORICAL" in intent: return "CATEGORICAL"
        if "ID" in intent: return "ID"
        if "NUMERIC" in intent: return "MEASURE"
        
        return "CATEGORICAL" # Safe fallback
        
    except Exception as e:
        print(f"LLM Mixed Type Error: {e}")
        return "CATEGORICAL"


# =====================================================
# 5. OUTLIER DETECTION METHODS
# =====================================================

def detect_outliers_iqr(values: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """IQR-based outlier detection. Returns boolean mask of outliers."""
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR <= 0:
        return pd.Series(False, index=values.index)
    
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    
    return (values < lower) | (values > upper)


def detect_outliers_zscore(values: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Z-score based outlier detection. Returns boolean mask of outliers."""
    mean = values.mean()
    std = values.std()
    
    if std <= 0:
        return pd.Series(False, index=values.index)
    
    z_scores = np.abs((values - mean) / std)
    return z_scores > threshold


def detect_outliers_isolation_forest(values: pd.Series, contamination: float = 0.05) -> pd.Series:
    """Isolation Forest outlier detection. Returns boolean mask of outliers."""
    valid_mask = values.notna()
    valid_values = values[valid_mask].values.reshape(-1, 1)
    
    if len(valid_values) < 10:
        return pd.Series(False, index=values.index)
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(valid_values)
    
    result = pd.Series(False, index=values.index)
    result.loc[valid_mask] = predictions == -1
    return result


# =====================================================
# 6. LANGGRAPH NODE FUNCTIONS
# =====================================================

def analyze_numeric_columns(state: OutlierState) -> OutlierState:
    """Node 1: Identify and analyze candidate numeric columns."""
    df = state["df"].copy()
    config = {**DEFAULT_CONFIG, **state.get("config", {})}
    numeric_analysis = {}
    errors = state.get("errors", [])
    
    for col in df.columns:
        try:
            parsed = None
            
            # First check if column is already numeric (int/float)
            if df[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int', 'float']:
                if config.get("process_native_numeric", True):
                    parsed = df[col].astype(float)
            else:
                # Check if column has enough numeric content
                digit_ratio = df[col].astype(str).str.contains(r'\d', regex=True).mean()
                
                if digit_ratio < config.get("digit_threshold", 0.4):
                    continue
                
                # Normalize and parse
                raw = df[col].astype(str)
                parsed = raw.map(normalize_numeric)
            
            if parsed is None:
                continue
                
            valid = parsed.dropna()
            
            if len(valid) < config.get("min_valid_samples", 20):
                continue
            
            # Compute statistics
            failed_parse_ratio = 0.0
            nan_before = df[col].isna().sum()
            nan_after = parsed.isna().sum()
            if len(df) > 0:
                failed_parse_ratio = (nan_after - nan_before) / len(df)

            numeric_analysis[col] = {
                "original_dtype": str(df[col].dtype),
                "parsed_count": len(valid),
                "nan_count": parsed.isna().sum(),
                "failed_parse_ratio": failed_parse_ratio,
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "std": float(valid.std()) if valid.std() > 0 else 0.0,
                "unique_ratio": valid.nunique() / len(valid) if len(valid) > 0 else 0,
                "samples": df[col].dropna().head(15).astype(str).tolist(),
                "mixed_samples": df[col][parsed.isna() & df[col].notna()].head(5).astype(str).tolist(), # Capture what failed to parse
                "parsed_series": parsed,
                "valid_indices": valid.index.tolist()  # Store valid indices for proper mapping
            }
            
        except Exception as e:
            errors.append({
                "column": col,
                "stage": "analyze",
                "error": str(e),
                "type": type(e).__name__
            })
    
    state["numeric_analysis"] = numeric_analysis
    state["errors"] = errors
    
    # DEBUG: Print what columns made it into numeric_analysis
    print(f"[DEBUG] Columns in numeric_analysis: {list(numeric_analysis.keys())}")
    if "Size" in numeric_analysis:
        print(f"[DEBUG] Size parsed_series sample: {numeric_analysis['Size']['parsed_series'].head(5).tolist()}")
    else:
        print(f"[DEBUG] Size NOT in numeric_analysis!")
    
    return state


def classify_column_intents(state: OutlierState) -> OutlierState:
    """Node 2: Classify the intent of each numeric column."""
    config = state.get("config", DEFAULT_CONFIG)
    column_intents = {}
    errors = state.get("errors", [])
    
    classify_fn = classify_intent_llm if config.get("use_llm", False) else classify_intent_heuristic
    
    for col, analysis in state["numeric_analysis"].items():
        try:
            stats = {
                "min": analysis["min"],
                "max": analysis["max"],
                "mean": analysis["mean"],
                "std": analysis["std"],
                "unique_ratio": analysis["unique_ratio"]
            }
            
            # === MIXED TYPE SAFETY CHECK ===
            # If significant data loss during parsing (>10%), run safety check
            if analysis.get("failed_parse_ratio", 0) > 0.10:
                print(f"[WARN] Column '{col}' has high parse failure ({analysis['failed_parse_ratio']:.1%}). Checking semantic type...")
                
                # Use LLM to distinguish between 'Model 320' (Categorical) vs '$1,000' (Dirty Numeric)
                mixed_intent = classify_mixed_type_llm(
                    col, 
                    analysis["samples"], 
                    analysis.get("mixed_samples", []),
                    stats
                )
                
                if mixed_intent in ["CATEGORICAL", "ID"]:
                    print(f"[INFO] Column '{col}' classified as {mixed_intent} (Mixed Safety)- SKIPPING outlier detection.")
                    column_intents[col] = mixed_intent
                    continue
                
                elif mixed_intent == "DIRTY_NUMERIC":
                    print(f"[INFO] Column '{col}' classified as DIRTY_NUMERIC. Applying aggresive re-parsing...")
                    new_parsed = df[col].map(extract_aggressive_numeric)
                    new_valid = new_parsed.dropna()
                    
                    # Update analysis with cleaner data if we recovered more
                    if len(new_valid) > len(valid):
                        print(f"[INFO] Aggressive parsing recovered {len(new_valid) - len(valid)} more samples for '{col}'.")
                        state["numeric_analysis"][col]["parsed_series"] = new_parsed
                        state["numeric_analysis"][col]["parsed_count"] = len(new_valid)
                        state["numeric_analysis"][col]["min"] = float(new_valid.min())
                        state["numeric_analysis"][col]["max"] = float(new_valid.max())
                        state["numeric_analysis"][col]["mean"] = float(new_valid.mean())
                        state["numeric_analysis"][col]["std"] = float(new_valid.std()) if new_valid.std() > 0 else 0.0
                        
                        # Update local vars for downstream classification
                        analysis = state["numeric_analysis"][col]

            
            # Normal Classification
            intent = classify_fn(col, analysis["samples"], stats)
            column_intents[col] = intent
            
        except Exception as e:
            errors.append({
                "column": col,
                "stage": "classify",
                "error": str(e),
                "type": type(e).__name__
            })
            column_intents[col] = "MEASURE"  # Default fallback
    
    state["column_intents"] = column_intents
    state["errors"] = errors
    return state


def detect_outliers(state: OutlierState) -> OutlierState:
    """Node 3: Detect outliers based on column intent - AGGRESSIVE MODE."""
    config = {**DEFAULT_CONFIG, **state.get("config", {})}
    outlier_report = {}
    errors = state.get("errors", [])
    iqr_mult = config.get("iqr_multiplier", 1.5)
    
    for col, intent in state["column_intents"].items():
        try:
            parsed = state["numeric_analysis"][col]["parsed_series"]
            valid = parsed.dropna()
            
            # Skip ONLY ID/CATEGORICAL columns - we want to clean everything else
            if intent in ["ID", "CATEGORICAL"]:
                outlier_report[col] = {
                    "intent": intent,
                    "action": "skip",
                    "reason": f"Column type {intent} is identifier/categorical, excluded"
                }
                continue
            
            # TIME and VERSION - skip for removal but still detect
            if intent in ["TIME", "VERSION"]:
                outlier_report[col] = {
                    "intent": intent,
                    "action": "skip",
                    "reason": f"Column type {intent} excluded from outlier detection"
                }
                continue
            
            # COUNT columns - SKIP outlier detection entirely
            # Values like SibSp=5 or Parch=4 are RARE, not ERRORS
            if intent == "COUNT":
                outlier_report[col] = {
                    "intent": intent,
                    "action": "skip",
                    "reason": f"COUNT column - rare discrete values are valid, not outliers"
                }
                continue
            
            # AGGRESSIVE detection for all other types
            if intent == "CURRENCY":
                # Use both IQR and Z-score, take union
                iqr_outliers = detect_outliers_iqr(valid, multiplier=iqr_mult)
                z_outliers = detect_outliers_zscore(valid, threshold=config.get("zscore_threshold", 2.5))
                outlier_mask = iqr_outliers | z_outliers
                method = "IQR + Z-score"
            elif intent in ["BOUNDED", "PERCENTAGE"]:
                # For bounded data - still use IQR to find outliers
                outlier_mask = detect_outliers_iqr(valid, multiplier=iqr_mult)
                method = "IQR"
            else:  # MEASURE, COUNT
                # Standard IQR
                outlier_mask = detect_outliers_iqr(valid, multiplier=iqr_mult)
                method = "IQR"
            
            Q1, Q3 = valid.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower = Q1 - iqr_mult * IQR
            upper = Q3 + iqr_mult * IQR
            
            # Create full-index outlier mask for proper row removal
            full_outlier_mask = pd.Series(False, index=parsed.index)
            full_outlier_mask.loc[valid.index] = outlier_mask.values
            
            outlier_report[col] = {
                "intent": intent,
                "method": method,
                "outlier_count": int(outlier_mask.sum()),
                "outlier_pct": round(outlier_mask.sum() / len(valid) * 100, 2) if len(valid) > 0 else 0,
                "Q1": float(Q1),
                "Q3": float(Q3),
                "IQR": float(IQR),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_mask": full_outlier_mask,  # Full index mask
                "outlier_indices": valid[outlier_mask].index.tolist()  # Explicit indices
            }
            
        except Exception as e:
            errors.append({
                "column": col,
                "stage": "detect",
                "error": str(e),
                "type": type(e).__name__
            })
    
    state["outlier_report"] = outlier_report
    state["errors"] = errors
    return state


def apply_treatment(state: OutlierState) -> OutlierState:
    """Node 4: Apply outlier treatment - Smart switch to Capping if data loss > 15%."""
    df = state["df"].copy()
    treatment_log = []
    errors = state.get("errors", [])
    config = {**DEFAULT_CONFIG, **state.get("config", {})}
    keep_mask = pd.Series(True, index=df.index)
    aggressive = config.get("aggressive_mode", True)
    
    # === SAFETY CHECK ===
    # Calculate total potential row loss if we were to drop everything
    potential_drop_mask = pd.Series(False, index=df.index)
    for col, report in state["outlier_report"].items():
        if report.get("action") == "skip": continue
        # Only consider columns that WOULD be dropped in aggressive mode
        intent = report["intent"]
        if intent not in ["BOUNDED", "PERCENTAGE"]: # These are already capped in conservative, but dropped in aggressive
             if "outlier_mask" in report:
                # Ensure mask aligns
                mask = report["outlier_mask"]
                if len(mask) == len(df):
                     potential_drop_mask |= mask

    potential_loss_pct = (potential_drop_mask.sum() / len(df)) * 100
    SAFE_LOSS_THRESHOLD = 15.0  # Industry standard safety limit
    
    # If potential loss is too high, FORCE switch to CAPPING
    if aggressive and potential_loss_pct > SAFE_LOSS_THRESHOLD:
        print(f"[WARN] Excessive outlier loss detected ({potential_loss_pct:.1f}%). Switching to CAPPING to preserve data.")
        aggressive = False  # Disable aggressive row dropping
        treatment_log.append({
            "action": "strategy_change",
            "reason": f"Potential data loss {potential_loss_pct:.1f}% > {SAFE_LOSS_THRESHOLD}%. Switched to Winsorization (Capping)."
        })
    elif aggressive:
         print(f"[INFO] Outlier loss acceptable ({potential_loss_pct:.1f}%). Proceeding with aggressive removal.")

    # === PRE-TREATMENT: Convert analyzed columns to their parsed numeric form ===
    # This ensures the cleaned CSV has proper numbers (e.g., 19M -> 19000000)
    # BUT we must respect LLM classification to preserve categorical columns
    for col, analysis in state["numeric_analysis"].items():
        intent = state["column_intents"].get(col, "MEASURE")
        
        # Skip columns that should NOT be converted to numeric
        # ID, TIME, VERSION, and CATEGORICAL columns should preserve original values
        if intent in ["ID", "TIME", "VERSION", "CATEGORICAL"]:
            print(f"[INFO] Column '{col}' classified as {intent} - preserving original values.")
            continue
        
        parsed = analysis.get("parsed_series")
        if parsed is not None:
            # Check if we actually have valid numeric values (not all NaN)
            valid_count = parsed.notna().sum()
            original_valid = df[col].notna().sum()
            
            # Only convert if we recovered a reasonable amount of data
            # If most values became NaN, this column might be categorical
            if valid_count > 0 and valid_count >= original_valid * 0.5:
                df[col] = parsed
                print(f"[INFO] Column '{col}' converted to numeric format ({valid_count}/{original_valid} values).")
            else:
                print(f"[WARN] Column '{col}' conversion skipped - too much data loss ({valid_count}/{original_valid}).")

    for col, report in state["outlier_report"].items():
        try:
            if report.get("action") == "skip":
                treatment_log.append({
                    "column": col,
                    "intent": report["intent"],
                    "action": "ignored",
                    "reason": report.get("reason", "Excluded type")
                })
                continue
            
            intent = report["intent"]
            parsed = state["numeric_analysis"][col]["parsed_series"]
            outlier_mask = report.get("outlier_mask", pd.Series(False, index=df.index))
            outlier_count = report.get("outlier_count", 0)
            
            # Get bounds
            lower = report["lower_bound"]
            upper = report["upper_bound"]
            
            # Re-evaluate logic based on potentially updated 'aggressive' flag
            if aggressive:
                # Remove rows with outliers
                keep_mask &= ~outlier_mask
                
                # Update column to parsed numeric values
                df[col] = parsed
                
                treatment_log.append({
                    "column": col,
                    "intent": intent,
                    "action": "outliers_removed",
                    "outliers_removed": int(outlier_count),
                    "bounds": [round(lower, 4), round(upper, 4)]
                })
            else:                   
                capped = parsed.clip(lower, upper)
                df[col] = capped
                treatment_log.append({
                    "column": col,
                    "intent": intent,
                    "action": "capped",
                    "outliers_capped": int(outlier_count),
                    "bounds": [round(lower, 4), round(upper, 4)]
                })
            
        except Exception as e:
            errors.append({
                "column": col,
                "stage": "apply",
                "error": str(e),
                "type": type(e).__name__
            })
    
    rows_before = len(df)
    df = df[keep_mask]
    rows_removed = rows_before - len(df)
    
    treatment_log.append({
        "summary": "row_removal",
        "rows_before": rows_before,
        "rows_after": len(df),
        "rows_removed": rows_removed
    })
    
    state["df"] = df
    state["treatment_log"] = treatment_log
    state["errors"] = errors
    return state


def generate_report(state: OutlierState) -> OutlierState:
    """Node 5: Generate final summary report."""
    summary = {
        "original_shape": state["original_shape"],
        "final_shape": state["df"].shape,
        "columns_analyzed": len(state["numeric_analysis"]),
        "columns_treated": len([t for t in state["treatment_log"] if t.get("action") not in ["ignored", None]]),
        "errors_encountered": len(state["errors"])
    }
    
    state["treatment_log"].insert(0, {"report_summary": summary})
    
    return state


# =====================================================
# 7. BUILD LANGGRAPH (if available)
# =====================================================

def build_outlier_graph():
    """Build the LangGraph for outlier treatment pipeline."""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph is not installed. Install with: pip install langgraph")
    
    graph = StateGraph(OutlierState)
    
    graph.add_node("analyze_numeric", analyze_numeric_columns)
    graph.add_node("classify_intents", classify_column_intents)
    graph.add_node("detect_outliers", detect_outliers)
    graph.add_node("apply_treatment", apply_treatment)
    graph.add_node("generate_report", generate_report)
    
    graph.set_entry_point("analyze_numeric")
    
    graph.add_edge("analyze_numeric", "classify_intents")
    graph.add_edge("classify_intents", "detect_outliers")
    graph.add_edge("detect_outliers", "apply_treatment")
    graph.add_edge("apply_treatment", "generate_report")
    graph.add_edge("generate_report", END)
    
    return graph.compile()


# =====================================================
# 8. STANDALONE NODE FUNCTION (for simple integration)
# =====================================================

def outlier_agent_node(state: dict) -> dict:
    """
    Main entry point for outlier treatment.
    Compatible with both simple dict state and full OutlierState.
    
    Args:
        state: dict with at least {"data": pd.DataFrame} or {"df": pd.DataFrame}
    
    Returns:
        Updated state with cleaned data and logs
    """
    if "data" in state:
        df = state["data"].copy()
    elif "df" in state:
        df = state["df"].copy()
    else:
        raise ValueError("State must contain 'data' or 'df' key with DataFrame")
    
    outlier_state: OutlierState = {
        "df": df,
        "original_shape": df.shape,
        "numeric_analysis": {},
        "column_intents": {},
        "outlier_report": {},
        "treatment_log": [],
        "errors": [],
        "config": state.get("config", DEFAULT_CONFIG)
    }
    
    outlier_state = analyze_numeric_columns(outlier_state)
    outlier_state = classify_column_intents(outlier_state)
    outlier_state = detect_outliers(outlier_state)
    outlier_state = apply_treatment(outlier_state)
    outlier_state = generate_report(outlier_state)
    
    if "Size" in outlier_state["df"].columns:
        print(f"[DEBUG] Size column after treatment: {outlier_state['df']['Size'].head(5).tolist()}")
    if "Installs" in outlier_state["df"].columns:
        print(f"[DEBUG] Installs column after treatment: {outlier_state['df']['Installs'].head(5).tolist()}")
    
    
    if "data" in state:
        state["data"] = outlier_state["df"]
    else:
        state["df"] = outlier_state["df"]
    
    state["log"] = outlier_state["treatment_log"]
    state["outlier_report"] = outlier_state["outlier_report"]
    state["errors"] = outlier_state["errors"]
    
    return state


# =====================================================
# 9. ENTRY POINT
# =====================================================

if __name__ == "__main__":
    import sys
    
   
    test_file = sys.argv[1] if len(sys.argv) > 1 else "../data/googleplaystore.csv"
    
    print(f"Loading: {test_file}")
    df = pd.read_csv(test_file)
    print(f"Original shape: {df.shape}")
    
    state = {"data": df}
    result = outlier_agent_node(state)
    
    print(f"\nFinal shape: {result['data'].shape}")
    print(f"\nTreatment Log:")
    for entry in result.get("log", []):
        print(f"  {entry}")
    
    if result.get("errors"):
        print(f"\nErrors:")
        for err in result["errors"]:
            print(f"  {err}")