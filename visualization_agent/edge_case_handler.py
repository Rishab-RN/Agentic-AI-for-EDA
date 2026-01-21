# =====================================================
# Edge Case Handler for Visualization Agent
# =====================================================
# Detects and resolves edge cases using rule-based logic
# first, then Groq LLM for semantic understanding when needed.
# =====================================================

import os
import re
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("[WARN] Groq not installed. Install with: pip install groq")


# =====================================================
# 1. EDGE CASE TYPE DEFINITIONS
# =====================================================

class EdgeCaseType(Enum):
    """Types of edge cases that can be detected."""
    # Rule-based fixable
    EMPTY_DATAFRAME = "empty_dataframe"
    SINGLE_ROW = "single_row"
    EMPTY_COLUMN = "empty_column"
    NEAR_EMPTY_COLUMN = "near_empty_column"
    SINGLE_VALUE_COLUMN = "single_value_column"
    ZERO_VARIANCE = "zero_variance"
    INF_VALUES = "inf_values"
    SPECIAL_CHARS_IN_NAME = "special_chars_in_name"
    LONG_COLUMN_NAME = "long_column_name"
    DUPLICATE_CORRELATION = "duplicate_correlation"
    SPARSE_DATA = "sparse_data"
    
    # LLM-required (semantic understanding)
    LIKELY_DATETIME = "likely_datetime"
    LIKELY_ID_COLUMN = "likely_id_column"
    LIKELY_TARGET = "likely_target"
    ORDINAL_CATEGORICAL = "ordinal_categorical"
    AMBIGUOUS_TYPE = "ambiguous_type"
    BINARY_NUMERIC = "binary_numeric"
    ZERO_SELECTION_FALLBACK = "zero_selection_fallback"
    RELATED_COLUMNS = "related_columns"


@dataclass
class EdgeCase:
    """Represents a detected edge case."""
    case_type: EdgeCaseType
    columns: List[str]
    details: Dict[str, Any]
    requires_llm: bool = False
    resolution: Optional[Dict] = None
    is_resolved: bool = False


@dataclass
class EdgeCaseConfig:
    """Configuration for edge case detection and resolution."""
    # Rule-based thresholds
    min_valid_count: int = 5
    min_unique_for_plot: int = 2
    max_missing_pct: float = 95.0
    duplicate_corr_threshold: float = 0.99
    max_column_name_length: int = 50
    
    # LLM settings (Groq)
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout: int = 30
    llm_batch_size: int = 30  # Max columns per LLM call
    
    # Fallback behavior
    min_plots_fallback: int = 5  # Minimum plots if nothing selected


# =====================================================
# 2. EDGE CASE DETECTOR
# =====================================================

class EdgeCaseDetector:
    """Detects edge cases in a DataFrame and its metadata."""
    
    def __init__(self, config: Optional[EdgeCaseConfig] = None):
        self.config = config or EdgeCaseConfig()
        
    def detect_all(self, df: pd.DataFrame, column_metadata: Dict[str, Dict]) -> List[EdgeCase]:
        """
        Detect all edge cases in the DataFrame.
        
        Args:
            df: The input DataFrame
            column_metadata: Pre-computed column metadata
            
        Returns:
            List of detected EdgeCase objects
        """
        edge_cases = []
        
        # Dataset-level checks
        edge_cases.extend(self._check_dataset_level(df))
        
        if len(df) < 2:
            return edge_cases  # Can't do much with empty/single-row
        
        # Column-level checks
        edge_cases.extend(self._check_column_level(df, column_metadata))
        
        # Semantic checks (LLM-required)
        edge_cases.extend(self._check_semantic(df, column_metadata))
        
        return edge_cases
    
    def _check_dataset_level(self, df: pd.DataFrame) -> List[EdgeCase]:
        """Check for dataset-level edge cases."""
        cases = []
        
        # Empty DataFrame
        if len(df) == 0:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.EMPTY_DATAFRAME,
                columns=[],
                details={"message": "DataFrame is empty"},
                requires_llm=False
            ))
        
        # Single row
        elif len(df) == 1:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.SINGLE_ROW,
                columns=[],
                details={"row_count": 1, "message": "DataFrame has only 1 row"},
                requires_llm=False
            ))
        
        return cases
    
    def _check_column_level(self, df: pd.DataFrame, column_metadata: Dict[str, Dict]) -> List[EdgeCase]:
        """Check for column-level edge cases."""
        cases = []
        
        empty_columns = []
        near_empty_columns = []
        single_value_columns = []
        inf_value_columns = []
        special_char_columns = []
        long_name_columns = []
        sparse_columns = []
        
        for col in df.columns:
            meta = column_metadata.get(col, {})
            
            # Check for 100% missing
            if meta.get("missing_pct", 0) >= 100:
                empty_columns.append(col)
                continue
            
            # Check for near-empty (>95% missing)
            if meta.get("missing_pct", 0) >= self.config.max_missing_pct:
                near_empty_columns.append(col)
                continue
            
            # Check valid count
            valid_count = meta.get("valid_count", len(df))
            if valid_count < self.config.min_valid_count:
                sparse_columns.append(col)
            
            # Check for single unique value
            if meta.get("is_numeric"):
                if meta.get("variance", 1) == 0:
                    single_value_columns.append(col)
            elif meta.get("cardinality", 2) == 1:
                single_value_columns.append(col)
            
            # Check for infinite values (numeric only)
            if meta.get("is_numeric") and valid_count > 0:
                if np.isinf(df[col].dropna()).any():
                    inf_value_columns.append(col)
            
            # Check column name for special characters
            if re.search(r'[\/\\:*?"<>|]', col):
                special_char_columns.append(col)
            
            # Check for long column names
            if len(col) > self.config.max_column_name_length:
                long_name_columns.append(col)
        
        # Create edge cases for each group
        if empty_columns:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.EMPTY_COLUMN,
                columns=empty_columns,
                details={"action": "skip", "reason": "100% missing values"},
                requires_llm=False
            ))
        
        if near_empty_columns:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.NEAR_EMPTY_COLUMN,
                columns=near_empty_columns,
                details={"action": "skip", "reason": f">{self.config.max_missing_pct}% missing"},
                requires_llm=False
            ))
        
        if single_value_columns:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.SINGLE_VALUE_COLUMN,
                columns=single_value_columns,
                details={"action": "skip", "reason": "Only one unique value"},
                requires_llm=False
            ))
        
        if inf_value_columns:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.INF_VALUES,
                columns=inf_value_columns,
                details={"action": "replace_with_nan", "reason": "Contains infinite values"},
                requires_llm=False
            ))
        
        if special_char_columns:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.SPECIAL_CHARS_IN_NAME,
                columns=special_char_columns,
                details={"action": "sanitize", "reason": "Invalid characters for filenames"},
                requires_llm=False
            ))
        
        if long_name_columns:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.LONG_COLUMN_NAME,
                columns=long_name_columns,
                details={"action": "truncate", "max_length": self.config.max_column_name_length},
                requires_llm=False
            ))
        
        if sparse_columns:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.SPARSE_DATA,
                columns=sparse_columns,
                details={"action": "skip", "reason": f"<{self.config.min_valid_count} valid values"},
                requires_llm=False
            ))
        
        return cases
    
    def _check_semantic(self, df: pd.DataFrame, column_metadata: Dict[str, Dict]) -> List[EdgeCase]:
        """Check for semantic edge cases that may require LLM."""
        cases = []
        
        # Patterns for heuristic detection
        datetime_patterns = r'(date|time|timestamp|created|updated|modified|dt|_at$)'
        id_patterns = r'(^id$|_id$|^pk$|^key$|^index$|_index$|_pk$|_key$)'
        target_patterns = r'(^target$|^label$|^y$|^outcome$|^result$|price|sales|revenue)'
        
        likely_datetime = []
        likely_id = []
        likely_target = []
        binary_numeric = []
        ambiguous_type = []
        ordinal_candidates = []
        
        for col in df.columns:
            meta = column_metadata.get(col, {})
            col_lower = col.lower()
            
            # Skip already problematic columns
            if meta.get("missing_pct", 0) >= self.config.max_missing_pct:
                continue
            
            # Check for likely datetime
            if re.search(datetime_patterns, col_lower, re.IGNORECASE):
                likely_datetime.append(col)
            
            # Check for likely ID column (numeric with high unique ratio)
            if re.search(id_patterns, col_lower, re.IGNORECASE):
                likely_id.append(col)
            elif meta.get("is_numeric") and meta.get("unique_ratio", 0) > 0.9:
                # High uniqueness numeric = possibly ID
                likely_id.append(col)
            
            # Check for likely target variable
            if re.search(target_patterns, col_lower, re.IGNORECASE):
                likely_target.append(col)
            
            # Check for binary numeric (0/1 or two values)
            if meta.get("is_numeric"):
                valid_vals = df[col].dropna()
                unique_vals = valid_vals.nunique()
                if unique_vals == 2:
                    binary_numeric.append(col)
                elif unique_vals <= 5 and all(valid_vals.apply(lambda x: float(x).is_integer() if pd.notna(x) else True)):
                    # Small set of integers - ambiguous
                    ambiguous_type.append(col)
            
            # Check for potential ordinal categorical
            if meta.get("is_categorical"):
                cardinality = meta.get("cardinality", 0)
                if 2 <= cardinality <= 10:
                    # Could be ordinal
                    ordinal_candidates.append(col)
        
        # Create LLM-required edge cases
        if likely_datetime:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.LIKELY_DATETIME,
                columns=likely_datetime,
                details={"detected_by": "pattern_matching"},
                requires_llm=True
            ))
        
        if likely_id:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.LIKELY_ID_COLUMN,
                columns=likely_id,
                details={"detected_by": "pattern_matching_or_high_uniqueness"},
                requires_llm=True
            ))
        
        if likely_target:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.LIKELY_TARGET,
                columns=likely_target,
                details={"detected_by": "pattern_matching"},
                requires_llm=True
            ))
        
        if binary_numeric:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.BINARY_NUMERIC,
                columns=binary_numeric,
                details={"recommendation_needed": "bar_or_histogram"},
                requires_llm=True
            ))
        
        if ambiguous_type:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.AMBIGUOUS_TYPE,
                columns=ambiguous_type,
                details={"recommendation_needed": "categorical_or_numeric"},
                requires_llm=True
            ))
        
        if ordinal_candidates:
            cases.append(EdgeCase(
                case_type=EdgeCaseType.ORDINAL_CATEGORICAL,
                columns=ordinal_candidates,
                details={"recommendation_needed": "ordering"},
                requires_llm=True
            ))
        
        return cases


# =====================================================
# 3. RULE-BASED RESOLVER
# =====================================================

class RuleBasedResolver:
    """Resolves edge cases using deterministic rules."""
    
    def __init__(self, config: Optional[EdgeCaseConfig] = None):
        self.config = config or EdgeCaseConfig()
    
    def resolve(self, edge_case: EdgeCase, df: pd.DataFrame) -> EdgeCase:
        """
        Resolve an edge case using rules.
        
        Args:
            edge_case: The edge case to resolve
            df: The DataFrame for context
            
        Returns:
            The edge case with resolution applied
        """
        case_type = edge_case.case_type
        
        resolution_map = {
            EdgeCaseType.EMPTY_DATAFRAME: self._resolve_empty_df,
            EdgeCaseType.SINGLE_ROW: self._resolve_single_row,
            EdgeCaseType.EMPTY_COLUMN: self._resolve_skip_columns,
            EdgeCaseType.NEAR_EMPTY_COLUMN: self._resolve_skip_columns,
            EdgeCaseType.SINGLE_VALUE_COLUMN: self._resolve_skip_columns,
            EdgeCaseType.SPARSE_DATA: self._resolve_skip_columns,
            EdgeCaseType.INF_VALUES: self._resolve_inf_values,
            EdgeCaseType.SPECIAL_CHARS_IN_NAME: self._resolve_special_chars,
            EdgeCaseType.LONG_COLUMN_NAME: self._resolve_long_names,
            EdgeCaseType.DUPLICATE_CORRELATION: self._resolve_duplicate_corr,
        }
        
        resolver = resolution_map.get(case_type)
        if resolver:
            edge_case.resolution = resolver(edge_case, df)
            edge_case.is_resolved = True
        
        return edge_case
    
    def _resolve_empty_df(self, edge_case: EdgeCase, df: pd.DataFrame) -> Dict:
        return {
            "action": "abort",
            "message": "Cannot generate visualizations for empty DataFrame",
            "skip_pipeline": True
        }
    
    def _resolve_single_row(self, edge_case: EdgeCase, df: pd.DataFrame) -> Dict:
        return {
            "action": "abort",
            "message": "Cannot generate meaningful visualizations for single-row DataFrame",
            "skip_pipeline": True
        }
    
    def _resolve_skip_columns(self, edge_case: EdgeCase, df: pd.DataFrame) -> Dict:
        return {
            "action": "skip",
            "columns_to_skip": edge_case.columns,
            "reason": edge_case.details.get("reason", "Insufficient data")
        }
    
    def _resolve_inf_values(self, edge_case: EdgeCase, df: pd.DataFrame) -> Dict:
        return {
            "action": "replace",
            "columns": edge_case.columns,
            "replace_inf_with": "nan",
            "reason": "Infinite values replaced with NaN for statistical analysis"
        }
    
    def _resolve_special_chars(self, edge_case: EdgeCase, df: pd.DataFrame) -> Dict:
        sanitized = {}
        for col in edge_case.columns:
            sanitized[col] = re.sub(r'[\/\\:*?"<>|]', '_', col)
        return {
            "action": "sanitize_filename",
            "column_filename_map": sanitized,
            "reason": "Special characters replaced with underscores for filenames"
        }
    
    def _resolve_long_names(self, edge_case: EdgeCase, df: pd.DataFrame) -> Dict:
        truncated = {}
        max_len = self.config.max_column_name_length
        for col in edge_case.columns:
            truncated[col] = col[:max_len-3] + "..." if len(col) > max_len else col
        return {
            "action": "truncate_filename",
            "column_filename_map": truncated,
            "reason": f"Column names truncated to {max_len} characters"
        }
    
    def _resolve_duplicate_corr(self, edge_case: EdgeCase, df: pd.DataFrame) -> Dict:
        return {
            "action": "skip_duplicate_pairs",
            "pairs_to_skip": edge_case.details.get("duplicate_pairs", []),
            "reason": "Near-perfect correlation indicates duplicate columns"
        }


# =====================================================
# 4. GROQ LLM RESOLVER
# =====================================================

class GroqLLMResolver:
    """Resolves semantic edge cases using Groq LLM."""
    
    def __init__(self, config: Optional[EdgeCaseConfig] = None):
        self.config = config or EdgeCaseConfig()
        self.client = None
        self._init_client()
        self.cache = {}  # Simple in-memory cache
    
    def _init_client(self):
        """Initialize Groq client."""
        if not GROQ_AVAILABLE:
            return
        
        api_key = self.config.groq_api_key or os.environ.get("GROQ_API_KEY")
        if api_key:
            self.client = Groq(api_key=api_key)
    
    def is_available(self) -> bool:
        """Check if Groq client is available."""
        return self.client is not None
    
    def resolve_batch(
        self, 
        edge_cases: List[EdgeCase], 
        df: pd.DataFrame,
        column_metadata: Dict[str, Dict]
    ) -> List[EdgeCase]:
        """
        Resolve multiple edge cases efficiently using batched LLM calls.
        
        Args:
            edge_cases: List of LLM-required edge cases
            df: DataFrame for context
            column_metadata: Pre-computed metadata
            
        Returns:
            List of resolved edge cases
        """
        if not self.is_available():
            print("[WARN] Groq not available. Using rule-based fallbacks for semantic edge cases.")
            return self._fallback_resolve_all(edge_cases, df)
        
        resolved_cases = []
        
        for edge_case in edge_cases:
            try:
                resolved = self._resolve_single(edge_case, df, column_metadata)
                resolved_cases.append(resolved)
            except Exception as e:
                print(f"[WARN] LLM resolution failed for {edge_case.case_type.value}: {e}")
                # Use fallback
                fallback = self._fallback_resolve(edge_case, df)
                resolved_cases.append(fallback)
        
        return resolved_cases
    
    def _resolve_single(
        self, 
        edge_case: EdgeCase, 
        df: pd.DataFrame,
        column_metadata: Dict[str, Dict]
    ) -> EdgeCase:
        """Resolve a single edge case using LLM."""
        
        # Build prompt based on case type
        prompt = self._build_prompt(edge_case, df, column_metadata)
        
        # Check cache
        cache_key = self._get_cache_key(edge_case, df)
        if cache_key in self.cache:
            edge_case.resolution = self.cache[cache_key]
            edge_case.is_resolved = True
            return edge_case
        
        # Call Groq
        try:
            response = self.client.chat.completions.create(
                model=self.config.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low for consistency
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            edge_case.resolution = result
            edge_case.is_resolved = True
            
            # Cache the result
            self.cache[cache_key] = result
            
        except Exception as e:
            print(f"[WARN] Groq API error: {e}")
            return self._fallback_resolve(edge_case, df)
        
        return edge_case
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for Groq."""
        return """You are an expert data analyst assistant helping with exploratory data analysis (EDA).
Your task is to analyze column information and provide recommendations for visualization.

ALWAYS respond with valid JSON in the following format:
{
    "decisions": [
        {
            "column": "column_name",
            "decision": "your_decision",
            "confidence": 0.95,
            "reason": "brief explanation"
        }
    ]
}

Be concise and precise. Focus on actionable decisions for data visualization."""

    def _build_prompt(
        self, 
        edge_case: EdgeCase, 
        df: pd.DataFrame,
        column_metadata: Dict[str, Dict]
    ) -> str:
        """Build a prompt for the specific edge case type."""
        
        case_type = edge_case.case_type
        columns = edge_case.columns
        
        # Get sample data for context (limited for efficiency)
        sample_info = {}
        for col in columns[:self.config.llm_batch_size]:  # Limit columns
            meta = column_metadata.get(col, {})
            sample_vals = df[col].dropna().head(5).tolist()
            sample_info[col] = {
                "dtype": meta.get("dtype", str(df[col].dtype)),
                "sample_values": [str(v) for v in sample_vals],
                "unique_count": meta.get("cardinality", df[col].nunique()),
                "missing_pct": meta.get("missing_pct", 0)
            }
        
        prompts = {
            EdgeCaseType.LIKELY_DATETIME: f"""
Analyze these columns and determine if they are datetime/timestamp columns that should be treated as time-series data.

Columns to analyze:
{json.dumps(sample_info, indent=2)}

For each column, decide: "is_datetime" (true/false)
If true, also suggest: "time_plot_type" (line, area, or none)
""",
            EdgeCaseType.LIKELY_ID_COLUMN: f"""
Analyze these columns and determine if they are ID/key columns that should NOT be visualized.

Columns to analyze:
{json.dumps(sample_info, indent=2)}

For each column, decide: "is_id_column" (true/false)
ID columns typically have: sequential numbers, UUID patterns, high uniqueness, names like "id", "key", "pk"
""",
            EdgeCaseType.LIKELY_TARGET: f"""
Analyze these columns and determine if any is likely a TARGET variable for prediction/analysis.

Columns to analyze:
{json.dumps(sample_info, indent=2)}

For each column, decide: "is_target" (true/false)
Target columns are typically: price, sales, outcome, result, label, y
If a target is found, it should get priority visualization.
""",
            EdgeCaseType.BINARY_NUMERIC: f"""
These columns contain only 2 unique numeric values (like 0/1).
Decide the best visualization for each.

Columns to analyze:
{json.dumps(sample_info, indent=2)}

For each column, decide: "plot_type" (bar, pie, or histogram)
Bar/pie works for flag columns, histogram for continuous that happens to have 2 values.
""",
            EdgeCaseType.AMBIGUOUS_TYPE: f"""
These numeric columns have very few unique values (like 1,2,3,4,5).
Decide if they should be treated as categorical or numeric.

Columns to analyze:
{json.dumps(sample_info, indent=2)}

For each column, decide: "treat_as" (categorical or numeric)
Examples: ratings (1-5) → categorical, year → could be either
""",
            EdgeCaseType.ORDINAL_CATEGORICAL: f"""
These categorical columns might have a natural ordering.
Determine the correct order if applicable.

Columns to analyze:
{json.dumps(sample_info, indent=2)}

For each column, decide:
- "is_ordinal" (true/false)
- "order" (list of values in correct order, if ordinal)

Examples: ["Low", "Medium", "High"], ["Small", "Medium", "Large"]
""",
            EdgeCaseType.RELATED_COLUMNS: f"""
Identify groups of related columns that might represent the same concept.

All columns: {list(df.columns)[:50]}

Look for patterns like:
- name, first_name, last_name (name group)
- city, state, zip, country (address group)
- start_date, end_date (date range)

Return: "column_groups" as list of lists
"""
        }
        
        return prompts.get(case_type, f"Analyze columns: {columns}")
    
    def _get_cache_key(self, edge_case: EdgeCase, df: pd.DataFrame) -> str:
        """Generate cache key for an edge case."""
        col_hash = hashlib.md5(str(sorted(edge_case.columns)).encode()).hexdigest()[:8]
        return f"{edge_case.case_type.value}_{col_hash}"
    
    def _fallback_resolve(self, edge_case: EdgeCase, df: pd.DataFrame) -> EdgeCase:
        """Provide rule-based fallback when LLM is unavailable."""
        case_type = edge_case.case_type
        
        fallback_resolutions = {
            EdgeCaseType.LIKELY_DATETIME: {
                "decisions": [{"column": col, "is_datetime": True, "confidence": 0.6} 
                             for col in edge_case.columns],
                "source": "fallback"
            },
            EdgeCaseType.LIKELY_ID_COLUMN: {
                "decisions": [{"column": col, "is_id_column": True, "confidence": 0.6}
                             for col in edge_case.columns],
                "source": "fallback"
            },
            EdgeCaseType.LIKELY_TARGET: {
                "decisions": [{"column": col, "is_target": col.lower() in ["target", "y", "label"]}
                             for col in edge_case.columns],
                "source": "fallback"
            },
            EdgeCaseType.BINARY_NUMERIC: {
                "decisions": [{"column": col, "plot_type": "bar", "confidence": 0.7}
                             for col in edge_case.columns],
                "source": "fallback"
            },
            EdgeCaseType.AMBIGUOUS_TYPE: {
                "decisions": [{"column": col, "treat_as": "categorical", "confidence": 0.5}
                             for col in edge_case.columns],
                "source": "fallback"
            },
            EdgeCaseType.ORDINAL_CATEGORICAL: {
                "decisions": [{"column": col, "is_ordinal": False}
                             for col in edge_case.columns],
                "source": "fallback"
            }
        }
        
        edge_case.resolution = fallback_resolutions.get(case_type, {"source": "fallback", "decisions": []})
        edge_case.is_resolved = True
        return edge_case
    
    def _fallback_resolve_all(self, edge_cases: List[EdgeCase], df: pd.DataFrame) -> List[EdgeCase]:
        """Apply fallback resolution to all edge cases."""
        return [self._fallback_resolve(ec, df) for ec in edge_cases]


# =====================================================
# 5. MASTER EDGE CASE HANDLER
# =====================================================

class EdgeCaseHandler:
    """
    Master handler that orchestrates edge case detection and resolution.
    
    Flow:
    1. Detect all edge cases
    2. Apply rule-based fixes for simple cases
    3. Call LLM for semantic cases (when rule-based can't handle)
    4. Return resolved cases for pipeline integration
    """
    
    def __init__(self, config: Optional[EdgeCaseConfig] = None):
        self.config = config or EdgeCaseConfig()
        self.detector = EdgeCaseDetector(self.config)
        self.rule_resolver = RuleBasedResolver(self.config)
        self.llm_resolver = GroqLLMResolver(self.config)
    
    def process(
        self, 
        df: pd.DataFrame, 
        column_metadata: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Process a DataFrame for edge cases.
        
        Args:
            df: Input DataFrame
            column_metadata: Pre-computed column metadata
            
        Returns:
            Dictionary with:
            - edge_cases: List of all detected edge cases
            - rule_based_resolutions: Resolutions from rules
            - llm_resolutions: Resolutions from LLM
            - columns_to_skip: List of columns to skip
            - column_type_overrides: Dict of column type corrections
            - filename_sanitization: Dict of column to safe filename
            - should_abort: Whether to abort pipeline
            - abort_reason: If aborting, why
        """
        print("\n[EDGE] Detecting edge cases...")
        
        # Step 1: Detect all edge cases
        all_cases = self.detector.detect_all(df, column_metadata)
        
        if not all_cases:
            print("[EDGE] No edge cases detected.")
            return {
                "edge_cases": [],
                "rule_based_resolutions": [],
                "llm_resolutions": [],
                "columns_to_skip": [],
                "column_type_overrides": {},
                "filename_sanitization": {},
                "should_abort": False,
                "abort_reason": None
            }
        
        print(f"[EDGE] Detected {len(all_cases)} edge case(s)")
        
        # Separate rule-based and LLM-required
        rule_based = [ec for ec in all_cases if not ec.requires_llm]
        llm_required = [ec for ec in all_cases if ec.requires_llm]
        
        print(f"[EDGE] Rule-based: {len(rule_based)}, LLM-required: {len(llm_required)}")
        
        # Step 2: Apply rule-based fixes
        rule_resolutions = []
        for edge_case in rule_based:
            resolved = self.rule_resolver.resolve(edge_case, df)
            rule_resolutions.append(resolved)
        
        # Check for abort conditions
        for res in rule_resolutions:
            if res.resolution and res.resolution.get("skip_pipeline"):
                return {
                    "edge_cases": all_cases,
                    "rule_based_resolutions": rule_resolutions,
                    "llm_resolutions": [],
                    "columns_to_skip": [],
                    "column_type_overrides": {},
                    "filename_sanitization": {},
                    "should_abort": True,
                    "abort_reason": res.resolution.get("message", "Critical edge case")
                }
        
        # Step 3: Resolve LLM-required cases
        llm_resolutions = []
        if llm_required:
            print(f"[EDGE] Resolving {len(llm_required)} semantic edge cases with Groq...")
            llm_resolutions = self.llm_resolver.resolve_batch(llm_required, df, column_metadata)
        
        # Step 4: Aggregate results
        result = self._aggregate_results(rule_resolutions, llm_resolutions)
        result["edge_cases"] = all_cases
        result["rule_based_resolutions"] = rule_resolutions
        result["llm_resolutions"] = llm_resolutions
        
        print(f"[EDGE] Columns to skip: {len(result['columns_to_skip'])}")
        print(f"[EDGE] Type overrides: {len(result['column_type_overrides'])}")
        
        return result
    
    def _aggregate_results(
        self, 
        rule_resolutions: List[EdgeCase], 
        llm_resolutions: List[EdgeCase]
    ) -> Dict[str, Any]:
        """Aggregate all resolutions into actionable directives."""
        
        columns_to_skip = set()
        column_type_overrides = {}
        filename_sanitization = {}
        id_columns = set()
        target_columns = set()
        datetime_columns = set()
        
        # Process rule-based resolutions
        for ec in rule_resolutions:
            if not ec.resolution:
                continue
            
            action = ec.resolution.get("action", "")
            
            if action == "skip":
                columns_to_skip.update(ec.resolution.get("columns_to_skip", ec.columns))
            
            elif action in ("sanitize_filename", "truncate_filename"):
                filename_sanitization.update(ec.resolution.get("column_filename_map", {}))
        
        # Process LLM resolutions
        for ec in llm_resolutions:
            if not ec.resolution:
                continue
            
            decisions = ec.resolution.get("decisions", [])
            
            for decision in decisions:
                col = decision.get("column", "")
                
                # ID column detection
                if decision.get("is_id_column"):
                    id_columns.add(col)
                    columns_to_skip.add(col)
                
                # Target detection
                if decision.get("is_target"):
                    target_columns.add(col)
                    column_type_overrides[col] = {"is_target": True, "priority_boost": 5}
                
                # Datetime detection
                if decision.get("is_datetime"):
                    datetime_columns.add(col)
                    column_type_overrides[col] = {"is_datetime": True, "plot_type": "timeseries"}
                
                # Ambiguous type resolution
                if "treat_as" in decision:
                    column_type_overrides[col] = {"treat_as": decision["treat_as"]}
                
                # Binary numeric
                if "plot_type" in decision:
                    column_type_overrides[col] = {"recommended_plot": decision["plot_type"]}
                
                # Ordinal ordering
                if decision.get("is_ordinal") and "order" in decision:
                    column_type_overrides[col] = {
                        "is_ordinal": True, 
                        "order": decision["order"]
                    }
        
        return {
            "columns_to_skip": list(columns_to_skip),
            "column_type_overrides": column_type_overrides,
            "filename_sanitization": filename_sanitization,
            "id_columns": list(id_columns),
            "target_columns": list(target_columns),
            "datetime_columns": list(datetime_columns),
            "should_abort": False,
            "abort_reason": None
        }


# =====================================================
# 6. ZERO-SELECTION FALLBACK HANDLER
# =====================================================

def handle_zero_selection_fallback(
    df: pd.DataFrame,
    column_metadata: Dict[str, Dict],
    priority_scores: Dict[str, Dict],
    config: EdgeCaseConfig
) -> List[Dict]:
    """
    Handle the case when no plots were selected by the main pipeline.
    Uses LLM to select the most meaningful columns for visualization.
    
    Args:
        df: DataFrame
        column_metadata: Column metadata
        priority_scores: Already computed scores
        config: Edge case configuration
        
    Returns:
        List of fallback plot selections
    """
    print("[EDGE] Zero plots selected - invoking fallback...")
    
    min_plots = config.min_plots_fallback
    
    # Try LLM first
    llm_resolver = GroqLLMResolver(config)
    
    if llm_resolver.is_available():
        try:
            # Build prompt for fallback selection
            column_summaries = []
            for col, meta in list(column_metadata.items())[:50]:  # Limit for token
                score = priority_scores.get(col, {}).get("score", 0)
                column_summaries.append({
                    "name": col,
                    "type": "numeric" if meta.get("is_numeric") else "categorical",
                    "score": score,
                    "unique_count": meta.get("cardinality", meta.get("valid_count", 0))
                })
            
            prompt = f"""
No visualizations were selected for this dataset because all columns scored below threshold.
Please select the {min_plots} MOST IMPORTANT columns to visualize anyway.

Consider:
- Business relevance (prices, counts, categories)
- Data quality (non-empty, varied)
- Column names that suggest importance

Available columns:
{json.dumps(column_summaries, indent=2)}

Return: "selected_columns" as a list of column names with "plot_type" for each.
"""
            
            response = llm_resolver.client.chat.completions.create(
                model=config.groq_model,
                messages=[
                    {"role": "system", "content": llm_resolver._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            selected = result.get("selected_columns", [])
            
            fallback_plots = []
            for sel in selected[:min_plots]:
                col = sel.get("column", sel.get("name", ""))
                plot_type = sel.get("plot_type", "histogram")
                
                if col in df.columns:
                    meta = column_metadata.get(col, {})
                    fallback_plots.append({
                        "column": col,
                        "plot_type": plot_type,
                        "category": "univariate_numeric" if meta.get("is_numeric") else "univariate_categorical",
                        "priority_score": 0,
                        "reason": "Fallback selection by LLM"
                    })
            
            if fallback_plots:
                return fallback_plots
                
        except Exception as e:
            print(f"[WARN] LLM fallback failed: {e}")
    
    # Rule-based fallback: select top N by simple heuristics
    print("[EDGE] Using rule-based fallback selection...")
    
    candidates = []
    for col, meta in column_metadata.items():
        if meta.get("missing_pct", 100) < 50:  # At least 50% present
            score = 0
            if meta.get("is_numeric"):
                score = meta.get("variance", 0)
            else:
                score = meta.get("cardinality", 0)
            candidates.append((col, score, meta))
    
    # Sort by score, take top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    fallback_plots = []
    for col, score, meta in candidates[:min_plots]:
        if meta.get("is_numeric"):
            plot_type = "histogram"
            category = "univariate_numeric"
        else:
            plot_type = "barplot"
            category = "univariate_categorical"
        
        fallback_plots.append({
            "column": col,
            "plot_type": plot_type,
            "category": category,
            "priority_score": 0,
            "reason": "Fallback selection (top by variance/cardinality)"
        })
    
    return fallback_plots


# =====================================================
# 7. CONVENIENCE FUNCTIONS
# =====================================================

def detect_edge_cases(df: pd.DataFrame, column_metadata: Dict[str, Dict]) -> List[EdgeCase]:
    """Convenience function to detect edge cases."""
    detector = EdgeCaseDetector()
    return detector.detect_all(df, column_metadata)


def process_edge_cases(
    df: pd.DataFrame, 
    column_metadata: Dict[str, Dict],
    groq_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to process all edge cases.
    
    Args:
        df: Input DataFrame
        column_metadata: Pre-computed column metadata
        groq_api_key: Optional Groq API key (defaults to env var)
        
    Returns:
        Dictionary with all edge case resolutions
    """
    config = EdgeCaseConfig(groq_api_key=groq_api_key)
    handler = EdgeCaseHandler(config)
    return handler.process(df, column_metadata)


def sanitize_column_name(col: str, max_length: int = 50) -> str:
    """Sanitize a column name for use in filenames."""
    # Replace special characters
    safe = re.sub(r'[\/\\:*?"<>|]', '_', col)
    # Replace spaces
    safe = safe.replace(' ', '_')
    # Truncate if needed
    if len(safe) > max_length:
        safe = safe[:max_length-3] + "..."
    return safe
