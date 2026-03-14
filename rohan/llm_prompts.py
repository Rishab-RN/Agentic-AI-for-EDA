# =====================================================
# LLM Prompts for Edge Case Resolution
# =====================================================
# Structured prompts for Groq Compound LLM
# Used by edge_case_handler.py for semantic understanding
# =====================================================

from typing import Dict, List, Any
import json


# =====================================================
# 1. SYSTEM PROMPTS
# =====================================================

SYSTEM_PROMPT_MAIN = """You are an expert data analyst assistant specializing in Exploratory Data Analysis (EDA).
Your task is to analyze column information and provide recommendations for visualization.

RULES:
1. ALWAYS respond with valid JSON
2. Be concise and actionable
3. Base decisions on column names, data types, and sample values
4. If uncertain, indicate lower confidence

OUTPUT FORMAT:
{
    "decisions": [
        {
            "column": "column_name",
            "decision": "your_decision",
            "confidence": 0.0-1.0,
            "reason": "brief explanation"
        }
    ]
}"""


SYSTEM_PROMPT_CLASSIFICATION = """You are a data type classification expert.
Analyze column metadata and classify columns into appropriate types for visualization.

COLUMN TYPES:
- datetime: Timestamp, date columns → time-series plots
- id: Identifiers, keys → should NOT be visualized
- target: Prediction target → priority visualization
- numeric: Continuous values → histograms, boxplots
- categorical: Discrete categories → bar charts, count plots
- ordinal: Ordered categories → ordered bar charts
- binary: Two values only → pie charts or bar charts
- text: Free-form text → should NOT be visualized

ALWAYS respond with valid JSON."""


SYSTEM_PROMPT_FALLBACK = """You are helping select the most important columns for visualization.
The standard pipeline selected zero columns, so you must pick the most meaningful ones.

CRITERIA FOR SELECTION:
1. Business relevance (prices, counts, revenue, sales)
2. Data quality (complete, varied values)
3. Interesting patterns (high variance, skewness)
4. Common analytical targets

Select columns that would provide the most insight to a data analyst.
ALWAYS respond with valid JSON."""


# =====================================================
# 2. TASK-SPECIFIC PROMPTS
# =====================================================

def get_datetime_detection_prompt(column_info: Dict[str, Any]) -> str:
    """
    Prompt for detecting datetime columns.
    
    Args:
        column_info: Dict with column names and sample values
        
    Returns:
        Formatted prompt string
    """
    return f"""Analyze these columns and determine if they are datetime/timestamp columns.

COLUMNS TO ANALYZE:
{json.dumps(column_info, indent=2)}

INDICATORS OF DATETIME:
- Column names: date, time, timestamp, created, updated, modified, _at, _on
- Value patterns: YYYY-MM-DD, ISO formats, Unix timestamps
- Data types: datetime64, object with date strings

FOR EACH COLUMN, RETURN:
{{
    "column": "column_name",
    "is_datetime": true/false,
    "datetime_format": "detected format or null",
    "time_plot_type": "line" | "area" | "none",
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}"""


def get_id_column_detection_prompt(column_info: Dict[str, Any]) -> str:
    """
    Prompt for detecting ID/key columns.
    
    Args:
        column_info: Dict with column names and sample values
        
    Returns:
        Formatted prompt string
    """
    return f"""Analyze these columns and determine if they are ID/key columns that should NOT be visualized.

COLUMNS TO ANALYZE:
{json.dumps(column_info, indent=2)}

INDICATORS OF ID COLUMNS:
- Names: id, _id, pk, key, index, uuid, guid, code
- Sequential integers: 1, 2, 3, 4, 5...
- UUID patterns: a1b2c3d4-e5f6-...
- High uniqueness ratio (>90% unique values)
- No analytical meaning

FOR EACH COLUMN, RETURN:
{{
    "column": "column_name",
    "is_id_column": true/false,
    "id_type": "sequential" | "uuid" | "code" | null,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}"""


def get_target_variable_prompt(column_info: Dict[str, Any]) -> str:
    """
    Prompt for detecting target/label columns.
    
    Args:
        column_info: Dict with column names and sample values
        
    Returns:
        Formatted prompt string
    """
    return f"""Analyze these columns and determine if any is a TARGET variable for prediction/analysis.

COLUMNS TO ANALYZE:
{json.dumps(column_info, indent=2)}

INDICATORS OF TARGET COLUMNS:
- Names: target, label, y, outcome, result, prediction, class
- Business metrics: price, sales, revenue, profit, count, amount
- Binary outcomes: survived, churn, default, fraud, success
- The column that other features would predict

FOR EACH COLUMN, RETURN:
{{
    "column": "column_name",
    "is_target": true/false,
    "target_type": "regression" | "classification" | null,
    "importance": "high" | "medium" | "low",
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}"""


def get_binary_numeric_prompt(column_info: Dict[str, Any]) -> str:
    """
    Prompt for handling binary numeric columns.
    
    Args:
        column_info: Dict with column names and sample values
        
    Returns:
        Formatted prompt string
    """
    return f"""These columns contain exactly 2 unique numeric values (like 0/1, 1/2).
Determine the best visualization approach for each.

COLUMNS TO ANALYZE:
{json.dumps(column_info, indent=2)}

DECISIONS:
- "bar": For flag/indicator columns (is_active, has_discount)
- "pie": For proportion visualization (gender, yes/no)
- "histogram": If the 2 values represent continuous scale endpoints

FOR EACH COLUMN, RETURN:
{{
    "column": "column_name",
    "plot_type": "bar" | "pie" | "histogram",
    "is_flag_column": true/false,
    "flag_meaning": "description of what 0 and 1 mean" or null,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}"""


def get_ambiguous_type_prompt(column_info: Dict[str, Any]) -> str:
    """
    Prompt for resolving ambiguous numeric/categorical columns.
    
    Args:
        column_info: Dict with column names and sample values
        
    Returns:
        Formatted prompt string
    """
    return f"""These numeric columns have very few unique integer values.
Determine if they should be treated as CATEGORICAL or NUMERIC.

COLUMNS TO ANALYZE:
{json.dumps(column_info, indent=2)}

GUIDANCE:
- Rating scales (1-5, 1-10) → CATEGORICAL (bar chart)
- Encoded categories (1=Low, 2=Medium, 3=High) → CATEGORICAL
- Counts or measurements → NUMERIC (histogram)
- Years → Could be either based on context
- Age groups (coded as 1, 2, 3) → CATEGORICAL

FOR EACH COLUMN, RETURN:
{{
    "column": "column_name",
    "treat_as": "categorical" | "numeric",
    "suggested_plot": "bar" | "histogram" | "boxplot",
    "possible_encoding": "description of what numbers might mean" or null,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}"""


def get_ordinal_ordering_prompt(column_info: Dict[str, Any]) -> str:
    """
    Prompt for determining ordinal category ordering.
    
    Args:
        column_info: Dict with column names and unique values
        
    Returns:
        Formatted prompt string
    """
    return f"""These categorical columns might have a natural ordering.
Determine the correct sort order if they are ordinal.

COLUMNS TO ANALYZE:
{json.dumps(column_info, indent=2)}

COMMON ORDINAL PATTERNS:
- Size: Small < Medium < Large < Extra Large
- Frequency: Never < Rarely < Sometimes < Often < Always
- Rating: Poor < Fair < Good < Excellent
- Priority: Low < Medium < High < Critical
- Education: High School < Bachelor < Master < PhD
- Income: Low < Middle < Upper Middle < High

FOR EACH COLUMN, RETURN:
{{
    "column": "column_name",
    "is_ordinal": true/false,
    "order": ["value1", "value2", "value3"] or null,
    "pattern_type": "size" | "frequency" | "rating" | "custom" | null,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}"""


def get_column_grouping_prompt(columns: List[str]) -> str:
    """
    Prompt for identifying related column groups.
    
    Args:
        columns: List of column names
        
    Returns:
        Formatted prompt string
    """
    return f"""Identify groups of related columns that represent the same concept or should be analyzed together.

ALL COLUMNS:
{json.dumps(columns, indent=2)}

COMMON GROUPINGS:
- Name components: first_name, last_name, full_name
- Address components: street, city, state, zip, country
- Date ranges: start_date, end_date, duration
- Contact info: email, phone, mobile
- Dimensions: length, width, height
- Prices: price, discount, tax, total

RETURN:
{{
    "column_groups": [
        {{
            "group_name": "descriptive name",
            "columns": ["col1", "col2", "col3"],
            "relationship": "components" | "range" | "variants" | "related"
        }}
    ],
    "ungrouped_columns": ["columns that don't belong to any group"]
}}"""


def get_fallback_selection_prompt(
    column_summaries: List[Dict[str, Any]], 
    num_to_select: int
) -> str:
    """
    Prompt for fallback column selection when nothing was selected.
    
    Args:
        column_summaries: List of column info dicts
        num_to_select: How many columns to select
        
    Returns:
        Formatted prompt string
    """
    return f"""The standard visualization pipeline selected ZERO columns for plotting.
Your job is to select the {num_to_select} MOST IMPORTANT columns to visualize anyway.

AVAILABLE COLUMNS:
{json.dumps(column_summaries, indent=2)}

SELECTION CRITERIA (in order of importance):
1. Business/analytical relevance (prices, sales, outcomes, counts)
2. Data completeness (fewer missing values is better)
3. Value variety (more unique values = more interesting)
4. Clear meaning (obvious column names over cryptic ones)

AVOID SELECTING:
- ID columns
- Columns with >50% missing values
- Constant columns (one value)
- Cryptic technical columns

RETURN:
{{
    "selected_columns": [
        {{
            "column": "column_name",
            "plot_type": "histogram" | "boxplot" | "bar" | "pie",
            "importance_rank": 1-{num_to_select},
            "reason": "why this column is important"
        }}
    ],
    "selection_rationale": "overall explanation of selection strategy"
}}"""


# =====================================================
# 3. PROMPT BUILDERS
# =====================================================

class PromptBuilder:
    """Builds prompts for different edge case types."""
    
    @staticmethod
    def build_column_info(
        df, 
        columns: List[str], 
        column_metadata: Dict[str, Dict],
        max_samples: int = 5,
        max_columns: int = 30
    ) -> Dict[str, Any]:
        """
        Build column information dict for prompts.
        
        Args:
            df: DataFrame
            columns: Columns to include
            column_metadata: Pre-computed metadata
            max_samples: Max sample values per column
            max_columns: Max columns to include (for token limits)
            
        Returns:
            Dict with column information
        """
        column_info = {}
        
        for col in columns[:max_columns]:
            meta = column_metadata.get(col, {})
            
            # Get sample values
            sample_vals = df[col].dropna().head(max_samples).tolist()
            sample_vals = [str(v) for v in sample_vals]  # Convert to strings
            
            column_info[col] = {
                "dtype": meta.get("dtype", str(df[col].dtype)),
                "sample_values": sample_vals,
                "unique_count": meta.get("cardinality", df[col].nunique()),
                "missing_pct": round(meta.get("missing_pct", 0), 2),
                "is_numeric": meta.get("is_numeric", False)
            }
            
            # Add stats for numeric columns
            if meta.get("is_numeric"):
                column_info[col]["min"] = meta.get("min")
                column_info[col]["max"] = meta.get("max")
                column_info[col]["mean"] = round(meta.get("mean", 0), 2) if meta.get("mean") else None
        
        return column_info
    
    @staticmethod
    def build_column_summaries(
        column_metadata: Dict[str, Dict],
        priority_scores: Dict[str, Dict],
        max_columns: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Build column summaries for fallback selection.
        
        Args:
            column_metadata: Pre-computed metadata
            priority_scores: Computed priority scores
            max_columns: Max columns to include
            
        Returns:
            List of column summary dicts
        """
        summaries = []
        
        for col, meta in list(column_metadata.items())[:max_columns]:
            score = priority_scores.get(col, {}).get("score", 0)
            
            summary = {
                "name": col,
                "type": "numeric" if meta.get("is_numeric") else "categorical",
                "score": score,
                "unique_count": meta.get("cardinality", meta.get("valid_count", 0)),
                "missing_pct": round(meta.get("missing_pct", 0), 2)
            }
            
            summaries.append(summary)
        
        return summaries


# =====================================================
# 4. RESPONSE PARSERS
# =====================================================

class ResponseParser:
    """Parses LLM responses into structured data."""
    
    @staticmethod
    def parse_decisions(response_json: Dict) -> List[Dict]:
        """
        Parse decisions from LLM response.
        
        Args:
            response_json: Parsed JSON from LLM
            
        Returns:
            List of decision dicts
        """
        decisions = response_json.get("decisions", [])
        
        # Handle case where response is already a list
        if isinstance(response_json, list):
            decisions = response_json
        
        # Handle single decision not in a list
        if not isinstance(decisions, list):
            decisions = [decisions] if decisions else []
        
        return decisions
    
    @staticmethod
    def parse_column_groups(response_json: Dict) -> Dict[str, List[str]]:
        """
        Parse column groupings from LLM response.
        
        Args:
            response_json: Parsed JSON from LLM
            
        Returns:
            Dict mapping group name to column list
        """
        groups = {}
        
        for group in response_json.get("column_groups", []):
            group_name = group.get("group_name", "unnamed")
            columns = group.get("columns", [])
            if columns:
                groups[group_name] = columns
        
        return groups
    
    @staticmethod
    def parse_fallback_selection(response_json: Dict) -> List[Dict]:
        """
        Parse fallback selection from LLM response.
        
        Args:
            response_json: Parsed JSON from LLM
            
        Returns:
            List of selected column dicts
        """
        return response_json.get("selected_columns", [])
    
    @staticmethod
    def validate_response(response: Dict, required_fields: List[str]) -> bool:
        """
        Validate that response contains required fields.
        
        Args:
            response: Response dict
            required_fields: List of required field names
            
        Returns:
            True if all required fields present
        """
        for field in required_fields:
            if field not in response:
                return False
        return True


# =====================================================
# 5. TESTING
# =====================================================

if __name__ == "__main__":
    # Test prompt generation
    print("=" * 60)
    print("LLM PROMPTS TEST")
    print("=" * 60)
    
    # Sample column info
    test_column_info = {
        "created_at": {
            "dtype": "object",
            "sample_values": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "unique_count": 100,
            "missing_pct": 0
        },
        "customer_id": {
            "dtype": "int64",
            "sample_values": ["1", "2", "3", "4", "5"],
            "unique_count": 1000,
            "missing_pct": 0
        },
        "price": {
            "dtype": "float64",
            "sample_values": ["10.5", "20.3", "15.7"],
            "unique_count": 500,
            "missing_pct": 2.5
        }
    }
    
    print("\n[TEST] Datetime Detection Prompt:")
    print("-" * 40)
    prompt = get_datetime_detection_prompt(test_column_info)
    print(prompt[:500] + "...")
    
    print("\n[TEST] ID Column Detection Prompt:")
    print("-" * 40)
    prompt = get_id_column_detection_prompt(test_column_info)
    print(prompt[:500] + "...")
    
    print("\n[TEST] Fallback Selection Prompt:")
    print("-" * 40)
    summaries = [
        {"name": "price", "type": "numeric", "score": 2, "unique_count": 100},
        {"name": "category", "type": "categorical", "score": 1, "unique_count": 5}
    ]
    prompt = get_fallback_selection_prompt(summaries, 2)
    print(prompt[:500] + "...")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
