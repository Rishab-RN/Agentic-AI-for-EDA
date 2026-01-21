import json
import numpy as np
import pandas as pd
from typing import TypedDict, Dict

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate



class EDAState(TypedDict):
    df: pd.DataFrame
    missing_report: Dict
    decision: Dict
    report: Dict
    uncertain_columns: list 
    llm_called: bool 



MISSING_TOKENS = {
    "", " ", "  ", ".", "..", "...", "-", "--","na", "n/a", "none", "null", "nil", "nan",
    "unknown", "no info","No Info","NULL", "NO INFO", "....","Null","Unknown","UNKNOWN",
}

def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    def clean_cell(x):
        if pd.isna(x):
            return np.nan
        x_str = str(x).strip().lower()
        return np.nan if x_str in MISSING_TOKENS else x

    for col in df.columns:
        df[col] = df[col].map(clean_cell)
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df




def analyze_missing_values(state: EDAState) -> EDAState:
    df = state["df"]
    report = {}

    for col in df.columns:
        report[col] = {
            "missing_pct": round(df[col].isna().mean() * 100, 2),
            "dtype": str(df[col].dtype)
        }

    
    report["_GLOBAL_"] = {
        "row_loss_pct": round(df.isna().any(axis=1).mean() * 100, 2)
    }

    state["missing_report"] = report
    
    
    state["uncertain_columns"] = identify_edge_cases(df, report)
    state["llm_called"] = False
    
    return state


def identify_edge_cases(df: pd.DataFrame, missing_report: Dict) -> list:
    """
    Identifies columns with borderline/ambiguous cases where LLM reasoning may help.
    
    Edge cases include:
    - Borderline missing percentage (30-40%): unclear if we should drop or impute
    - Mixed data types: column has both numbers and text
    - Potential ID columns: might be identifiers that shouldn't be imputed
    - Unusual patterns: very high cardinality categorical columns
    """
    uncertain = []
    
    for col in df.columns:
        if col == "_GLOBAL_" or col not in missing_report:
            continue
            
        info = missing_report[col]
        missing_pct = info["missing_pct"]
        
        # Edge Case 1: Borderline missing percentage (30-40%)
        if 30 <= missing_pct <= 40:
            uncertain.append({
                "column": col,
                "reason": f"borderline_missing_pct ({missing_pct}%)",
                "missing_pct": missing_pct
            })
            continue
        
       
        if df[col].dtype == "object":
            unique_ratio = df[col].nunique() / len(df[col].dropna()) if len(df[col].dropna()) > 0 else 0
            if unique_ratio > 0.9 and missing_pct > 0:
                uncertain.append({
                    "column": col,
                    "reason": "potential_id_column",
                    "missing_pct": missing_pct
                })
                continue
        
       
        special_keywords = ['id', 'key', 'code', 'uuid', 'date', 'time', 'timestamp']
        if any(kw in col.lower() for kw in special_keywords) and missing_pct > 5:
            uncertain.append({
                "column": col,
                "reason": "special_column_name",
                "missing_pct": missing_pct
            })
    
    return uncertain




def compress_missing_report(missing_report: dict) -> str:
    lines = []

    for col, info in missing_report.items():
        if col == "_GLOBAL_":
            continue
        lines.append(
            f"{col}: {info['missing_pct']}% missing, dtype={info['dtype']}"
        )

    global_loss = missing_report.get("_GLOBAL_", {}).get("row_loss_pct", 0)
    lines.append(f"GLOBAL_ROW_LOSS: {global_loss}%")

    return "\n".join(lines)




llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

def decide_missing_value_strategy(state: EDAState) -> EDAState:
    """
    CONDITIONAL LLM CALLING:
    - If there are uncertain/edge cases → Call LLM for those specific columns
    - Otherwise → Skip LLM entirely (rules will handle it)
    """
    uncertain_columns = state.get("uncertain_columns", [])
    
    if not uncertain_columns:
        print("[OK] No edge cases detected. Using rule-based approach (LLM skipped).")
        state["decision"] = {}
        state["llm_called"] = False
        return state
    
    print(f"[WARN] Edge cases detected in {len(uncertain_columns)} column(s). Consulting LLM...")
    
   
    edge_case_summary = "\n".join([
        f"- {item['column']}: {item['missing_pct']}% missing, reason={item['reason']}"
        for item in uncertain_columns
    ])
    
    edge_case_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert data analyst. The following columns have BORDERLINE/UNCERTAIN cases:

    {edge_cases}
    
    Full dataset context:
    {missing_report}

    IMPORTANT RULES:
    1. ID/KEY COLUMNS (columns with 'id', 'key', 'uuid', 'code' in name):
       - ALWAYS use "drop_rows" (rows without ID are useless - they can't be identified)
       - Never impute ID columns (you cannot guess an ID)
       - Never drop the column itself (ID is essential for the dataset)
    
    2. TIME/DATE COLUMNS:
       - Consider "forward_fill" for time-series continuity
       - Or "drop_rows" if temporal order doesn't matter
    
    3. BORDERLINE COLUMNS (30-40% missing):
       - "drop_column" if not critical to analysis
       - Impute with "median"/"mean"/"mode" if important feature

    Available actions:
    - "drop_rows": Remove rows with missing values (USE THIS FOR ID COLUMNS)
    - "drop_column": Remove the entire column
    - "median"/"mean": Numerical imputation
    - "mode": Categorical imputation
    - "forward_fill": Time-series imputation

    Return ONLY valid JSON for the uncertain columns.
    Example: {{ "id": {{ "action": "drop_rows", "reason": "ID column - rows without ID are useless" }} }}
    """
    )
    
    full_summary = compress_missing_report(state["missing_report"])
    prompt = edge_case_prompt.format(
        edge_cases=edge_case_summary,
        missing_report=full_summary
    )

    response = llm.invoke(prompt)
    state["llm_called"] = True

    try:
        decision = json.loads(response.content)
        print(f"[LLM] LLM recommendations: {decision}")
    except json.JSONDecodeError:
        print(f"[WARN] LLM returned invalid JSON. Falling back to rules.")
        decision = {}

    state["decision"] = decision
    return state




# def apply_missing_value_strategy(state: EDAState) -> EDAState:
#     df = state["df"]
#     report = state["report"]

#     report["column_actions"] = {}
#     report["imputed_values"] = {}
#     report["dropped_columns"] = []
#     report["rows_dropped"] = 0

#     original_rows = len(df)
#     global_loss = state["missing_report"]["_GLOBAL_"]["row_loss_pct"]

#     for col, rule in state["decision"].items():
#         if col not in df.columns:
#             continue

#         action = rule["action"]
#         missing_before = df[col].isna().sum()

#         # SAFETY OVERRIDE
#         if action == "drop_rows" and global_loss > 5:
#             action = "median" if df[col].dtype != "object" else "mode"

#         report["column_actions"][col] = action

#         if action == "drop_rows":
#             df = df.dropna(subset=[col])

#         elif action == "drop_column":
#             df = df.drop(columns=[col])
#             report["dropped_columns"].append(col)

#         elif action == "median":
#             df[col] = df[col].fillna(df[col].median())
#             report["imputed_values"][col] = missing_before

#         elif action == "mean":
#             df[col] = df[col].fillna(df[col].mean())
#             report["imputed_values"][col] = missing_before

#         elif action == "mode":
#             df[col] = df[col].fillna(df[col].mode()[0])
#             report["imputed_values"][col] = missing_before

#     report["rows_dropped"] = original_rows - len(df)
#     report["final_shape"] = df.shape
#     report["global_row_loss_pct"] = global_loss

#     state["df"] = df
#     state["report"] = report
#     return state


def is_skewed(series: pd.Series) -> bool:
    """
    Determines whether a numerical column is skewed.
    """
    
    clean = series.dropna()

    # Not enough data → unknown distribution
    if len(clean) < 10:
        return True  # treat as skewed

    skewness = abs(clean.skew())

    return skewness > 0.5


# def resolve_final_action(col, df, missing_pct, global_loss, suggested_action):
#     """
#     Deterministic and statistically correct decision maker.
#     """

#     # Rule 1: Drop column if too many missing values
#     if missing_pct > 40:
#         return "drop_column"

#     # Rule 2: Drop rows only if global loss allows
#     if missing_pct < 5 and global_loss <= 5:
#         return "drop_rows"

#     # Rule 3: Categorical → mode
#     if df[col].dtype == "object":
#         return "mode"

#     # Rule 4: Numerical → mean or median
#     if is_skewed(df[col]):
#         return "median"
#     else:
#         return "mean"


from typing import List, Tuple, Set


def select_columns_for_row_drop(
    df: pd.DataFrame,
    candidate_columns: List[str],
    max_row_loss_pct: float = 5.0
) -> Tuple[List[str], Set[int]]:
    """
    Selects a subset of columns for which row deletion is allowed
    such that total row loss does NOT exceed max_row_loss_pct.
    """

    total_rows = len(df)
    max_rows_allowed = int((max_row_loss_pct / 100) * total_rows)

    rows_to_drop: Set[int] = set()
    allowed_columns: List[str] = []

    sorted_columns = sorted(
        candidate_columns,
        key=lambda col: df[col].isna().sum()
    )

    for col in sorted_columns:
        missing_rows = set(df.index[df[col].isna()])
        new_rows = missing_rows - rows_to_drop

        projected_loss = len(rows_to_drop | new_rows)

        if projected_loss <= max_rows_allowed:
            allowed_columns.append(col)
            rows_to_drop |= new_rows

    return allowed_columns, rows_to_drop



# def apply_missing_value_strategy(state: EDAState) -> EDAState:
#     df = state["df"]
#     report = state["report"]

#     report["column_actions"] = {}
#     report["imputed_values"] = {}
#     report["dropped_columns"] = []
#     report["rows_dropped"] = 0

#     original_rows = len(df)
#     global_loss = state["missing_report"]["_GLOBAL_"]["row_loss_pct"]

#     for col in df.columns:
#         info = state["missing_report"][col]
#         missing_pct = info["missing_pct"]

#         # LLM suggestion (optional)
#         suggested = state["decision"].get(col, {}).get("action", "median")

#         # 🔒 FINAL AUTHORITY
#         action = resolve_final_action(
#             col, df, missing_pct, global_loss, suggested
#         )

#         report["column_actions"][col] = action
#         missing_before = df[col].isna().sum()

#         if action == "drop_rows":
#             df = df.dropna(subset=[col])

#         elif action == "drop_column":
#             df = df.drop(columns=[col])
#             report["dropped_columns"].append(col)

#         elif action == "median":
#             df[col] = df[col].fillna(df[col].median())
#             report["imputed_values"][col] = missing_before

#         elif action == "mean":
#             df[col] = df[col].fillna(df[col].mean())
#             report["imputed_values"][col] = missing_before

#         elif action == "mode":
#             df[col] = df[col].fillna(df[col].mode()[0])
#             report["imputed_values"][col] = missing_before

#     report["rows_dropped"] = original_rows - len(df)
#     report["final_shape"] = df.shape
#     report["global_row_loss_pct"] = global_loss

#     state["df"] = df
#     state["report"] = report
#     return state


def apply_missing_value_strategy(state: EDAState) -> EDAState:
    """
    Applies cleaning strategy using:
    1. LLM decisions for edge cases (if available)
    2. Rule-based logic for everything else
    """
    df = state["df"]
    report = state["report"]
    llm_decisions = state.get("decision", {})
    uncertain_col_names = [item["column"] for item in state.get("uncertain_columns", [])]

    report["column_actions"] = {}
    report["imputed_values"] = {}
    report["dropped_columns"] = []
    report["rows_dropped"] = 0
    report["llm_guided_columns"] = [] 

    original_rows = len(df)
    global_loss = state["missing_report"]["_GLOBAL_"]["row_loss_pct"]

    candidate_drop_cols = [
        col for col, info in state["missing_report"].items()
        if col != "_GLOBAL_" and info["missing_pct"] < 5
    ]

    allowed_drop_cols, rows_to_drop = select_columns_for_row_drop(
        df=df,
        candidate_columns=candidate_drop_cols,
        max_row_loss_pct=5
    )

    for col in df.columns:
        if col not in state["missing_report"]:
            continue

        missing_pct = state["missing_report"][col]["missing_pct"]
        missing_before = df[col].isna().sum()  # Count missing values
        
        if missing_before == 0:
            report["column_actions"][col] = "no_missing"
            continue
        
        if col in uncertain_col_names and col in llm_decisions:
            llm_action = llm_decisions[col].get("action", None)
            
            if llm_action:
                report["llm_guided_columns"].append(col)
                
                if llm_action == "drop_column":
                    df = df.drop(columns=[col])
                    report["column_actions"][col] = "drop_column (LLM)"
                    report["dropped_columns"].append(col)
                    continue
                    
                elif llm_action == "drop_rows":
                    df = df.dropna(subset=[col])
                    report["column_actions"][col] = f"drop_rows (LLM) - removed {missing_before} rows"
                    continue
                    
                elif llm_action == "keep_null":
                    report["column_actions"][col] = "keep_null (LLM)"
                    continue
                    
                elif llm_action == "forward_fill":
                    df[col] = df[col].ffill()
                    report["column_actions"][col] = f"forward_fill (LLM) - filled {missing_before} values"
                    report["imputed_values"][col] = missing_before
                    continue
                    
                elif llm_action in ["median", "mean", "mode"]:
                    if llm_action == "median" and df[col].dtype != "object":
                        df[col] = df[col].fillna(df[col].median())
                    elif llm_action == "mean" and df[col].dtype != "object":
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        mode_vals = df[col].mode()
                        if len(mode_vals) > 0:
                            df[col] = df[col].fillna(mode_vals[0])
                    
                    report["column_actions"][col] = f"{llm_action} (LLM) - filled {missing_before} values"
                    report["imputed_values"][col] = missing_before
                    continue

        
        if col in allowed_drop_cols:
            df = df.dropna(subset=[col])
            report["column_actions"][col] = f"drop_rows - removed {missing_before} rows"
            continue

        if missing_pct > 40:
            df = df.drop(columns=[col])
            report["column_actions"][col] = f"drop_column ({missing_pct}% missing)"
            report["dropped_columns"].append(col)
            continue

        # Imputation
        if df[col].dtype == "object":
            mode_values = df[col].mode()
            if len(mode_values) > 0:
                df[col] = df[col].fillna(mode_values[0])
                report["column_actions"][col] = f"mode - filled {missing_before} values"
            else:
                df[col] = df[col].fillna("Unknown")
                report["column_actions"][col] = f"mode_unknown - filled {missing_before} values"
        else:
            if is_skewed(df[col]):
                df[col] = df[col].fillna(df[col].median())
                report["column_actions"][col] = f"median - filled {missing_before} values"
            else:
                df[col] = df[col].fillna(df[col].mean())
                report["column_actions"][col] = f"mean - filled {missing_before} values"

        report["imputed_values"][col] = missing_before

    report["rows_dropped"] = original_rows - len(df)
    report["final_shape"] = df.shape
    report["global_row_loss_pct"] = global_loss
    report["llm_was_called"] = state.get("llm_called", False)

    state["df"] = df
    state["report"] = report
    return state





def build_graph():
    graph = StateGraph(EDAState)

    graph.add_node("analyze_missing", analyze_missing_values)
    graph.add_node("decide_strategy", decide_missing_value_strategy)
    graph.add_node("apply_cleaning", apply_missing_value_strategy)

    graph.set_entry_point("analyze_missing")

    graph.add_edge("analyze_missing", "decide_strategy")
    graph.add_edge("decide_strategy", "apply_cleaning")
    graph.add_edge("apply_cleaning", END)

    return graph.compile()




def run_agent(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        engine="python",
        sep=",",
        quotechar='"',
        escapechar="\\",
        encoding="utf-8",
        on_bad_lines="warn"
    )

    df = normalize_missing_values(df)

    state: EDAState = {
        "df": df,
        "missing_report": {},
        "decision": {},
        "report": {
            "original_shape": df.shape
        },
        "uncertain_columns": [],  
        "llm_called": False       
    }

    graph = build_graph()
    final_state = graph.invoke(state)

    print("\n[REPORT] CLEANING REPORT")
    print("=" * 60)
    
   
    if final_state["report"].get("llm_was_called", False):
        llm_guided = final_state["report"].get("llm_guided_columns", [])
        print(f"[LLM] LLM was consulted for {len(llm_guided)} edge case column(s): {llm_guided}")
    else:
        print("[FAST] Pure rule-based cleaning (LLM not needed)")
    
    print("-" * 60)
    for k, v in final_state["report"].items():
        if k not in ["llm_was_called", "llm_guided_columns"]:
            print(f"{k}: {v}")

    return final_state["df"], final_state["report"]




def get_dataset_path() -> str:
    """
    Interactively prompts the user for a dataset path and validates it.
    """
    import os
    
    while True:
        print("\n" + "=" * 60)
        print("🔍 AGENTIC MISSING VALUE CLEANER")
        print("=" * 60)
        
        dataset_path = input("\n Enter the path to your CSV dataset: ").strip()
        
        dataset_path = dataset_path.strip('"').strip("'")
        
        if not dataset_path:
            print(" Error: Please provide a valid file path.")
            continue
        
        if not os.path.exists(dataset_path):
            print(f"Error: File not found at '{dataset_path}'")
            print("   Please check the path and try again.")
            continue
        
        if not dataset_path.lower().endswith('.csv'):
            print(" Warning: The file does not have a .csv extension.")
            confirm = input("   Do you want to continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        return dataset_path


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f" Error: File not found at '{input_file}'")
            sys.exit(1)
    else:
        input_file = get_dataset_path()

    print(f"\n Processing: {input_file}")
    print("-" * 60)

    try:
        cleaned_df, cleaning_report = run_agent(input_file)

        input_dir = os.path.dirname(os.path.abspath(input_file))
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(input_dir, f"{base_name}_cleaned.csv")

        cleaned_df.to_csv(output_file, index=False)

        print("\n Cleaning completed successfully!")
        print(f" Output saved as: {output_file}")
        print(f" Original shape: {cleaning_report.get('original_shape', 'N/A')}")
        print(f" Final shape: {cleaning_report.get('final_shape', 'N/A')}")
        print(f"  Rows dropped: {cleaning_report.get('rows_dropped', 0)}")
        print(f"  Columns dropped: {cleaning_report.get('dropped_columns', [])}")

    except FileNotFoundError as e:
        print(f" Error: Could not find the file - {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(" Error: The CSV file is empty.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f" Error: Failed to parse CSV file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f" Unexpected error: {e}")
        raise


