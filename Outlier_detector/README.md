# Outlier Detector Agent

An intelligent, LLM-powered outlier detection and treatment module for data cleaning pipelines.

## Features

- **Robust Numeric Parsing**: Handles messy formats like "19M", "10,000+", "$500", "50kg"
- **LLM-Powered Classification**: Uses LLM to distinguish between:
  - **Categorical data** (e.g., "BMW 320") - preserved as-is
  - **Dirty numeric data** (e.g., "19M", "Varies with device") - cleaned and converted
- **Multiple Detection Methods**: IQR, Z-score, Isolation Forest
- **Smart Treatment**: Automatic switching between row removal and capping based on data loss threshold

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Basic Usage

```python
import pandas as pd
from outlier_agent_node import outlier_agent_node

# Load your data
df = pd.read_csv("your_data.csv")

# Run the outlier agent
state = {"data": df}
result = outlier_agent_node(state)

# Get cleaned data
cleaned_df = result["data"]

# View treatment log
for entry in result.get("log", []):
    print(entry)

# Save cleaned data
cleaned_df.to_csv("cleaned_data.csv", index=False)
```

### With Custom Config

```python
state = {
    "data": df,
    "config": {
        "aggressive_mode": True,      # Remove outliers vs cap them
        "iqr_multiplier": 1.5,        # IQR outlier threshold
        "zscore_threshold": 2.5,      # Z-score threshold
        "use_llm": True,              # Use LLM for classification
    }
}
result = outlier_agent_node(state)
```

## File Structure

```
Outlier_detector/
├── outlier_agent_node.py    # Main agent logic
├── graph_runner.py          # LangGraph workflow runner
├── llm_utils.py             # LLM utility functions
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | Isolation Forest algorithm |
| langchain-groq | LLM integration (optional) |
| langgraph | Workflow orchestration (optional) |
| python-dotenv | Environment variable loading |

## Degraded Mode

The module works without LLM/LangGraph dependencies:
- Falls back to heuristic-based column classification
- Core outlier detection still fully functional

## License

MIT License
