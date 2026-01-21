# Visualization Agent

An intelligent, LLM-powered visualization agent for automated EDA (Exploratory Data Analysis).

## Features

- **Intelligent Plot Selection**: Uses LLM + local heuristics to choose the most relevant visualizations
- **Data-Driven Insights**: Generates specific statistical insights for each plot (skewness, correlation, top categories)
- **Multiple Plot Types**: Histograms, Scatter plots, Box plots, Violin plots, Heatmaps, Bar charts, KDE plots, Q-Q plots
- **Edge Case Handling**: Robust handling of missing data, zero-inflation, high cardinality
- **Plot Curation**: Filters redundant plots and prioritizes impactful visualizations

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
from visualization_agent import visualization_agent_node, generate_all_selected_plots

# Load your data
df = pd.read_csv("your_data.csv")

# Step 1: Get plot selections from the agent
state = {"df": df}
result = visualization_agent_node(state)

# Step 2: Generate the selected plots
selected_plots = result.get("selected_plots", [])
output_dir = "./plots"
generation_results = generate_all_selected_plots(df, selected_plots, output_dir)

# View generated plots
for plot in generation_results.get("plots_generated", []):
    print(f"Plot: {plot['filepath']}")
    print(f"Insight: {plot['inference']}")
```

### With Custom Target Variable

```python
state = {
    "df": df,
    "target_column": "Price",  # Specify target for bivariate analysis
    "config": {
        "max_plots": 15,
        "include_correlations": True,
    }
}
result = visualization_agent_node(state)
```

## File Structure

```
visualization_agent/
├── __init__.py              # Package exports
├── viz_agent_node.py        # Main agent logic (49KB)
├── agentic_viz_agent.py     # LLM-powered agent (41KB)
├── plot_generator.py        # Plot generation functions (47KB)
├── local_intel.py           # Local semantic engine (11KB)
├── llm_prompts.py           # LLM prompt templates (16KB)
├── edge_case_handler.py     # Edge case detection (42KB)
├── graph_runner.py          # LangGraph workflow (13KB)
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas, numpy | Data manipulation |
| scipy | Statistical calculations |
| matplotlib, seaborn | Visualization |
| langchain-groq | LLM integration (optional) |
| langgraph | Workflow orchestration (optional) |
| python-dotenv | Environment variable loading |

## Plot Types Supported

| Plot Type | Use Case |
|-----------|----------|
| Histogram | Distribution of numeric variables |
| KDE Plot | Smooth density estimation |
| Box Plot | Outlier detection and quartiles |
| Violin Plot | Distribution shape comparison |
| Scatter Plot | Correlation between numeric variables |
| Correlation Heatmap | Feature relationships |
| Bar Chart | Categorical variable counts |
| Count Plot | Frequency distribution |
| Q-Q Plot | Normality check |
| Missing Value Heatmap | Data quality assessment |

## Degraded Mode

The module works without LLM/LangGraph dependencies:
- Falls back to heuristic-based plot selection
- Core plot generation still fully functional

## License

MIT License
