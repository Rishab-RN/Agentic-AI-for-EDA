# =====================================================
# Intelligent Visualization Agent Package
# =====================================================
# Agentic EDA System - Visualization Selection Framework
#
# This agent intelligently selects a limited set of meaningful
# plots based on data characteristics such as variance, outliers,
# missingness, and correlation, ensuring scalability to 
# high-dimensional datasets (100+ columns).
#
# AGENTIC MODE: LLM decides which plots to generate (like a data analyst)
# RULE-BASED MODE: Fixed logic fallback when LLM unavailable
# =====================================================

from .viz_agent_node import (
    visualization_agent_node,
    build_visualization_graph,
    extract_column_metadata,
    process_edge_cases,
    compute_correlation_matrix,
    compute_priority_scores,
    determine_plot_eligibility,
    select_plots_with_budget,
    select_plots_agentic,      # NEW: LLM-driven selection
    select_plots_rule_based,   # NEW: Rule-based fallback
    generate_visualization_report,
    VisualizationState,
    DEFAULT_CONFIG,
    LANGGRAPH_AVAILABLE,
    AGENTIC_AGENT_AVAILABLE    # NEW: Check if agentic mode available
)

from .plot_generator import (
    generate_histogram,
    generate_boxplot,
    generate_violin,
    generate_kde,
    generate_barplot,
    generate_countplot,
    generate_scatter,
    generate_correlation_heatmap,
    generate_pie_chart,
    generate_all_selected_plots,
    generate_plot,
    # Senior analyst plot types
    generate_pairplot,
    generate_missing_value_heatmap,
    generate_qq_plot,
    generate_distribution_comparison,
    generate_outlier_summary,
    generate_box_by_category
)

from .graph_runner import (
    run_pipeline,
    run_pipeline_simple,
    run_pipeline_langgraph
)

# Optional edge case handler imports
try:
    from .edge_case_handler import (
        EdgeCaseHandler,
        EdgeCaseConfig,
        EdgeCaseDetector,
        handle_zero_selection_fallback,
        sanitize_column_name,
        process_edge_cases as process_edge_cases_standalone
    )
    EDGE_CASE_HANDLER_AVAILABLE = True
except ImportError:
    EDGE_CASE_HANDLER_AVAILABLE = False

# Optional agentic agent imports
try:
    from .agentic_viz_agent import (
        AgenticVisualizationAgent,
        get_agentic_plot_recommendations,
        AgentDecisionResult
    )
except ImportError:
    pass  # Already handled in viz_agent_node

__version__ = "1.1.0"
__author__ = "Agentic EDA System"

__all__ = [
    # Main entry points
    "visualization_agent_node",
    "run_pipeline",
    "run_pipeline_simple",
    "run_pipeline_langgraph",
    
    # Pipeline nodes
    "extract_column_metadata",
    "process_edge_cases",
    "compute_correlation_matrix",
    "compute_priority_scores",
    "determine_plot_eligibility",
    "select_plots_with_budget",
    "generate_visualization_report",
    
    # Plot generators
    "generate_histogram",
    "generate_boxplot",
    "generate_violin",
    "generate_kde",
    "generate_barplot",
    "generate_countplot",
    "generate_scatter",
    "generate_correlation_heatmap",
    "generate_pie_chart",
    "generate_all_selected_plots",
    "generate_plot",
    
    # Graph builder
    "build_visualization_graph",
    
    # Edge case handling
    "EdgeCaseHandler",
    "EdgeCaseConfig",
    "EdgeCaseDetector",
    "handle_zero_selection_fallback",
    "sanitize_column_name",
    "EDGE_CASE_HANDLER_AVAILABLE",
    
    # Types and configs
    "VisualizationState",
    "DEFAULT_CONFIG",
    "LANGGRAPH_AVAILABLE",
]

