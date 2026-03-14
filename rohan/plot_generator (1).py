# =====================================================
# Plot Generator Utilities for Visualization Agent
# =====================================================
# Generates actual visualizations using matplotlib and seaborn
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any
from datetime import datetime

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =====================================================
# 1. CONFIGURATION
# =====================================================

DEFAULT_FIGURE_SIZE = (10, 6)
DEFAULT_DPI = 100
DEFAULT_STYLE = "whitegrid"


def setup_plot_style():
    """Configure matplotlib and seaborn for consistent styling."""
    plt.rcParams.update({
        'figure.figsize': DEFAULT_FIGURE_SIZE,
        'figure.dpi': DEFAULT_DPI,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    sns.set_style(DEFAULT_STYLE)


def sanitize_column_name(col: str, max_length: int = 50) -> str:
    """
    Sanitize a column name for use in filenames.
    
    Args:
        col: Column name to sanitize
        max_length: Maximum length for the output
        
    Returns:
        Safe filename string
    """
    import re
    # Replace special characters with underscore
    safe = re.sub(r'[\/\\:*?"<>|]', '_', col)
    # Replace spaces
    safe = safe.replace(' ', '_')
    # Remove any double underscores
    safe = re.sub(r'_+', '_', safe)
    # Truncate if needed
    if len(safe) > max_length:
        safe = safe[:max_length-3] + "..."
    return safe


def truncate_title(title: str, max_length: int = 60) -> str:
    """Truncate a title for display in plots."""
    if len(title) > max_length:
        return title[:max_length-3] + "..."
    return title


# =====================================================
# 2. UNIVARIATE PLOT GENERATORS
# =====================================================

def generate_histogram(
    df: pd.DataFrame, 
    column: str, 
    output_dir: str,
    bins: int = 30,
    kde: bool = True
) -> Optional[str]:
    """
    Generate a histogram for a numeric column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to plot
        output_dir: Directory to save the plot
        bins: Number of histogram bins
        kde: Whether to overlay KDE curve
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        
        data = df[column].dropna()
        
        # Create histogram with optional KDE
        sns.histplot(data, bins=bins, kde=kde, ax=ax, color='steelblue', alpha=0.7)
        
        # Add statistics annotation
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        filename = f"histogram_{column.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"Error generating histogram for {column}: {e}")
        plt.close('all')
        return None


def generate_boxplot(
    df: pd.DataFrame, 
    column: str, 
    output_dir: str
) -> Optional[str]:
    """
    Generate a boxplot for a numeric column (good for outlier visualization).
    
    Args:
        df: DataFrame containing the data
        column: Column name to plot
        output_dir: Directory to save the plot
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data = df[column].dropna()
        
        # Create boxplot
        box = ax.boxplot(data, patch_artist=True, vert=True)
        box['boxes'][0].set_facecolor('lightblue')
        box['boxes'][0].set_alpha(0.7)
        
        # Add statistics
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
        
        stats_text = f"Q1: {Q1:.2f}\nQ3: {Q3:.2f}\nIQR: {IQR:.2f}\nOutliers: {outlier_count}"
        ax.text(1.15, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'Box Plot of {column}', fontsize=14, fontweight='bold')
        ax.set_ylabel(column)
        ax.set_xticklabels([column])
        
        plt.tight_layout()
        
        # Save plot
        filename = f"boxplot_{column.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"Error generating boxplot for {column}: {e}")
        plt.close('all')
        return None


def generate_violin(
    df: pd.DataFrame, 
    column: str, 
    output_dir: str
) -> Optional[str]:
    """
    Generate a violin plot for a numeric column (good for skewed distributions).
    
    Args:
        df: DataFrame containing the data
        column: Column name to plot
        output_dir: Directory to save the plot
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data = df[column].dropna()
        
        # Create violin plot
        parts = ax.violinplot(data, positions=[1], showmeans=True, showmedians=True)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)
        
        # Add skewness annotation
        skewness = data.skew()
        ax.text(1.15, 0.5, f"Skewness: {skewness:.2f}", transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'Violin Plot of {column}', fontsize=14, fontweight='bold')
        ax.set_ylabel(column)
        ax.set_xticks([1])
        ax.set_xticklabels([column])
        
        plt.tight_layout()
        
        # Save plot
        filename = f"violin_{column.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"Error generating violin plot for {column}: {e}")
        plt.close('all')
        return None


def generate_kde(
    df: pd.DataFrame, 
    column: str, 
    output_dir: str
) -> Optional[str]:
    """
    Generate a KDE (Kernel Density Estimate) plot for a numeric column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to plot
        output_dir: Directory to save the plot
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        
        data = df[column].dropna()
        
        # Create KDE plot
        sns.kdeplot(data, ax=ax, fill=True, color='steelblue', alpha=0.5)
        
        ax.set_title(f'Density Plot of {column}', fontsize=14, fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Density')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"kde_{column.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"Error generating KDE plot for {column}: {e}")
        plt.close('all')
        return None


def generate_barplot(
    df: pd.DataFrame, 
    column: str, 
    output_dir: str,
    top_n: int = 15
) -> Optional[str]:
    """
    Generate a bar plot for a categorical column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to plot
        output_dir: Directory to save the plot
        top_n: Maximum number of categories to show
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get value counts
        value_counts = df[column].value_counts().head(top_n)
        
        # Create bar plot
        bars = ax.bar(range(len(value_counts)), value_counts.values, color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, val in zip(bars, value_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(val), ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"barplot_{column.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"Error generating bar plot for {column}: {e}")
        plt.close('all')
        return None


def generate_countplot(
    df: pd.DataFrame, 
    column: str, 
    output_dir: str,
    top_n: int = 15
) -> Optional[str]:
    """
    Generate a count plot for a categorical column using seaborn.
    
    Args:
        df: DataFrame containing the data
        column: Column name to plot
        output_dir: Directory to save the plot
        top_n: Maximum number of categories to show
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get top N categories
        top_categories = df[column].value_counts().head(top_n).index.tolist()
        data = df[df[column].isin(top_categories)]
        
        # Create count plot
        sns.countplot(data=data, x=column, ax=ax, order=top_categories, palette='husl')
        
        ax.set_title(f'Count Plot of {column}', fontsize=14, fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"countplot_{column.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"Error generating count plot for {column}: {e}")
        plt.close('all')
        return None


def generate_pie_chart(
    df: pd.DataFrame, 
    column: str, 
    output_dir: str,
    top_n: int = 8
) -> Optional[str]:
    """
    Generate a pie chart for a categorical column.
    Best for low-cardinality columns (< 10 categories).
    
    Args:
        df: DataFrame containing the data
        column: Column name to plot
        output_dir: Directory to save the plot
        top_n: Maximum number of categories to show
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get value counts
        value_counts = df[column].value_counts().head(top_n)
        
        # If more categories exist, add "Other"
        if len(df[column].value_counts()) > top_n:
            other_count = df[column].value_counts()[top_n:].sum()
            value_counts['Other'] = other_count
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        wedges, texts, autotexts = ax.pie(
            value_counts.values, 
            labels=value_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.02] * len(value_counts)
        )
        
        # Styling
        for autotext in autotexts:
            autotext.set_fontsize(9)
        
        ax.set_title(truncate_title(f'Distribution of {column}'), fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        safe_col = sanitize_column_name(column)
        filename = f"pie_{safe_col}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"Error generating pie chart for {column}: {e}")
        plt.close('all')
        return None


# =====================================================
# 3. BIVARIATE PLOT GENERATORS
# =====================================================

def generate_scatter(
    df: pd.DataFrame, 
    col1: str, 
    col2: str, 
    output_dir: str,
    add_regression: bool = True
) -> Optional[str]:
    """
    Generate a scatter plot for two numeric columns.
    
    Args:
        df: DataFrame containing the data
        col1: First column (X-axis)
        col2: Second column (Y-axis)
        output_dir: Directory to save the plot
        add_regression: Whether to add regression line
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        
        # Drop rows with NaN in either column
        data = df[[col1, col2]].dropna()
        
        # Create scatter plot
        ax.scatter(data[col1], data[col2], alpha=0.5, c='steelblue', edgecolors='white', linewidth=0.5)
        
        # Add regression line if requested
        if add_regression and len(data) > 2:
            from numpy.polynomial import polynomial as P
            z = np.polyfit(data[col1], data[col2], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[col1].min(), data[col1].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend line')
        
        # Calculate and display correlation
        correlation = data[col1].corr(data[col2])
        ax.text(0.95, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'{col1} vs {col2}', fontsize=14, fontweight='bold')
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        
        if add_regression:
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        filename = f"scatter_{col1.replace(' ', '_')}_vs_{col2.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"Error generating scatter plot for {col1} vs {col2}: {e}")
        plt.close('all')
        return None


def generate_correlation_heatmap(
    df: pd.DataFrame, 
    columns: Optional[List[str]], 
    output_dir: str,
    max_columns: int = 20
) -> Optional[str]:
    """
    Generate a correlation heatmap for numeric columns.
    
    Args:
        df: DataFrame containing the data
        columns: List of columns to include (None = all numeric)
        output_dir: Directory to save the plot
        max_columns: Maximum number of columns to plot
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        setup_plot_style()
        
        # Select numeric columns
        if columns is None:
            numeric_df = df.select_dtypes(include=[np.number])
        else:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        
        # Limit columns if too many
        if len(numeric_df.columns) > max_columns:
            # Select columns with highest variance
            variances = numeric_df.var().sort_values(ascending=False)
            selected_cols = variances.head(max_columns).index.tolist()
            numeric_df = numeric_df[selected_cols]
        
        if len(numeric_df.columns) < 2:
            print("Not enough numeric columns for correlation heatmap")
            return None
        
        # Calculate size based on number of columns
        n_cols = len(numeric_df.columns)
        fig_size = max(10, n_cols * 0.5)
        
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True if n_cols <= 15 else False, 
                    fmt='.2f', cmap='RdBu_r', center=0, 
                    square=True, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8})
        
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        filename = "correlation_heatmap.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"Error generating correlation heatmap: {e}")
        plt.close('all')
        return None


# =====================================================
# 4. CATEGORICAL VS NUMERIC PLOTS
# =====================================================

def generate_box_by_category(
    df: pd.DataFrame, 
    numeric_col: str, 
    category_col: str, 
    output_dir: str,
    max_categories: int = 10
) -> Optional[str]:
    """
    Generate a boxplot of a numeric column grouped by a categorical column.
    
    Args:
        df: DataFrame containing the data
        numeric_col: Numeric column to plot
        category_col: Categorical column for grouping
        output_dir: Directory to save the plot
        max_categories: Maximum number of categories to show
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get top categories
        top_cats = df[category_col].value_counts().head(max_categories).index.tolist()
        data = df[df[category_col].isin(top_cats)]
        
        # Create boxplot
        sns.boxplot(data=data, x=category_col, y=numeric_col, ax=ax, palette='husl')
        
        ax.set_title(f'{numeric_col} by {category_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(category_col)
        ax.set_ylabel(numeric_col)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"boxplot_{numeric_col.replace(' ', '_')}_by_{category_col.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"Error generating categorical boxplot for {numeric_col} by {category_col}: {e}")
        plt.close('all')
        return None


# =====================================================
# 5. MASTER PLOT GENERATOR
# =====================================================

def generate_plot(
    df: pd.DataFrame,
    plot_config: Dict[str, Any],
    output_dir: str
) -> Optional[str]:
    """
    Generate a plot based on configuration.
    
    Args:
        df: DataFrame containing the data
        plot_config: Plot configuration from selected_plots
        output_dir: Directory to save plots
    
    Returns:
        Path to saved plot or None if failed
    """
    plot_type = plot_config.get("plot_type", "").lower()
    
    if plot_type == "histogram":
        return generate_histogram(df, plot_config["column"], output_dir)
    
    elif plot_type == "boxplot":
        return generate_boxplot(df, plot_config["column"], output_dir)
    
    elif plot_type == "violin":
        return generate_violin(df, plot_config["column"], output_dir)
    
    elif plot_type == "kde":
        return generate_kde(df, plot_config["column"], output_dir)
    
    elif plot_type == "barplot":
        return generate_barplot(df, plot_config["column"], output_dir)
    
    elif plot_type == "countplot":
        return generate_countplot(df, plot_config["column"], output_dir)
    
    elif plot_type == "pie":
        return generate_pie_chart(df, plot_config["column"], output_dir)
    
    elif plot_type == "bar":  # Alias for barplot
        return generate_barplot(df, plot_config["column"], output_dir)
    
    elif plot_type == "scatter":
        return generate_scatter(df, plot_config["column1"], plot_config["column2"], output_dir)
    
    elif plot_type == "heatmap":
        return generate_correlation_heatmap(df, None, output_dir)
    
    else:
        print(f"Unknown plot type: {plot_type}")
        return None


def generate_all_selected_plots(
    df: pd.DataFrame,
    selected_plots: List[Dict],
    output_dir: str
) -> Dict[str, Any]:
    """
    Generate all selected plots and return results.
    
    Args:
        df: DataFrame containing the data
        selected_plots: List of plot configurations from visualization agent
        output_dir: Directory to save plots
    
    Returns:
        Dictionary with generation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "plots_generated": [],
        "plots_failed": [],
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n[PLOT] Generating {len(selected_plots)} plots...")
    print(f"   Output directory: {output_dir}")
    print("-" * 50)
    
    for i, plot_config in enumerate(selected_plots, 1):
        plot_type = plot_config.get("plot_type", "unknown")
        
        if plot_config.get("category") == "bivariate":
            plot_desc = f"{plot_type}: {plot_config.get('column1')} vs {plot_config.get('column2')}"
        elif plot_config.get("category") == "correlation_overview":
            plot_desc = f"{plot_type}: Correlation overview"
        else:
            plot_desc = f"{plot_type}: {plot_config.get('column', 'N/A')}"
        
        print(f"   [{i}/{len(selected_plots)}] Generating {plot_desc}...", end=" ")
        
        filepath = generate_plot(df, plot_config, output_dir)
        
        if filepath:
            print("[OK]")
            results["plots_generated"].append({
                "config": plot_config,
                "filepath": filepath
            })
        else:
            print("[FAIL]")
            results["plots_failed"].append({
                "config": plot_config,
                "error": "Generation failed"
            })
    
    print("-" * 50)
    print(f"[SUCCESS] Successfully generated: {len(results['plots_generated'])} plots")
    if results["plots_failed"]:
        print(f"[WARN] Failed: {len(results['plots_failed'])} plots")
    
    return results


# =====================================================
# 6. ENTRY POINT FOR TESTING
# =====================================================

if __name__ == "__main__":
    import sys
    
    # Test with sample data
    test_file = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\rohan\Antigravity\EL_sem3\Outlier_detector\AmesHousing_cleaned.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./test_plots"
    
    print(f"Loading: {test_file}")
    df = pd.read_csv(test_file)
    print(f"Dataset shape: {df.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample plots
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
    
    print("\n[PLOT] Generating sample plots...")
    
    for col in numeric_cols[:3]:
        path = generate_histogram(df, col, output_dir)
        if path:
            print(f"   [OK] Histogram: {path}")
        
        path = generate_boxplot(df, col, output_dir)
        if path:
            print(f"   [OK] Boxplot: {path}")
    
    # Generate correlation heatmap
    path = generate_correlation_heatmap(df, None, output_dir)
    if path:
        print(f"   [OK] Heatmap: {path}")
    
    print("\n[SUCCESS] Test completed!")
