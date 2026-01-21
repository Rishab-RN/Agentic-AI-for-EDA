# =====================================================
# Plot Generator Utilities for Visualization Agent
# =====================================================
# Generates actual visualizations using matplotlib and seaborn
# =====================================================

import os
import pandas as pd
import numpy as np

# IMPORTANT: Set non-interactive backend BEFORE importing pyplot
# This fixes "main thread is not in main loop" errors when running in Flask
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple
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
        skew_val = data.skew()
        
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
        
        # Generate Insight
        skew_desc = "approximately symmetric"
        if skew_val > 1: skew_desc = "right-skewed (positive skew)"
        elif skew_val < -1: skew_desc = "left-skewed (negative skew)"
        elif skew_val > 0.5: skew_desc = "moderately right-skewed"
        elif skew_val < -0.5: skew_desc = "moderately left-skewed"
            
        inference = (f"The distribution of {column} is {skew_desc} (skewness: {skew_val:.2f}). "
                     f"The data is centered around {median_val:.2f}, with values ranging from {data.min():.2f} to {data.max():.2f}.")
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating histogram for {column}: {e}")
        return None, ""
        



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
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = df[column].dropna()
        
        # Create boxplot
        box = ax.boxplot(data, patch_artist=True, vert=True)
        # Handle box styling safely (box['boxes'] might be list)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        # Add statistics
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        median_val = data.median()
        IQR = Q3 - Q1
        outlier_count = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
        
        # Add stats text to plot
        stats_text = f"Q1: {Q1:.2f}\nQ3: {Q3:.2f}\nIQR: {IQR:.2f}\nOutliers: {outlier_count}"
        # Position text slightly outside right axis
        ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
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
        
        # Generate Insight
        inference = (f"Boxplot analysis for {column} reveals a median value of {median_val:.2f}. "
                     f"Detected {outlier_count} potential outliers beyond the whiskers. "
                     f"The Interquartile Range (IQR) is {IQR:.2f}, indicating the spread of the middle 50% of data.")
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating boxplot for {column}: {e}")
        plt.close('all')
        return None, ""


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
        
        # Generate Insight
        median_val = data.median()
        skew_val = data.skew()
        skew_desc = "right-skewed" if skew_val > 0.5 else "left-skewed" if skew_val < -0.5 else "symmetric"
        
        inference = (f"Violin plot reveals a {skew_desc} distribution (skew: {skew_val:.2f}) centered at {median_val:.2f}. "
                     f"The shape indicates the density of data at different values.")
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating violin plot for {column}: {e}")
        plt.close('all')
        return None, ""


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
        
        # Generate Insight
        mean_val = data.mean()
        std_val = data.std()
        
        inference = (f"KDE plot estimates the probability density. The distribution has a mean of {mean_val:.2f} "
                     f"and a standard deviation of {std_val:.2f}, visualizing the data's spread and shape.")
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating KDE plot for {column}: {e}")
        plt.close('all')
        return None, ""


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
        
        # Generate Insight
        top_cat = value_counts.index[0]
        top_val = value_counts.values[0]
        total = df[column].count()
        pct = (top_val / total) * 100
        
        inference = (f"Bar plot analysis: '{top_cat}' is the most frequent category, accounting for {top_val} entries ({pct:.1f}%). "
                     f"The plot shows the relative frequency of the top {min(top_n, len(value_counts))} values.")
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating bar plot for {column}: {e}")
        plt.close('all')
        return None, ""


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
        
        # Generate Insight
        top_cat = top_categories[0]
        # value_counts() is needed again as 'data' is filtered
        all_counts = df[column].value_counts()
        top_val = all_counts[top_cat]
        total = len(df)
        pct = (top_val / total) * 100
        
        inference = (f"Count plot shows '{top_cat}' as the dominant category ({top_val} occurrences, {pct:.1f}%). "
                     f"This visualization highlights the frequency distribution across top categories.")
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating count plot for {column}: {e}")
        plt.close('all')
        return None, ""


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
        Tuple of (path to saved plot, inference) or (None, "") if failed
    """
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get value counts
        value_counts = df[column].value_counts().head(top_n)
        total_count = len(df[column].dropna())
        
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
        
        # Generate Insight
        top_category = value_counts.index[0]
        top_percentage = (value_counts.values[0] / total_count) * 100
        num_categories = len(df[column].value_counts())
        inference = (f"Pie chart shows the distribution of {column} across {num_categories} categories. "
                     f"'{top_category}' dominates with {top_percentage:.1f}% of the data.")
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating pie chart for {column}: {e}")
        plt.close('all')
        return None, ""



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
        
        # Generate Insight
        strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
        direction = "positive" if correlation > 0 else "negative"
        inference = (f"Scatter plot shows a {strength} {direction} correlation (r={correlation:.3f}) between {col1} and {col2}. "
                     f"As {col1} increases, {col2} tends to {'increase' if correlation > 0 else 'decrease'}.")
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating scatter plot for {col1} vs {col2}: {e}")
        plt.close('all')
        return None, ""


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
        
        # Generate Insight (Find top correlations)
        corr_unstacked = corr_matrix.unstack()
        # Remove self-correlations and duplicates
        pairs = corr_unstacked[corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)]
        if not pairs.empty:
            strongest_pair = pairs.abs().idxmax()
            max_corr = pairs[strongest_pair]
            inference = (f"Correlation heatmap visualizes linear relationships. "
                         f"Strongest correlation found between '{strongest_pair[0]}' and '{strongest_pair[1]}' (r={max_corr:.2f}).")
        else:
            inference = "Correlation heatmap shows relationships between variables. No significant strong correlations detected."

        return filepath, inference
        
    except Exception as e:
        print(f"Error generating correlation heatmap: {e}")
        plt.close('all')
        return None, ""


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
# 4.5 ADVANCED SENIOR ANALYST PLOTS
# =====================================================

def generate_pairplot(
    df: pd.DataFrame,
    columns: List[str],
    output_dir: str,
    max_columns: int = 6,
    hue: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Generate a pairplot for multiple numeric columns.
    Essential for understanding relationships between features.
    
    Args:
        df: DataFrame containing the data
        columns: List of numeric columns to include
        output_dir: Directory to save the plot
        max_columns: Maximum columns to include (for readability)
        hue: Optional categorical column for color grouping
        
    Returns:
        Tuple of (path to saved plot, inference) or (None, "") if failed
    """
    try:
        setup_plot_style()
        
        # Limit columns for readability
        cols_to_use = columns[:max_columns]
        
        # Filter to only use valid columns
        valid_cols = [c for c in cols_to_use if c in df.columns]
        if len(valid_cols) < 2:
            return None, ""
        
        # Create pairplot
        if hue and hue in df.columns:
            g = sns.pairplot(df[valid_cols + [hue]].dropna(), hue=hue, diag_kind='kde')
        else:
            g = sns.pairplot(df[valid_cols].dropna(), diag_kind='kde')
        
        g.fig.suptitle(f'Pairplot: {len(valid_cols)} Features', y=1.02, fontsize=14, fontweight='bold')
        
        # Save plot
        filename = f"pairplot_{len(valid_cols)}_features.png"
        filepath = os.path.join(output_dir, filename)
        g.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close('all')
        
        # Generate inference
        inference = f"Pairplot shows relationships between {len(valid_cols)} key features. Diagonal shows individual distributions (KDE), off-diagonal shows pairwise scatter plots."
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating pairplot: {e}")
        plt.close('all')
        return None, ""


def generate_missing_value_heatmap(
    df: pd.DataFrame,
    output_dir: str,
    max_columns: int = 30
) -> Tuple[Optional[str], str]:
    """
    Generate a missing value heatmap showing patterns of missingness.
    Critical for understanding data quality issues.
    
    Args:
        df: DataFrame containing the data
        output_dir: Directory to save the plot
        max_columns: Maximum columns to show
        
    Returns:
        Tuple of (path to saved plot, inference) or (None, "") if failed
    """
    try:
        setup_plot_style()
        
        # Calculate missing percentages
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        
        # Only include columns with missing values
        cols_with_missing = missing_pct[missing_pct > 0]
        
        if len(cols_with_missing) == 0:
            return None, "No missing values in dataset"
        
        # Limit columns
        cols_to_show = cols_with_missing.head(max_columns)
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(cols_to_show) * 0.4)))
        
        # Create horizontal bar chart of missing percentages
        colors = ['#ff6b6b' if x > 50 else '#ffa94d' if x > 20 else '#69db7c' for x in cols_to_show.values]
        bars = ax.barh(range(len(cols_to_show)), cols_to_show.values, color=colors)
        
        ax.set_yticks(range(len(cols_to_show)))
        ax.set_yticklabels(cols_to_show.index)
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Value Analysis', fontsize=14, fontweight='bold')
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Critical (50%)')
        ax.axvline(x=20, color='orange', linestyle='--', alpha=0.5, label='Warning (20%)')
        ax.legend()
        
        # Add percentage labels
        for bar, pct in zip(bars, cols_to_show.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        filename = "missing_value_analysis.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        # Generate inference
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        overall_pct = (total_missing / total_cells) * 100
        
        worst_col = cols_with_missing.index[0]
        worst_pct = cols_with_missing.values[0]
        
        inference = f"Missing value analysis: {len(cols_with_missing)} columns have missing data ({overall_pct:.1f}% overall). '{worst_col}' has the most missing ({worst_pct:.1f}%)."
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating missing value heatmap: {e}")
        plt.close('all')
        return None, ""


def generate_qq_plot(
    df: pd.DataFrame,
    column: str,
    output_dir: str
) -> Tuple[Optional[str], str]:
    """
    Generate a Q-Q plot to assess normality of a numeric column.
    Essential for understanding if data follows normal distribution.
    
    Args:
        df: DataFrame containing the data
        column: Column name to check
        output_dir: Directory to save the plot
        
    Returns:
        Tuple of (path to saved plot, inference) or (None, "") if failed
    """
    try:
        from scipy import stats
        
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Get data and drop NaN
        data = df[column].dropna()
        
        if len(data) < 10:
            return None, ""
        
        # Create Q-Q plot
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: {column}', fontsize=14, fontweight='bold')
        ax.get_lines()[0].set_markerfacecolor('#3498db')
        ax.get_lines()[0].set_markersize(5)
        ax.get_lines()[1].set_color('#e74c3c')
        
        plt.tight_layout()
        
        # Save plot
        safe_col = sanitize_column_name(column)
        filename = f"qqplot_{safe_col}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        # Perform Shapiro-Wilk test (sample if too large)
        sample = data.sample(min(5000, len(data)), random_state=42) if len(data) > 5000 else data
        stat, p_value = stats.shapiro(sample)
        
        if p_value < 0.05:
            inference = f"Q-Q plot for {column}: Data significantly deviates from normal distribution (p={p_value:.4f}). Consider transformation for parametric tests."
        else:
            inference = f"Q-Q plot for {column}: Data approximately follows normal distribution (p={p_value:.4f}). Safe for parametric statistical tests."
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating Q-Q plot for {column}: {e}")
        plt.close('all')
        return None, ""


def generate_distribution_comparison(
    df: pd.DataFrame,
    numeric_col: str,
    category_col: str,
    output_dir: str,
    max_categories: int = 5
) -> Tuple[Optional[str], str]:
    """
    Generate overlapping distribution plots for a numeric column by category.
    Useful for comparing distributions across groups.
    
    Args:
        df: DataFrame containing the data
        numeric_col: Numeric column to compare
        category_col: Categorical column for grouping
        output_dir: Directory to save the plot
        max_categories: Maximum number of categories to show
        
    Returns:
        Tuple of (path to saved plot, inference) or (None, "") if failed
    """
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get top categories
        top_cats = df[category_col].value_counts().head(max_categories).index.tolist()
        
        # Plot KDE for each category
        for cat in top_cats:
            data = df[df[category_col] == cat][numeric_col].dropna()
            if len(data) > 5:
                sns.kdeplot(data=data, label=str(cat), ax=ax, fill=True, alpha=0.3)
        
        ax.set_title(f'Distribution of {numeric_col} by {category_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(numeric_col)
        ax.set_ylabel('Density')
        ax.legend(title=category_col)
        
        plt.tight_layout()
        
        # Save plot
        safe_num = sanitize_column_name(numeric_col)
        safe_cat = sanitize_column_name(category_col)
        filename = f"distribution_{safe_num}_by_{safe_cat}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        # Generate inference
        inference = f"Distribution comparison of {numeric_col} across {category_col} categories. Visual differences indicate potential group effects."
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating distribution comparison: {e}")
        plt.close('all')
        return None, ""


def generate_outlier_summary(
    df: pd.DataFrame,
    output_dir: str,
    max_columns: int = 15
) -> Tuple[Optional[str], str]:
    """
    Generate a summary visualization of outliers across all numeric columns.
    Critical for data quality assessment.
    
    Args:
        df: DataFrame containing the data
        output_dir: Directory to save the plot
        max_columns: Maximum columns to show
        
    Returns:
        Tuple of (path to saved plot, inference) or (None, "") if failed
    """
    try:
        setup_plot_style()
        
        # Calculate outlier counts for each numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
                if outliers > 0:
                    outlier_counts[col] = {
                        'count': outliers,
                        'percentage': (outliers / len(data)) * 100
                    }
        
        if not outlier_counts:
            return None, "No outliers detected in numeric columns"
        
        # Sort by outlier count
        sorted_cols = sorted(outlier_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:max_columns]
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_cols) * 0.5)))
        
        cols = [x[0] for x in sorted_cols]
        counts = [x[1]['count'] for x in sorted_cols]
        pcts = [x[1]['percentage'] for x in sorted_cols]
        
        # Color based on severity
        colors = ['#e74c3c' if p > 10 else '#f39c12' if p > 5 else '#3498db' for p in pcts]
        
        bars = ax.barh(range(len(cols)), counts, color=colors)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)
        ax.set_xlabel('Number of Outliers')
        ax.set_title('Outlier Analysis by Column', fontsize=14, fontweight='bold')
        
        # Add percentage labels
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        filename = "outlier_summary.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close(fig)
        
        # Generate inference
        total_outliers = sum(x[1]['count'] for x in sorted_cols)
        worst_col = sorted_cols[0][0]
        worst_count = sorted_cols[0][1]['count']
        
        inference = f"Outlier analysis: {len(outlier_counts)} columns have outliers. '{worst_col}' has the most ({worst_count}). Total outliers: {total_outliers}."
        
        return filepath, inference
        
    except Exception as e:
        print(f"Error generating outlier summary: {e}")
        plt.close('all')
        return None, ""


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
    
    # === SCATTER PLOTS (with aliases) ===
    elif plot_type in ["scatter", "scatter_plot", "scatterplot"]:
        return generate_scatter(df, plot_config["column1"], plot_config["column2"], output_dir)
    
    # === HEATMAP (with aliases) ===
    elif plot_type in ["heatmap", "correlation_heatmap", "corr_heatmap"]:
        return generate_correlation_heatmap(df, None, output_dir)
    
    # === GROUPED BARPLOT (for categorical comparisons) ===
    elif plot_type in ["grouped_barplot", "stacked_bar", "category_comparison"]:
        # For now, route to regular barplot
        col = plot_config.get("column", plot_config.get("columns", [None])[0])
        if col:
            return generate_barplot(df, col, output_dir)
        return None, ""
    
    # === MISSING VALUE HEATMAP (with aliases) ===
    elif plot_type in ["missing_value_heatmap", "missing_heatmap"]:
        return generate_missing_value_heatmap(df, output_dir)
    
    # New senior analyst plot types
    elif plot_type == "pairplot":
        columns = plot_config.get("columns", [])
        hue = plot_config.get("hue")
        return generate_pairplot(df, columns, output_dir, hue=hue)
    
    elif plot_type == "missing_value" or plot_type == "missing":
        return generate_missing_value_heatmap(df, output_dir)
    
    elif plot_type == "qqplot" or plot_type == "qq":
        return generate_qq_plot(df, plot_config["column"], output_dir)
    
    elif plot_type == "distribution_comparison":
        # Handle multiple key formats
        cols = plot_config.get("columns", [])
        if len(cols) >= 2:
            numeric_col, category_col = cols[0], cols[1]
        else:
            numeric_col = plot_config.get("numeric_col", plot_config.get("column1", cols[0] if cols else None))
            category_col = plot_config.get("category_col", plot_config.get("column2"))
        if numeric_col and category_col:
            return generate_distribution_comparison(df, numeric_col, category_col, output_dir)
        else:
            print(f"Missing columns for distribution_comparison: {plot_config}")
            return None, ""
    
    elif plot_type == "outlier_summary" or plot_type == "outliers":
        return generate_outlier_summary(df, output_dir)
    
    elif plot_type == "grouped_boxplot" or plot_type == "box_by_category":
        # Handle multiple key formats
        cols = plot_config.get("columns", [])
        if len(cols) >= 2:
            numeric_col, category_col = cols[0], cols[1]
        else:
            numeric_col = plot_config.get("numeric_col", plot_config.get("column1", cols[0] if cols else None))
            category_col = plot_config.get("category_col", plot_config.get("column2"))
        if numeric_col and category_col:
            return generate_box_by_category(df, numeric_col, category_col, output_dir)
        else:
            print(f"Missing columns for grouped_boxplot: {plot_config}")
            return None, ""
    
    else:
        print(f"Unknown plot type: {plot_type}")
        return None, ""


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
        
        plot_desc = f"{plot_type}"
        print(f"   [{i}/{len(selected_plots)}] Generating {plot_desc}...", end=" ")
        
        # Ensure we handle both tuple return (path, inference) and string return (path)
        result = generate_plot(df, plot_config, output_dir)
        if isinstance(result, tuple):
            filepath, inference = result
        else:
            filepath, inference = result, ""
        
        if filepath:
            print("[OK]")
            results["plots_generated"].append({
                "config": plot_config,
                "filepath": filepath,
                "inference": inference 
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
