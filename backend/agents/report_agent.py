import os
import json
from datetime import datetime
from typing import Generator, Dict, Any, List


WEASYPRINT_AVAILABLE = None  

from dotenv import load_dotenv
import pathlib
_backend_dir = pathlib.Path(__file__).parent.parent
load_dotenv(_backend_dir / '.env')

GROQ_API_KEY = os.getenv("GROQ_API_KEY_REPORT")


class ReportAgent:
    """
    Report Agent that generates streaming reports and industry-level PDFs
    with AI-powered insights using Groq LLM.
    """
    
    def __init__(self):
        self.report_sections = []
        self.groq_client = None
        self._init_groq()
    
    def _init_groq(self):
        """Initialize Groq client for AI insights."""
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=GROQ_API_KEY)
        except ImportError:
            print("[ReportAgent] Groq not installed. Using fallback insights.")
            self.groq_client = None
        except Exception as e:
            print(f"[ReportAgent] Groq init error: {e}")
            self.groq_client = None
    
    def _generate_ai_insight(self, context: str, data_summary: dict, insight_type: str = "general") -> str:
        """
        Generate professional AI-powered insight using Groq LLM.
        
        Args:
            context: Description of what we're analyzing
            data_summary: Relevant data metrics
            insight_type: Type of insight (overview, missing, outlier, visualization, correlation)
        
        Returns:
            Professional insight text
        """
        if not self.groq_client:
            return self._get_fallback_insight(insight_type, data_summary)
        
        try:
            if insight_type == "overview":
                prompt = f"""You are a senior data scientist writing an executive summary for a data quality report.
                
Dataset Overview:
{json.dumps(data_summary, indent=2, default=str)}

Context: {context}

Write a 2-3 sentence professional executive insight that:
1. Summarizes the data quality state
2. Highlights the most important finding
3. Provides a brief recommendation

Be concise, professional, and data-driven. Use specific numbers from the data."""

            elif insight_type == "missing":
                prompt = f"""You are a data quality expert analyzing missing value treatment results.

Missing Value Analysis:
{json.dumps(data_summary, indent=2, default=str)}

Context: {context}

Write a 2-3 sentence professional insight that:
1. Explains the impact of missing values on data quality
2. Justifies the treatment strategies used
3. Notes any potential data integrity concerns

Be specific about the numbers and methods used."""

            elif insight_type == "outlier":
                prompt = f"""You are a statistical analyst reviewing outlier detection and treatment.

Outlier Analysis:
{json.dumps(data_summary, indent=2, default=str)}

Context: {context}

Write a 2-3 sentence professional insight that:
1. Explains the significance of outliers found
2. Justifies the treatment approach
3. Discusses impact on downstream analysis

Use specific statistics and be technically accurate."""

            elif insight_type == "visualization":
                prompt = f"""You are a data visualization expert interpreting an EDA chart.

Visualization Details:
{json.dumps(data_summary, indent=2, default=str)}

Context: {context}

Write a 2-3 sentence professional interpretation that:
1. Describes what the visualization reveals about the data
2. Highlights key patterns or anomalies
3. Suggests actionable insights

Be specific and insightful, avoiding generic observations."""

            else:
                prompt = f"""You are a data scientist providing professional insights.

Data:
{json.dumps(data_summary, indent=2, default=str)}

Context: {context}

Write a 2-3 sentence professional insight that is specific, actionable, and data-driven."""

            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a senior data scientist providing professional insights for data quality reports. Be concise, specific, and use exact numbers from the data provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[ReportAgent] Groq API error: {e}")
            return self._get_fallback_insight(insight_type, data_summary)
    
    def _get_fallback_insight(self, insight_type: str, data_summary: dict) -> str:
        """Fallback insights when Groq is unavailable."""
        if insight_type == "overview":
            return f"The dataset underwent comprehensive automated analysis. Data quality assessment and cleaning operations were performed to ensure analytical readiness."
        elif insight_type == "missing":
            return "Missing value treatment was applied based on data type and distribution characteristics to preserve data integrity while maximizing usable records."
        elif insight_type == "outlier":
            return "Outliers were detected using statistical methods and treated according to their impact on data distribution and analytical objectives."
        elif insight_type == "visualization":
            return "This visualization provides insights into the data distribution and relationships within the dataset."
        else:
            return "Analysis completed successfully with data quality measures applied."
    
    def generate_report_stream(self, job_data: dict) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that yields report sections as they're analyzed.
        
        Yields:
            Dict with keys: stage, status, text, details (optional)
        """
        stages = job_data.get("stages", {})
        analysis = job_data.get("analysis", {})
        
        # === HEADER ===
        yield {
            "stage": "header",
            "status": "info",
            "text": "📊 Agentic AI System for EDA",
            "details": {
                "generated_at": datetime.now().isoformat(),
                "job_id": job_data.get("id", "unknown")
            }
        }
        
        # === DATASET OVERVIEW ===
        yield {
            "stage": "overview",
            "status": "analyzing",
            "text": "[Analyzing...] Dataset Overview"
        }
        
        yield {
            "stage": "overview",
            "status": "complete",
            "text": f"→ Dataset: {analysis.get('total_rows', 0):,} rows × {analysis.get('total_columns', 0)} columns",
            "details": {
                "numeric_columns": analysis.get("numeric_columns", 0),
                "categorical_columns": analysis.get("categorical_columns", 0)
            }
        }
        
        # === MISSING VALUE STAGE ===
        missing_stage = stages.get("missing", {})
        missing_result = missing_stage.get("result", {})
        
        yield {
            "stage": "missing",
            "status": "analyzing",
            "text": "[Analyzing...] Missing Value Stage"
        }
        
        if missing_stage.get("status") == "skipped":
            yield {
                "stage": "missing",
                "status": "skipped",
                "text": "→ No missing values detected - stage skipped"
            }
        elif missing_stage.get("status") == "done":
            original = missing_result.get("original_shape", [0, 0])
            final = missing_result.get("final_shape", [0, 0])
            rows_dropped = missing_result.get("rows_dropped", 0)
            cols_dropped = missing_result.get("columns_dropped", [])
            imputed = missing_result.get("imputed_values", {})
            
            yield {
                "stage": "missing",
                "status": "info",
                "text": f"→ Original shape: {original[0]:,} rows × {original[1]} columns"
            }
            
            if cols_dropped:
                for col in cols_dropped[:5]:
                    yield {
                        "stage": "missing",
                        "status": "action",
                        "text": f"→ Dropped column '{col}' (too many missing values)"
                    }
                if len(cols_dropped) > 5:
                    yield {
                        "stage": "missing",
                        "status": "info",
                        "text": f"→ ...and {len(cols_dropped) - 5} more columns dropped"
                    }
            
            if rows_dropped > 0:
                yield {
                    "stage": "missing",
                    "status": "action",
                    "text": f"→ Dropped {rows_dropped:,} rows with missing values"
                }
            
            if imputed:
                for col, method in list(imputed.items())[:5]:
                    yield {
                        "stage": "missing",
                        "status": "action",
                        "text": f"→ Imputed '{col}' using {method}"
                    }
                if len(imputed) > 5:
                    yield {
                        "stage": "missing",
                        "status": "info",
                        "text": f"→ ...and {len(imputed) - 5} more columns imputed"
                    }
            
            yield {
                "stage": "missing",
                "status": "complete",
                "text": f"→ Final shape: {final[0]:,} rows × {final[1]} columns"
            }
        else:
            yield {
                "stage": "missing",
                "status": "error",
                "text": f"→ Error: {missing_result.get('error', 'Unknown error')}"
            }
        
        # === OUTLIER STAGE ===
        outlier_stage = stages.get("outlier", {})
        outlier_result = outlier_stage.get("result", {})
        
        yield {
            "stage": "outlier",
            "status": "analyzing",
            "text": "[Analyzing...] Outlier Detection Stage"
        }
        
        if outlier_stage.get("status") == "skipped":
            yield {
                "stage": "outlier",
                "status": "skipped",
                "text": "→ No outliers detected - stage skipped"
            }
        elif outlier_stage.get("status") == "done":
            rows_removed = outlier_result.get("rows_removed", 0)
            treatment_log = outlier_result.get("treatment_log", [])
            
            for entry in treatment_log[:8]:
                if isinstance(entry, dict):
                    action = entry.get("action", "")
                    col = entry.get("column", "")
                    count = entry.get("outliers_treated", entry.get("count", 0))
                    if action and col:
                        yield {
                            "stage": "outlier",
                            "status": "action",
                            "text": f"→ {action.replace('_', ' ').title()}: '{col}' ({count} outliers)"
                        }
            
            if len(treatment_log) > 8:
                yield {
                    "stage": "outlier",
                    "status": "info",
                    "text": f"→ ...and {len(treatment_log) - 8} more treatments applied"
                }
            
            yield {
                "stage": "outlier",
                "status": "complete",
                "text": f"→ Removed {rows_removed:,} extreme outlier rows"
            }
        else:
            yield {
                "stage": "outlier",
                "status": "error",
                "text": f"→ Error: {outlier_result.get('error', 'Unknown error')}"
            }
        
        # === VISUALIZATION STAGE ===
        viz_stage = stages.get("visualize", {})
        viz_result = viz_stage.get("result", {})
        
        yield {
            "stage": "visualize",
            "status": "analyzing",
            "text": "[Analyzing...] Visualization Stage"
        }
        
        if viz_stage.get("status") == "done":
            plots_generated = viz_result.get("plots_generated", 0)
            selected_plots = viz_result.get("selected_plots", [])
            
            yield {
                "stage": "visualize",
                "status": "info",
                "text": f"→ Generated {plots_generated} visualization plots"
            }
            
            plot_types = set()
            for plot in selected_plots[:10]:
                if isinstance(plot, dict):
                    plot_types.add(plot.get("plot_type", "unknown"))
            
            for pt in list(plot_types)[:5]:
                yield {
                    "stage": "visualize",
                    "status": "action",
                    "text": f"→ Created {pt.replace('_', ' ').title()} plots"
                }
            
            yield {
                "stage": "visualize",
                "status": "complete",
                "text": "→ Visualization generation complete"
            }
        else:
            yield {
                "stage": "visualize",
                "status": "error",
                "text": f"→ Error: {viz_result.get('error', 'Unknown error')}"
            }
        
        # === COMPLETION ===
        yield {
            "stage": "complete",
            "status": "complete",
            "text": "✅ Report generation complete"
        }
    
    def generate_pdf(self, job_data: dict, plots_dir: str, output_path: str) -> Dict[str, Any]:
        """
        Generate industry-level PDF report with embedded graphs and inferences.
        """
        def log_debug(msg):
            try:
                with open("backend_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now()}] {msg}\n")
            except:
                pass

        log_debug(f"Starting PDF generation for Job {job_data.get('id')} to {output_path}")

        # Use xhtml2pdf (pure Python, no GTK required)
        try:
            from xhtml2pdf import pisa
        except ImportError:
            log_debug("xhtml2pdf not installed")
            return {
                "status": "error",
                "error": "xhtml2pdf not installed. Run: pip install xhtml2pdf"
            }
        
        try:
            # Build HTML
            log_debug("Building HTML content...")
            html_content = self._build_html(job_data, plots_dir)
            log_debug(f"HTML build successful. Length: {len(html_content)}")
            
            # Save HTML for debugging
            debug_html_path = output_path.replace('.pdf', '.html')
            with open(debug_html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            log_debug(f"Saved debug HTML to {debug_html_path}")

            # Generate PDF using xhtml2pdf
            log_debug("Starting pisa.CreatePDF...")
            with open(output_path, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
            
            if pisa_status.err:
                log_debug(f"PISA Error: {pisa_status.err}")
                return {
                    "status": "error",
                    "error": f"PDF generation failed with {pisa_status.err} errors"
                }
            
            log_debug("PDF generation successful")
            return {
                "status": "success",
                "path": output_path,
                "size_bytes": os.path.getsize(output_path)
            }
            
        except Exception as e:
            import traceback
            log_debug(f"EXCEPTION in generate_pdf: {str(e)}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_data_quality_score(self, job_data: dict) -> dict:
        """Calculate overall data quality score based on pipeline results."""
        stages = job_data.get("stages", {})
        analysis = job_data.get("analysis", {})
        
        score = 100
        issues = []
        
        # Missing value impact
        missing_result = stages.get("missing", {}).get("result", {})
        if missing_result:
            original_rows = missing_result.get("original_shape", [0, 0])[0]
            final_rows = missing_result.get("final_shape", [0, 0])[0]
            cols_dropped = len(missing_result.get("columns_dropped", []))
            
            if original_rows > 0:
                row_loss_pct = ((original_rows - final_rows) / original_rows) * 100
                if row_loss_pct > 5:
                    score -= min(15, row_loss_pct)
                    issues.append(f"{row_loss_pct:.1f}% rows removed due to missing values")
            
            if cols_dropped > 0:
                score -= min(10, cols_dropped * 2)
                issues.append(f"{cols_dropped} columns dropped due to excessive missing values")
        
        # Outlier impact
        outlier_result = stages.get("outlier", {}).get("result", {})
        if outlier_result:
            rows_removed = outlier_result.get("rows_removed", 0)
            
            # Robustly determine original rows count before outlier treatment
            # Try getting it from missing result's final shape, or raw totals
            final_shape_missing = missing_result.get("final_shape")
            
            if final_shape_missing and isinstance(final_shape_missing, (list, tuple)) and len(final_shape_missing) > 0:
                base_rows = final_shape_missing[0]
            else:
                base_rows = analysis.get("total_rows", 1000)
                
            if base_rows > 0 and rows_removed > 0:
                outlier_pct = (rows_removed / base_rows) * 100
                if outlier_pct > 2:
                    score -= min(10, outlier_pct * 2)
                    issues.append(f"{outlier_pct:.1f}% rows removed as outliers")
        
        score = max(0, min(100, score))
        
        # Determine quality level
        if score >= 90:
            level = "Excellent"
            color = "#10b981"
        elif score >= 75:
            level = "Good"
            color = "#3b82f6"
        elif score >= 60:
            level = "Fair"
            color = "#f59e0b"
        else:
            level = "Needs Attention"
            color = "#ef4444"
        
        return {
            "score": round(score),
            "level": level,
            "color": color,
            "issues": issues[:5]
        }
    
    def _get_plot_inference(self, plot_filename: str) -> str:
        """Generate intelligent inference for a plot based on its type."""
        name_lower = plot_filename.lower()
        
        if "histogram" in name_lower:
            return "This histogram shows the frequency distribution of values. A symmetric bell shape indicates normal distribution, while skewness suggests potential transformation needs."
        elif "boxplot" in name_lower:
            return "The boxplot displays the five-number summary (min, Q1, median, Q3, max). Points beyond the whiskers indicate potential outliers that may require investigation."
        elif "violin" in name_lower:
            return "Violin plots combine box plots with kernel density estimation, revealing the full distribution shape. Wider sections indicate higher data concentration."
        elif "scatter" in name_lower:
            return "This scatter plot reveals the relationship between two variables. The trend line indicates correlation strength and direction."
        elif "heatmap" in name_lower or "correlation" in name_lower:
            return "The correlation heatmap shows pairwise relationships between numeric features. Values near +1 or -1 indicate strong positive or negative correlations respectively."
        elif "barplot" in name_lower or "countplot" in name_lower:
            return "This bar chart shows the frequency distribution across categories. Imbalanced distributions may affect model performance and require sampling strategies."
        elif "kde" in name_lower:
            return "The KDE (Kernel Density Estimation) plot provides a smoothed view of the data distribution, useful for identifying multi-modal patterns."
        else:
            return "This visualization provides insights into the data structure and relationships within the dataset."
    
    def _get_strategy_justification(self, column: str, strategy: str) -> str:
        """Generate justification for imputation strategy."""
        strategy_lower = str(strategy).lower()
        
        if "median" in strategy_lower:
            return "Median imputation chosen for robustness against outliers and skewed distributions."
        elif "mean" in strategy_lower:
            return "Mean imputation selected for normally distributed data to preserve central tendency."
        elif "mode" in strategy_lower:
            return "Mode imputation applied to preserve the most frequent categorical value."
        elif "drop" in strategy_lower:
            return "Rows dropped due to critical missing values that cannot be reliably imputed."
        elif "forward" in strategy_lower or "ffill" in strategy_lower:
            return "Forward fill applied for time-series data to maintain temporal consistency."
        elif "backward" in strategy_lower or "bfill" in strategy_lower:
            return "Backward fill applied for time-series data to maintain temporal consistency."
        elif "knn" in strategy_lower:
            return "KNN imputation used to leverage similar records for more accurate value estimation."
        else:
            return f"Strategy '{strategy}' applied based on data characteristics and domain requirements."
    
    def _safe_format(self, value: Any) -> str:
        """Safely format a number with commas, returning invalid values as strings."""
        try:
            if isinstance(value, (int, float)):
                return f"{value:,}"
            elif isinstance(value, str) and value.replace(',', '').isnumeric():
                return f"{int(value.replace(',', '')):,}"
            return str(value)
        except:
            return str(value)

    def _build_html(self, job_data: dict, plots_dir: str) -> str:
        """Build comprehensive HTML for industry-level PDF generation."""
        
        analysis = job_data.get("analysis", {})
        stages = job_data.get("stages", {})
        
        # Calculate data quality score
        quality = self._calculate_data_quality_score(job_data)
        
        # xhtml2pdf-compatible CSS - Modern Purple Theme
        css = """
        <style>
            @page { 
                size: A4; 
                margin: 1.5cm;
            }
            
            body { 
                font-family: Helvetica, Arial, sans-serif; 
                line-height: 1.6; 
                color: #1e1b4b;
                font-size: 10pt;
                background-color: #faf8ff;
            }
            
            /* Header - Premium Purple Gradient Look */
            .cover-header {
                background-color: #8b5cf6;
                color: white;
                padding: 35px 30px;
                text-align: center;
                margin: -1.5cm -1.5cm 25px -1.5cm;
                border-bottom: 4px solid #7c3aed;
            }
            .cover-header h1 { 
                margin: 0; 
                font-size: 26pt;
                font-weight: bold;
                letter-spacing: -0.5px;
            }
            .cover-header .subtitle { 
                margin: 10px 0 0 0; 
                font-size: 12pt;
                opacity: 0.9;
            }
            .cover-header .meta p {
                margin: 5px 0 0 0;
                font-size: 9pt;
                opacity: 0.8;
            }
            
            /* Section Styling */
            .section { 
                margin: 25px 0;
                padding: 15px;
                background-color: white;
                border: 1px solid #e9d5ff;
                border-radius: 8px;
            }
            .section h2 { 
                color: #7c3aed; 
                font-size: 15pt;
                border-bottom: 3px solid #e9d5ff; 
                padding-bottom: 8px;
                margin: 0 0 15px 0;
            }
            .section h3 {
                color: #6b7280;
                font-size: 11pt;
                margin: 15px 0 8px 0;
                border-left: 3px solid #a78bfa;
                padding-left: 10px;
            }
            
            /* Executive Summary Box */
            .exec-summary {
                background-color: #f5f0ff;
                border: 2px solid #e9d5ff;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 15px;
            }
            .score-box {
                display: inline-block;
                width: 70px;
                height: 70px;
                line-height: 70px;
                text-align: center;
                font-size: 24pt;
                font-weight: bold;
                color: white;
                border-radius: 10px;
                margin-right: 20px;
            }
            
            /* Tables - Modern Styling */
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 12px 0;
                border-radius: 8px;
                overflow: hidden;
            }
            th {
                background-color: #8b5cf6;
                color: white;
                padding: 12px 10px;
                text-align: left;
                font-size: 9pt;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            td {
                padding: 10px;
                border-bottom: 1px solid #e9d5ff;
                font-size: 9pt;
                background-color: white;
            }
            tr:nth-child(even) td {
                background-color: #faf8ff;
            }
            
            /* Stat boxes using table */
            .stat-table {
                margin: 15px 0;
            }
            .stat-table td {
                text-align: center;
                padding: 20px 15px;
                border: 2px solid #e9d5ff;
                background-color: #f5f0ff;
                vertical-align: top;
            }
            .stat-value { 
                font-size: 20pt; 
                font-weight: bold; 
                color: #7c3aed;
                display: block;
                margin-bottom: 5px;
            }
            .stat-label { 
                font-size: 8pt; 
                color: #6b7280; 
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            /* Badges */
            .badge {
                padding: 4px 10px;
                font-size: 8pt;
                font-weight: bold;
                border-radius: 4px;
            }
            .badge-success { background-color: #dcfce7; color: #166534; }
            .badge-warning { background-color: #fef3c7; color: #92400e; }
            .badge-info { background-color: #ede9fe; color: #5b21b6; }
            .badge-error { background-color: #fee2e2; color: #991b1b; }
            
            /* Key Findings */
            .key-findings {
                background-color: #faf8ff;
                border-left: 4px solid #8b5cf6;
                padding: 12px 15px;
                margin: 15px 0;
            }
            .key-findings h3 {
                margin: 0 0 8px 0;
                color: #7c3aed;
                border: none;
                padding: 0;
            }
            .key-findings ul {
                margin: 0;
                padding-left: 18px;
            }
            .key-findings li {
                margin: 5px 0;
                color: #374151;
            }
            
            /* Plots */
            .plot-container {
                margin: 20px 0;
                page-break-inside: avoid;
                background-color: white;
                border: 1px solid #e9d5ff;
                border-radius: 8px;
                padding: 15px;
            }
            .plot-container img { 
                width: 100%;
                max-width: 450px;
                display: block;
                margin: 0 auto;
            }
            .plot-caption {
                font-size: 9pt;
                color: #6b7280;
                margin-top: 10px;
                padding: 10px;
                background-color: #faf8ff;
                border-radius: 5px;
                font-style: italic;
            }
            .plot-title {
                font-weight: bold;
                color: #7c3aed;
                margin-bottom: 10px;
                font-size: 11pt;
            }
            
            /* Footer */
            .footer {
                margin-top: 30px;
                padding-top: 20px;
                border-top: 2px solid #e9d5ff;
                text-align: center;
                font-size: 9pt;
                color: #6b7280;
            }
            .footer strong {
                color: #7c3aed;
            }
            
            /* Lists */
            ul { margin: 8px 0; padding-left: 20px; }
            li { margin: 5px 0; color: #374151; }
            
            /* Emphasis */
            strong { color: #1e1b4b; }
        </style>
        """
        
        # Build HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Agentic AI System for EDA</title>
            {css}
        </head>
        <body>
            <!-- Cover Header -->
            <div class="cover-header">
                <h1>Agentic AI System for EDA</h1>
                <p class="subtitle">Automated Data Quality & Analysis Report</p>
                <div class="meta">
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                    <p>Job ID: {job_data.get('id', 'N/A')}</p>
                </div>
            </div>
            
            <!-- Executive Summary -->
            <div class="section executive-summary">
                <h2>Executive Summary</h2>
                <div class="exec-summary">
                    <div>
                        <span class="score-box" style="background-color: {quality['color']}">{quality['score']}</span>
                        <span style="font-size: 14pt; font-weight: bold;">Data Quality Score</span>
                    </div>
                </div>
                """
                
        # Prepare key findings metrics
        total_rows = self._safe_format(analysis.get('total_rows', 0))
        total_cols = analysis.get('total_columns', 0)
        num_cols = analysis.get('numeric_columns', 0)
        cat_cols = analysis.get('categorical_columns', 0)
        missing_cols = analysis.get('missing_analysis', {}).get('columns_with_missing', 0)
        outlier_cols = analysis.get('outlier_analysis', {}).get('columns_with_outliers', 0)
        
        # Generate AI Executive Insight
        overview_data = {
            "total_rows": analysis.get('total_rows', 0),
            "total_columns": total_cols,
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "columns_with_missing": missing_cols,
            "columns_with_outliers": outlier_cols,
            "quality_score": quality['score'],
            "quality_level": quality['level']
        }
        ai_executive_insight = self._generate_ai_insight(
            f"Analyzing dataset with {total_rows} rows and {total_cols} columns. Quality score: {quality['score']}/100 ({quality['level']})",
            overview_data,
            "overview"
        )

        html += f"""
            <div class="key-findings">
                <h3>Executive Insight:</h3>
                <p style="color: #374151; line-height: 1.7;">{ai_executive_insight}</p>
            </div>
            <div class="key-findings">
                <h3>Key Findings:</h3>
                <ul>
                    <li>Dataset contains <strong>{total_rows}</strong> rows and <strong>{total_cols}</strong> columns.</li>
                    <li>Identified <strong>{num_cols}</strong> numeric and <strong>{cat_cols}</strong> categorical features.</li>
                    <li>Missing values detected in <strong>{missing_cols}</strong> columns.</li>
                    <li>Outliers treated in <strong>{outlier_cols}</strong> columns.</li>
                </ul>
            </div>
        </div>
        """
            
        # Missing Value Section
        html += '<div class="section"><h2>Missing Value Analysis</h2>'
        try:
            missing_stage = stages.get("missing", {})
            missing_result = missing_stage.get("result", {})
            
            if missing_stage.get("status") == "skipped":
                html += """
                    <p><span class="badge badge-success">No Missing Values</span></p>
                    <p>The dataset is complete with no missing values detected.</p>
                """
            elif missing_stage.get("status") == "done":
                # Safe extraction of shape
                original = missing_result.get("original_shape", [0, 0])
                final = missing_result.get("final_shape", [0, 0])
                rows_dropped = missing_result.get("rows_dropped", 0)
                cols_dropped = missing_result.get("columns_dropped", [])
                imputed = missing_result.get("imputed_values", {})
                
                html += f"""
                    <table class="stat-table">
                        <tr>
                            <td>
                                <div class="stat-value">{self._safe_format(original[0])} → {self._safe_format(final[0])}</div>
                                <div class="stat-label">Rows (Before → After)</div>
                            </td>
                            <td>
                                <div class="stat-value">{self._safe_format(rows_dropped)}</div>
                                <div class="stat-label">Rows Removed</div>
                            </td>
                            <td>
                                <div class="stat-value">{len(cols_dropped)}</div>
                                <div class="stat-label">Columns Removed</div>
                            </td>
                        </tr>
                    </table>
                """
                
                # Generate AI Insight for Missing Values
                missing_summary = {
                    "original_rows": original[0] if isinstance(original, list) else 0,
                    "final_rows": final[0] if isinstance(final, list) else 0,
                    "rows_dropped": rows_dropped,
                    "columns_dropped": len(cols_dropped),
                    "columns_imputed": len(imputed_strategies) if 'imputed_strategies' in dir() else 0
                }
                ai_missing_insight = self._generate_ai_insight(
                    f"Missing value treatment: Dropped {rows_dropped} rows and {len(cols_dropped)} columns. Imputed values in remaining columns.",
                    missing_summary,
                    "missing"
                )
                html += f"""
                <div class="key-findings" style="margin-top: 15px;">
                    <h3>Insight:</h3>
                    <p style="color: #374151; line-height: 1.7;">{ai_missing_insight}</p>
                </div>
                """
                
                # Column Treatment Table - Show ALL columns
                column_actions = missing_result.get("column_actions", {})
                
                # Debug: Print what we received
                print(f"[DEBUG] missing_result keys: {list(missing_result.keys())}")
                print(f"[DEBUG] column_actions: {column_actions}")
                
                if column_actions:
                    html += """
                    <h3>Column Treatment Summary</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Action Taken</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    
                    # Separate columns by treatment type for better organization
                    treated_cols = []
                    clean_cols = []
                    dropped_cols = []
                    
                    for col, action in column_actions.items():
                        if action == "no_missing":
                            clean_cols.append((col, action))
                        elif "drop_column" in str(action):
                            dropped_cols.append((col, action))
                        else:
                            treated_cols.append((col, action))
                    
                    # First show treated columns (most important)
                    for col, action in sorted(treated_cols):
                        badge_class = "badge-warning" if "drop" in str(action) else "badge-info"
                        html += f"""
                            <tr>
                                <td><strong>{col}</strong></td>
                                <td><span class="badge {badge_class}">{action}</span></td>
                                <td>Treated</td>
                            </tr>
                        """
                    
                    # Then show dropped columns
                    for col, action in sorted(dropped_cols):
                        html += f"""
                            <tr>
                                <td><strong>{col}</strong></td>
                                <td><span class="badge badge-error">{action}</span></td>
                                <td>Removed</td>
                            </tr>
                        """
                    
                    # Finally show clean columns (no missing)
                    for col, action in sorted(clean_cols):
                        html += f"""
                            <tr>
                                <td>{col}</td>
                                <td><span class="badge badge-success">No Missing Values</span></td>
                                <td>Clean</td>
                            </tr>
                        """
                    
                    html += "</tbody></table>"
                
                # Dropped columns table
                if cols_dropped:
                    html += """
                    <h3>Dropped Columns</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Column Name</th>
                                <th>Reason</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    for col in cols_dropped:
                        html += f"""
                            <tr>
                                <td><strong>{col}</strong></td>
                                <td>Excessive missing values (>50% threshold)</td>
                            </tr>
                        """
                    html += "</tbody></table>"
            else:
                error = missing_result.get("error", "Unknown error")
                html += f'<p class="error">Error in Missing Analysis: {error}</p>'
        except Exception as e:
            html += f'<p class="error">Could not generate Missing Value section: {str(e)}</p>'
        html += "</div>"
        
        # Outlier Section
        html += '<div class="section"><h2>Outlier Detection & Treatment</h2>'
        try:
            outlier_stage = stages.get("outlier", {})
            outlier_result = outlier_stage.get("result", {})
            
            if outlier_stage.get("status") == "skipped":
                html += """
                    <p><span class="badge badge-success">No Outliers Detected</span></p>
                    <p>No significant outliers were detected.</p>
                """
            elif outlier_stage.get("status") == "done":
                rows_removed = outlier_result.get("rows_removed", 0)
                treatment_log = outlier_result.get("treatment_log", [])
                original = outlier_result.get("original_shape", [0, 0])
                final = outlier_result.get("final_shape", [0, 0])
                
                orig_rows = original[0] if isinstance(original, (list, tuple)) and len(original) > 0 else 'N/A'
                final_rows = final[0] if isinstance(final, (list, tuple)) and len(final) > 0 else 'N/A'
                
                html += f"""
                    <table class="stat-table">
                        <tr>
                            <td>
                                <div class="stat-value">{self._safe_format(orig_rows)} → {self._safe_format(final_rows)}</div>
                                <div class="stat-label">Rows (Before → After)</div>
                            </td>
                            <td>
                                <div class="stat-value">{self._safe_format(rows_removed)}</div>
                                <div class="stat-label">Outlier Rows Removed</div>
                            </td>
                            <td>
                                <div class="stat-value">{len(treatment_log)}</div>
                                <div class="stat-label">Columns Treated</div>
                            </td>
                        </tr>
                    </table>
                """
                
                # Generate AI Insight for Outliers
                outlier_summary = {
                    "original_rows": orig_rows,
                    "final_rows": final_rows,
                    "rows_removed": rows_removed,
                    "columns_treated": len(treatment_log),
                    "treatment_actions": [e.get("action", "unknown") for e in treatment_log if isinstance(e, dict)][:5]
                }
                ai_outlier_insight = self._generate_ai_insight(
                    f"Outlier treatment: Removed {rows_removed} extreme rows and treated outliers in {len(treatment_log)} columns.",
                    outlier_summary,
                    "outlier"
                )
                html += f"""
                <div class="key-findings" style="margin-top: 15px;">
                    <h3>Insight:</h3>
                    <p style="color: #374151; line-height: 1.7;">{ai_outlier_insight}</p>
                </div>
                """
                
                if treatment_log:
                    html += """
                    <h3>Treatment Details</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Action</th>
                                <th>Outliers Treated</th>
                                <th>Intent</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    for entry in treatment_log:
                        if isinstance(entry, dict):
                            if "summary" in entry: continue
                            
                            col = entry.get("column", "N/A")
                            action = entry.get("action", "treated").replace("_", " ").title()
                            count = entry.get("outliers_capped", 
                                     entry.get("outliers_flagged", 
                                     entry.get("outliers_treated", 
                                     entry.get("outlier_count",
                                     entry.get("count", 0)))))
                            intent = entry.get("intent", "MEASURE")
                            
                            badge_class = "badge-warning" if "remove" in action.lower() or "flag" in action.lower() else "badge-info"
                            html += f"""
                                <tr>
                                    <td><strong>{col}</strong></td>
                                    <td><span class="badge {badge_class}">{action}</span></td>
                                    <td>{self._safe_format(count)}</td>
                                    <td>{intent}</td>
                                </tr>
                            """
                    html += "</tbody></table>"
            else:
                error = outlier_result.get("error", "Unknown error")
                html += f'<p class="error">Error in Outlier Analysis: {error}</p>'
        except Exception as e:
            html += f'<p class="error">Could not generate Outlier section: {str(e)}</p>'
        html += "</div>"
        
        # Visualization Section
        if plots_dir and os.path.exists(plots_dir):
            html += '<div class="section"><h2>Visualization Gallery</h2>'
            try:
                # Prepare inference map from job_data for specific insights
                viz_result = stages.get("visualize", {}).get("result", {})
                generated_details = viz_result.get("generated_plots_details", [])
                inference_map = {}
                for item in generated_details:
                    fname = os.path.basename(item.get("filepath", ""))
                    inf = item.get("inference", "")
                    if fname and inf:
                        inference_map[fname] = inf

                plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
                if plot_files:
                    html += "<p>The following visualizations were automatically generated to provide insights into the data characteristics and distributions.</p>"
                    import base64
                    
                    all_plot_insights = []
                    
                    for plot_file in sorted(plot_files):
                        try:
                            plot_path = os.path.join(plots_dir, plot_file)
                            title = plot_file.replace('.png', '').replace('_', ' ').title()
                            
                            # Generate AI insight for this specific visualization
                            plot_context = {
                                "plot_type": title,
                                "filename": plot_file,
                                "dataset_rows": analysis.get('total_rows', 0),
                                "dataset_cols": analysis.get('total_columns', 0),
                                "numeric_columns": analysis.get('numeric_columns', 0),
                                "categorical_columns": analysis.get('categorical_columns', 0)
                            }
                            
                            # Generate detailed AI interpretation for this plot
                            plot_insight = self._generate_ai_insight(
                                f"Interpreting visualization: {title}. This is a data visualization from EDA of a dataset with {analysis.get('total_rows', 0)} rows.",
                                plot_context,
                                "visualization"
                            )
                            all_plot_insights.append({"title": title, "insight": plot_insight})
                        
                            with open(plot_path, 'rb') as img_file:
                                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            img_src = f"data:image/png;base64,{img_data}"
                            
                            html += f"""
                            <div class="plot-container">
                                <div class="plot-title">{title}</div>
                                <img src="{img_src}" alt="{title}" style="width:100%; max-width:500px;">
                                <div class="plot-caption">
                                    <strong>Interpretation:</strong> {plot_insight}
                                </div>
                            </div>
                            """
                        except Exception as e:
                            html += f"<p>Error loading plot {plot_file}: {str(e)}</p>"
                    
                    # Generate overall visualization summary
                    if all_plot_insights:
                        # Collect all insights for overall summary generation
                        insight_texts = [p["insight"] for p in all_plot_insights if p["insight"]]
                        viz_summary_data = {
                            "total_plots": len(all_plot_insights),
                            "plot_types": [p["title"] for p in all_plot_insights],
                            "dataset_size": analysis.get('total_rows', 0),
                            "sample_insights": insight_texts[:3]  # Sample of insights
                        }
                        overall_viz_insight = self._generate_ai_insight(
                            f"Summarize the key findings from {len(all_plot_insights)} visualizations. Focus on data quality, distributions, and notable patterns.",
                            viz_summary_data,
                            "visualization"
                        )
                        html += f"""
                        <div class="key-findings" style="margin-top: 20px;">
                            <h3>Overall Visualization Summary:</h3>
                            <p style="color: #374151; line-height: 1.7;">{overall_viz_insight}</p>
                        </div>
                        """
            except Exception as e:
                html += f'<p class="error">Could not generate Visualization section: {str(e)}</p>'
            html += "</div>"
    
        # Footer
        html += """
            <div class="footer">
                <p><strong>Agentic AI System for EDA</strong> | Automated Data Quality & Analysis Report</p>
                <p>This report was generated using intelligent multi-agent data analysis.</p>
            </div>
        </body>
        </html>
        """
        
        return html


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def format_sse_event(data: dict) -> str:
    """Format data for Server-Sent Events."""
    return f"data: {json.dumps(data)}\n\n"


# =====================================================
# STANDALONE TEST
# =====================================================

if __name__ == "__main__":
    # Test with mock data
    mock_job = {
        "id": "test123",
        "analysis": {
            "total_rows": 1460,
            "total_columns": 81,
            "numeric_columns": 38,
            "categorical_columns": 43
        },
        "stages": {
            "missing": {
                "status": "done",
                "result": {
                    "original_shape": [1460, 81],
                    "final_shape": [1458, 79],
                    "rows_dropped": 2,
                    "columns_dropped": ["Alley", "PoolQC"],
                    "imputed_values": {
                        "LotFrontage": "median",
                        "MasVnrArea": "mean"
                    }
                }
            },
            "outlier": {
                "status": "done",
                "result": {
                    "original_shape": [1458, 79],
                    "final_shape": [1443, 79],
                    "rows_removed": 15,
                    "treatment_log": [
                        {"action": "capped", "column": "SalePrice", "outliers_treated": 23, "method": "IQR"},
                        {"action": "removed", "column": "LotArea", "count": 8, "method": "Z-Score"}
                    ]
                }
            },
            "visualize": {
                "status": "done",
                "result": {
                    "plots_generated": 12,
                    "selected_plots": [
                        {"plot_type": "histogram"},
                        {"plot_type": "correlation_heatmap"},
                        {"plot_type": "boxplot"}
                    ]
                }
            }
        }
    }
    
    agent = ReportAgent()
    
    print("=" * 60)
    print("REPORT AGENT TEST - Streaming Output")
    print("=" * 60)
    
    for section in agent.generate_report_stream(mock_job):
        print(f"[{section['status']:10}] {section['text']}")
    
    print("\n" + "=" * 60)
