# =====================================================
# Flask Backend - EDA Agent Pipeline
# =====================================================
# Main Flask application with REST API endpoints
# =====================================================

# IMPORTANT: Set matplotlib backend before any imports
# This prevents "main thread is not in main loop" errors when plotting in threads
import matplotlib
matplotlib.use('Agg')

import os
import uuid
import json
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import agents
from agents import MasterAgent, run_missing_value_agent, run_outlier_agent, run_visualization_agent, run_correlation_agent, ReportAgent


# =====================================================
# CUSTOM JSON ENCODER FOR NUMPY TYPES
# =====================================================

class NumpyJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# =====================================================
# APP CONFIGURATION
# =====================================================

app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider  # Use custom JSON provider
app.json = NumpyJSONProvider(app)  # Initialize it
CORS(app)  # Enable CORS for React frontend

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Job storage (in production, use Redis or database)
jobs = {}


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_job_dir(job_id):
    """Get or create job output directory."""
    job_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


# =====================================================
# API ROUTES
# =====================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a CSV file for processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Only CSV files are allowed"}), 400
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save file
    filename = secure_filename(file.filename)
    job_dir = get_job_dir(job_id)
    filepath = os.path.join(job_dir, f"original_{filename}")
    file.save(filepath)
    
    # Initialize job
    jobs[job_id] = {
        "id": job_id,
        "status": "uploaded",
        "filename": filename,
        "filepath": filepath,
        "created_at": datetime.now().isoformat(),
        "stages": {}
    }
    
    return jsonify({
        "job_id": job_id,
        "filename": filename,
        "status": "uploaded",
        "message": "File uploaded successfully"
    })


@app.route('/api/analyze/<job_id>', methods=['GET'])
def analyze_data(job_id):
    """Run master agent analysis on uploaded data."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    
    try:
        import pandas as pd
        df = pd.read_csv(job["filepath"])
        
        # Run master agent analysis
        master = MasterAgent()
        analysis = master.analyze_data(df)
        
        # Update job
        job["analysis"] = analysis
        job["status"] = "analyzed"
        
        return jsonify({
            "job_id": job_id,
            "status": "analyzed",
            "analysis": analysis
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/pipeline/start', methods=['POST'])
def start_pipeline():
    """Start the intelligent EDA pipeline (Phase 1: Cleaning & Visualization)."""
    data = request.get_json()
    job_id = data.get("job_id")
    
    # Note: model_type and target_column are now collected in Phase 2
    
    if not job_id or job_id not in jobs:
        return jsonify({"error": "Invalid job_id"}), 400
    
    job = jobs[job_id]
    job["status"] = "running"
    
    try:
        import pandas as pd
        
        # Load original data
        df = pd.read_csv(job["filepath"])
        job_dir = get_job_dir(job_id)
        
        # Initialize stage tracking
        job["stages"] = {
            "analysis": {"status": "running"},
            "missing": {"status": "pending"},
            "outlier": {"status": "pending"},
            "visualize": {"status": "pending"},
            "correlation": {"status": "pending"} # Will be run later
        }
        
        # === STAGE 1: Master Agent Analysis ===
        master = MasterAgent()
        analysis = master.analyze_data(df)
        job["analysis"] = analysis
        job["stages"]["analysis"] = {"status": "done", "result": analysis}
        
        current_csv = job["filepath"]
        
        # === STAGE 2: Missing Value Agent (if needed) ===
        if "missing_value_agent" in analysis["agents_to_run"]:
            job["stages"]["missing"]["status"] = "running"
            
            output_csv = os.path.join(job_dir, "stage1_missing_cleaned.csv")
            result = run_missing_value_agent(current_csv, output_csv)
            
            job["stages"]["missing"] = {
                "status": "done" if result["status"] == "success" else "error",
                "result": result
            }
            
            if result["status"] == "success":
                current_csv = output_csv
        else:
            job["stages"]["missing"] = {
                "status": "skipped",
                "reason": "No missing values detected"
            }
        
        # === STAGE 3: Outlier Agent (if needed) ===
        if "outlier_agent" in analysis["agents_to_run"]:
            job["stages"]["outlier"]["status"] = "running"
            
            output_csv = os.path.join(job_dir, "stage2_outlier_cleaned.csv")
            result = run_outlier_agent(current_csv, output_csv)
            
            job["stages"]["outlier"] = {
                "status": "done" if result["status"] == "success" else "error",
                "result": result
            }
            
            if result["status"] == "success":
                current_csv = output_csv
        else:
            job["stages"]["outlier"] = {
                "status": "skipped",
                "reason": "No outliers detected"
            }
        
        # === STAGE 4: Visualization Agent (always runs) ===
        job["stages"]["visualize"]["status"] = "running"
        
        result = run_visualization_agent(current_csv, job_dir)
        
        job["stages"]["visualize"] = {
            "status": "done" if result["status"] == "success" else "error",
            "result": result
        }
        
        # STOP HERE for Phase 1
        job["current_csv"] = current_csv  # Save for Phase 2
        job["status"] = "ready_for_correlation"
        
        # Save intermediate final CSV
        final_csv = os.path.join(job_dir, "final_cleaned.csv")
        pre_corr_csv = os.path.join(job_dir, "pre_correlation_cleaned.csv")
        import shutil
        shutil.copy(current_csv, final_csv)
        shutil.copy(current_csv, pre_corr_csv)  # For separate download
        job["final_csv"] = final_csv
        job["pre_correlation_csv"] = pre_corr_csv
        
        return jsonify({
            "job_id": job_id,
            "status": "ready_for_correlation",
            "stages": job["stages"],
            "agents_run": analysis["agents_to_run"],
            "agents_skipped": analysis["agents_skipped"]
        })
        
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/pipeline/correlation', methods=['POST'])
def run_correlation_stage():
    """Run the correlation agent (Phase 2)."""
    data = request.get_json()
    job_id = data.get("job_id")
    model_type = data.get("model_type", "tree")
    target_column = data.get("target_column", "")
    
    if not job_id or job_id not in jobs:
        return jsonify({"error": "Invalid job_id"}), 400
        
    job = jobs[job_id]
    
    # Ensure Phase 1 is done
    if "current_csv" not in job:
         return jsonify({"error": "Pipeline Phase 1 not completed"}), 400
         
    job["status"] = "running_correlation"
    job["model_type"] = model_type
    job["target_column"] = target_column
    
    try:
        current_csv = job["current_csv"]
        job_dir = get_job_dir(job_id)
        
        # === STAGE 5: Correlation Agent ===
        job["stages"]["correlation"]["status"] = "running"
        
        output_csv = os.path.join(job_dir, "stage4_correlation_cleaned.csv")
        result = run_correlation_agent(
            current_csv, output_csv, model_type, target_column, job_dir
        )
        
        job["stages"]["correlation"] = {
            "status": "done" if result["status"] == "success" else "error",
            "result": result
        }
        
        final_csv_path = current_csv
        if result["status"] == "success":
            final_csv_path = output_csv
        
        # Save final cleaned CSV
        final_csv = os.path.join(job_dir, "final_cleaned.csv")
        import shutil
        shutil.copy(final_csv_path, final_csv)
        
        job["final_csv"] = final_csv
        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()
        
        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "stages": job["stages"]
        })
        
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        # Even if correlation fails, we can still consider the pipeline 'completed' with errors?
        # But let's mark as completed so user can download partial results
        job["status"] = "completed" 
        job["stages"]["correlation"]["status"] = "error"
        job["stages"]["correlation"]["error"] = str(e)
        return jsonify({"error": str(e)}), 500


@app.route('/api/pipeline/status/<job_id>', methods=['GET'])
def get_pipeline_status(job_id):
    """Get current pipeline status."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    
    return jsonify({
        "job_id": job_id,
        "status": job.get("status"),
        "stages": job.get("stages", {}),
        "analysis": job.get("analysis", {}),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at")
    })


@app.route('/api/pipeline/results/<job_id>', methods=['GET'])
def get_pipeline_results(job_id):
    """Get full pipeline results."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    
    if job.get("status") not in ["completed", "ready_for_correlation"]:
        return jsonify({"error": "Pipeline not completed yet"}), 400
    
    # Get list of generated plots
    job_dir = get_job_dir(job_id)
    plots_dir = os.path.join(job_dir, "plots")
    plots = []
    
    if os.path.exists(plots_dir):
        plots = [f"/api/plots/{job_id}/{f}" for f in os.listdir(plots_dir) if f.endswith('.png')]
    
    return jsonify({
        "job_id": job_id,
        "status": job.get("status"),
        "download_url": f"/api/download/{job_id}/final_cleaned.csv",
        "plots": plots,
        "stages": job.get("stages", {}),
        "analysis": job.get("analysis", {})
    })


@app.route('/api/download/<job_id>/<filename>', methods=['GET'])
def download_file(job_id, filename):
    """Download a result file."""
    job_dir = get_job_dir(job_id)
    filepath = os.path.join(job_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(filepath, as_attachment=True)


@app.route('/api/plots/<job_id>/<filename>', methods=['GET'])
def get_plot(job_id, filename):
    """Get a generated plot image."""
    job_dir = get_job_dir(job_id)
    plots_dir = os.path.join(job_dir, "plots")
    
    return send_from_directory(plots_dir, filename)


# =====================================================
# REPORT ENDPOINTS (SSE Streaming + PDF)
# =====================================================

@app.route('/api/report/stream/<job_id>', methods=['GET'])
def stream_report(job_id):
    """Stream report updates via Server-Sent Events."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    
    if job.get("status") != "completed":
        return jsonify({"error": "Pipeline not completed yet"}), 400
    
    def generate():
        report_agent = ReportAgent()
        
        for section in report_agent.generate_report_stream(job):
            yield f"data: {json.dumps(section)}\n\n"
        
        # Signal end of stream
        yield f"data: {json.dumps({'stage': 'done', 'status': 'complete'})}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )


@app.route('/api/report/pdf/<job_id>', methods=['GET'])
def download_pdf_report(job_id):
    """Generate and download PDF report with embedded graphs."""
    import traceback
    
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    
    if job.get("status") not in ["completed", "ready_for_correlation"]:
        return jsonify({"error": "Pipeline not completed yet"}), 400
    
    try:
        job_dir = get_job_dir(job_id)
        plots_dir = os.path.join(job_dir, "plots")
        pdf_path = os.path.join(job_dir, "eda_report.pdf")
        
        print(f"[PDF] Generating PDF for job {job_id}")
        print(f"[PDF] Plots dir: {plots_dir}, exists: {os.path.exists(plots_dir)}")
        
        report_agent = ReportAgent()
        result = report_agent.generate_pdf(job, plots_dir, pdf_path)
        
        print(f"[PDF] Result: {result}")
        
        if result["status"] == "error":
            print(f"[PDF] Error: {result['error']}")
            return jsonify({"error": result["error"]}), 500
        
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"eda_report_{job_id}.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        print(f"[PDF] Exception: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/report/correlation-pdf/<job_id>', methods=['GET'])
def download_correlation_pdf_report(job_id):
    """Generate and download PDF report for correlation analysis using dynamic job data."""
    import traceback
    import base64
    
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    
    if job.get("status") != "completed":
        return jsonify({"error": "Pipeline not completed yet"}), 400
    
    try:
        job_dir = get_job_dir(job_id)
        
        # Get correlation results from job stages
        correlation_result = job.get("stages", {}).get("correlation", {}).get("result", {})
        
        # Try to load detailed metadata from saved files
        metadata_path = os.path.join(job_dir, "correlation_metadata.json")
        llm_summary_path = os.path.join(job_dir, "correlation_llm_summary.txt")
        
        metadata = {}
        llm_summary = correlation_result.get("llm_summary", "")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        if os.path.exists(llm_summary_path):
            with open(llm_summary_path, 'r', encoding='utf-8') as f:
                llm_summary = f.read()
        
        # Generate styled HTML report from job data
        html_content = _generate_dynamic_correlation_report_html(
            job=job,
            correlation_result=correlation_result,
            metadata=metadata,
            llm_summary=llm_summary,
            job_dir=job_dir
        )
        
        # Convert to PDF using WeasyPrint
        pdf_path = os.path.join(job_dir, "correlation_report.pdf")
        
        try:
            from weasyprint import HTML, CSS
            HTML(string=html_content).write_pdf(pdf_path)
        except ImportError:
            # Fallback: Save as HTML if WeasyPrint not available
            html_path = os.path.join(job_dir, "correlation_report.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return send_file(
                html_path,
                as_attachment=True,
                download_name=f"correlation_report_{job_id}.html",
                mimetype="text/html"
            )
        
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"correlation_report_{job_id}.pdf",
            mimetype="application/pdf"
        )
        
    except Exception as e:
        print(f"[CORRELATION PDF] Exception: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _generate_dynamic_correlation_report_html(job: dict, correlation_result: dict, 
                                               metadata: dict, llm_summary: str, 
                                               job_dir: str) -> str:
    """Generate styled HTML from dynamic correlation analysis data."""
    from datetime import datetime
    import base64
    
    job_id = job.get("id", "N/A")
    model_type = job.get("model_type", "N/A")
    target_col = job.get("target_column", "Not specified") or "Not specified"
    
    # Extract data from correlation result or metadata
    original_shape = correlation_result.get("original_shape", metadata.get("original_shape", [0, 0]))
    final_shape = correlation_result.get("final_shape", metadata.get("final_shape", [0, 0]))
    columns_removed = correlation_result.get("columns_removed", len(metadata.get("removed_columns", [])))
    removed_columns = metadata.get("removed_columns", correlation_result.get("removed_columns", []))
    redundant_pairs = metadata.get("redundant_pairs", [])
    vif_results = metadata.get("vif", [])
    
    # Build HTML sections
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Correlation Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background: #ffffff;
                color: #1f2937;
                line-height: 1.6;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
            .header p {{ margin: 5px 0; opacity: 0.9; }}
            .section {{ margin: 25px 0; padding: 20px; background: #f9fafb; border-radius: 8px; border-left: 4px solid #4f46e5; }}
            h2 {{ color: #4f46e5; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; margin-top: 0; }}
            h3 {{ color: #6366f1; margin-top: 20px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
            .stat-box {{ background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #4f46e5; }}
            .stat-label {{ font-size: 12px; color: #6b7280; margin-top: 5px; }}
            .data-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
            .data-table th {{ background: #4f46e5; color: white; padding: 10px; text-align: left; }}
            .data-table td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }}
            .data-table tr:nth-child(even) {{ background: #f3f4f6; }}
            .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }}
            .badge-warning {{ background: #fef3c7; color: #92400e; }}
            .badge-error {{ background: #fee2e2; color: #dc2626; }}
            .badge-info {{ background: #dbeafe; color: #1e40af; }}
            .llm-summary {{ background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .llm-summary h3 {{ margin-top: 0; }}
            .llm-content {{ color: #374151; }}
            .llm-content h2 {{ color: #4f46e5; font-size: 18px; margin-top: 25px; border-bottom: 1px solid #e5e7eb; padding-bottom: 5px; }}
            .llm-content h3 {{ color: #6366f1; font-size: 16px; margin-top: 20px; }}
            .llm-content h4 {{ color: #818cf8; font-size: 14px; margin-top: 15px; }}
            .llm-content strong {{ color: #4338ca; }}
            .llm-content ol {{ margin-left: 20px; }}
            .llm-content li {{ margin-bottom: 8px; }}
            .plot-container {{ margin: 20px 0; text-align: center; }}
            .plot-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; text-align: center; color: #6b7280; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔗 Correlation Analysis Report</h1>
            <p><strong>Job ID:</strong> {job_id}</p>
            <p><strong>Model Type:</strong> {model_type}</p>
            <p><strong>Target Column:</strong> {target_col}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <!-- Summary Statistics -->
        <div class="section">
            <h2>📊 Analysis Summary</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{original_shape[0] if isinstance(original_shape, list) else 0} × {original_shape[1] if isinstance(original_shape, list) else 0}</div>
                    <div class="stat-label">Original Shape</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{final_shape[0] if isinstance(final_shape, list) else 0} × {final_shape[1] if isinstance(final_shape, list) else 0}</div>
                    <div class="stat-label">Final Shape</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{columns_removed}</div>
                    <div class="stat-label">Columns Removed</div>
                </div>
            </div>
        </div>
    """
    
    # Removed Columns Section
    if removed_columns:
        html += """
        <div class="section">
            <h2>🗑️ Removed Features</h2>
            <p>The following features were removed due to high correlation or multicollinearity:</p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Reason</th>
                        <th>Metric</th>
                    </tr>
                </thead>
                <tbody>
        """
        for col in removed_columns:
            if isinstance(col, dict):
                feature = col.get("removed", "Unknown")
                reason = col.get("reason", "Unknown").replace("_", " ").title()
                metric = col.get("metric", "N/A")
                if isinstance(metric, float):
                    metric = f"{metric:.4f}"
                badge_class = "badge-error" if "vif" in str(col.get("reason", "")).lower() else "badge-warning"
                html += f"""
                    <tr>
                        <td><strong>{feature}</strong></td>
                        <td><span class="badge {badge_class}">{reason}</span></td>
                        <td>{metric}</td>
                    </tr>
                """
            else:
                html += f"""
                    <tr>
                        <td><strong>{col}</strong></td>
                        <td><span class="badge badge-warning">High Correlation</span></td>
                        <td>N/A</td>
                    </tr>
                """
        html += "</tbody></table></div>"
    
    # VIF Results Section
    if vif_results:
        html += """
        <div class="section">
            <h2>📈 VIF Analysis (Variance Inflation Factor)</h2>
            <p>VIF measures multicollinearity. Values > 10 indicate high multicollinearity.</p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>VIF Value</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        for vif in vif_results[:20]:  # Limit to top 20
            feature = vif.get("feature", "Unknown")
            vif_val = vif.get("vif", 0)
            if vif_val is None or vif_val == float('inf'):
                vif_display = "∞ (Infinite)"
                status = '<span class="badge badge-error">Remove</span>'
            elif vif_val > 10:
                vif_display = f"{vif_val:.2f}"
                status = '<span class="badge badge-error">High</span>'
            elif vif_val > 5:
                vif_display = f"{vif_val:.2f}"
                status = '<span class="badge badge-warning">Moderate</span>'
            else:
                vif_display = f"{vif_val:.2f}"
                status = '<span class="badge badge-info">OK</span>'
            
            html += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{vif_display}</td>
                    <td>{status}</td>
                </tr>
            """
        html += "</tbody></table></div>"
    
    # Redundant Pairs Section
    if redundant_pairs:
        html += """
        <div class="section">
            <h2>🔄 Highly Correlated Pairs</h2>
            <p>Feature pairs with correlation above threshold (|r| > 0.85):</p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Feature 1</th>
                        <th>Feature 2</th>
                        <th>Correlation</th>
                    </tr>
                </thead>
                <tbody>
        """
        for pair in redundant_pairs[:15]:  # Limit to top 15
            col1 = pair.get("col1", "Unknown")
            col2 = pair.get("col2", "Unknown")
            corr = pair.get("correlation", 0)
            if isinstance(corr, float):
                corr_display = f"{corr:.4f}"
            else:
                corr_display = str(corr)
            html += f"""
                <tr>
                    <td>{col1}</td>
                    <td>{col2}</td>
                    <td><strong>{corr_display}</strong></td>
                </tr>
            """
        html += "</tbody></table></div>"
    
    # LLM Summary Section - Convert markdown to HTML
    if llm_summary:
        # Convert markdown to HTML
        formatted_summary = _convert_markdown_to_html(llm_summary)
        html += f"""
        <div class="llm-summary">
            <h3>🤖 AI Analysis Summary</h3>
            <div class="llm-content">{formatted_summary}</div>
        </div>
        """
    
    # Correlation Plots Section
    plots_dir = os.path.join(job_dir, "plots", "correlation")
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        if plot_files:
            html += """
            <div class="section">
                <h2>📊 Correlation Visualizations</h2>
                <p>Scatter plots showing highly correlated feature pairs:</p>
            """
            import base64
            for plot_file in sorted(plot_files)[:10]:  # Limit to 10 plots
                plot_path = os.path.join(plots_dir, plot_file)
                try:
                    with open(plot_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    plot_title = plot_file.replace('.png', '').replace('_', ' ').replace('correlation ', '').title()
                    html += f"""
                    <div class="plot-container">
                        <img src="data:image/png;base64,{img_data}" alt="{plot_title}">
                        <p><strong>{plot_title}</strong></p>
                    </div>
                    """
                except Exception:
                    pass
            html += "</div>"
    
    # Footer
    html += """
        <div class="footer">
            <p><strong>Agentic AI EDA System</strong> | Correlation Analysis Module</p>
            <p>This report was generated using intelligent feature selection and multicollinearity detection.</p>
        </div>
    </body>
    </html>
    """
    
    return html


def _convert_markdown_to_html(markdown_text: str) -> str:
    """Convert markdown formatting to HTML for display in reports."""
    import re
    
    html = markdown_text
    
    # Convert bold text: **text** -> <strong>text</strong>
    html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html)
    
    # Convert headers
    html = re.sub(r'^# (.+)$', r'<h2 style="color:#4f46e5; margin-top:20px;">\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h3 style="color:#6366f1; margin-top:15px;">\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h4 style="color:#818cf8; margin-top:10px;">\1</h4>', html, flags=re.MULTILINE)
    
    # Convert numbered lists: 1. item -> <ol><li>item</li></ol>
    lines = html.split('\n')
    result_lines = []
    in_list = False
    
    for line in lines:
        numbered_match = re.match(r'^(\d+)\.\s+(.+)$', line)
        if numbered_match:
            if not in_list:
                result_lines.append('<ol style="margin-left:20px; line-height:1.8;">')
                in_list = True
            result_lines.append(f'<li>{numbered_match.group(2)}</li>')
        else:
            if in_list and line.strip() == '':
                result_lines.append('</ol>')
                in_list = False
            elif in_list and not numbered_match:
                result_lines.append('</ol>')
                in_list = False
                result_lines.append(line)
            else:
                result_lines.append(line)
    
    if in_list:
        result_lines.append('</ol>')
    
    html = '\n'.join(result_lines)
    
    # Convert markdown tables to HTML tables
    lines = html.split('\n')
    in_table = False
    table_html = []
    result_lines = []
    
    for line in lines:
        # Check if line is a table row
        if '|' in line and line.strip().startswith('|') and line.strip().endswith('|'):
            if not in_table:
                in_table = True
                table_html = ['<table style="width:100%; border-collapse:collapse; margin:15px 0; font-size:13px;">']
            
            cells = [c.strip() for c in line.split('|')[1:-1]]
            
            # Skip separator row (contains only dashes)
            if all(c.replace('-', '').replace(':', '').strip() == '' for c in cells):
                continue
            
            if len(table_html) == 1:
                # First row is header
                table_html.append('<thead><tr style="background:#4f46e5; color:white;">')
                for cell in cells:
                    table_html.append(f'<th style="padding:10px; text-align:left;">{cell}</th>')
                table_html.append('</tr></thead><tbody>')
            else:
                table_html.append('<tr style="border-bottom:1px solid #e5e7eb;">')
                for cell in cells:
                    table_html.append(f'<td style="padding:8px 10px;">{cell}</td>')
                table_html.append('</tr>')
        else:
            if in_table:
                table_html.append('</tbody></table>')
                result_lines.append('\n'.join(table_html))
                table_html = []
                in_table = False
            result_lines.append(line)
    
    if in_table:
        table_html.append('</tbody></table>')
        result_lines.append('\n'.join(table_html))
    
    html = '\n'.join(result_lines)
    
    # Convert line breaks to paragraphs
    paragraphs = html.split('\n\n')
    formatted_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith('<'):
            formatted_paragraphs.append(f'<p style="line-height:1.7; margin:10px 0;">{p}</p>')
        elif p:
            formatted_paragraphs.append(p)
    
    html = '\n'.join(formatted_paragraphs)
    
    # Convert remaining single line breaks to <br>
    html = re.sub(r'(?<!</p>)\n(?!<)', '<br>\n', html)
    
    return html


# =====================================================
# MAIN ENTRY POINT
# =====================================================

if __name__ == '__main__':
    print("=" * 60)
    print("EDA Agent Pipeline - Flask Backend")
    print("=" * 60)
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    print("=" * 60)
    
    # Disable threading and reloader to prevent matplotlib Tkinter errors
    # threaded=False ensures all requests run on main thread (required for matplotlib)
    # use_reloader=False prevents child process spawning issues
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False, use_reloader=False)

