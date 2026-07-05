
import os
import traceback
from flask import Blueprint, current_app, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
from .database import save_run, get_runs, delete_run, clear_history
from .engine import rank_resumes, allowed_file

main = Blueprint("main", __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            jd_text = request.form.get("jd_text", "").strip()
            files = [f for f in request.files.getlist("resumes") if f and f.filename]
            if not jd_text:
                return render_template("index.html", error="Please paste or write a Job Description.")
            if not files:
                return render_template("index.html", error="Please upload at least one resume.")
            invalid = [f.filename for f in files if not allowed_file(f.filename)]
            if invalid:
                return render_template("index.html", error="Only PDF, DOCX and TXT resumes are allowed.")

            result = rank_resumes(jd_text, files, current_app.config["UPLOAD_FOLDER"], current_app.config["OUTPUT_FOLDER"], current_app.config["EXPORT_FOLDER"])

            if not result["resumes"]:
                msg = "No readable resumes found. Use text-based PDF/DOCX/TXT files."
                if result.get("errors"):
                    msg += " Warnings: " + " | ".join(result["errors"][:5])
                return render_template("index.html", error=msg)

            top = result["resumes"][0]
            save_run(current_app.config["DB_PATH"], result["job_title"], jd_text, result["total"], top["candidate_name"], top["final_score"], result["csv_filename"], result["pdf_filename"], result["zip_filename"])
            return render_template("result.html", result=result)

        except Exception as e:
            traceback.print_exc()
            return render_template("index.html", error=f"Ranking failed safely. Error: {str(e)}")
    return render_template("index.html")


@main.route("/history")
def history():
    return render_template("history.html", rows=get_runs(current_app.config["DB_PATH"]))


@main.route("/history/delete/<int:run_id>", methods=["POST"])
def delete_history(run_id):
    delete_run(current_app.config["DB_PATH"], run_id)
    return redirect(url_for("main.history"))


@main.route("/history/clear", methods=["POST"])
def clear_all_history():
    clear_history(current_app.config["DB_PATH"])
    return redirect(url_for("main.history"))


@main.route("/download/<file_type>/<filename>")
def download_file(file_type, filename):
    safe_name = secure_filename(filename)
    if file_type in {"csv", "pdf", "zip"}:
        path = os.path.join(current_app.config["EXPORT_FOLDER"], safe_name)
    else:
        return "Invalid file type", 400
    if not os.path.exists(path):
        return "File not found. Please run ranking again.", 404
    return send_file(path, as_attachment=True, download_name=safe_name)


@main.route("/download/resume/<filename>")
def download_resume(filename):
    safe_name = secure_filename(filename)
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], safe_name)
    if not os.path.exists(path):
        return "Resume file not found.", 404
    return send_file(path, as_attachment=True, download_name=safe_name)


@main.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "message": "AI Resume Ranking Engine is running"})


@main.route("/api/history")
def api_history():
    return jsonify(get_runs(current_app.config["DB_PATH"]))
