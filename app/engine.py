
import os
import re
import shutil
import zipfile
import hashlib
from collections import Counter
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

USE_DEEP_LEARNING = os.getenv("USE_DEEP_LEARNING", "auto").lower()
USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "false").lower() == "true"
USE_FAISS = os.getenv("USE_FAISS", "auto").lower()

_sbert_model = None
_cross_encoder = None
_sbert_failed = False
_cross_failed = False
_faiss_failed = False

SKIP_FILENAME_KEYWORDS = [
    "ranking_report", "resume_ranking_report", "ai_report", "analysis_report",
    "combined_document", "entities_export", "sorted_resumes", "report_total",
    "ranking_results", "job_description", "job_desc", "jd_", "_jd", "requirements"
]

SKILLS_BANK = [
    "python", "sql", "machine learning", "deep learning", "nlp", "computer vision",
    "flask", "fastapi", "django", "rest api", "api", "docker", "git", "github",
    "aws", "azure", "gcp", "linux", "pandas", "numpy", "scikit-learn", "sklearn",
    "tensorflow", "pytorch", "keras", "opencv", "transformers", "bert", "llm", "rag",
    "faiss", "mysql", "postgresql", "mongodb", "power bi", "tableau", "excel",
    "html", "css", "javascript", "react", "node.js", "data analysis", "data visualization",
    "statistics", "eda", "feature engineering", "xgboost", "random forest", "svm",
    "logistic regression", "linear regression", "classification", "regression",
    "communication", "problem solving", "data cleaning", "data preprocessing",
    "matplotlib", "seaborn", "jupyter", "analytics", "data wrangling", "visualization",
    "business intelligence", "etl", "dashboard", "data science", "model training",
    "model evaluation", "neural network", "cnn", "rnn", "lstm", "transformer",
    "fastapi", "streamlit", "mlops", "deployment", "cloud", "rest", "backend"
]

BAD_NAME_WORDS = {
    "resume", "curriculum", "vitae", "summary", "profile", "aspiring", "business",
    "computer", "applications", "application", "ranking", "report", "total",
    "data", "analyst", "developer", "engineer", "intern", "student", "project",
    "job", "title", "details", "skills", "education"
}

ROLE_WEIGHT_MAP = {
    "ai": {
        "Python": 10, "Machine Learning": 10, "Deep Learning": 10, "Pytorch": 9, "Tensorflow": 9,
        "Scikit-Learn": 8, "NLP": 8, "Computer Vision": 8, "Pandas": 7, "Numpy": 7,
        "SQL": 6, "Flask": 5, "FastAPI": 5, "Docker": 5, "Git": 4, "AWS": 4, "Excel": 2
    },
    "data": {
        "Python": 9, "SQL": 10, "Pandas": 10, "Numpy": 8, "Excel": 9, "Power Bi": 9,
        "Tableau": 8, "Data Analysis": 10, "Data Visualization": 9, "Statistics": 8,
        "EDA": 8, "Machine Learning": 5
    },
    "web": {
        "HTML": 8, "CSS": 8, "Javascript": 9, "React": 10, "Node.Js": 9, "Django": 7,
        "Flask": 7, "FastAPI": 7, "Git": 6, "SQL": 6, "Docker": 5
    }
}


def normalize_skill_label(skill):
    value = skill.title()
    replace_map = {
        "Api": "API", "Sql": "SQL", "Nlp": "NLP", "Aws": "AWS", "Gcp": "GCP",
        "LlM": "LLM", "Rag": "RAG", "Eda": "EDA", "Cnn": "CNN", "Rnn": "RNN",
        "Lstm": "LSTM", "Html": "HTML", "Css": "CSS", "Pytorch": "PyTorch",
        "Tensorflow": "TensorFlow", "Numpy": "NumPy", "Fastapi": "FastAPI",
        "Node.Js": "Node.js", "Power Bi": "Power BI"
    }
    for k, v in replace_map.items():
        value = value.replace(k, v)
    return value


def get_role_type(jd_text):
    low = jd_text.lower()
    if any(x in low for x in ["machine learning", "deep learning", "ai/ml", "artificial intelligence", "pytorch", "tensorflow", "nlp", "computer vision"]):
        return "ai"
    if any(x in low for x in ["data analyst", "power bi", "tableau", "excel", "business intelligence", "dashboard"]):
        return "data"
    if any(x in low for x in ["frontend", "backend", "react", "javascript", "node", "web developer"]):
        return "web"
    return "ai"


def get_skill_weight(skill, role_type):
    weights = ROLE_WEIGHT_MAP.get(role_type, {})
    return weights.get(skill, 5)


def get_sbert_model():
    global _sbert_model, _sbert_failed
    if USE_DEEP_LEARNING == "off":
        return None
    if _sbert_failed:
        return None
    if _sbert_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _sbert_failed = True
            return None
    return _sbert_model


def get_cross_encoder():
    global _cross_encoder, _cross_failed
    if not USE_CROSS_ENCODER:
        return None
    if _cross_failed:
        return None
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _cross_failed = True
            return None
    return _cross_encoder


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def should_skip_file(filename):
    low = filename.lower()
    if any(k in low for k in SKIP_FILENAME_KEYWORDS):
        return True
    if low.startswith("000") or "_score_" in low:
        return True
    return False


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def compact_text(text):
    return re.sub(r"\s+", " ", text).strip()


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_fingerprint(text):
    normalized = re.sub(r"\W+", "", text.lower())[:3000]
    return hashlib.md5(normalized.encode("utf-8", errors="ignore")).hexdigest()


def extract_pdf_text(path, max_pages=8):
    text = ""
    try:
        reader = PdfReader(path)
        for page in reader.pages[:max_pages]:
            text += (page.extract_text() or "") + "\n"
    except Exception:
        return ""
    return text


def extract_docx_text(path):
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""


def extract_txt_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def extract_resume_text(path):
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return clean_text(extract_pdf_text(path))
    if ext == "docx":
        return clean_text(extract_docx_text(path))
    if ext == "txt":
        return clean_text(extract_txt_text(path))
    return ""


def is_bad_name(candidate):
    c = candidate.strip()
    if len(c) < 4 or len(c) > 45:
        return True
    low_words = set(c.lower().split())
    if low_words & BAD_NAME_WORDS:
        return True
    if any(ch.isdigit() for ch in c):
        return True
    return False


def extract_candidate_name(filename, text):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for line in lines[:8]:
        cleaned = re.sub(r"[^A-Za-z\s]", " ", line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if 2 <= len(cleaned.split()) <= 4 and cleaned.isupper() and not is_bad_name(cleaned.title()):
            return cleaned.title()

    first = compact_text(text[:400])
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", first)
    for cand in candidates:
        if not is_bad_name(cand):
            return cand

    base = os.path.splitext(filename)[0]
    base = re.sub(r"[_\-]+", " ", base)
    base = re.sub(r"\b(resume|cv|data|analyst|developer|python|compressed|final|new|updated|ml|ai|pm|qa)\b", " ", base, flags=re.I)
    base = re.sub(r"\s+", " ", base).strip()
    if base and not is_bad_name(base.title()):
        return base.title()
    return os.path.splitext(filename)[0].replace("_", " ").title()


def extract_email(text):
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return emails[0] if emails else ""


def extract_phone(text):
    phones = re.findall(r"(?:\+91[-\s]?)?[6-9]\d{9}", text)
    return phones[0] if phones else ""


def extract_skills(text):
    lower = text.lower()
    found = []
    for skill in SKILLS_BANK:
        if skill in lower:
            found.append(normalize_skill_label(skill))
    return sorted(set(found))


def infer_job_title(jd):
    lines = [x.strip() for x in jd.strip().splitlines() if x.strip()]
    patterns = [r"job title\s*[:\-]\s*(.+)", r"role\s*[:\-]\s*(.+)", r"position\s*[:\-]\s*(.+)", r"hiring\s+(.+)"]
    for p in patterns:
        m = re.search(p, jd, re.I)
        if m:
            return m.group(1).strip()[:80]
    for line in lines[:8]:
        low = line.lower()
        if len(line) <= 90 and any(w in low for w in ["engineer", "developer", "analyst", "scientist", "intern", "manager", "specialist"]):
            return line[:80]
    return "Job Description"


def tfidf_scores(jd_text, resume_texts):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000, min_df=1, max_df=0.98, sublinear_tf=True)
    matrix = vectorizer.fit_transform([jd_text] + resume_texts)
    raw = cosine_similarity(matrix[0:1], matrix[1:]).flatten() * 100
    return [float(x) for x in raw]


def faiss_scores(jd_emb, resume_embs):
    global _faiss_failed
    if USE_FAISS == "off" or _faiss_failed:
        return None
    try:
        import faiss
        dim = resume_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(resume_embs.astype("float32"))
        scores, ids = index.search(jd_emb.astype("float32"), len(resume_embs))
        result = [0.0] * len(resume_embs)
        for idx, score in zip(ids[0], scores[0]):
            if idx >= 0:
                result[int(idx)] = float(score * 100)
        return result
    except Exception:
        _faiss_failed = True
        return None


def deep_learning_scores(jd_text, resume_texts):
    model = get_sbert_model()
    if model is None:
        return None, "TF-IDF Fallback Ranking", False

    try:
        texts = [jd_text] + resume_texts
        embeddings = model.encode(texts, batch_size=8, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        jd_emb = embeddings[0:1]
        resume_embs = embeddings[1:]

        faiss_result = faiss_scores(jd_emb, resume_embs)
        if faiss_result is not None:
            return faiss_result, "Deep Learning SBERT + FAISS Vector Search", True

        scores = cosine_similarity(jd_emb, resume_embs).flatten() * 100
        return [float(x) for x in scores], "Deep Learning SBERT Embedding Ranking", False
    except Exception:
        return None, "TF-IDF Fallback Ranking", False


def normalize_scores(raw_scores):
    if not raw_scores:
        return []
    max_score = max(raw_scores)
    if max_score <= 0:
        return [0.0 for _ in raw_scores]
    normalized = []
    for s in raw_scores:
        relative = (s / max_score) * 100
        blended = (s * 0.30) + (relative * 0.70)
        normalized.append(float(min(100, blended)))
    return normalized


def hire_recommendation(score):
    if score >= 85:
        return "Strong Hire"
    if score >= 70:
        return "Hire"
    if score >= 55:
        return "Consider"
    if score >= 40:
        return "Weak Match"
    return "Reject"


def weighted_skill_score(jd_skills, resume_skills, role_type):
    if not jd_skills:
        return 0, [], []
    matched = sorted(set(jd_skills).intersection(set(resume_skills)))
    missing = sorted(set(jd_skills).difference(set(resume_skills)))
    total_weight = sum(get_skill_weight(s, role_type) for s in jd_skills)
    matched_weight = sum(get_skill_weight(s, role_type) for s in matched)
    score = (matched_weight / total_weight * 100) if total_weight else 0
    return round(score, 2), matched, missing


def build_explanation(candidate, jd_skills, role_type):
    matched = candidate["matched_skills"]
    missing = candidate["missing_skills"]
    top_matched = sorted(matched, key=lambda s: get_skill_weight(s, role_type), reverse=True)[:6]
    critical_missing = sorted(missing, key=lambda s: get_skill_weight(s, role_type), reverse=True)[:6]

    reasons = []
    if top_matched:
        reasons.append("Strong match on: " + ", ".join(top_matched))
    if candidate["semantic_score"] >= 75:
        reasons.append("High semantic similarity with the job description")
    elif candidate["semantic_score"] >= 55:
        reasons.append("Moderate semantic similarity with the job description")
    else:
        reasons.append("Low-to-moderate semantic similarity")

    if candidate["skill_score"] >= 70:
        reasons.append("Good weighted skill coverage")
    elif candidate["skill_score"] >= 40:
        reasons.append("Partial skill coverage")
    else:
        reasons.append("Limited skill coverage")

    return {
        "why_ranked": reasons,
        "critical_missing": critical_missing,
        "top_matched": top_matched,
    }


def score_resume(jd_skills, resume_skills, semantic_score, role_type):
    skill_score, matched, missing = weighted_skill_score(jd_skills, resume_skills, role_type)
    final_score = (semantic_score * 0.65) + (skill_score * 0.35)
    final_score = min(100, max(0, final_score))
    return round(final_score, 2), matched, missing, skill_score


def cross_encoder_rerank(jd_text, resumes):
    cross = get_cross_encoder()
    if cross is None or not resumes:
        return resumes, False
    try:
        top_n = min(25, len(resumes))
        pairs = [(jd_text, r["text_for_rerank"][:2500]) for r in resumes[:top_n]]
        ce_scores = cross.predict(pairs)
        ce_scores = [float(x) for x in ce_scores]
        min_s, max_s = min(ce_scores), max(ce_scores)
        for i, ce in enumerate(ce_scores):
            ce_norm = ((ce - min_s) / (max_s - min_s) * 100) if max_s > min_s else 50.0
            resumes[i]["cross_encoder_score"] = round(ce_norm, 2)
            resumes[i]["final_score"] = round((resumes[i]["final_score"] * 0.70) + (ce_norm * 0.30), 2)
            resumes[i]["recommendation"] = hire_recommendation(resumes[i]["final_score"])
        return sorted(resumes, key=lambda x: x["final_score"], reverse=True), True
    except Exception:
        return resumes, False


def analytics_summary(resumes):
    total = len(resumes)
    if not total:
        return {}
    scores = [r["final_score"] for r in resumes]
    rec_counts = Counter(r["recommendation"] for r in resumes)
    skill_counts = Counter()
    for r in resumes:
        skill_counts.update(r["matched_skills"])
    return {
        "total": total,
        "average_score": round(sum(scores) / total, 2),
        "highest_score": round(max(scores), 2),
        "strong_hire": rec_counts.get("Strong Hire", 0),
        "hire": rec_counts.get("Hire", 0),
        "consider": rec_counts.get("Consider", 0),
        "weak_match": rec_counts.get("Weak Match", 0),
        "reject": rec_counts.get("Reject", 0),
        "top_skills": skill_counts.most_common(10),
        "score_buckets": {
            "85-100": sum(1 for s in scores if s >= 85),
            "70-84": sum(1 for s in scores if 70 <= s < 85),
            "55-69": sum(1 for s in scores if 55 <= s < 70),
            "40-54": sum(1 for s in scores if 40 <= s < 55),
            "0-39": sum(1 for s in scores if s < 40),
        }
    }


def rank_resumes(jd_text, uploaded_files, upload_folder, output_folder, export_folder):
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(export_folder, exist_ok=True)

    run_folder = os.path.join(output_folder, "latest_sorted")
    if os.path.exists(run_folder):
        shutil.rmtree(run_folder)
    os.makedirs(run_folder, exist_ok=True)

    for old in os.listdir(upload_folder):
        old_path = os.path.join(upload_folder, old)
        try:
            if os.path.isfile(old_path):
                os.remove(old_path)
        except Exception:
            pass

    jd_text = clean_text(jd_text)
    jd_skills = extract_skills(jd_text)
    role_type = get_role_type(jd_text)

    parsed, errors, seen_hashes, seen_texts = [], [], set(), set()

    for file in uploaded_files:
        if not file or not file.filename:
            continue
        if not allowed_file(file.filename):
            errors.append(f"{file.filename}: unsupported file type")
            continue
        if should_skip_file(file.filename):
            errors.append(f"{file.filename}: skipped because it looks like JD/report/export, not a resume")
            continue

        safe_name = secure_filename(file.filename)
        saved_path = os.path.join(upload_folder, safe_name)

        try:
            file.save(saved_path)
            h = file_hash(saved_path)
            if h in seen_hashes:
                errors.append(f"{safe_name}: duplicate file skipped")
                continue
            seen_hashes.add(h)
            text = extract_resume_text(saved_path)
        except Exception as e:
            errors.append(f"{safe_name}: {str(e)}")
            continue

        if not text or len(text) < 25:
            errors.append(f"{safe_name}: no readable text found")
            continue

        tfp = text_fingerprint(text)
        if tfp in seen_texts:
            errors.append(f"{safe_name}: duplicate resume content skipped")
            continue
        seen_texts.add(tfp)

        parsed.append({
            "filename": safe_name,
            "source_path": saved_path,
            "text": text,
            "candidate_name": extract_candidate_name(safe_name, text),
            "email": extract_email(text),
            "phone": extract_phone(text),
            "resume_skills": extract_skills(text),
        })

    if not parsed:
        return {
            "job_title": infer_job_title(jd_text), "jd_text": jd_text, "jd_skills": jd_skills,
            "resumes": [], "csv_filename": "", "pdf_filename": "", "zip_filename": "",
            "total": 0, "method": "Deep Learning SBERT Ranking", "errors": errors,
            "deep_learning_used": False, "cross_encoder_used": False, "faiss_used": False,
            "analytics": {}, "role_type": role_type
        }

    resume_texts = [x["text"] for x in parsed]
    raw_scores, method, faiss_used = deep_learning_scores(jd_text, resume_texts)
    deep_used = raw_scores is not None

    if raw_scores is None:
        raw_scores = tfidf_scores(jd_text, resume_texts)

    normalized_scores = normalize_scores(raw_scores)

    resumes = []
    for item, semantic_score in zip(parsed, normalized_scores):
        final_score, matched, missing, skill_score = score_resume(jd_skills, item["resume_skills"], semantic_score, role_type)
        candidate = {
            "filename": item["filename"], "candidate_name": item["candidate_name"], "email": item["email"], "phone": item["phone"],
            "semantic_score": round(semantic_score, 2), "skill_score": skill_score, "final_score": final_score,
            "recommendation": hire_recommendation(final_score), "matched_skills": matched, "missing_skills": missing,
            "resume_skills": item["resume_skills"], "text_preview": item["text"][:800],
            "text_for_rerank": item["text"], "source_path": item["source_path"], "cross_encoder_score": None,
        }
        candidate["explanation"] = build_explanation(candidate, jd_skills, role_type)
        resumes.append(candidate)

    resumes = sorted(resumes, key=lambda x: x["final_score"], reverse=True)
    resumes, cross_used = cross_encoder_rerank(jd_text, resumes)

    if cross_used:
        method = "Deep Learning SBERT + CrossEncoder Reranking"
    elif faiss_used:
        method = "Deep Learning SBERT + FAISS Vector Search"

    for r in resumes:
        r["recommendation"] = hire_recommendation(r["final_score"])
        r["explanation"] = build_explanation(r, jd_skills, role_type)

    for idx, item in enumerate(resumes, start=1):
        ext = item["filename"].rsplit(".", 1)[-1].lower()
        rank_name = secure_filename(f"{idx:04d}_Score_{item['final_score']}_{item['candidate_name']}.{ext}")
        dest = os.path.join(run_folder, rank_name)
        try:
            shutil.copy2(item["source_path"], dest)
        except Exception:
            pass
        item["sorted_file"] = rank_name

    csv_filename = "resume_ranking_results.csv"
    pdf_filename = "resume_ranking_report.pdf"
    zip_filename = "sorted_resumes_by_rank.zip"

    export_csv(os.path.join(export_folder, csv_filename), resumes, method)
    export_pdf(os.path.join(export_folder, pdf_filename), resumes, jd_text, jd_skills, method, errors, deep_used, cross_used, faiss_used)
    make_sorted_zip(os.path.join(export_folder, zip_filename), run_folder)

    for r in resumes:
        r.pop("text_for_rerank", None)

    return {
        "job_title": infer_job_title(jd_text), "jd_text": jd_text, "jd_skills": jd_skills,
        "resumes": resumes, "csv_filename": csv_filename, "pdf_filename": pdf_filename,
        "zip_filename": zip_filename, "total": len(resumes), "method": method,
        "errors": errors, "deep_learning_used": deep_used, "cross_encoder_used": cross_used,
        "faiss_used": faiss_used, "analytics": analytics_summary(resumes), "role_type": role_type
    }


def export_csv(path, resumes, method):
    rows = []
    for idx, r in enumerate(resumes, start=1):
        rows.append({
            "rank": idx, "candidate_name": r["candidate_name"], "filename": r["filename"],
            "email": r["email"], "phone": r["phone"], "final_score": r["final_score"],
            "recommendation": r["recommendation"], "sbert_similarity_score": r["semantic_score"],
            "cross_encoder_score": r.get("cross_encoder_score"), "weighted_skill_score": r["skill_score"],
            "matched_skills": ", ".join(r["matched_skills"]), "missing_skills": ", ".join(r["missing_skills"]),
            "why_ranked": " | ".join(r["explanation"]["why_ranked"]),
            "critical_missing": ", ".join(r["explanation"]["critical_missing"]),
            "ranking_method": method, "sorted_file": r.get("sorted_file", "")
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def export_pdf(path, resumes, jd_text, jd_skills, method, errors=None, deep_used=False, cross_used=False, faiss_used=False):
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph("Deep Learning Resume Ranking Report", styles["Title"]), Spacer(1, 10)]
    story.append(Paragraph(f"<b>Total Resumes Ranked:</b> {len(resumes)}", styles["Normal"]))
    story.append(Paragraph(f"<b>Ranking Method:</b> {method}", styles["Normal"]))
    story.append(Paragraph(f"<b>SBERT Used:</b> {deep_used} | <b>FAISS Used:</b> {faiss_used} | <b>CrossEncoder Used:</b> {cross_used}", styles["Normal"]))
    story.append(Paragraph(f"<b>JD Skills Detected:</b> {', '.join(jd_skills) if jd_skills else 'Not found'}", styles["Normal"]))
    story.append(Spacer(1, 12))

    table_data = [["Rank", "Candidate", "Score", "Recommendation", "Matched Skills"]]
    for idx, r in enumerate(resumes[:30], start=1):
        table_data.append([str(idx), r["candidate_name"][:24], str(r["final_score"]), r["recommendation"], ", ".join(r["matched_skills"][:5])])

    table = Table(table_data, colWidths=[30, 125, 50, 85, 220])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)

    if errors:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Skipped Files / Warnings", styles["Heading2"]))
        for err in errors[:20]:
            story.append(Paragraph(str(err), styles["Normal"]))
    doc.build(story)


def make_sorted_zip(zip_path, folder):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for file in files:
                full = os.path.join(root, file)
                arc = os.path.relpath(full, folder)
                z.write(full, arc)
