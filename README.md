# AI Resume–Job Semantic Ranking Engine V2

Final deploy-ready Deep Learning version.

## Added in V2

- SBERT / Sentence-BERT deep learning embeddings
- Optional FAISS vector search
- Optional CrossEncoder reranking
- Weighted skill scoring based on JD role type
- Explainable AI panel: why ranked here + critical missing skills
- Recruiter analytics dashboard
- Score distribution
- Top matched skills
- Search/filter table
- Duplicate detection
- JD/report/export file skipping
- CSV/PDF/sorted ZIP export
- History page

## Run Locally

```bash
pip install -r requirements.txt
python run.py
```

Open:

```text
http://127.0.0.1:5000
```

## Optional Environment

Create `.env` or set variables:

```text
USE_DEEP_LEARNING=auto
USE_CROSS_ENCODER=false
USE_FAISS=auto
```

CrossEncoder is heavy. Use only on stronger laptops/servers.

## Resume Bullet

Built a Deep Learning–based Resume–Job Semantic Ranking Engine using Sentence-BERT, optional FAISS vector search, optional CrossEncoder reranking, weighted skill scoring, and explainable candidate ranking to process 500+ resumes with CSV/PDF/ZIP exports and recruiter analytics.
