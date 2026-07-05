import sqlite3
from datetime import datetime

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS ranking_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT,
            jd_preview TEXT,
            total_resumes INTEGER,
            top_candidate TEXT,
            top_score REAL,
            csv_file TEXT,
            pdf_file TEXT,
            zip_file TEXT,
            created_at TEXT
        )
        '''
    )
    conn.commit()
    conn.close()


def save_run(db_path, job_title, jd_preview, total_resumes, top_candidate, top_score, csv_file, pdf_file, zip_file):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        '''
        INSERT INTO ranking_runs
        (job_title, jd_preview, total_resumes, top_candidate, top_score, csv_file, pdf_file, zip_file, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            job_title,
            jd_preview[:500],
            total_resumes,
            top_candidate,
            float(top_score),
            csv_file,
            pdf_file,
            zip_file,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )
    conn.commit()
    conn.close()


def get_runs(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM ranking_runs ORDER BY id DESC")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def delete_run(db_path, run_id):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM ranking_runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()


def clear_history(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM ranking_runs")
    conn.commit()
    conn.close()
