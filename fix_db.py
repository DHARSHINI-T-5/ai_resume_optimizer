"""
Quick fix: drops and recreates resume_ai.db with the correct schema.
Run once: python fix_db.py
"""
import sqlite3, os

DB = "resume_ai.db"

if os.path.exists(DB):
    os.remove(DB)
    print(f"Deleted old {DB}")

con = sqlite3.connect(DB)
con.executescript("""
CREATE TABLE IF NOT EXISTS users (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    email   TEXT UNIQUE NOT NULL,
    created TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS analyses (
    id               TEXT PRIMARY KEY,
    user_email       TEXT,
    ats_score        REAL,
    match_level      TEXT,
    skill_scores     TEXT,
    matched_keywords TEXT,
    missing_keywords TEXT,
    suggestions      TEXT,
    templates        TEXT,
    resume_text      TEXT,
    jd_text          TEXT,
    created          TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS generated_resumes (
    id          TEXT PRIMARY KEY,
    analysis_id TEXT,
    template    TEXT,
    pdf_bytes   BLOB,
    created     TEXT DEFAULT CURRENT_TIMESTAMP
);
""")
con.commit()
con.close()
print(f"Created fresh {DB} with correct schema.")
print("Now run:  python app.py")
