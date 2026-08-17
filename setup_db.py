"""
ResumeAI Pro — Database Setup & Migration Script
================================================
Run this ONCE to create or reset the database.
  python setup_db.py           # create fresh DB
  python setup_db.py --reset   # drop and recreate all tables
  python setup_db.py --info    # show table info
"""

import sqlite3, sys, os

DB = "resumeai.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT UNIQUE NOT NULL,
    name        TEXT DEFAULT '',
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyses (
    id               TEXT PRIMARY KEY,
    user_email       TEXT DEFAULT 'anon',
    filename         TEXT DEFAULT '',
    ats_score        REAL DEFAULT 0,
    match_level      TEXT DEFAULT '',
    skill_scores     TEXT DEFAULT '{}',
    matched_keywords TEXT DEFAULT '[]',
    missing_keywords TEXT DEFAULT '[]',
    suggestions      TEXT DEFAULT '[]',
    templates        TEXT DEFAULT '[]',
    resume_text      TEXT DEFAULT '',
    jd_text          TEXT DEFAULT '',
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS generated_resumes (
    id           TEXT PRIMARY KEY,
    analysis_id  TEXT DEFAULT '',
    user_email   TEXT DEFAULT 'anon',
    template_id  TEXT DEFAULT '',
    content_json TEXT DEFAULT '{}',
    pdf_bytes    BLOB,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email TEXT DEFAULT 'anon',
    role       TEXT NOT NULL,
    message    TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

def create_db():
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    print(f"✓ Database '{DB}' ready with all tables.")

def reset_db():
    if os.path.exists(DB):
        os.remove(DB)
        print(f"✓ Deleted old '{DB}'")
    create_db()
    print("✓ Fresh database created.")

def info_db():
    if not os.path.exists(DB):
        print(f"✗ '{DB}' does not exist. Run: python setup_db.py")
        return
    con = sqlite3.connect(DB)
    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f"\n  Database: {DB}")
    print(f"  Tables: {len(tables)}")
    for (t,) in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cols  = con.execute(f"PRAGMA table_info({t})").fetchall()
        print(f"\n  [{t}] — {count} rows — {len(cols)} columns")
        for col in cols:
            print(f"    {col[1]:25s} {col[2]}")
    con.close()

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv)>1 else ""
    if arg == "--reset":
        reset_db()
    elif arg == "--info":
        info_db()
    else:
        create_db()
    print("\nNext step: python app.py")
