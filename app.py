"""
ResumeAI Pro — Complete Backend v3
===================================
APIs:
  POST /analyze           → ATS score, skills, keywords, suggestions
  POST /generate_resume   → AI-enhanced resume in selected template
  GET  /templates         → All 15+ templates with metadata
  POST /chat              → AI chatbot assistant
  POST /voice             → Voice query → AI text response
  GET  /download_resume   → PDF download
  POST /signup            → Biometric enrolment
  POST /login             → Biometric verification
  GET  /history           → User analysis history
  GET  /ping              → Health check

Dependencies: flask flask-cors PyPDF2/pypdf scikit-learn numpy reportlab
Optional:     openai python-docx
"""

import os, re, json, uuid, sqlite3, io, math, time
from datetime import datetime
from flask import Flask, request, jsonify, send_file, g
from flask_cors import CORS
import numpy as np

# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader as _PR; _PDF = "pypdf"
except ImportError:
    try:
        from PyPDF2 import PdfReader as _PR; _PDF = "PyPDF2"
    except ImportError:
        _PR = None; _PDF = None

try:
    import docx as _docx; _DOCX = True
except ImportError:
    _DOCX = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cs
    _SK = True
except ImportError:
    _SK = False

try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib import colors as rlc
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    HRFlowable, Table, TableStyle, KeepTogether)
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    _RL = True
except ImportError:
    _RL = False

try:
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    _OAI = bool(openai.api_key)
except ImportError:
    _OAI = False

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

DB_PATH   = "resumeai.db"
FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def get_db():
    db = getattr(g, "_db", None)
    if db is None:
        db = g._db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(_):
    db = getattr(g, "_db", None)
    if db: db.close()

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
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
    """)
    con.commit()
    con.close()

init_db()

# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
TECH_SKILLS = [
    "python","java","javascript","typescript","c++","c#","go","rust","kotlin","swift",
    "php","ruby","r","scala","dart","html","css","sql","nosql","bash","react","angular",
    "vue","next.js","nuxt","node.js","express","django","flask","fastapi","spring",
    "tensorflow","pytorch","keras","scikit-learn","pandas","numpy","opencv","spark",
    "hadoop","kafka","airflow","aws","gcp","azure","docker","kubernetes","terraform",
    "jenkins","git","linux","rest","graphql","grpc","microservices","ci/cd","devops",
    "mlops","machine learning","deep learning","nlp","computer vision","data science",
    "big data","postgresql","mysql","mongodb","redis","elasticsearch","cassandra",
    "tableau","power bi","excel","jira","selenium","cypress","jest","pytest",
    "agile","scrum","firebase","flutter","unity","blockchain","solidity",
]
SOFT_SKILLS = [
    "communication","leadership","teamwork","problem solving","critical thinking",
    "time management","adaptability","creativity","collaboration","analytical",
    "attention to detail","decision making","interpersonal","presentation",
    "negotiation","project management","mentoring","strategic thinking",
    "innovation","accountability","self-motivated","organized","multitasking",
]
EXP_VERBS = [
    "built","designed","led","managed","implemented","delivered","maintained",
    "optimized","architected","launched","reduced","improved","increased",
    "automated","deployed","scaled","integrated","developed","created","established",
    "achieved","mentored","streamlined","migrated","refactored","engineered",
    "spearheaded","orchestrated","accelerated","transformed","drove",
]
EDU_TERMS = [
    "bachelor","master","phd","degree","university","college","b.tech","m.tech",
    "b.e","b.sc","m.sc","mba","bca","mca","engineering","science","gpa","cgpa",
]
SECTION_HEADERS = [
    "experience","education","skills","projects","certifications","summary",
    "objective","achievements","publications","awards","work history",
    "professional experience","technical skills","profile","about",
]
STOP_WORDS = {
    "the","and","for","are","was","this","that","with","have","from","will","can",
    "all","but","not","you","has","had","its","our","your","their","they","also",
    "each","more","been","into","than","then","when","which","would","about",
    "after","before","should","could","these","those","over","what","where","who",
    "how","why","use","using","used","work","working","per","via","etc","may",
}

# ══════════════════════════════════════════════════════════════════════════════
# 15+ TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════
TEMPLATES = [
    # TECH
    {"id":"tech_modern","name":"Tech Modern","category":"tech","sub":"Software Engineer",
     "icon":"💻","color":"#00ff88","accent":"#020617",
     "description":"Clean two-column with skills sidebar. Perfect for SWE, backend, fullstack.",
     "ats_score":95,"use_for":["software","developer","engineer","backend","frontend","fullstack"],
     "highlights":["Skills sidebar","GitHub/LinkedIn prominent","Project cards"]},
    {"id":"tech_minimal","name":"Tech Minimal","category":"tech","sub":"DevOps / Cloud",
     "icon":"⚙️","color":"#60a5fa","accent":"#0f172a",
     "description":"Ultra-clean single column. Great for DevOps, SRE, cloud architects.",
     "ats_score":98,"use_for":["devops","cloud","sre","infrastructure","platform"],
     "highlights":["Maximum ATS compatibility","Clean sections","Tech stack list"]},
    {"id":"data_science","name":"Data Science Pro","category":"tech","sub":"Data / ML / AI",
     "icon":"📊","color":"#a78bfa","accent":"#0f172a",
     "description":"Metrics-first layout. Ideal for data scientists, ML engineers, analysts.",
     "ats_score":96,"use_for":["data","machine learning","ai","analyst","scientist","nlp"],
     "highlights":["Metrics dashboard section","Skills proficiency bars","Publication section"]},
    {"id":"fullstack_pro","name":"Fullstack Pro","category":"tech","sub":"Full Stack",
     "icon":"🌐","color":"#34d399","accent":"#064e3b",
     "description":"Stack-organized layout showcasing frontend + backend + database skills.",
     "ats_score":94,"use_for":["fullstack","web developer","react","angular","vue","node"],
     "highlights":["Tech stack grid","Live project links","Dual-column skills"]},
    # BUSINESS
    {"id":"biz_executive","name":"Executive","category":"business","sub":"Management / C-Suite",
     "icon":"🏢","color":"#1e3a5f","accent":"#f8fafc",
     "description":"Authoritative single-column. Ideal for directors, VPs, C-suite executives.",
     "ats_score":97,"use_for":["manager","director","vp","head","executive","ceo","cto","lead"],
     "highlights":["Executive summary","P&L / metrics focus","Board experience section"]},
    {"id":"biz_marketing","name":"Marketing Pro","category":"business","sub":"Marketing / Growth",
     "icon":"📈","color":"#f97316","accent":"#1c0a00",
     "description":"Campaign-focused with ROI metrics. For marketers, growth hackers, CMOs.",
     "ats_score":92,"use_for":["marketing","growth","seo","sem","campaign","brand","content"],
     "highlights":["Campaign metrics","Channel expertise","Brand portfolio"]},
    {"id":"biz_sales","name":"Sales Champion","category":"business","sub":"Sales / BD",
     "icon":"💰","color":"#fbbf24","accent":"#111827",
     "description":"Revenue-first layout. Perfect for AEs, BDRs, sales managers.",
     "ats_score":91,"use_for":["sales","business development","account","revenue","quota"],
     "highlights":["Quota achievement","Revenue metrics","Territory management"]},
    {"id":"biz_finance","name":"Finance Elite","category":"business","sub":"Finance / Banking",
     "icon":"📑","color":"#0ea5e9","accent":"#0c1445",
     "description":"Conservative ATS-safe layout for finance, banking, consulting roles.",
     "ats_score":99,"use_for":["finance","banking","accounting","cfa","audit","investment"],
     "highlights":["Maximum ATS score","Conservative formatting","CPA/CFA section"]},
    # CREATIVE
    {"id":"creative_designer","name":"Creative Designer","category":"creative","sub":"UX/UI / Design",
     "icon":"🎨","color":"#ec4899","accent":"#1a0012",
     "description":"Portfolio-forward layout for designers, UX researchers, creative directors.",
     "ats_score":85,"use_for":["designer","ux","ui","figma","product design","creative"],
     "highlights":["Portfolio links","Design tools section","Awards & recognition"]},
    {"id":"creative_media","name":"Media Pro","category":"creative","sub":"Media / Content",
     "icon":"🎬","color":"#f43f5e","accent":"#0f0014",
     "description":"Visual-forward for content creators, video producers, social media managers.",
     "ats_score":83,"use_for":["content","media","video","social","journalism","writer"],
     "highlights":["Platform metrics","Content portfolio","Viral campaigns"]},
    {"id":"creative_modern","name":"Modern Creative","category":"creative","sub":"Brand / Advertising",
     "icon":"✨","color":"#8b5cf6","accent":"#0d0020",
     "description":"Bold typographic layout for advertising, brand strategy, creative agencies.",
     "ats_score":84,"use_for":["brand","advertising","copywriter","creative director","agency"],
     "highlights":["Brand case studies","Client roster","Campaign results"]},
    # FRESHER
    {"id":"fresher_classic","name":"Fresher Classic","category":"fresher","sub":"Entry Level / Graduate",
     "icon":"🎓","color":"#6366f1","accent":"#f0f9ff",
     "description":"Education-first clean layout. Perfect for fresh graduates and interns.",
     "ats_score":93,"use_for":["fresher","graduate","intern","entry level","student","trainee"],
     "highlights":["Education prominent","Projects section","Internship highlights"]},
    {"id":"fresher_tech","name":"Tech Graduate","category":"fresher","sub":"CS / Engineering Graduate",
     "icon":"👨‍💻","color":"#10b981","accent":"#022c22",
     "description":"Tech-focused graduate template with GitHub, projects and skills front.",
     "ats_score":94,"use_for":["cs student","engineering student","btech","mtech","computer science"],
     "highlights":["GitHub projects","Competitive programming","Open source"]},
    {"id":"fresher_mba","name":"MBA Graduate","category":"fresher","sub":"MBA / Business Graduate",
     "icon":"📚","color":"#0891b2","accent":"#042f2e",
     "description":"Business-focused graduate template for MBA, BCom, BBA graduates.",
     "ats_score":92,"use_for":["mba","bba","bcom","business graduate","management","pgdm"],
     "highlights":["Internship focus","Leadership roles","Case competitions"]},
    {"id":"universal_pro","name":"Universal Pro","category":"universal","sub":"Any Role",
     "icon":"⭐","color":"#f59e0b","accent":"#1c1008",
     "description":"Versatile professional template that works across all industries.",
     "ats_score":96,"use_for":["any","general","professional","career change"],
     "highlights":["Works for all roles","Maximum compatibility","Clean structure"]},
    # ── NEW BATCH: IT / Tech ──────────────────────────────────────────────────
    {"id":"cybersec_pro","name":"Cybersecurity Pro","category":"tech","sub":"Security / SOC / Pentesting",
     "icon":"🔐","color":"#ef4444","accent":"#0f0505",
     "description":"Dark accent two-column template for security engineers, SOC analysts, and pentesters.",
     "ats_score":95,"use_for":["security","cybersecurity","soc","pentesting","owasp","siem","devsecops"],
     "highlights":["CVE/Vulnerability section","Certifications front","Tool proficiency grid"]},
    {"id":"mobile_dev","name":"Mobile Developer","category":"tech","sub":"iOS / Android / Flutter",
     "icon":"📱","color":"#6366f1","accent":"#0d0b3f",
     "description":"App-store-ready template for iOS, Android and cross-platform developers.",
     "ats_score":93,"use_for":["mobile","ios","android","flutter","react native","swift","kotlin"],
     "highlights":["App store links","Platform badges","Downloads/ratings metrics"]},
    {"id":"cloud_architect","name":"Cloud Architect","category":"tech","sub":"Cloud / Infrastructure",
     "icon":"☁️","color":"#38bdf8","accent":"#0c2340",
     "description":"Architecture-focused template for cloud architects and platform engineers.",
     "ats_score":97,"use_for":["cloud","architect","aws","gcp","azure","infrastructure","platform","sre"],
     "highlights":["Cert badges prominent","Architecture projects","Cost-optimisation metrics"]},
    {"id":"ai_ml_engineer","name":"AI/ML Engineer","category":"tech","sub":"Artificial Intelligence / LLM",
     "icon":"🤖","color":"#818cf8","accent":"#0f0f2e",
     "description":"Research-meets-engineering template for AI/ML engineers and LLM specialists.",
     "ats_score":96,"use_for":["ai","llm","nlp","generative","langchain","huggingface","ml engineer"],
     "highlights":["Model performance metrics","Research papers","Open-source contributions"]},
    # ── NEW BATCH: Non-IT / Business ─────────────────────────────────────────
    {"id":"hr_specialist","name":"HR Professional","category":"business","sub":"Human Resources / Talent",
     "icon":"🧑‍💼","color":"#0891b2","accent":"#04202e",
     "description":"Warm, structured layout for HR managers, talent acquisition, and L&D professionals.",
     "ats_score":94,"use_for":["hr","human resources","recruiter","talent","l&d","hrbp","payroll"],
     "highlights":["Hiring metrics","Employee count managed","HRIS tools section"]},
    {"id":"project_manager","name":"Project Manager","category":"business","sub":"PMO / Scrum Master",
     "icon":"📋","color":"#10b981","accent":"#012218",
     "description":"Delivery-focused template for project managers, scrum masters, and PMO leads.",
     "ats_score":95,"use_for":["project manager","pmo","scrum","agile","delivery","prince2","pmp"],
     "highlights":["Budget managed","Team sizes","On-time delivery rates"]},
    {"id":"operations_mgr","name":"Operations Manager","category":"business","sub":"Operations / Supply Chain",
     "icon":"⚙️","color":"#64748b","accent":"#0f1520",
     "description":"Process-oriented layout for operations, supply chain and logistics professionals.",
     "ats_score":93,"use_for":["operations","supply chain","logistics","procurement","warehouse","lean","six sigma"],
     "highlights":["Cost savings","Process improvements","KPI achievements"]},
    {"id":"legal_professional","name":"Legal Professional","category":"business","sub":"Law / Compliance",
     "icon":"⚖️","color":"#1e293b","accent":"#f8fafc",
     "description":"Conservative, formal layout for lawyers, compliance officers and legal analysts.",
     "ats_score":98,"use_for":["lawyer","legal","attorney","compliance","contract","litigation","paralegal"],
     "highlights":["Bar admissions","Cases won","Practice areas"]},
    # ── NEW BATCH: Creative ───────────────────────────────────────────────────
    {"id":"video_editor","name":"Video Editor / Motion","category":"creative","sub":"Video / Motion Graphics",
     "icon":"🎬","color":"#dc2626","accent":"#1a0000",
     "description":"Cinematic-inspired layout for video editors, motion designers, and cinematographers.",
     "ats_score":82,"use_for":["video editor","motion graphics","premiere","after effects","davinci","vfx","cinematographer"],
     "highlights":["Reel link prominent","Software stack","Project credits"]},
    {"id":"graphic_designer","name":"Graphic Designer","category":"creative","sub":"Graphic / Visual Design",
     "icon":"🖌️","color":"#d946ef","accent":"#1a0020",
     "description":"Bold, typographic layout for graphic designers, illustrators and art directors.",
     "ats_score":83,"use_for":["graphic designer","illustrator","branding","typography","print","photoshop","indesign"],
     "highlights":["Portfolio link featured","Brand clients","Software proficiency"]},
    {"id":"architect_pro","name":"Architect / Interior","category":"creative","sub":"Architecture / Interior Design",
     "icon":"🏛️","color":"#92400e","accent":"#fef3c7",
     "description":"Structural, grid-based layout for architects and interior designers.",
     "ats_score":85,"use_for":["architect","interior design","autocad","revit","sketchup","bim","3d rendering"],
     "highlights":["Project portfolio links","Software tools","Awards & recognition"]},
    # ── NEW BATCH: Academic ───────────────────────────────────────────────────
    {"id":"professor_academic","name":"Academic / Professor","category":"academic","sub":"Professor / Researcher",
     "icon":"🎓","color":"#1d4ed8","accent":"#eff6ff",
     "description":"Publication-first academic CV template for professors, researchers and PhD candidates.",
     "ats_score":90,"use_for":["professor","researcher","phd","postdoc","lecturer","academic","faculty"],
     "highlights":["Publications list","Grants & funding","Teaching history"]},
    {"id":"research_scientist","name":"Research Scientist","category":"academic","sub":"R&D / Laboratory",
     "icon":"🔬","color":"#0d9488","accent":"#042f2e",
     "description":"Lab-focused template for research scientists, chemists, biologists, and clinical researchers.",
     "ats_score":91,"use_for":["research scientist","lab","chemistry","biology","clinical","pharma","r&d","biotech"],
     "highlights":["Lab techniques","Publications","Patents"]},
    # ── NEW BATCH: Executive ──────────────────────────────────────────────────
    {"id":"cto_ciso","name":"CTO / CISO","category":"executive","sub":"Technology Executive",
     "icon":"🏆","color":"#1e3a5f","accent":"#f0f9ff",
     "description":"Board-room ready template for CTOs, CISOs, and VP Engineering roles.",
     "ats_score":97,"use_for":["cto","ciso","vp engineering","technology executive","chief","svp"],
     "highlights":["P&L/budget ownership","Team scaling","Strategic vision"]},
    {"id":"startup_founder","name":"Startup / Founder","category":"executive","sub":"Founder / Entrepreneur",
     "icon":"🚀","color":"#7c3aed","accent":"#0d0025",
     "description":"Narrative-driven layout for founders, entrepreneurs and startup operators.",
     "ats_score":88,"use_for":["founder","startup","entrepreneur","ceo","co-founder","venture","angel"],
     "highlights":["Funding raised","Revenue milestones","Team built"]},
    # ── NEW BATCH: ATS-Friendly ───────────────────────────────────────────────
    {"id":"ats_pure","name":"ATS Pure Text","category":"ats","sub":"Maximum ATS Compatibility",
     "icon":"🤖","color":"#374151","accent":"#f9fafb",
     "description":"Zero-design, pure text layout optimized for ATS parsers. Highest pass rate.",
     "ats_score":100,"use_for":["ats","large company","fortune500","government","banking","conservative"],
     "highlights":["100% ATS parse rate","Zero graphics","Plain section headers"]},
    {"id":"ats_modern","name":"ATS Modern Clean","category":"ats","sub":"ATS-Safe + Readable",
     "icon":"📄","color":"#2563eb","accent":"#eff6ff",
     "description":"Subtle design with guaranteed ATS compatibility. Best of both worlds.",
     "ats_score":99,"use_for":["ats","corporate","recruiter","applicant tracking","linkedin"],
     "highlights":["ATS-safe fonts","Standard headers","Thin accent line only"]},
    # ── NEW BATCH: Photo ──────────────────────────────────────────────────────
    {"id":"with_photo","name":"Professional with Photo","category":"photo","sub":"Photo + Credentials",
     "icon":"🖼️","color":"#0f766e","accent":"#f0fdfa",
     "description":"Modern layout with circular photo placeholder for markets where photo resumes are expected.",
     "ats_score":80,"use_for":["photo","europe","middle east","asia","hospitality","consulting","photo resume"],
     "highlights":["Photo circle","Contact sidebar","Award section"]},
    {"id":"linkedin_style","name":"LinkedIn Style","category":"photo","sub":"Social / Networking Resume",
     "icon":"🔗","color":"#0077b5","accent":"#f8fbff",
     "description":"LinkedIn-inspired layout with banner, photo, and endorsements section.",
     "ats_score":85,"use_for":["linkedin","networking","social","endorsements","recommendations","modern"],
     "highlights":["LinkedIn-style header","Skills endorsement section","Recommendation quotes"]},
]

# ══════════════════════════════════════════════════════════════════════════════
# AI CHATBOT KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
CHAT_KB = {
    "ats":       "ATS (Applicant Tracking System) is software employers use to filter resumes. Your ATS score shows how well your resume matches the job. Scores above 70% are good; above 85% is excellent.",
    "template":  "We have 15+ templates across Tech, Business, Creative and Fresher categories. The system auto-recommends the best template based on your job description and experience level.",
    "upload":    "Go to the Analysis section, upload your resume (PDF, DOCX, or TXT), paste the job description, and click Analyze. You'll get a full ATS score + keyword analysis in seconds.",
    "keywords":  "Keywords are specific skills and terms from the job description. Adding missing keywords naturally to your resume significantly improves your ATS score.",
    "score":     "Your ATS score is calculated from 5 factors: semantic similarity (30%), keyword match (25%), technical skills (20%), section structure (15%), and readability (10%).",
    "improve":   "To improve your resume: 1) Add missing keywords naturally 2) Use action verbs 3) Quantify achievements with numbers 4) Ensure proper section headers 5) Keep formatting clean.",
    "download":  "After generating your resume, click the Download PDF button. Your resume is formatted in the selected template and ready to submit.",
    "voice":     "Click the microphone button to ask questions by voice. I'll respond in text and speech. You can ask anything about resume writing, templates, or improving your score.",
    "generate":  "Select a template from the recommendations, click 'Generate Resume', and the AI will reformat your content into the chosen template with improved bullet points and structure.",
    "hello":     "Hello! I'm your ResumeAI assistant. I can help you analyze your resume, choose templates, improve your ATS score, and guide you through the entire process. What would you like to do?",
    "hi":        "Hi there! Ready to build an ATS-optimized resume? Upload your resume and job description to get started, or ask me anything!",
    "help":      "Here's what I can help with:\n• Analyze your resume ATS score\n• Suggest missing keywords\n• Recommend the best template\n• Guide resume improvement\n• Explain ATS systems\n• Answer resume questions\n\nWhat do you need?",
    "default":   "Great question! For the best results, make sure to: upload a text-based PDF, paste the full job description, and follow the keyword suggestions. Would you like me to explain any specific feature?",
}

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def extract_text(file_obj, filename: str = "") -> str:
    fname = (filename or "").lower()
    if fname.endswith(".docx") and _DOCX:
        doc = _docx.Document(file_obj)
        return "\n".join(p.text for p in doc.paragraphs)
    if _PR:
        reader = _PR(file_obj)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    raw = file_obj.read()
    return re.sub(r'\s+', ' ', re.sub(rb'[^\x20-\x7e\n]', b' ', raw).decode('ascii','ignore'))

def bow_cosine(a: str, b: str) -> float:
    def f(s):
        d = {}
        for w in re.findall(r'\b\w+\b', s.lower()):
            d[w] = d.get(w,0)+1
        return d
    fa, fb = f(a), f(b)
    keys = set(fa)|set(fb)
    va = [fa.get(k,0) for k in keys]
    vb = [fb.get(k,0) for k in keys]
    dot  = sum(x*y for x,y in zip(va,vb))
    norm = math.sqrt(sum(x**2 for x in va))*math.sqrt(sum(x**2 for x in vb))
    return dot/norm if norm else 0.0

def semantic(a: str, b: str) -> float:
    if _SK:
        try:
            v = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=1)
            m = v.fit_transform([a,b])
            return float(_cs(m[0:1],m[1:2])[0][0])*100
        except Exception:
            pass
    return bow_cosine(a,b)*100

def parse_sections(text: str) -> dict:
    r = text.lower()
    s = {"name":"","email":"","phone":"","linkedin":"","github":"","website":"",
         "summary":"","skills":[],"experience":[],"education":[],"projects":[],"certifications":[]}
    m = re.search(r'[\w.+-]+@[\w.-]+\.\w{2,}', text)
    if m: s["email"] = m.group()
    m = re.search(r'(\+?[\d\-\s\(\)]{9,15})', text)
    if m: s["phone"] = m.group().strip()
    m = re.search(r'linkedin\.com/in/[\w\-]+', text, re.I)
    if m: s["linkedin"] = m.group()
    m = re.search(r'github\.com/[\w\-]+', text, re.I)
    if m: s["github"] = m.group()
    for line in text.split('\n'):
        line = line.strip()
        if 3 < len(line) < 55 and re.match(r'^[A-Za-z\s\.\-]+$',line) and not any(t in line.lower() for t in SECTION_HEADERS):
            s["name"] = line; break
    bullets = re.findall(r'[•\-\*►▶●]\s*(.{15,})', text)
    s["experience"] = [b.strip() for b in bullets[:16]]
    sk = re.search(r'(?:skills?|technologies?)[:\s]*([^\n]{20,300})', text, re.I)
    if sk:
        s["skills"] = [x.strip() for x in re.split(r'[,|•;/]', sk.group(1)) if x.strip() and len(x.strip())>1][:24]
    edu_lines = [l.strip() for l in text.split('\n') if any(t in l.lower() for t in EDU_TERMS[:8]) and l.strip() and len(l.strip())>5]
    s["education"] = edu_lines[:5]
    proj_m = re.search(r'(?:projects?)[:\n](.*?)(?:\n\n|\Z)', text, re.I|re.S)
    if proj_m:
        proj_lines = [l.strip() for l in proj_m.group(1).split('\n') if l.strip() and len(l.strip())>10]
        s["projects"] = proj_lines[:6]
    cert_m = re.search(r'(?:certifications?|certificates?)[:\n](.*?)(?:\n\n|\Z)', text, re.I|re.S)
    if cert_m:
        cert_lines = [l.strip() for l in cert_m.group(1).split('\n') if l.strip() and len(l.strip())>5]
        s["certifications"] = cert_lines[:6]
    return s

# ══════════════════════════════════════════════════════════════════════════════
# ATS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_ats(resume_text: str, jd_text: str) -> dict:
    r, j = resume_text.lower(), jd_text.lower()
    sem = semantic(r, j)
    jd_tok  = {w for w in re.findall(r'\b[a-z][a-z0-9+#\-./]{1,}\b', j) if w not in STOP_WORDS and len(w)>2}
    res_tok = {w for w in re.findall(r'\b[a-z][a-z0-9+#\-./]{1,}\b', r) if w not in STOP_WORDS and len(w)>2}
    matched_kw = sorted(jd_tok & res_tok)
    missing_kw = sorted(jd_tok - res_tok)
    kw_pct = (len(matched_kw)/len(jd_tok)*100) if jd_tok else 0

    jd_tech   = [s for s in TECH_SKILLS if s in j]
    res_tech  = [s for s in TECH_SKILLS if s in r]
    tech_hit  = [s for s in jd_tech if s in res_tech]
    tech_miss = [s for s in jd_tech if s not in res_tech]
    tech_pct  = (len(tech_hit)/len(jd_tech)*100) if jd_tech else (55 if res_tech else 20)

    jd_soft  = [s for s in SOFT_SKILLS if s in j]
    res_soft = [s for s in SOFT_SKILLS if s in r]
    soft_hit = [s for s in jd_soft if s in res_soft]
    soft_pct = (len(soft_hit)/len(jd_soft)*100) if jd_soft else (45 if res_soft else 20)

    verb_hits = sum(1 for v in EXP_VERBS if v in r)
    years     = len(re.findall(r'\b(20\d{2}|19\d{2})\b', resume_text))
    exp_pct   = min(100, verb_hits*5 + years*8 + sem*0.2)

    edu_jd  = [t for t in EDU_TERMS if t in j]
    edu_res = [t for t in EDU_TERMS if t in r]
    edu_pct = (len([t for t in edu_jd if t in edu_res])/len(edu_jd)*100) if edu_jd else (65 if edu_res else 40)

    found_sec  = [s for s in SECTION_HEADERS if s in r]
    struct_pct = min(100, len(found_sec)/6*100)

    bullets  = len(re.findall(r'[•\-\*●►]', resume_text))
    numbers  = len(re.findall(r'\d+', resume_text))
    read_pct = min(100, bullets*2.5 + numbers*0.8 + verb_hits*3)

    ats = round(min(sem*0.30 + kw_pct*0.25 + tech_pct*0.20 + struct_pct*0.15 + read_pct*0.10, 99.0), 1)
    level = "Elite" if ats>85 else "Strong" if ats>70 else "Average" if ats>45 else "Weak"

    skill_scores = {
        "technical_skills":     round(min(tech_pct,100),1),
        "soft_skills":          round(min(soft_pct,100),1),
        "experience_relevance": round(min(exp_pct,100),1),
        "education_match":      round(min(edu_pct,100),1),
        "keyword_match":        round(min(kw_pct,100),1),
        "readability":          round(min(read_pct,100),1),
    }

    suggestions = _build_suggestions(ats, skill_scores, tech_miss, missing_kw, found_sec, bullets, numbers, verb_hits)
    tpls = _match_templates(jd_text, ats, struct_pct, found_sec)
    msg = {"Elite":"Excellent! Your resume is highly optimised.",
           "Strong":"Good match — a few tweaks will push it to Elite.",
           "Average":"Moderate match. Follow suggestions to improve significantly.",
           "Weak":"Needs work. Use the suggestions and templates below."}.get(level,"")
    learn_suggestions = _skill_learning_suggestions(jd_text, tech_miss)

    return {
        "ats_score":ats,"match_level":level,"message":msg,
        "skill_scores":skill_scores,
        "matched_keywords":matched_kw[:25],"missing_keywords":missing_kw[:20],
        "tech_matched":tech_hit[:15],"tech_missing":tech_miss[:12],
        "tech_all_resume":res_tech[:20],      # ← NEW: ALL skills found in resume
        "learn_suggestions":learn_suggestions, # ← NEW: personalised learning picks
        "soft_matched":soft_hit[:10],"found_sections":found_sec,
        "suggestions":suggestions,"templates":tpls,
    }
def _build_suggestions(ats, ss, tech_miss, missing_kw, found_sec, bullets, numbers, verbs):
    s = []
    if ats >= 70:
        s.append({"type":"positive","priority":0,
                  "title":"Your resume is strong and ATS-friendly! ✓",
                  "detail":"Well-aligned with the job description. Only minor refinements suggested."})
        if ss["soft_skills"] < 55:
            s.append({"type":"minor","priority":3,"title":"Add 2–3 soft skill keywords",
                      "detail":"Include terms like 'leadership', 'collaboration', 'problem-solving' where genuine."})
        if numbers < 4:
            s.append({"type":"minor","priority":3,"title":"Quantify your achievements",
                      "detail":"E.g. 'Improved performance by 35%' · 'Managed team of 8' · 'Saved $50K annually'."})
        return s
    if tech_miss:
        s.append({"type":"critical","priority":1,
                  "title":f"Add missing tech skills: {', '.join(tech_miss[:6])}",
                  "detail":"These appear in the JD but not your resume. Add to Skills section if applicable."})
    kw_adds = [k for k in missing_kw if k not in tech_miss and len(k)>3][:6]
    if kw_adds:
        s.append({"type":"high","priority":1,
                  "title":f"Include job-specific keywords: {', '.join(kw_adds)}",
                  "detail":"Mirror the job description language naturally in bullet points."})
    if "experience" not in found_sec:
        s.append({"type":"high","priority":2,"title":"Add a clearly labelled 'Experience' section",
                  "detail":"ATS parsers require a recognisable section header to extract work history."})
    if "skills" not in found_sec:
        s.append({"type":"high","priority":2,"title":"Add a dedicated 'Skills' section",
                  "detail":"List technical and soft skills in one scannable section."})
    if "summary" not in found_sec:
        s.append({"type":"medium","priority":3,"title":"Add a professional summary (3–4 lines)",
                  "detail":"Open with a targeted summary mirroring key phrases from the job description."})
    if verbs < 4:
        s.append({"type":"medium","priority":3,"title":"Use strong action verbs",
                  "detail":"Start bullets with: Built, Designed, Led, Reduced, Improved, Automated, Delivered, Scaled."})
    if numbers < 3:
        s.append({"type":"medium","priority":3,"title":"Quantify every achievement",
                  "detail":"'Reduced latency by 40%' beats 'improved performance'. Numbers get past ATS."})
    if bullets < 5:
        s.append({"type":"medium","priority":3,"title":"Use bullet points throughout",
                  "detail":"ATS systems and recruiters both prefer scannable bullets over dense paragraphs."})
    if "projects" not in found_sec:
        s.append({"type":"low","priority":4,"title":"Add a Projects section with links",
                  "detail":"Essential for freshers; valuable for everyone — projects prove practical skills."})
    if "certifications" not in found_sec:
        s.append({"type":"low","priority":4,"title":"Add relevant certifications",
                  "detail":"AWS, Google, Azure certs, TF Developer, PMP, etc. stand out to recruiters."})
    return s

def _match_templates(jd_text: str, ats: float, struct_pct: float, found_sec: list) -> list:
    """
    Smart Template Recommendation Engine v2.
    Returns TOP 3 recommended + others, ensuring distinct categories.
    """
    j = jd_text.lower()
    scored = []

    # Experience-level signals
    is_fresher = any(w in j for w in ["fresher","graduate","entry level","intern","0-1 year","0-2 year","junior"])
    is_senior  = any(w in j for w in ["senior","lead","principal","director","head of","vp ","10+ year","8+ year"])
    is_mid     = not is_fresher and not is_senior

    # Industry signals
    is_tech     = any(w in j for w in ["software","developer","engineer","coding","programming","api","backend","frontend","fullstack"])
    is_data     = any(w in j for w in ["data","machine learning","ai","analytics","ml","nlp","data scientist"])
    is_design   = any(w in j for w in ["designer","ux","ui","figma","creative","visual","brand"])
    is_biz      = any(w in j for w in ["manager","director","business","strategy","marketing","sales","executive","finance"])
    is_cloud    = any(w in j for w in ["aws","gcp","azure","cloud","devops","infrastructure","sre","platform"])
    is_security = any(w in j for w in ["security","cybersecurity","soc","pentesting","infosec"])
    is_mobile   = any(w in j for w in ["mobile","ios","android","flutter","react native"])

    for t in TEMPLATES:
        score = 0
        # Keyword match (primary)
        for kw in t["use_for"]:
            if kw in j: score += 3
        # Experience level bonus
        if is_fresher and t["category"] == "fresher": score += 4
        if is_senior  and t["id"] in ("biz_executive","cloud_architect","cybersec_pro"): score += 3
        if is_mid     and t["ats_score"] >= 94: score += 1
        # ATS priority when score is low
        if ats < 60 and t["ats_score"] >= 97: score += 2
        # Structural completeness bonus
        if struct_pct < 50 and t["category"] == "fresher": score += 1
        # Industry fit bonus
        if is_tech    and t["category"] == "tech":    score += 2
        if is_data    and "data" in t["id"]:           score += 3
        if is_design  and t["category"] == "creative": score += 3
        if is_biz     and t["category"] == "business": score += 2
        if is_cloud   and t["id"] in ("cloud_architect","tech_minimal"): score += 3
        if is_security and t["id"] == "cybersec_pro": score += 4
        if is_mobile  and t["id"] == "mobile_dev":    score += 4

        scored.append({**t, "match_score": score, "recommended": False})

    scored.sort(key=lambda x: (-x["match_score"], x["id"]))

    # Ensure top-3 are from different categories for variety
    top3, seen_cats, rest = [], set(), []
    for t in scored:
        if len(top3) < 3 and t["category"] not in seen_cats:
            top3.append(t)
            seen_cats.add(t["category"])
        else:
            rest.append(t)

    # If we couldn't fill top3 with unique categories, fill from rest
    while len(top3) < 3 and rest:
        top3.append(rest.pop(0))

    # Mark top 3 as recommended with reason text
    reasons = [
        "Best match for your role & experience level",
        "Strong ATS compatibility for this industry",
        "Alternative style — high recruiter appeal",
    ]
    for i, t in enumerate(top3):
        t["recommended"] = True
        t["recommendation_reason"] = reasons[i] if i < len(reasons) else "Recommended"

    # Return top 3 + up to 3 more alternatives (different from top3 ids)
    top3_ids = {t["id"] for t in top3}
    alternatives = [t for t in rest if t["id"] not in top3_ids][:3]
    return top3 + alternatives

# ── SKILL LEARNING SUGGESTIONS ────────────────────────────────────────────────
_LEARN_MAP = {
    # if JD needs these, recommend related learnable skills
    "machine learning":  ["scikit-learn","pytorch","tensorflow","mlops","huggingface"],
    "deep learning":     ["pytorch","tensorflow","keras","cuda","transformers"],
    "nlp":               ["spacy","huggingface","bert","langchain","nltk"],
    "data science":      ["pandas","numpy","matplotlib","seaborn","sql","tableau"],
    "aws":               ["terraform","cdk","lambda","s3","ec2","iam"],
    "gcp":               ["bigquery","cloud run","vertex ai","gke"],
    "azure":             ["azure devops","aks","cosmos db","azure functions"],
    "docker":            ["kubernetes","helm","docker compose","containerd"],
    "kubernetes":        ["helm","istio","argocd","prometheus","grafana"],
    "devops":            ["jenkins","github actions","ansible","terraform","prometheus"],
    "react":             ["next.js","redux","tailwind","typescript","vite"],
    "angular":           ["rxjs","typescript","ngrx","nx"],
    "vue":               ["nuxt","pinia","vite","quasar"],
    "node.js":           ["express","fastify","prisma","graphql","redis"],
    "python":            ["fastapi","sqlalchemy","celery","pytest","pydantic"],
    "java":              ["spring","spring boot","maven","gradle","junit"],
    "postgresql":        ["prisma","sqlalchemy","pgvector","redis","supabase"],
    "mongodb":           ["mongoose","atlas","aggregation pipeline","redis"],
    "blockchain":        ["solidity","ethers.js","hardhat","web3.js","ipfs"],
    "security":          ["owasp","penetration testing","burp suite","siem","soc"],
    "product management":["jira","confluence","figma","amplitude","mixpanel"],
    "marketing":         ["google analytics","sem","hubspot","mailchimp","crm"],
    "data engineering":  ["apache spark","airflow","dbt","kafka","snowflake"],
}

def _skill_learning_suggestions(jd_text: str, tech_missing: list) -> list:
    """Return personalised 'skills to learn' based on JD context + resume gaps."""
    j = jd_text.lower()
    seen, recs = set(), []

    # Priority 1: directly map from missing tech skills
    for skill in tech_missing[:8]:
        for domain, suggestions in _LEARN_MAP.items():
            if domain in skill or skill in domain:
                for s in suggestions:
                    if s not in seen and s not in tech_missing:
                        seen.add(s); recs.append(s)
                break

    # Priority 2: scan JD for domain keywords
    for domain, suggestions in _LEARN_MAP.items():
        if domain in j:
            for s in suggestions:
                if s not in seen:
                    seen.add(s); recs.append(s)

    return recs[:12]

# ══════════════════════════════════════════════════════════════════════════════
# PDF BUILDER — Multi-template
# ══════════════════════════════════════════════════════════════════════════════
def build_pdf(resume_dict: dict) -> bytes:
    """Multi-template PDF builder — each template renders a distinct layout."""
    if not _RL:
        sec = resume_dict.get("sections", {})
        c   = sec.get("contact", {})
        lines = [c.get("name",""), c.get("email",""), c.get("phone",""), "",
                 "SKILLS", ", ".join(sec.get("skills", [])), "",
                 "EXPERIENCE", *sec.get("experience", []), "",
                 "EDUCATION",  *sec.get("education",  []), ""]
        return "\n".join(str(l) for l in lines).encode()

    template  = resume_dict.get("template", "universal_pro")
    sec       = resume_dict.get("sections", {})
    color_hex = resume_dict.get("meta", {}).get("color", "#1e3a5f").lstrip("#")
    try:
        rgb = tuple(int(color_hex[i:i+2], 16) / 255 for i in (0, 2, 4))
    except Exception:
        rgb = (0.12, 0.23, 0.37)
    accent = rlc.Color(*rgb)

    buf = io.BytesIO()

    # ── Dispatch to layout ──────────────────────────────────────────────────
    if template in ("tech_modern", "fullstack_pro"):
        _pdf_two_column(buf, sec, accent, template)
    elif template in ("biz_executive", "biz_sales"):
        _pdf_executive(buf, sec, accent, template)
    elif template in ("biz_marketing", "creative_media", "creative_modern"):
        _pdf_marketing(buf, sec, accent, template)
    elif template == "creative_designer":
        _pdf_designer(buf, sec, accent)
    elif template == "data_science":
        _pdf_data_science(buf, sec, accent)
    elif template in ("fresher_classic", "fresher_mba"):
        _pdf_fresher(buf, sec, accent, template)
    elif template == "fresher_tech":
        _pdf_fresher_tech(buf, sec, accent)
    else:
        # tech_minimal, biz_finance, universal_pro, and any new IDs
        _pdf_minimal(buf, sec, accent, template)

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _safe_para(text, style):
    """Wrap text in a Paragraph, escaping problematic chars."""
    from reportlab.platypus import Paragraph
    text = str(text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    try:
        return Paragraph(text, style)
    except Exception:
        return Paragraph("", style)


def _wrap_list(items, style, bullet="•", max_chars=110):
    """Return a list of Paragraphs from a list of strings, with word-wrap."""
    from reportlab.platypus import Paragraph
    story = []
    for item in (items or []):
        text = str(item or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        prefix = f"{bullet} " if bullet else ""
        story.append(_safe_para(prefix + text, style))
    return story


def _section_header(title, style):
    from reportlab.platypus import HRFlowable, Paragraph, Spacer
    from reportlab.lib import colors as rlc
    return [
        Spacer(1, 6),
        _safe_para(title.upper(), style),
        HRFlowable(width="100%", thickness=0.5, color=rlc.lightgrey, spaceAfter=3, spaceBefore=0),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT 1 — MINIMAL (tech_minimal, biz_finance, universal_pro)
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_minimal(buf, sec, accent, template_id):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rlc
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    HRFlowable, KeepTogether)

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=1.6*cm, bottomMargin=1.6*cm)

    name_s  = ParagraphStyle('N', fontName='Helvetica-Bold', fontSize=20, textColor=accent,
                             spaceAfter=2, alignment=TA_CENTER)
    cont_s  = ParagraphStyle('C', fontName='Helvetica', fontSize=8.5, textColor=rlc.grey,
                             spaceAfter=4, alignment=TA_CENTER)
    sec_s   = ParagraphStyle('S', fontName='Helvetica-Bold', fontSize=10, textColor=accent,
                             spaceBefore=8, spaceAfter=2)
    body_s  = ParagraphStyle('B', fontName='Helvetica', fontSize=9.5, spaceAfter=3, leading=13)
    bul_s   = ParagraphStyle('U', fontName='Helvetica', fontSize=9.5, spaceAfter=3,
                             leading=13, leftIndent=12)
    foot_s  = ParagraphStyle('F', fontName='Helvetica', fontSize=7, textColor=rlc.grey,
                             alignment=TA_CENTER)

    c = sec.get("contact", {})
    story = [
        _safe_para(c.get("name") or "Your Name", name_s),
        _safe_para("  |  ".join(x for x in [c.get("email"), c.get("phone"),
                                             c.get("linkedin"), c.get("github")] if x), cont_s),
        HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=4, spaceBefore=2),
    ]

    if sec.get("summary"):
        story += _section_header("Summary", sec_s)
        story.append(_safe_para(str(sec["summary"]), body_s))

    skills = sec.get("skills", [])
    if skills:
        story += _section_header("Skills", sec_s)
        rows = [skills[i:i+5] for i in range(0, len(skills), 5)]
        for row in rows:
            story.append(_safe_para("  ·  ".join(row), body_s))

    if sec.get("experience"):
        story += _section_header("Experience", sec_s)
        story += _wrap_list(sec["experience"], bul_s)

    if sec.get("education"):
        story += _section_header("Education", sec_s)
        story += _wrap_list(sec["education"], body_s, bullet="")

    if sec.get("projects"):
        story += _section_header("Projects", sec_s)
        story += _wrap_list(sec["projects"], bul_s)

    if sec.get("certifications"):
        story += _section_header("Certifications", sec_s)
        story += _wrap_list(sec["certifications"], body_s, bullet="")

    story += [
        Spacer(1, 0.4*cm),
        HRFlowable(width="100%", thickness=0.3, color=rlc.lightgrey),
        _safe_para(f"Generated by ResumeAI Pro · Template: {template_id.replace('_',' ').title()}", foot_s),
    ]
    doc.build(story)


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT 2 — TWO-COLUMN SIDEBAR (tech_modern, fullstack_pro)
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_two_column(buf, sec, accent, template_id):
    """Left sidebar for skills/contact; right main column for experience.
    FIXED: Proper A4 two-column layout with no overflow or misalignment."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rlc
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    HRFlowable, Table, TableStyle)

    PAGE_W, PAGE_H = A4
    # Fixed margins for clean layout
    LM, RM, TM, BM = 0*cm, 0*cm, 0*cm, 0*cm
    SIDEBAR = 5.6*cm
    TOTAL_W = PAGE_W
    MAIN    = TOTAL_W - SIDEBAR

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=0, leftMargin=0,
                            topMargin=0, bottomMargin=0)

    # Sidebar styles (white text on dark bg)
    name_s   = ParagraphStyle('N2c', fontName='Helvetica-Bold', fontSize=13,
                               textColor=accent, spaceAfter=4, leading=16,
                               wordWrap='CJK')
    role_s   = ParagraphStyle('R2c', fontName='Helvetica', fontSize=7.5,
                               textColor=rlc.HexColor('#94a3b8'), spaceAfter=3, leading=10)
    sh_s     = ParagraphStyle('SH2c', fontName='Helvetica-Bold', fontSize=7.5,
                               textColor=accent, spaceBefore=10, spaceAfter=4,
                               borderPadding=(0,0,2,0))
    sb_s     = ParagraphStyle('SB2c', fontName='Helvetica', fontSize=7.5,
                               textColor=rlc.HexColor('#e2e8f0'), spaceAfter=3, leading=10)
    # Main column styles (dark text on white)
    mh_s     = ParagraphStyle('MH2c', fontName='Helvetica-Bold', fontSize=9.5,
                               textColor=accent, spaceBefore=10, spaceAfter=3)
    mb_s     = ParagraphStyle('MB2c', fontName='Helvetica', fontSize=9,
                               spaceAfter=3, leading=12, textColor=rlc.HexColor('#1e293b'))
    mbul_s   = ParagraphStyle('MU2c', fontName='Helvetica', fontSize=8.5,
                               spaceAfter=3, leading=12, leftIndent=10,
                               textColor=rlc.HexColor('#1e293b'))
    sum_s    = ParagraphStyle('SM2c', fontName='Helvetica', fontSize=9,
                               spaceAfter=4, leading=13, textColor=rlc.HexColor('#334155'))
    foot_s   = ParagraphStyle('F2c', fontName='Helvetica', fontSize=6.5,
                               textColor=rlc.grey, alignment=TA_CENTER)

    c = sec.get("contact", {})
    dark = rlc.HexColor('#0f172a')

    def side_section(title, items, bullet="▸"):
        parts = [
            HRFlowable(width=SIDEBAR-0.6*cm, thickness=0.3, color=rlc.HexColor('#334155'),
                       spaceAfter=2, spaceBefore=2),
            _safe_para(title.upper(), sh_s)
        ]
        for it in (items or []):
            txt = str(it or "").strip()
            if len(txt) > 40: txt = txt[:40] + "…"
            parts.append(_safe_para(f"{bullet} {txt}" if bullet else txt, sb_s))
        return parts

    def main_section(title, items, is_bul=True):
        parts = [
            HRFlowable(width=MAIN-0.6*cm, thickness=0.4, color=rlc.HexColor('#e2e8f0'),
                       spaceAfter=2, spaceBefore=2),
            _safe_para(title.upper(), mh_s)
        ]
        for it in (items or []):
            txt = str(it or "").strip()
            # Truncate very long bullets
            if len(txt) > 120: txt = txt[:120] + "…"
            parts.append(_safe_para(
                ("▸  " if is_bul else "") + txt,
                mbul_s if is_bul else mb_s
            ))
        return parts

    # Build sidebar
    side_items = [Spacer(1, 0.5*cm)]
    name_text = (c.get("name") or "Your Name")
    if len(name_text) > 22: name_text = name_text[:22] + "…"
    side_items.append(_safe_para(name_text, name_s))
    side_items.append(HRFlowable(width=SIDEBAR-0.6*cm, thickness=1, color=accent,
                                  spaceAfter=4, spaceBefore=2))
    for field in [c.get("email"), c.get("phone"), c.get("linkedin"), c.get("github")]:
        if field:
            f = str(field)
            if len(f) > 28: f = f[:28] + "…"
            side_items.append(_safe_para(f, role_s))

    if sec.get("skills"):
        side_items += side_section("Technical Skills", sec["skills"][:14])
    if sec.get("certifications"):
        side_items += side_section("Certifications", sec["certifications"][:4], bullet="✓")

    # Build main column
    main_items = [Spacer(1, 0.5*cm)]
    if sec.get("summary"):
        main_items.append(_safe_para("PROFESSIONAL SUMMARY", mh_s))
        main_items.append(HRFlowable(width=MAIN-0.6*cm, thickness=0.4,
                                      color=rlc.HexColor('#e2e8f0'), spaceAfter=3))
        main_items.append(_safe_para(str(sec["summary"]), sum_s))
    if sec.get("experience"):
        main_items += main_section("Experience", sec["experience"])
    if sec.get("projects"):
        main_items += main_section("Projects", sec["projects"])
    if sec.get("education"):
        main_items += main_section("Education", sec["education"], is_bul=False)

    # Render two-column table — full page width, no outer margins
    tbl = Table([[side_items, main_items]],
                colWidths=[SIDEBAR, MAIN],
                rowHeights=None)
    tbl.setStyle(TableStyle([
        ('VALIGN',       (0,0), (-1,-1), 'TOP'),
        ('BACKGROUND',   (0,0), (0,-1), dark),
        ('LEFTPADDING',  (0,0), (0,-1), 12),
        ('RIGHTPADDING', (0,0), (0,-1), 8),
        ('LEFTPADDING',  (1,0), (1,-1), 14),
        ('RIGHTPADDING', (1,0), (1,-1), 16),
        ('TOPPADDING',   (0,0), (-1,-1), 0),
        ('BOTTOMPADDING',(0,0), (-1,-1), 8),
    ]))

    footer_tbl = Table(
        [[_safe_para(f"ResumeAI Pro · {template_id.replace('_',' ').title()} · ATS Optimized", foot_s)]],
        colWidths=[TOTAL_W]
    )
    footer_tbl.setStyle(TableStyle([
        ('TOPPADDING',   (0,0),(-1,-1), 4),
        ('BOTTOMPADDING',(0,0),(-1,-1), 4),
        ('LEFTPADDING',  (0,0),(-1,-1), 0),
        ('RIGHTPADDING', (0,0),(-1,-1), 0),
    ]))

    doc.build([tbl, Spacer(1, 0.1*cm), footer_tbl])


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT 3 — EXECUTIVE (biz_executive, biz_sales)
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_executive(buf, sec, accent, template_id):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rlc
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, HRFlowable)

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2.5*cm, leftMargin=2.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    name_s  = ParagraphStyle('N', fontName='Helvetica-Bold', fontSize=26, textColor=accent,
                             spaceAfter=0, alignment=TA_CENTER, letterSpacing=2)
    cont_s  = ParagraphStyle('C', fontName='Helvetica',      fontSize=8.5, textColor=rlc.grey,
                             spaceAfter=6, alignment=TA_CENTER)
    sec_s   = ParagraphStyle('S', fontName='Helvetica-Bold', fontSize=11, textColor=accent,
                             spaceBefore=12, spaceAfter=3, letterSpacing=1)
    body_s  = ParagraphStyle('B', fontName='Helvetica',      fontSize=10, spaceAfter=4, leading=14)
    bul_s   = ParagraphStyle('U', fontName='Helvetica',      fontSize=10, spaceAfter=4,
                             leading=14, leftIndent=14)
    sum_s   = ParagraphStyle('SM', fontName='Helvetica',     fontSize=10.5, spaceAfter=4,
                             leading=15, textColor=rlc.HexColor('#222222'))
    foot_s  = ParagraphStyle('F', fontName='Helvetica',      fontSize=7,  textColor=rlc.grey,
                             alignment=TA_CENTER)

    c = sec.get("contact", {})
    story = [
        _safe_para(c.get("name") or "Your Name", name_s),
        HRFlowable(width="100%", thickness=3, color=accent, spaceAfter=4, spaceBefore=4),
        _safe_para("  |  ".join(x for x in [c.get("email"), c.get("phone"),
                                             c.get("linkedin")] if x), cont_s),
        HRFlowable(width="100%", thickness=0.5, color=rlc.lightgrey, spaceAfter=6),
    ]

    if sec.get("summary"):
        story += [_safe_para("EXECUTIVE SUMMARY", sec_s),
                  _safe_para(str(sec["summary"]), sum_s),
                  HRFlowable(width="100%", thickness=0.5, color=rlc.lightgrey, spaceAfter=4)]

    if sec.get("skills"):
        story += [_safe_para("CORE COMPETENCIES", sec_s)]
        # 3-column skill grid
        from reportlab.platypus import Table, TableStyle
        skills = sec["skills"]
        rows = [skills[i:i+3] for i in range(0, len(skills), 3)]
        rows = [r + ['']*(3-len(r)) for r in rows]  # pad
        tbl = Table(rows, colWidths=['33%','33%','34%'])
        tbl.setStyle(TableStyle([
            ('FONTNAME',  (0,0),(-1,-1),'Helvetica'),
            ('FONTSIZE',  (0,0),(-1,-1),9.5),
            ('TOPPADDING',(0,0),(-1,-1),3),
            ('BOTTOMPADDING',(0,0),(-1,-1),3),
        ]))
        story.append(tbl)

    if sec.get("experience"):
        story += [_safe_para("PROFESSIONAL EXPERIENCE", sec_s)]
        story += _wrap_list(sec["experience"], bul_s)

    if sec.get("education"):
        story += [_safe_para("EDUCATION", sec_s)]
        story += _wrap_list(sec["education"], body_s, bullet="")

    if sec.get("certifications"):
        story += [_safe_para("CERTIFICATIONS & AWARDS", sec_s)]
        story += _wrap_list(sec["certifications"], body_s, bullet="")

    story += [Spacer(1, 0.5*cm),
              HRFlowable(width="100%", thickness=0.3, color=rlc.lightgrey),
              _safe_para(f"ResumeAI Pro · {template_id.replace('_',' ').title()}", foot_s)]
    doc.build(story)


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT 4 — MARKETING/CREATIVE STRIPE (biz_marketing, creative_media, creative_modern)
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_marketing(buf, sec, accent, template_id):
    """Coloured accent bar down the left edge via a borderless table."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rlc
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    HRFlowable, Table, TableStyle)

    STRIPE = 0.7*cm
    PAGE_W, _ = A4
    LM = 1.5*cm; RM = 1.5*cm; TM = 1.4*cm; BM = 1.4*cm
    CONTENT_W = PAGE_W - LM - RM - STRIPE - 0.3*cm

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=RM, leftMargin=LM,
                            topMargin=TM, bottomMargin=BM)

    name_s  = ParagraphStyle('N', fontName='Helvetica-Bold', fontSize=22, textColor=accent, spaceAfter=2)
    cont_s  = ParagraphStyle('C', fontName='Helvetica',      fontSize=8.5, textColor=rlc.grey, spaceAfter=5)
    sec_s   = ParagraphStyle('S', fontName='Helvetica-Bold', fontSize=10, textColor=accent, spaceBefore=8, spaceAfter=2)
    body_s  = ParagraphStyle('B', fontName='Helvetica',      fontSize=9.5, spaceAfter=3, leading=13)
    bul_s   = ParagraphStyle('U', fontName='Helvetica',      fontSize=9.5, spaceAfter=3, leading=13, leftIndent=10)
    foot_s  = ParagraphStyle('F', fontName='Helvetica',      fontSize=7,  textColor=rlc.grey, alignment=TA_CENTER)

    c = sec.get("contact", {})
    content = [
        _safe_para(c.get("name") or "Your Name", name_s),
        _safe_para("  |  ".join(x for x in [c.get("email"), c.get("phone"),
                                             c.get("linkedin")] if x), cont_s),
        HRFlowable(width=CONTENT_W, thickness=0.5, color=rlc.lightgrey, spaceAfter=4),
    ]
    if sec.get("summary"):
        content += [_safe_para("PROFILE", sec_s), _safe_para(str(sec["summary"]), body_s)]
    if sec.get("skills"):
        content += [_safe_para("EXPERTISE", sec_s),
                    _safe_para("  ·  ".join(sec["skills"][:16]), body_s)]
    if sec.get("experience"):
        content += [_safe_para("EXPERIENCE", sec_s)] + _wrap_list(sec["experience"], bul_s)
    if sec.get("projects"):
        content += [_safe_para("CAMPAIGNS / PROJECTS", sec_s)] + _wrap_list(sec["projects"], bul_s)
    if sec.get("education"):
        content += [_safe_para("EDUCATION", sec_s)] + _wrap_list(sec["education"], body_s, bullet="")

    stripe_col = [Spacer(STRIPE, 1)]  # coloured column (background set via TableStyle)

    tbl = Table([[stripe_col, content]], colWidths=[STRIPE, CONTENT_W + 0.3*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (0, -1), accent),
        ('VALIGN',      (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (1, 0), (1, -1), 10),
        ('TOPPADDING',  (0, 0), (-1, -1), 0),
    ]))
    story = [tbl, Spacer(1, 0.3*cm),
             HRFlowable(width="100%", thickness=0.3, color=rlc.lightgrey),
             _safe_para(f"ResumeAI Pro · {template_id.replace('_',' ').title()}", foot_s)]
    doc.build(story)


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT 5 — DESIGNER / PORTFOLIO (creative_designer)
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_designer(buf, sec, accent):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rlc
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    HRFlowable, Table, TableStyle)

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=1.8*cm, leftMargin=1.8*cm,
                            topMargin=1.4*cm, bottomMargin=1.4*cm)

    name_s  = ParagraphStyle('N', fontName='Helvetica-Bold', fontSize=24, textColor=accent,
                             spaceAfter=2, alignment=TA_CENTER)
    tag_s   = ParagraphStyle('T', fontName='Helvetica',      fontSize=9,  textColor=rlc.grey,
                             spaceAfter=6, alignment=TA_CENTER)
    sec_s   = ParagraphStyle('S', fontName='Helvetica-Bold', fontSize=10, textColor=accent,
                             spaceBefore=8, spaceAfter=2)
    body_s  = ParagraphStyle('B', fontName='Helvetica',      fontSize=9.5, spaceAfter=3, leading=13)
    bul_s   = ParagraphStyle('U', fontName='Helvetica',      fontSize=9.5, spaceAfter=3,
                             leading=13, leftIndent=10)
    box_s   = ParagraphStyle('BX', fontName='Helvetica-Bold', fontSize=9, textColor=accent,
                             spaceAfter=2)
    foot_s  = ParagraphStyle('F', fontName='Helvetica',      fontSize=7, textColor=rlc.grey,
                             alignment=TA_CENTER)

    c = sec.get("contact", {})
    story = [
        _safe_para(c.get("name") or "Your Name", name_s),
        _safe_para("UX / UI Designer  ·  Creative Director", tag_s),
        HRFlowable(width="100%", thickness=2, color=accent, spaceAfter=2, spaceBefore=0),
        _safe_para("  |  ".join(x for x in [c.get("email"), c.get("phone"),
                                             c.get("linkedin"), c.get("github")] if x), tag_s),
        HRFlowable(width="100%", thickness=0.5, color=rlc.lightgrey, spaceAfter=6),
    ]

    if sec.get("summary"):
        story += _section_header("Design Philosophy", sec_s)
        story.append(_safe_para(str(sec["summary"]), body_s))

    # Portfolio callout box
    portfolio_url = c.get("linkedin") or c.get("github") or "portfolio.example.com"
    tbl = Table([[_safe_para(f"🔗 Portfolio: {portfolio_url}", box_s)]],
                colWidths=["100%"])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), rlc.Color(0.95,0.95,1.0)),
        ('BOX',        (0,0), (-1,-1), 0.8, accent),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0),(-1,-1), 6),
        ('LEFTPADDING',(0,0), (-1,-1), 10),
    ]))
    story += [tbl, Spacer(1, 6)]

    if sec.get("skills"):
        story += _section_header("Design Tools & Skills", sec_s)
        story.append(_safe_para("  ·  ".join(sec["skills"][:16]), body_s))

    if sec.get("experience"):
        story += _section_header("Experience", sec_s)
        story += _wrap_list(sec["experience"], bul_s)

    if sec.get("projects"):
        story += _section_header("Featured Projects", sec_s)
        story += _wrap_list(sec["projects"], bul_s)

    if sec.get("education"):
        story += _section_header("Education", sec_s)
        story += _wrap_list(sec["education"], body_s, bullet="")

    story += [Spacer(1, 0.4*cm),
              HRFlowable(width="100%", thickness=0.3, color=rlc.lightgrey),
              _safe_para("ResumeAI Pro · Creative Designer Template", foot_s)]
    doc.build(story)


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT 6 — DATA SCIENCE (metrics-bar header)
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_data_science(buf, sec, accent):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rlc
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    HRFlowable, Table, TableStyle)

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=1.8*cm, leftMargin=1.8*cm,
                            topMargin=1.4*cm, bottomMargin=1.4*cm)

    name_s  = ParagraphStyle('N', fontName='Helvetica-Bold', fontSize=20, textColor=rlc.white,
                             spaceAfter=2, alignment=TA_CENTER)
    sub_s   = ParagraphStyle('SU', fontName='Helvetica',     fontSize=9,  textColor=rlc.lightgrey,
                             spaceAfter=0, alignment=TA_CENTER)
    sec_s   = ParagraphStyle('S', fontName='Helvetica-Bold', fontSize=10, textColor=accent,
                             spaceBefore=8, spaceAfter=2)
    body_s  = ParagraphStyle('B', fontName='Helvetica',      fontSize=9.5, spaceAfter=3, leading=13)
    bul_s   = ParagraphStyle('U', fontName='Helvetica',      fontSize=9.5, spaceAfter=3,
                             leading=13, leftIndent=10)
    met_s   = ParagraphStyle('M', fontName='Helvetica-Bold', fontSize=9,  textColor=accent,
                             alignment=TA_CENTER, spaceAfter=2)
    foot_s  = ParagraphStyle('F', fontName='Helvetica',      fontSize=7, textColor=rlc.grey,
                             alignment=TA_CENTER)

    dark = rlc.Color(0.06, 0.06, 0.14)
    c = sec.get("contact", {})

    # Dark header banner
    header_tbl = Table([
        [_safe_para(c.get("name") or "Your Name", name_s)],
        [_safe_para("Data Scientist  ·  ML Engineer  ·  AI Researcher", sub_s)],
        [_safe_para("  |  ".join(x for x in [c.get("email"), c.get("phone"),
                                              c.get("linkedin"), c.get("github")] if x), sub_s)],
    ], colWidths=["100%"])
    header_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), dark),
        ('TOPPADDING',    (0,0),(-1,-1), 10),
        ('BOTTOMPADDING', (0,0),(-1,-1), 10),
        ('LEFTPADDING',   (0,0),(-1,-1), 14),
    ]))

    story = [header_tbl, Spacer(1, 8)]

    if sec.get("summary"):
        story += _section_header("Profile", sec_s)
        story.append(_safe_para(str(sec["summary"]), body_s))

    # Skills as inline tags
    if sec.get("skills"):
        story += _section_header("Technical Stack", sec_s)
        story.append(_safe_para("  ·  ".join(sec["skills"][:20]), body_s))

    if sec.get("experience"):
        story += _section_header("Experience", sec_s)
        story += _wrap_list(sec["experience"], bul_s)

    if sec.get("projects"):
        story += _section_header("Research / Projects", sec_s)
        story += _wrap_list(sec["projects"], bul_s)

    if sec.get("education"):
        story += _section_header("Education", sec_s)
        story += _wrap_list(sec["education"], body_s, bullet="")

    if sec.get("certifications"):
        story += _section_header("Publications & Certifications", sec_s)
        story += _wrap_list(sec["certifications"], body_s, bullet="")

    story += [Spacer(1, 0.4*cm),
              HRFlowable(width="100%", thickness=0.3, color=rlc.lightgrey),
              _safe_para("ResumeAI Pro · Data Science Pro Template", foot_s)]
    doc.build(story)


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT 7 — FRESHER CLASSIC / MBA (education-first)
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_fresher(buf, sec, accent, template_id):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rlc
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    HRFlowable, Table, TableStyle)

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=1.6*cm, bottomMargin=1.6*cm)

    name_s  = ParagraphStyle('N', fontName='Helvetica-Bold', fontSize=22, textColor=accent,
                             spaceAfter=2, alignment=TA_CENTER)
    cont_s  = ParagraphStyle('C', fontName='Helvetica',      fontSize=8.5, textColor=rlc.grey,
                             spaceAfter=5, alignment=TA_CENTER)
    edu_box = ParagraphStyle('E', fontName='Helvetica-Bold', fontSize=10,  textColor=accent,
                             spaceAfter=2)
    sec_s   = ParagraphStyle('S', fontName='Helvetica-Bold', fontSize=10, textColor=accent,
                             spaceBefore=8, spaceAfter=2)
    body_s  = ParagraphStyle('B', fontName='Helvetica',      fontSize=9.5, spaceAfter=3, leading=13)
    bul_s   = ParagraphStyle('U', fontName='Helvetica',      fontSize=9.5, spaceAfter=3,
                             leading=13, leftIndent=10)
    foot_s  = ParagraphStyle('F', fontName='Helvetica',      fontSize=7,  textColor=rlc.grey,
                             alignment=TA_CENTER)

    c = sec.get("contact", {})
    story = [
        _safe_para(c.get("name") or "Your Name", name_s),
        _safe_para("  |  ".join(x for x in [c.get("email"), c.get("phone"),
                                             c.get("linkedin")] if x), cont_s),
        HRFlowable(width="100%", thickness=2, color=accent, spaceAfter=6, spaceBefore=2),
    ]

    # Education FIRST (fresher layout hallmark)
    if sec.get("education"):
        story += _section_header("Education", sec_s)
        for edu in sec["education"]:
            # Highlight education in a subtle box
            tbl = Table([[_safe_para(str(edu), edu_box)]], colWidths=["100%"])
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0),(-1,-1), rlc.Color(0.96,0.98,1.0)),
                ('LEFTPADDING',(0,0),(-1,-1), 10),
                ('TOPPADDING', (0,0),(-1,-1), 5),
                ('BOTTOMPADDING',(0,0),(-1,-1), 5),
            ]))
            story += [tbl, Spacer(1, 4)]

    if sec.get("summary"):
        story += _section_header("Objective", sec_s)
        story.append(_safe_para(str(sec["summary"]), body_s))

    if sec.get("skills"):
        story += _section_header("Skills", sec_s)
        story.append(_safe_para("  ·  ".join(sec["skills"][:18]), body_s))

    if sec.get("experience"):
        story += _section_header("Internships / Experience", sec_s)
        story += _wrap_list(sec["experience"], bul_s)

    if sec.get("projects"):
        story += _section_header("Academic Projects", sec_s)
        story += _wrap_list(sec["projects"], bul_s)

    if sec.get("certifications"):
        story += _section_header("Certifications & Achievements", sec_s)
        story += _wrap_list(sec["certifications"], body_s, bullet="")

    story += [Spacer(1, 0.4*cm),
              HRFlowable(width="100%", thickness=0.3, color=rlc.lightgrey),
              _safe_para(f"ResumeAI Pro · {template_id.replace('_',' ').title()}", foot_s)]
    doc.build(story)


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT 8 — FRESHER TECH (GitHub/projects-forward)
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_fresher_tech(buf, sec, accent):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rlc
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    HRFlowable, Table, TableStyle)

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=1.8*cm, leftMargin=1.8*cm,
                            topMargin=1.4*cm, bottomMargin=1.4*cm)

    name_s  = ParagraphStyle('N', fontName='Helvetica-Bold', fontSize=20, textColor=accent,
                             spaceAfter=1)
    tag_s   = ParagraphStyle('T', fontName='Helvetica',      fontSize=9, textColor=rlc.grey,
                             spaceAfter=3)
    sec_s   = ParagraphStyle('S', fontName='Helvetica-Bold', fontSize=10, textColor=accent,
                             spaceBefore=8, spaceAfter=2)
    body_s  = ParagraphStyle('B', fontName='Helvetica',      fontSize=9.5, spaceAfter=3, leading=13)
    bul_s   = ParagraphStyle('U', fontName='Helvetica',      fontSize=9.5, spaceAfter=3,
                             leading=13, leftIndent=10)
    gh_s    = ParagraphStyle('GH', fontName='Helvetica-Bold', fontSize=9, textColor=accent,
                             spaceAfter=2)
    foot_s  = ParagraphStyle('F', fontName='Helvetica',      fontSize=7, textColor=rlc.grey,
                             alignment=TA_CENTER)

    c = sec.get("contact", {})
    story = [
        _safe_para(c.get("name") or "Your Name", name_s),
        _safe_para("  |  ".join(x for x in [c.get("email"), c.get("phone"),
                                             c.get("linkedin"), c.get("github")] if x), tag_s),
        HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=4, spaceBefore=2),
    ]

    # GitHub/links callout
    if c.get("github"):
        tbl = Table([[_safe_para(f"👨‍💻 GitHub: {c['github']}", gh_s)]], colWidths=["100%"])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,-1), rlc.Color(0.05,0.1,0.05)),
            ('BOX',        (0,0),(-1,-1), 0.6, accent),
            ('LEFTPADDING',(0,0),(-1,-1), 10),
            ('TOPPADDING', (0,0),(-1,-1), 5),
            ('BOTTOMPADDING',(0,0),(-1,-1), 5),
        ]))
        story += [tbl, Spacer(1, 6)]

    if sec.get("skills"):
        story += _section_header("Technical Skills", sec_s)
        story.append(_safe_para("  ·  ".join(sec["skills"][:20]), body_s))

    # Projects BEFORE experience for tech freshers
    if sec.get("projects"):
        story += _section_header("Projects & Open Source", sec_s)
        story += _wrap_list(sec["projects"], bul_s)

    if sec.get("experience"):
        story += _section_header("Internships / Work Experience", sec_s)
        story += _wrap_list(sec["experience"], bul_s)

    if sec.get("education"):
        story += _section_header("Education", sec_s)
        story += _wrap_list(sec["education"], body_s, bullet="")

    if sec.get("certifications"):
        story += _section_header("Certifications", sec_s)
        story += _wrap_list(sec["certifications"], body_s, bullet="")

    story += [Spacer(1, 0.4*cm),
              HRFlowable(width="100%", thickness=0.3, color=rlc.lightgrey),
              _safe_para("ResumeAI Pro · Tech Graduate Template", foot_s)]
    doc.build(story)

# ══════════════════════════════════════════════════════════════════════════════
# AI CHAT (OpenAI or rule-based fallback)
# ══════════════════════════════════════════════════════════════════════════════
def ai_chat_response(user_msg: str, context: dict = None) -> str:
    if _OAI:
        try:
            sys_prompt = """You are ResumeAI Pro assistant. You help users build ATS-optimized resumes.
You know about: ATS scoring, resume templates, keyword optimization, job descriptions, resume writing best practices.
Keep responses concise (2-4 sentences), helpful, and action-oriented.
If the user has analysis data, reference it specifically."""
            if context and context.get("ats_score"):
                sys_prompt += f"\nUser's current ATS score: {context['ats_score']}%"
                sys_prompt += f"\nMatch level: {context.get('match_level','')}"
                sys_prompt += f"\nTop missing keywords: {', '.join(context.get('missing_keywords',[])[:5])}"

            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":sys_prompt},
                    {"role":"user","content":user_msg}
                ],
                max_tokens=200, temperature=0.7
            )
            return resp.choices[0].message.content
        except Exception as e:
            pass  # Fall through to rule-based

    # Rule-based fallback
    lower = user_msg.lower()
    for key, resp in CHAT_KB.items():
        if key in lower:
            if context and context.get("ats_score") and key in ("score","ats","improve"):
                score = context["ats_score"]
                miss  = context.get("missing_keywords",[])[:3]
                return f"{resp}\n\nYour current score: {score}%. Top missing keywords: {', '.join(miss)}."
            return resp

    # Context-aware default
    if context and context.get("ats_score"):
        return (f"Based on your analysis, your ATS score is {context['ats_score']}% ({context.get('match_level','')})."
                f" Your top priorities are: add missing keywords, use more action verbs, and quantify achievements."
                f" Would you like specific advice on any of these?")
    return CHAT_KB["default"]

# ══════════════════════════════════════════════════════════════════════════════
# FACE AUTH
# ══════════════════════════════════════════════════════════════════════════════
def _dp(email): return os.path.join(FACES_DIR, f"{email}.json")

def save_desc(email, desc):
    with open(_dp(email), "w") as f: json.dump({"email":email,"descriptor":desc},f)

def load_desc(email):
    p = _dp(email)
    return json.load(open(p)).get("descriptor") if os.path.exists(p) else None

def euclid(a, b):
    return float(np.linalg.norm(np.array(a,float)-np.array(b,float)))

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/ping")
def ping():
    return jsonify({"status":"ok","version":"3.0",
                    "features":{"pdf":bool(_PDF),"docx":_DOCX,"sklearn":_SK,"reportlab":_RL,"openai":_OAI}})

@app.route("/templates", methods=["GET"])
def get_templates():
    cat = request.args.get("category","")
    tpls = [t for t in TEMPLATES if not cat or t["category"]==cat]
    return jsonify({"status":"ok","templates":tpls,"total":len(tpls)})

@app.route("/recommend_templates", methods=["POST"])
def recommend_templates():
    d   = request.json or {}
    role     = d.get("role", "").lower()
    exp_lvl  = d.get("experience_level", "mid").lower()
    industry = d.get("industry", "").lower()
    pref     = d.get("preference", "").lower()
    jd       = d.get("job_description", "")
    synthetic = f"{role} {exp_lvl} {industry} {pref} {jd}".lower()
    ats_est = 40 if exp_lvl == "fresher" else (70 if exp_lvl == "mid" else 80)
    struct  = 60 if exp_lvl != "fresher" else 40
    recs = _match_templates(synthetic, ats_est, struct, [])
    if pref == "ats": recs.sort(key=lambda t: -t.get("ats_score", 0))
    elif pref == "creative":
        recs = [t for t in recs if t["category"]=="creative"] + [t for t in recs if t["category"]!="creative"]
    elif pref == "minimal":
        recs = [t for t in recs if "minimal" in t["id"] or t["id"]=="universal_pro"] + [t for t in recs if "minimal" not in t["id"] and t["id"]!="universal_pro"]
    reasons = ["Best for your role — ATS optimized layout","Strong match for your industry — high recruiter appeal","Alternative style — great for creative differentiation"]
    for i, t in enumerate(recs[:3]):
        t["recommended"] = True
        if "recommendation_reason" not in t: t["recommendation_reason"] = reasons[min(i, 2)]
    return jsonify({"status": "ok","recommended": recs[:3],"others": recs[3:6]})

@app.route("/signup", methods=["POST"])
def signup():
    d = request.json or {}
    email = d.get("email","").strip().lower()
    name  = d.get("name","").strip()
    desc  = d.get("descriptor")
    if not email: return jsonify({"status":"error","message":"Email required"}), 400
    if desc and len(desc)>64: save_desc(email, desc)
    db = get_db()
    db.execute("INSERT OR IGNORE INTO users(email,name) VALUES(?,?)", (email, name))
    db.commit()
    return jsonify({"status":"success","message":"Account registered"})

@app.route("/login", methods=["POST"])
def login():
    d = request.json or {}
    email = d.get("email","").strip().lower()
    desc  = d.get("descriptor")
    if not email: return jsonify({"status":"error","message":"Email required"}), 400
    stored = load_desc(email)
    if stored is None:
        return jsonify({"status":"error","message":"User not found. Please sign up."}), 404
    if not desc:
        return jsonify({"status":"error","message":"No face descriptor provided"}), 400
    dist = euclid(desc, stored)
    if dist < 0.50:
        return jsonify({"status":"success","message":"Identity verified","distance":round(dist,4)})
    return jsonify({"status":"error","message":f"Face mismatch (dist={dist:.3f}). Try better lighting."}), 401

@app.route("/google_signup", methods=["POST"])
def google_signup():
    """
    Handles Google OAuth signup verification.
    Body JSON: { email, name, google_id }
    Creates or updates user account.
    Returns: { status, message, email, name }
    """
    d = request.json or {}
    email     = d.get("email","").strip().lower()
    name      = d.get("name","").strip()
    google_id = d.get("google_id","").strip()
    if not email:
        return jsonify({"status":"error","message":"Email required"}), 400
    if not email.endswith((".com",".org",".net",".edu",".io",".co",".in")):
        return jsonify({"status":"error","message":"Invalid email domain"}), 400
    db = get_db()
    db.execute("INSERT OR IGNORE INTO users(email,name) VALUES(?,?)", (email, name))
    db.commit()
    return jsonify({"status":"success","message":"Google account linked","email":email,"name":name or email.split("@")[0]})

@app.route("/feedback", methods=["POST"])
def get_feedback():
    """
    AI-based resume feedback: missing skills, improvements, summary suggestions.
    Body JSON: { analysis_id } or { sections, job_description }
    """
    try:
        d = request.json or {}
        analysis_id = d.get("analysis_id","")
        sections    = d.get("sections",{})
        jd          = d.get("job_description","")

        if analysis_id:
            cache = app.config.get("_rc",{})
            if analysis_id in cache:
                resume_text = cache[analysis_id]["text"]
                jd = jd or cache[analysis_id].get("jd","")
            else:
                row = get_db().execute("SELECT resume_text,jd_text FROM analyses WHERE id=?", (analysis_id,)).fetchone()
                if not row:
                    return jsonify({"status":"error","message":"Analysis not found"}), 404
                resume_text = row["resume_text"]
                jd = jd or row["jd_text"]
            parsed = parse_sections(resume_text)
            sections = {
                "summary": parsed["summary"],
                "skills": parsed["skills"],
                "experience": parsed["experience"],
                "education": parsed["education"],
                "projects": parsed["projects"],
            }

        r = " ".join([sections.get("summary",""), " ".join(sections.get("skills",[])),
                      " ".join(sections.get("experience",[]))]).lower()
        j = jd.lower()

        tech_in_resume = [s for s in TECH_SKILLS if s in r]
        tech_in_jd     = [s for s in TECH_SKILLS if s in j]
        missing_tech   = [s for s in tech_in_jd if s not in tech_in_resume][:8]

        soft_in_resume = [s for s in SOFT_SKILLS if s in r]
        missing_soft   = [s for s in SOFT_SKILLS if s not in soft_in_resume][:5]

        verb_hits  = sum(1 for v in EXP_VERBS if v in r)
        numbers    = len(re.findall(r'\d+', " ".join(sections.get("experience",[]))))
        bullets    = len(sections.get("experience",[]))

        feedback_items = []

        if missing_tech:
            feedback_items.append({
                "category": "Missing Technical Skills",
                "icon": "🔧",
                "severity": "high",
                "items": missing_tech,
                "advice": f"Add these skills to your Skills section if you have experience: {', '.join(missing_tech[:5])}."
            })

        if missing_soft:
            feedback_items.append({
                "category": "Soft Skills to Highlight",
                "icon": "🤝",
                "severity": "medium",
                "items": missing_soft[:4],
                "advice": "Weave these into experience bullets naturally — e.g. 'Led cross-functional team (leadership)'"
            })

        if numbers < 3:
            feedback_items.append({
                "category": "Quantify Achievements",
                "icon": "📊",
                "severity": "high",
                "items": ["Add % improvements", "Revenue/cost figures", "Team sizes", "Time saved"],
                "advice": "Change 'improved performance' → 'improved performance by 42%'. Numbers pass ATS and impress recruiters."
            })

        if verb_hits < 4:
            feedback_items.append({
                "category": "Weak Action Verbs",
                "icon": "⚡",
                "severity": "medium",
                "items": ["Built", "Led", "Reduced", "Automated", "Delivered", "Scaled"],
                "advice": "Start every bullet with a strong past-tense action verb."
            })

        summary = sections.get("summary","")
        if not summary or len(summary) < 60:
            feedback_items.append({
                "category": "Professional Summary Missing / Weak",
                "icon": "📝",
                "severity": "high",
                "items": ["Write 3-4 targeted sentences", "Mirror JD language", "State years of experience", "Name top 2-3 skills"],
                "advice": "A strong summary is the #1 way to pass the 6-second recruiter scan."
            })

        if bullets < 4:
            feedback_items.append({
                "category": "Experience Bullets Too Few",
                "icon": "📋",
                "severity": "medium",
                "items": ["Aim for 4-6 bullets per role", "Each bullet = one achievement", "Use STAR format"],
                "advice": "More context-rich bullets = higher ATS score + recruiter interest."
            })

        score_boost = min(25, len(missing_tech)*2 + (5 if not summary else 0) + (8 if numbers<3 else 0))
        return jsonify({
            "status": "success",
            "feedback": feedback_items,
            "total_issues": len(feedback_items),
            "potential_score_boost": score_boost,
            "summary_tip": f"Fix {len(feedback_items)} issues to potentially boost your ATS score by ~{score_boost} points."
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "resume" not in request.files:
            return jsonify({"status":"error","message":"No resume file attached"}), 400
        file = request.files["resume"]
        jd   = request.form.get("job_description","").strip()
        if not jd: return jsonify({"status":"error","message":"Job description is empty"}), 400

        try:
            resume_text = extract_text(file, file.filename)
        except Exception as e:
            return jsonify({"status":"error","message":f"Could not read file: {e}"}), 400
        if not resume_text.strip():
            return jsonify({"status":"error","message":"File appears empty. Use a text-based PDF."}), 400

        result      = run_ats(resume_text, jd)
        analysis_id = str(uuid.uuid4())
        email       = request.form.get("email","anon")

        db = get_db()
        db.execute("""
            INSERT INTO analyses
              (id,user_email,filename,ats_score,match_level,skill_scores,
               matched_keywords,missing_keywords,suggestions,templates,resume_text,jd_text)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """,(analysis_id, email, file.filename, result["ats_score"], result["match_level"],
             json.dumps(result["skill_scores"]), json.dumps(result["matched_keywords"]),
             json.dumps(result["missing_keywords"]), json.dumps(result["suggestions"]),
             json.dumps(result["templates"]), resume_text[:6000], jd[:2000]))
        db.commit()

        app.config.setdefault("_rc",{})[analysis_id] = {"text":resume_text,"jd":jd}

        return jsonify({"status":"success","analysis_id":analysis_id,
                        "ats_score":result["ats_score"],"match_level":result["match_level"],
                        "message":result["message"],"skill_scores":result["skill_scores"],
                        "matched_keywords":result["matched_keywords"],
                        "missing_keywords":result["missing_keywords"],
                        "tech_matched":result["tech_matched"],"tech_missing":result["tech_missing"],
                        "tech_all_resume":result.get("tech_all_resume",[]),
                        "learn_suggestions":result.get("learn_suggestions",[]),
                        "found_sections":result["found_sections"],
                        "suggestions":result["suggestions"],"templates":result["templates"]})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/generate_resume", methods=["POST"])
def generate_resume():
    try:
        d           = request.json or {}
        analysis_id = d.get("analysis_id","")
        template_id = d.get("template_id","universal_pro")
        email       = d.get("email","anon")

        cache = app.config.get("_rc",{})
        if analysis_id in cache:
            rt = cache[analysis_id]["text"]
        else:
            row = get_db().execute("SELECT resume_text FROM analyses WHERE id=?",
                                   (analysis_id,)).fetchone()
            if not row: return jsonify({"status":"error","message":"Analysis not found. Re-upload."}), 404
            rt = row["resume_text"]

        tpl_meta = next((t for t in TEMPLATES if t["id"]==template_id), TEMPLATES[-1])
        sections = parse_sections(rt)
        resume_dict = {
            "template": template_id,
            "meta":{"color":tpl_meta["color"],"style":"single-column","name":tpl_meta["name"]},
            "sections":{
                "contact":{"name":sections["name"],"email":sections["email"],
                           "phone":sections["phone"],"linkedin":sections["linkedin"],
                           "github":sections["github"]},
                "summary":  sections["summary"] or (sections["experience"][0] if sections["experience"] else ""),
                "skills":   sections["skills"],
                "experience":sections["experience"],
                "education": sections["education"],
                "projects":  sections["projects"],
                "certifications":sections["certifications"],
            }
        }

        pdf_bytes = build_pdf(resume_dict)
        gen_id    = str(uuid.uuid4())
        db = get_db()
        db.execute("INSERT INTO generated_resumes(id,analysis_id,user_email,template_id,content_json,pdf_bytes) VALUES(?,?,?,?,?,?)",
                   (gen_id, analysis_id, email, template_id, json.dumps(resume_dict), pdf_bytes))
        db.commit()
        app.config.setdefault("_gc",{})[gen_id] = pdf_bytes

        return jsonify({"status":"success","gen_id":gen_id,"template":tpl_meta["name"],
                        "resume_content":resume_dict,"download_url":f"/download_resume?id={gen_id}"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500

@app.route("/download_resume", methods=["GET"])
def download_resume():
    gen_id    = request.args.get("id","")
    pdf_bytes = app.config.get("_gc",{}).get(gen_id)
    if not pdf_bytes:
        row = get_db().execute("SELECT pdf_bytes FROM generated_resumes WHERE id=?",
                               (gen_id,)).fetchone()
        if not row: return jsonify({"status":"error","message":"Not found. Please generate first."}), 404
        pdf_bytes = row["pdf_bytes"]
    buf  = io.BytesIO(pdf_bytes if isinstance(pdf_bytes,bytes) else pdf_bytes.encode())
    buf.seek(0)
    mime = "application/pdf" if _RL else "text/plain"
    ext  = "pdf" if _RL else "txt"
    return send_file(buf, mimetype=mime, as_attachment=True,
                     download_name=f"resume_optimized_{gen_id[:8]}.{ext}")

@app.route("/chat", methods=["POST"])
def chat():
    d       = request.json or {}
    message = d.get("message","").strip()
    email   = d.get("email","anon")
    context = d.get("context",{})   # {ats_score, match_level, missing_keywords}
    if not message: return jsonify({"status":"error","message":"Message required"}), 400

    response = ai_chat_response(message, context)

    db = get_db()
    db.execute("INSERT INTO chat_history(user_email,role,message) VALUES(?,?,?)",(email,"user",message))
    db.execute("INSERT INTO chat_history(user_email,role,message) VALUES(?,?,?)",(email,"assistant",response))
    db.commit()

    return jsonify({"status":"success","response":response,"timestamp":datetime.now().isoformat()})

@app.route("/voice", methods=["POST"])
def voice():
    """Accepts voice transcript text, returns AI response for TTS."""
    d       = request.json or {}
    text    = d.get("text","").strip()
    context = d.get("context",{})
    if not text: return jsonify({"status":"error","message":"Voice text required"}), 400
    response = ai_chat_response(text, context)
    return jsonify({"status":"success","query":text,"response":response,
                    "speak":True,"timestamp":datetime.now().isoformat()})

@app.route("/history", methods=["GET"])
def history():
    email = request.args.get("email","anon")
    db    = get_db()
    rows  = db.execute(
        "SELECT id,filename,ats_score,match_level,created_at FROM analyses WHERE user_email=? ORDER BY created_at DESC LIMIT 10",
        (email,)).fetchall()
    return jsonify({"status":"success","analyses":[dict(r) for r in rows]})

@app.route("/save_resume_draft", methods=["POST"])
def save_resume_draft():
    """
    Saves inline-edited resume from the frontend editor.
    Body JSON: { email, template_id, sections:{contact,summary,skills,experience,education,projects,certifications}, gen_id? }
    Returns: { status, gen_id, download_url }
    """
    try:
        d           = request.json or {}
        email       = d.get("email","anon")
        template_id = d.get("template_id","universal_pro")
        sections    = d.get("sections",{})
        existing_id = d.get("gen_id")          # re-save existing draft

        tpl_meta    = next((t for t in TEMPLATES if t["id"]==template_id), TEMPLATES[-1])
        resume_dict = {
            "template": template_id,
            "meta":{"color":tpl_meta["color"],"name":tpl_meta["name"]},
            "sections": sections,
        }
        pdf_bytes = build_pdf(resume_dict)
        gen_id    = existing_id or str(uuid.uuid4())

        db = get_db()
        if existing_id:
            db.execute(
                "UPDATE generated_resumes SET template_id=?,content_json=?,pdf_bytes=?,created_at=? WHERE id=?",
                (template_id,json.dumps(resume_dict),pdf_bytes,datetime.now().isoformat(),existing_id))
        else:
            db.execute(
                "INSERT INTO generated_resumes(id,analysis_id,user_email,template_id,content_json,pdf_bytes) VALUES(?,?,?,?,?,?)",
                (gen_id,"",email,template_id,json.dumps(resume_dict),pdf_bytes))
        db.commit()
        app.config.setdefault("_gc",{})[gen_id] = pdf_bytes
        return jsonify({"status":"success","gen_id":gen_id,"template":tpl_meta["name"],
                        "download_url":f"/download_resume?id={gen_id}"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}),500


@app.route("/scratch_resume", methods=["POST"])
def scratch_resume():
    """
    Parses an uploaded resume file and returns structured sections for the inline editor.
    Body: multipart with file field 'resume'
    Returns: { status, sections }
    """
    try:
        if "resume" not in request.files:
            return jsonify({"status":"error","message":"No file attached"}),400
        file = request.files["resume"]
        try: text = extract_text(file,file.filename)
        except Exception as e: return jsonify({"status":"error","message":f"Could not read: {e}"}),400
        if not text.strip(): return jsonify({"status":"error","message":"File appears empty"}),400
        s = parse_sections(text)
        return jsonify({"status":"success","sections":{
            "contact":{"name":s["name"],"email":s["email"],"phone":s["phone"],"linkedin":s["linkedin"],"github":s["github"]},
            "summary":s["summary"],"skills":s["skills"],
            "experience":s["experience"],"education":s["education"],
            "projects":s["projects"],"certifications":s["certifications"],
        }})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}),500


# ══════════════════════════════════════════════════════════════════════════════
# NEW v5 ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/ats_optimize", methods=["POST"])
def ats_optimize():
    """
    Takes user-provided sections + job_description and returns ATS-boosted
    bullet points, role keywords inserted, and action-verb upgrades.
    Body JSON: { sections, job_description }
    Returns: { status, sections_optimized, keywords_added, score_estimate }
    """
    try:
        d  = request.json or {}
        jd = d.get("job_description","")
        s  = d.get("sections",{})
        if not jd or not s:
            return jsonify({"status":"error","message":"sections and job_description required"}),400

        # Extract JD keywords not already in resume text
        res_text = " ".join([
            s.get("summary",""),
            " ".join(s.get("skills",[])),
            " ".join(s.get("experience",[])),
        ]).lower()
        jd_lower = jd.lower()
        jd_tok   = {w for w in re.findall(r"\b[a-z][a-z0-9+#\-./]{1,}\b", jd_lower)
                    if w not in STOP_WORDS and len(w)>3}
        res_tok  = {w for w in re.findall(r"\b[a-z][a-z0-9+#\-./]{1,}\b", res_text)
                    if w not in STOP_WORDS}
        missing  = sorted(jd_tok - res_tok)[:10]

        # Upgrade experience bullets: add action verb prefix if missing
        ACTION_PREFIXES = ["Delivered","Engineered","Optimised","Achieved","Drove","Scaled",
                           "Automated","Implemented","Led","Reduced","Improved","Built"]
        exp_upgraded = []
        for i, bullet in enumerate(s.get("experience",[])):
            b = str(bullet).strip().lstrip("•-* ")
            first_word = b.split()[0] if b.split() else ""
            if first_word.lower() not in [v.lower() for v in EXP_VERBS]:
                b = f"{ACTION_PREFIXES[i % len(ACTION_PREFIXES)]} {b}"
            exp_upgraded.append(b)

        # Inject top missing keywords into summary if summary exists
        summary = s.get("summary","")
        if missing and summary:
            keyword_phrase = ", ".join(missing[:4])
            if keyword_phrase.lower() not in summary.lower():
                summary = summary.rstrip(".") + f". Proficient with {keyword_phrase}."

        # Build optimized sections
        s_opt = dict(s)
        s_opt["experience"] = exp_upgraded
        s_opt["summary"]    = summary
        # Merge missing tech keywords into skills if not already there
        current_skills_lower = [x.lower() for x in s.get("skills",[])]
        new_skills = [k for k in missing[:6] if k not in current_skills_lower]
        s_opt["skills"] = s.get("skills",[]) + new_skills

        # Estimate score improvement
        score_est = min(99, 55 + len(exp_upgraded)*2 + len(s_opt["skills"])*1.5 + (20 if summary else 0))

        return jsonify({
            "status":            "success",
            "sections_optimized": s_opt,
            "keywords_added":     new_skills,
            "bullets_upgraded":   len(exp_upgraded),
            "score_estimate":     round(score_est,1),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}),500


@app.route("/preview_html", methods=["POST"])
def preview_html():
    """
    Returns a complete standalone A4 HTML resume preview for the given sections+template.
    Body JSON: { template_id, sections }
    Returns: { status, html }
    """
    try:
        d           = request.json or {}
        template_id = d.get("template_id","universal_pro")
        sections    = d.get("sections",{})
        tpl_meta    = next((t for t in TEMPLATES if t["id"]==template_id), TEMPLATES[-1])
        color       = tpl_meta.get("color","#1e3a5f")
        html        = _build_preview_html(sections, template_id, color, tpl_meta["name"])
        return jsonify({"status":"success","html":html})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}),500


def _esc(t):
    return str(t or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


def _build_preview_html(sec, template_id, color, tpl_name):
    """Build a self-contained A4 HTML string for live preview. Template-aware layout."""
    c  = sec.get("contact",{})
    is_two_col = template_id in ("tech_modern","fullstack_pro")
    is_dark_hdr= template_id in ("data_science","biz_finance")
    is_stripe  = template_id in ("biz_marketing","creative_media","creative_modern")
    is_fresher = template_id in ("fresher_classic","fresher_mba","fresher_tech")
    is_exec    = template_id in ("biz_executive","biz_sales")

    skills_html = "".join(
        f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
        f'font-size:11px;font-weight:600;background:{color}18;color:{color};'
        f'border:1px solid {color}40;margin:2px">{_esc(sk)}</span>'
        for sk in sec.get("skills",[])[:20]
    )
    exp_html = "".join(
        f'<div style="margin-bottom:6px;font-size:12.5px;padding-left:14px;position:relative;line-height:1.5;color:#2d2d2d">'
        f'<span style="position:absolute;left:0;color:{color}">▸</span>{_esc(e)}</div>'
        for e in sec.get("experience",[])[:8]
    )
    edu_html = "".join(
        f'<div style="font-size:12.5px;margin-bottom:5px;color:#2d2d2d">{_esc(e)}</div>'
        for e in sec.get("education",[])[:4]
    )
    proj_html = "".join(
        f'<div style="margin-bottom:5px;font-size:12.5px;padding-left:14px;position:relative;color:#2d2d2d">'
        f'<span style="position:absolute;left:0;color:{color}">▸</span>{_esc(p)}</div>'
        for p in sec.get("projects",[])[:4]
    )
    cert_html = "".join(
        f'<div style="font-size:12px;margin-bottom:3px;color:#444">{_esc(ct)}</div>'
        for ct in sec.get("certifications",[])[:4]
    )

    def sec_title(label):
        return (f'<div style="font-size:10px;font-weight:800;text-transform:uppercase;'
                f'letter-spacing:.1em;color:{color};padding-bottom:4px;'
                f'border-bottom:2px solid {color};margin:16px 0 8px">{label}</div>')

    contact_line = " &nbsp;|&nbsp; ".join(
        _esc(x) for x in [c.get("email"),c.get("phone"),c.get("linkedin"),c.get("github")] if x
    )

    # ── TWO-COLUMN layout ──────────────────────────────────────────────────────
    if is_two_col:
        sbg = "#064e3b" if template_id == "fullstack_pro" else "#0f172a"
        sidebar_skills = "".join(
            f'<div style="font-size:10.5px;margin-bottom:5px;padding:4px 9px;'
            f'background:{color}22;border-radius:8px;color:#e2e8f0;'
            f'word-break:break-word;overflow-wrap:break-word">{_esc(sk)}</div>'
            for sk in sec.get("skills",[])[:16]
        )
        sidebar_certs = "".join(
            f'<div style="font-size:10px;margin-bottom:4px;color:#94a3b8;'
            f'word-break:break-word;line-height:1.4">✓ {_esc(ct)}</div>'
            for ct in sec.get("certifications",[])[:4]
        )
        sidebar_contact = "".join(
            f'<div style="font-size:10px;margin-bottom:4px;color:#94a3b8;'
            f'word-break:break-all;overflow-wrap:break-all">{_esc(x)}</div>'
            for x in [c.get("email"),c.get("phone"),c.get("linkedin"),c.get("github")] if x
        )
        main_body = ""
        if sec.get("summary"):
            main_body += (sec_title("Professional Summary") +
                          f'<p style="font-size:12px;color:#334155;line-height:1.6;margin-bottom:6px">{_esc(sec["summary"])}</p>')
        if sec.get("experience"):  main_body += sec_title("Experience") + exp_html
        if sec.get("projects"):    main_body += sec_title("Projects")    + proj_html
        if sec.get("education"):   main_body += sec_title("Education")   + edu_html

        sidebar_certs_section = (
            f'<div style="font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;'
            f'color:{color};margin:14px 0 6px;border-bottom:1px solid {color}44;padding-bottom:3px">Certifications</div>'
            + sidebar_certs
        ) if sidebar_certs else ""

        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:Arial,sans-serif;background:#f4f4f4;display:flex;justify-content:center;padding:20px}}
.a4{{width:210mm;min-height:297mm;background:#fff;box-shadow:0 4px 24px rgba(0,0,0,.18);
     display:flex;flex-direction:row;overflow:hidden}}
.sidebar{{width:160px;min-width:160px;max-width:160px;background:{sbg};padding:24px 14px;
          flex-shrink:0;min-height:297mm;overflow:hidden}}
.maincol{{flex:1;padding:24px 28px;overflow:hidden;min-width:0}}
</style></head>
<body><div class="a4">
<div class="sidebar">
  <div style="font-size:16px;font-weight:800;color:{color};margin-bottom:6px;
              line-height:1.25;word-break:break-word">{_esc(c.get("name","Your Name"))}</div>
  <div style="height:2px;background:{color};margin:8px 0"></div>
  {sidebar_contact}
  <div style="font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;
              color:{color};margin:14px 0 6px;border-bottom:1px solid {color}44;padding-bottom:3px">Skills</div>
  {sidebar_skills}
  {sidebar_certs_section}
</div>
<div class="maincol">{main_body}</div>
</div></body></html>"""

    # ── EXECUTIVE layout ───────────────────────────────────────────────────────
    if is_exec:
        competency_rows = ""
        skills = sec.get("skills",[])
        for i in range(0, min(len(skills),12), 3):
            row = skills[i:i+3]
            competency_rows += ("<tr>" +
                "".join(f'<td style="padding:3px 6px;font-size:12px;color:#333">{_esc(sk)}</td>' for sk in row) +
                "</tr>")
        main_body = ""
        if sec.get("summary"):
            main_body += (sec_title("Executive Summary") +
                f'<p style="font-size:13px;color:#333;line-height:1.7;margin-bottom:6px">{_esc(sec["summary"])}</p>')
        if skills:
            main_body += (sec_title("Core Competencies") +
                f'<table style="width:100%;border-collapse:collapse">{competency_rows}</table>')
        if sec.get("experience"):  main_body += sec_title("Professional Experience") + exp_html
        if sec.get("education"):   main_body += sec_title("Education") + edu_html
        if sec.get("certifications"): main_body += sec_title("Awards & Certifications") + cert_html

        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Plus Jakarta Sans',Arial,sans-serif;background:#f4f4f4;display:flex;justify-content:center;padding:20px}}
.a4{{width:210mm;min-height:297mm;background:#fff;box-shadow:0 4px 24px rgba(0,0,0,.18);padding:40px 50px}}</style></head>
<body><div class="a4">
  <div style="text-align:center;border-bottom:4px solid {color};padding-bottom:18px;margin-bottom:4px">
    <div style="font-size:26px;font-weight:800;color:{color};letter-spacing:.03em">{_esc(c.get("name","Your Name"))}</div>
    <div style="font-size:12px;color:#666;margin-top:5px">{contact_line}</div>
  </div>
  {main_body}
</div></body></html>"""

    # ── STRIPE layout ──────────────────────────────────────────────────────────
    if is_stripe:
        main_body = ""
        if sec.get("summary"):
            main_body += (sec_title("Profile") +
                f'<p style="font-size:12.5px;color:#333;line-height:1.6">{_esc(sec["summary"])}</p>')
        if sec.get("skills"):
            main_body += sec_title("Expertise") + f'<div style="margin-bottom:6px">{skills_html}</div>'
        if sec.get("experience"):  main_body += sec_title("Experience") + exp_html
        if sec.get("projects"):    main_body += sec_title("Projects")   + proj_html
        if sec.get("education"):   main_body += sec_title("Education")  + edu_html
        if sec.get("certifications"): main_body += sec_title("Certifications") + cert_html

        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Plus Jakarta Sans',Arial,sans-serif;background:#f4f4f4;display:flex;justify-content:center;padding:20px}}
.a4{{width:210mm;min-height:297mm;background:#fff;box-shadow:0 4px 24px rgba(0,0,0,.18);display:grid;grid-template-columns:12px 1fr}}</style></head>
<body><div class="a4">
<div style="background:{color};min-height:297mm"></div>
<div style="padding:28px 32px">
  <div style="margin-bottom:16px;border-bottom:1px solid #e5e7eb;padding-bottom:12px">
    <div style="font-size:22px;font-weight:800;color:{color}">{_esc(c.get("name","Your Name"))}</div>
    <div style="font-size:11.5px;color:#666;margin-top:4px">{contact_line}</div>
  </div>
  {main_body}
</div></div></body></html>"""

    # ── FRESHER layout ─────────────────────────────────────────────────────────
    if is_fresher:
        edu_boxed = "".join(
            f'<div style="background:{color}0d;border-left:3px solid {color};'
            f'padding:8px 12px;margin-bottom:6px;border-radius:0 8px 8px 0;font-size:12.5px;color:#333">{_esc(e)}</div>'
            for e in sec.get("education",[])[:3]
        )
        main_body = edu_boxed
        if sec.get("summary"):
            main_body += (sec_title("Objective") +
                f'<p style="font-size:12.5px;color:#333;line-height:1.6">{_esc(sec["summary"])}</p>')
        if sec.get("skills"):   main_body += sec_title("Skills") + f'<div>{skills_html}</div>'
        if sec.get("experience"): main_body += sec_title("Internships / Experience") + exp_html
        if sec.get("projects"):   main_body += sec_title("Academic Projects") + proj_html
        if sec.get("certifications"): main_body += sec_title("Achievements") + cert_html

        return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Plus Jakarta Sans',Arial,sans-serif;background:#f4f4f4;display:flex;justify-content:center;padding:20px}}
.a4{{width:210mm;min-height:297mm;background:#fff;box-shadow:0 4px 24px rgba(0,0,0,.18);padding:32px 40px}}</style></head>
<body><div class="a4">
  <div style="text-align:center;border-bottom:3px solid {color};padding-bottom:14px;margin-bottom:4px">
    <div style="font-size:22px;font-weight:800;color:{color}">{_esc(c.get("name","Your Name"))}</div>
    <div style="font-size:11.5px;color:#666;margin-top:4px">{contact_line}</div>
  </div>
  <div style="font-size:10.5px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{color};margin:14px 0 8px;border-bottom:2px solid {color};padding-bottom:4px">Education</div>
  {main_body}
</div></body></html>"""

    # ── DEFAULT / MINIMAL layout ───────────────────────────────────────────────
    main_body = ""
    if sec.get("summary"):
        main_body += (sec_title("Professional Summary") +
            f'<p style="font-size:12.5px;color:#333;line-height:1.6;margin-bottom:4px">{_esc(sec["summary"])}</p>')
    if sec.get("skills"):
        main_body += sec_title("Skills") + f'<div style="margin-bottom:4px">{skills_html}</div>'
    if sec.get("experience"):    main_body += sec_title("Experience") + exp_html
    if sec.get("education"):     main_body += sec_title("Education")  + edu_html
    if sec.get("projects"):      main_body += sec_title("Projects")   + proj_html
    if sec.get("certifications"): main_body += sec_title("Certifications") + cert_html

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Plus Jakarta Sans',Arial,sans-serif;background:#f4f4f4;display:flex;justify-content:center;padding:20px}}
.a4{{width:210mm;min-height:297mm;background:#fff;box-shadow:0 4px 24px rgba(0,0,0,.18);padding:32px 40px}}</style></head>
<body><div class="a4">
  <div style="border-bottom:2px solid {color};padding-bottom:14px;margin-bottom:2px">
    <div style="font-size:22px;font-weight:800;color:{color};margin-bottom:4px">{_esc(c.get("name","Your Name"))}</div>
    <div style="font-size:11.5px;color:#666">{contact_line}</div>
  </div>
  {main_body}
</div></body></html>"""


@app.route("/auto_fill", methods=["POST"])
def auto_fill():
    """
    FIX 1: Auto-fill empty/sparse resume sections with ATS-optimized content.
    Body JSON: { sections, job_description, template_id }
    Returns: { status, sections_filled, fill_log }
    """
    try:
        d = request.json or {}
        sec = d.get("sections", {})
        jd  = d.get("job_description", "") or ""
        tpl = d.get("template_id", "universal_pro")

        # Safe jd_lower used throughout this function
        jd_lower = jd.lower()
        j_lower  = jd_lower   # alias kept for role-detection block below
        fill_log = []

        # ── Detect role from resume content + JD (comprehensive multi-signal) ──
        def _auto_fill_detect_role(sec_data, jd_text):
            all_resume = " ".join([
                str(sec_data.get("summary", "")),
                " ".join(sec_data.get("skills", [])),
                " ".join(sec_data.get("experience", [])),
                str(sec_data.get("job_title", "")),
            ]).lower()
            combined = all_resume + " " + jd_text.lower()
            ROLE_SIGNALS = {
                "QA / Test Engineer": ("selenium","automation testing","test cases","bug tracking","defect","qa engineer","quality assurance","regression testing","test plan","sdet","manual testing"),
                "HR / Human Resources": ("recruitment","talent acquisition","onboarding","employee relations","payroll","hr operations","hris","performance management","hr manager","human resources","workforce"),
                "Digital Marketing": ("seo","sem","google ads","social media marketing","content marketing","email marketing","campaign","lead generation","google analytics","meta ads","ppc","hubspot"),
                "UI/UX Designer": ("figma","wireframe","prototyping","user research","usability testing","design system","ux designer","ui designer","adobe xd","interaction design","user journey"),
                "Cybersecurity": ("penetration testing","vulnerability assessment","siem","soc","threat analysis","firewall","ethical hacking","cybersecurity","information security","kali linux","incident response"),
                "Data Analyst": ("sql","power bi","tableau","excel","data visualization","dashboard","data analyst","business intelligence","etl","kpi","statistical analysis","google data studio"),
                "Data Scientist / ML Engineer": ("machine learning","deep learning","neural network","tensorflow","pytorch","scikit-learn","data science","nlp","computer vision","xgboost","mlops","generative ai"),
                "DevOps / Cloud Engineer": ("kubernetes","docker","terraform","jenkins","ci/cd","github actions","aws","devops","site reliability","infrastructure as code","helm","ansible","cloudformation"),
                "Banking / Finance": ("banking","financial operations","compliance","kyc","aml","loan processing","credit analysis","treasury","risk management","financial analyst","cfa","accounting","gaap"),
                "Teacher / Educator": ("lesson plan","classroom management","curriculum","student engagement","pedagogy","teacher","educator","tutor","e-learning","learning objectives","lms","moodle"),
                "Nurse / Healthcare": ("patient care","clinical","nursing","healthcare","medication administration","ehr","emr","triage","registered nurse","bls","acls","hospital","clinical support"),
                "Civil / Mechanical Engineer": ("autocad","structural","civil engineering","construction","site supervision","mechanical engineering","cad","revit","hvac","manufacturing","piping","fabrication"),
                "Sales / Business Development": ("sales","business development","crm","lead generation","cold calling","revenue","quota","account management","b2b","b2c","salesforce","deal closure"),
                "Product Manager": ("product roadmap","product manager","go-to-market","user stories","okr","product strategy","backlog","stakeholder","product owner","mvp","feature prioritization"),
                "Graphic Designer": ("adobe photoshop","illustrator","indesign","graphic design","branding","typography","logo design","visual identity","canva","color theory","creative brief"),
                "Software Engineer": ("software engineer","backend","frontend","full stack","api development","rest api","microservices","node.js","react","django","flask","spring boot","java","python developer"),
            }
            scores = {r: sum(1 for kw in kws if kw in combined) for r, kws in ROLE_SIGNALS.items()}
            best = max(scores, key=scores.get)
            return best if scores[best] > 0 else "Software Engineer"

        role = _auto_fill_detect_role(sec, jd)

        # ── 1. Professional Summary ──────────────────────────────
        SUMMARIES = {
            "QA / Test Engineer": "Detail-oriented QA / Test Engineer with proven expertise in designing and executing comprehensive test strategies. Skilled in Selenium automation, JIRA defect tracking, regression testing, and Agile QA workflows. Consistently delivered defect-free releases through rigorous quality assurance practices.",
            "HR / Human Resources": "Dynamic HR Professional with strong experience in talent acquisition, onboarding, employee relations, and HR operations. Skilled in HRIS management, payroll coordination, and performance management cycles. Committed to building engaged, high-performing teams aligned with organizational goals.",
            "Digital Marketing": "Results-driven Digital Marketing Specialist with expertise in SEO, Google Ads, social media management, and content strategy. Proven track record in growing organic traffic, improving campaign ROI, and driving lead generation through data-backed multi-channel strategies.",
            "UI/UX Designer": "Creative UI/UX Designer with expertise in Figma, user research, wireframing, and high-fidelity prototyping. Passionate about delivering intuitive, accessible digital experiences that meet user needs and business goals. Experienced in usability testing and design systems.",
            "Cybersecurity": "Motivated Cybersecurity professional with hands-on experience in vulnerability assessment, penetration testing, and SOC monitoring. Proficient in SIEM tools and ethical hacking methodologies. Committed to identifying and mitigating threats across enterprise environments.",
            "Data Analyst": "Analytical Data Analyst with strong SQL, Power BI, Tableau, and Excel skills for transforming data into actionable business intelligence. Experienced in building dashboards, conducting statistical analyses, and delivering insights that support strategic decision-making.",
            "Data Scientist / ML Engineer": "Innovative Data Scientist proficient in Python, TensorFlow, and scikit-learn for building production-grade machine learning models. Experienced across the full ML lifecycle — from data preprocessing and feature engineering to model deployment and performance monitoring.",
            "DevOps / Cloud Engineer": "Cloud-native DevOps Engineer with expertise in CI/CD automation, Kubernetes, Docker, and Terraform. Proven ability to improve deployment frequency, system reliability, and observability. Advocate for DevSecOps culture and infrastructure as code best practices.",
            "Banking / Finance": "Detail-oriented Finance / Banking professional with strong expertise in financial operations, KYC/AML compliance, credit analysis, and regulatory reporting. Proficient in financial modeling, reconciliation, and banking software. Committed to accuracy and audit-ready financial management.",
            "Teacher / Educator": "Dedicated Educator with proven experience in curriculum development, lesson planning, and student-centered instruction. Skilled at adapting teaching methods to diverse learning styles, managing classrooms effectively, and leveraging e-learning tools for improved outcomes.",
            "Nurse / Healthcare": "Compassionate Nursing professional with hands-on experience in patient care, clinical assessment, and healthcare documentation. Proficient in medication administration, vital signs monitoring, and collaborating with multidisciplinary teams to deliver high-quality, patient-centered care.",
            "Civil / Mechanical Engineer": "Skilled Civil/Mechanical Engineer with hands-on project experience in design, site supervision, and quality control. Proficient in AutoCAD, STAAD Pro, and construction management. Detail-oriented professional committed to delivering projects safely, on time, and within budget.",
            "Sales / Business Development": "Results-oriented Sales & Business Development professional skilled in lead generation, client relationship management, and deal closure. Experienced with CRM tools, B2B/B2C strategies, and pipeline management. Consistently exceeded revenue targets through consultative selling.",
            "Product Manager": "Strategic Product Manager with a track record of launching digital products through data-driven prioritization and cross-functional leadership. Expert in Agile/Scrum, OKR frameworks, and go-to-market execution. Passionate about building user-centric products that deliver measurable business value.",
            "Graphic Designer": "Creative Graphic Designer with expertise in brand identity, visual communication, and digital design. Proficient in Adobe Photoshop, Illustrator, InDesign, and Canva. Experienced in producing compelling print and digital assets aligned to brand briefs and campaign objectives.",
            "Software Engineer": "Results-driven Software Engineer with a strong foundation in full-stack development, API design, and cloud-native applications. Experienced in delivering scalable solutions that improve system performance. Passionate about clean code architecture, Agile practices, and continuous learning.",
        }
        if not sec.get("summary") or len(str(sec.get("summary","")).strip()) < 50:
            sec["summary"] = SUMMARIES.get(role, (
                "Dedicated professional with strong domain expertise and a commitment to delivering high-quality results. "
                "Experienced in collaborating across teams, solving complex challenges, and contributing meaningfully to organizational goals."
            ))
            fill_log.append(f"✓ Professional Summary generated for {role}")

        # ── 1b. Enhance existing weak summary ──────────────────
        elif len(str(sec.get("summary","")).strip()) < 200:
            existing = str(sec["summary"]).strip()
            role_prefix = {
                "QA / Test Engineer": "Experienced QA professional.",
                "HR / Human Resources": "Skilled HR professional.",
                "Digital Marketing": "Data-driven marketing specialist.",
                "UI/UX Designer": "Creative UI/UX designer.",
                "Cybersecurity": "Security-focused professional.",
                "Data Analyst": "Analytical data professional.",
                "Software Engineer": "Results-driven engineer.",
            }
            prefix = role_prefix.get(role, "Dedicated professional.")
            sec["summary"] = f"{prefix} {existing} Committed to excellence, continuous improvement, and delivering measurable impact in every project undertaken."
            fill_log.append("✓ Summary enhanced with professional language")

        # ── 2. Skills expansion ──────────────────────────────────
        SKILLS_MAP = {
            "QA / Test Engineer": ["Selenium WebDriver","Manual Testing","Automation Testing","TestNG","JUnit","JIRA / Bugzilla","Test Case Design","Regression Testing","API Testing (Postman)","Agile QA","Defect Tracking","SQL (DB Testing)","Cucumber / BDD","Smoke Testing","Functional Testing"],
            "HR / Human Resources": ["Talent Acquisition","End-to-End Recruitment","Onboarding & Induction","Employee Relations","Payroll Processing","HRIS (SAP / Workday)","Performance Appraisal","Labor Law Compliance","Workforce Planning","Employee Engagement","HR Operations","MIS Reporting"],
            "Digital Marketing": ["SEO / On-page & Off-page","Google Ads","Meta Ads","Google Analytics 4","Content Marketing","Email Marketing","Social Media Management","Lead Generation","HubSpot","Keyword Research","CRO","SEM","Marketing Funnels","Canva"],
            "UI/UX Designer": ["Figma","Adobe XD","Wireframing","Prototyping","User Research","Usability Testing","Design Systems","Information Architecture","User Journey Mapping","Accessibility (WCAG)","Sketch","A/B Testing","Typography","Responsive Design"],
            "Cybersecurity": ["Penetration Testing","Vulnerability Assessment","SIEM (Splunk / QRadar)","SOC Monitoring","Network Security","Kali Linux","Nessus / Nmap","Metasploit","Incident Response","OWASP Top 10","Firewall Configuration","Threat Intelligence","CEH"],
            "Data Analyst": ["SQL (Advanced)","Power BI","Tableau","Excel (Advanced)","Python (Pandas / NumPy)","Data Visualization","ETL Pipelines","Dashboard Development","Statistical Analysis","KPI Tracking","Google Analytics","Data Cleaning","Business Intelligence"],
            "Data Scientist / ML Engineer": ["Python","TensorFlow / PyTorch","Scikit-learn","Machine Learning","Deep Learning","NLP","Feature Engineering","Model Deployment","SQL","Pandas / NumPy","MLOps","XGBoost / LightGBM","Computer Vision","A/B Testing"],
            "DevOps / Cloud Engineer": ["AWS / Azure / GCP","Kubernetes","Docker","Terraform","Jenkins","GitHub Actions","CI/CD Pipelines","Linux / Bash","Ansible","Helm","Prometheus / Grafana","ELK Stack","Infrastructure as Code","DevSecOps"],
            "Banking / Finance": ["Financial Operations","KYC / AML Compliance","Credit Analysis","Loan Processing","Risk Management","Financial Modeling (Excel)","Regulatory Reporting","Reconciliation","Banking Software","Portfolio Analysis","GAAP / IFRS","Treasury Operations"],
            "Teacher / Educator": ["Lesson Planning","Curriculum Development","Classroom Management","Student Assessment","Differentiated Instruction","Google Classroom","Moodle / Canvas","Bloom's Taxonomy","Formative Assessment","Student Engagement","Educational Technology"],
            "Nurse / Healthcare": ["Patient Care & Assessment","Medication Administration","Vital Signs Monitoring","EHR / EMR Systems","BLS / ACLS","Clinical Documentation","Wound Care","Infection Control","Patient Education","ICU / Ward Care","Triage","Healthcare Compliance"],
            "Civil / Mechanical Engineer": ["AutoCAD","STAAD Pro / ETABS","Site Supervision","Structural Design","Construction Management","BOQ Estimation","Quality Control","Project Scheduling","Safety Compliance","Revit / BIM","HVAC Systems","MS Project"],
            "Sales / Business Development": ["B2B / B2C Sales","Lead Generation","CRM (Salesforce / HubSpot)","Cold Calling","Client Relationship Management","Deal Negotiation","Revenue Forecasting","Pipeline Management","Upselling / Cross-selling","Market Research","Key Account Management"],
            "Product Manager": ["Product Roadmapping","Agile / Scrum","JIRA / Confluence","User Story Writing","OKR & KPI Frameworks","Go-to-Market Strategy","Stakeholder Management","Product Analytics","A/B Testing","Competitive Analysis","SQL","Backlog Prioritization"],
            "Graphic Designer": ["Adobe Photoshop","Adobe Illustrator","Adobe InDesign","Canva","Brand Identity Design","Typography","Color Theory","Logo Design","Print & Digital Design","Social Media Graphics","Motion Graphics","Packaging Design"],
            "Software Engineer": ["Python","JavaScript","TypeScript","React","Node.js","REST APIs","Docker","AWS","PostgreSQL","Git","Agile","Microservices","CI/CD","Unit Testing","Redis"],
        }
        current_skills = sec.get("skills", [])
        if len(current_skills) < 8:
            role_skills = SKILLS_MAP.get(role, SKILLS_MAP["Software Engineer"])
            jd_tokens = set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#\-./]{1,}\b', jd_lower))
            jd_skills = [s for s in role_skills if any(t in s.lower() for t in jd_tokens)]
            other_skills = [s for s in role_skills if s not in jd_skills]
            merged = list(dict.fromkeys(current_skills + jd_skills + other_skills))
            sec["skills"] = merged[:16]
            fill_log.append(f"✓ Skills expanded to {len(sec['skills'])} {role}-specific keywords")

        # ── 3. Experience with metrics ────────────────────────────
        EXP_MAP = {
            "QA / Test Engineer": [
                "Designed and executed 200+ test cases (functional, regression, smoke) for web and mobile applications",
                "Built Selenium + TestNG automation framework reducing manual regression effort by 60%",
                "Tracked and resolved 150+ defects in JIRA, coordinating with developers to ensure timely resolution",
                "Performed API testing using Postman, validating 50+ endpoints for data accuracy and response times",
                "Collaborated with cross-functional Agile teams in sprint ceremonies, contributing to zero critical post-release bugs",
            ],
            "HR / Human Resources": [
                "Managed end-to-end recruitment for 40+ positions across engineering, sales, and operations functions",
                "Designed and delivered structured onboarding program reducing new-hire ramp-up time by 30%",
                "Administered monthly payroll for 200+ employees ensuring 100% accuracy and statutory compliance",
                "Conducted quarterly performance appraisal cycles and facilitated calibration meetings for 3 departments",
                "Implemented employee engagement initiatives reducing voluntary attrition by 18% year-over-year",
            ],
            "Digital Marketing": [
                "Executed SEO strategy increasing organic traffic by 120% and improving keyword rankings for 50+ target terms",
                "Managed Google Ads campaigns with ₹4L monthly budget, achieving 3.5× ROAS through audience optimization",
                "Grew brand social media presence by 35K followers in 6 months via content calendar and influencer partnerships",
                "Designed email marketing sequences with 32% open rate and 8% CTR, surpassing industry benchmarks",
                "Produced weekly analytics reports using Google Analytics 4, identifying top-performing channels for budget reallocation",
            ],
            "UI/UX Designer": [
                "Led end-to-end UX design for mobile banking app redesign, improving task completion rate by 40%",
                "Conducted 15+ user interviews and synthesized findings into actionable design improvements validated by A/B testing",
                "Built reusable Figma component library used by 3 product teams, reducing design-to-dev handoff time by 50%",
                "Facilitated usability testing sessions, identified 10+ critical pain points, and iterated designs to resolve them",
                "Collaborated with developers in Agile sprints to ensure pixel-perfect implementation of design specifications",
            ],
            "Cybersecurity": [
                "Performed OWASP Top 10 web application penetration testing for 5 client environments, reporting critical vulnerabilities before go-live",
                "Monitored SOC alerts using Splunk SIEM, triaging 100+ events daily and escalating 12 confirmed incidents",
                "Conducted vulnerability assessments using Nessus, producing detailed reports with prioritized remediation steps",
                "Developed security awareness training program for 150+ employees, reducing phishing simulation click-through by 65%",
                "Assisted in incident response for ransomware containment, restoring operations within 4 hours of detection",
            ],
            "Data Analyst": [
                "Built interactive Power BI dashboard for regional sales teams, reducing manual reporting time from 8 hours to 25 minutes/week",
                "Designed SQL queries to extract, clean, and transform 2M+ transaction records for monthly performance reporting",
                "Conducted A/B analysis on checkout funnel, identifying UX bottleneck that increased conversion rate by 19%",
                "Automated Excel-based KPI tracker saving finance team 6 hours/week in manual data consolidation",
                "Presented data-driven business intelligence reports to senior leadership, influencing ₹1Cr+ budget allocation decisions",
            ],
            "Software Engineer": [
                "Designed and deployed RESTful APIs using FastAPI/Python serving 50K+ daily requests with 99.9% uptime",
                "Reduced page load time by 42% through code splitting, lazy loading, and CDN optimization",
                "Led migration of monolithic backend to microservices architecture, improving deployment frequency by 3×",
                "Implemented automated testing suite increasing code coverage from 45% to 87%",
                "Collaborated with cross-functional teams in Agile sprints to deliver 12 product features in Q2",
            ],
        }
        if len(sec.get("experience", [])) < 3:
            new_exp = EXP_MAP.get(role, EXP_MAP["Software Engineer"])
            sec["experience"] = (sec.get("experience", []) + new_exp)[:6]
            fill_log.append(f"✓ Experience bullets enriched with {role}-specific metrics")

        # ── 4. Projects (auto-generate if missing) ───────────────
        PROJECTS_MAP = {
            "QA / Test Engineer": [
                "Automation Test Suite — Selenium + TestNG framework for e-commerce portal · Reduced manual regression effort by 65%",
                "API Testing Framework — Postman + Newman REST API test suite integrated with CI/CD for nightly runs",
                "Bug Tracking Dashboard — Configured JIRA workflow for defect lifecycle · Improved closure rate by 30%",
            ],
            "HR / Human Resources": [
                "Campus Recruitment Drive — Coordinated hiring for 50+ positions across 8 colleges · 95% offer acceptance rate",
                "Digital Onboarding Portal — Zoho HR workflow design · Reduced new-hire paperwork time by 40%",
                "Employee Engagement Survey — 300+ employee survey · Identified top 5 attrition drivers for policy revision",
            ],
            "Digital Marketing": [
                "SEO Campaign — On-page & off-page strategy for B2B client · Increased organic traffic by 140% in 6 months",
                "Google Ads Campaign — Managed ₹5L/month PPC budget · Improved ROAS by 3.2× via A/B ad copy testing",
                "Social Media Growth — Grew brand Instagram from 2K to 25K followers in 4 months via content + influencer strategy",
            ],
            "UI/UX Designer": [
                "Mobile Banking App Redesign — Led UX overhaul (50K+ users) · Reduced task completion time by 35%",
                "E-Commerce Design System — Reusable Figma component library for 3 teams · Cut design-dev handoff by 50%",
                "Usability Study — 12-participant moderated testing · Identified 8 critical pain points for next sprint",
            ],
            "Cybersecurity": [
                "Web App Pentest — OWASP Top 10 assessment for fintech client · Discovered 4 critical vulnerabilities pre-launch",
                "SIEM Dashboard — Deployed Splunk for network monitoring · Reduced MTTD by 45%",
                "Security Awareness Training — Phishing simulation for 200+ employees · Reduced click-through rate by 70%",
            ],
            "Data Analyst": [
                "Sales Intelligence Dashboard — Power BI dashboard · Reduced weekly reporting from 8 hours to 25 minutes",
                "Customer Churn Analysis — SQL + Python analysis · Insights reduced monthly churn by 18%",
                "Inventory Optimization Report — 3-year supply chain analysis · Saved ₹12L in excess stock costs annually",
            ],
            "Software Engineer": [
                "ResumeAI Pro — AI-powered resume optimizer (Python/Flask, React, SQLite) · 95% ATS improvement rate",
                "TaskFlow — Real-time project management SaaS (Node.js, Socket.IO, PostgreSQL) · 200+ active users",
                "DevMetrics Dashboard — Infrastructure monitoring tool (Grafana, Prometheus, Kubernetes) · 4-hour MTTR reduction",
            ],
        }
        if len(sec.get("projects", [])) < 2:
            fallback_projects = [
                f"Domain Project 1 — Applied {role} skills to solve real-world business problem · Delivered measurable results",
                f"Domain Project 2 — End-to-end {role} project showcasing technical and analytical expertise",
            ]
            sec["projects"] = (sec.get("projects", []) + PROJECTS_MAP.get(role, fallback_projects))[:4]
            fill_log.append(f"✓ Projects section auto-generated for {role}")

        # ── 5. Certifications ─────────────────────────────────────
        CERTS_MAP = {
            "QA / Test Engineer": ["ISTQB Foundation Level Certified Tester","Selenium WebDriver with Java – Udemy","API Testing with Postman – Coursera"],
            "HR / Human Resources": ["SHRM-CP (Society for Human Resource Management)","Diploma in Human Resource Management – NIPM","Recruitment & Talent Acquisition – LinkedIn Learning"],
            "Digital Marketing": ["Google Ads Search Certification","HubSpot Inbound Marketing Certification","Google Analytics Individual Qualification (GAIQ)"],
            "UI/UX Designer": ["Google UX Design Professional Certificate – Coursera","Figma UI UX Design Essentials – Udemy","Nielsen Norman Group UX Certificate"],
            "Cybersecurity": ["Certified Ethical Hacker (CEH) – EC-Council","CompTIA Security+","Google Cybersecurity Professional Certificate – Coursera"],
            "Data Analyst": ["Google Data Analytics Certificate – Coursera","Microsoft Power BI Data Analyst (PL-300)","Tableau Desktop Specialist"],
            "Data Scientist / ML Engineer": ["IBM Data Science Professional Certificate – Coursera","Deep Learning Specialization – DeepLearning.AI","AWS Certified Machine Learning – Specialty"],
            "DevOps / Cloud Engineer": ["AWS Solutions Architect – Associate","Certified Kubernetes Administrator (CKA)","HashiCorp Certified: Terraform Associate"],
            "Banking / Finance": ["CFA Level I – CFA Institute","Certified Financial Planner (CFP)","NISM Series-V-A: Mutual Fund Distributors Certification"],
            "Teacher / Educator": ["B.Ed (Bachelor of Education)","Google Certified Educator Level 1","Learning How to Learn – McMaster University (Coursera)"],
            "Nurse / Healthcare": ["BLS & ACLS Certification – American Heart Association","Registered Nurse (RN) License","Critical Care Nursing Certificate"],
            "Civil / Mechanical Engineer": ["AutoCAD Certified Professional","Project Management Professional (PMP) – PMI","STAAD Pro Structural Analysis – Bentley"],
            "Sales / Business Development": ["Salesforce Certified Sales Cloud Consultant","HubSpot Sales Software Certification","Strategic Sales Management – Coursera"],
            "Product Manager": ["Certified Scrum Product Owner (CSPO) – Scrum Alliance","Product Management Certificate – Pragmatic Institute","Product-Led Growth Certification – Product-Led Institute"],
            "Graphic Designer": ["Adobe Certified Professional – Visual Design","Graphic Design Specialization – CalArts (Coursera)","Canva Design School Certification"],
            "Software Engineer": ["AWS Certified Developer – Associate (2024)","MongoDB University: M001 Basics (2023)","Meta Front-End Developer Certificate – Coursera (2023)"],
        }
        if len(sec.get("certifications", [])) < 2:
            sec["certifications"] = (sec.get("certifications", []) + CERTS_MAP.get(role, CERTS_MAP["Software Engineer"]))[:4]
            fill_log.append(f"✓ Certifications added for {role} career path")

        # ── 6. Career Objective / Value Section ──────────────────
        VALUE_MAP = {
            "QA / Test Engineer": "What I Bring · I champion quality at every stage of the SDLC. I help teams ship defect-free software faster through systematic automation, clear test documentation, and proactive defect prevention.",
            "HR / Human Resources": "What I Bring · I connect the right people to the right roles and build cultures where talent thrives. I improve organizational outcomes through strategic recruitment, structured onboarding, and data-driven HR practices.",
            "Digital Marketing": "What I Bring · I translate data into growth. I help businesses maximize ROI through targeted campaigns, optimized content, and performance-driven marketing strategies aligned to business objectives.",
            "UI/UX Designer": "What I Bring · I design experiences that people love to use. I improve product outcomes by centering every design decision on user research, accessibility, and measurable usability improvements.",
            "Cybersecurity": "What I Bring · I protect organizations from evolving threats. I improve security posture through proactive vulnerability management, rapid incident response, and a zero-trust security mindset.",
            "Data Analyst": "What I Bring · I transform raw data into strategic decisions. I improve business outcomes by delivering accurate forecasts, uncovering hidden revenue opportunities, and enabling self-service analytics.",
            "Software Engineer": "What I Bring · I architect maintainable, scalable systems. I improve business outcomes by reducing technical debt, accelerating feature delivery, and ensuring system reliability.",
        }
        if len(str(sec.get("summary","")).strip()) < 400:
            value = VALUE_MAP.get(role, f"What I Bring · Passionate {role} committed to delivering measurable results, continuous improvement, and meaningful contributions to every team and project I join.")
            sec["summary"] = sec.get("summary","") + "\n\n" + value
            fill_log.append("✓ Career value statement appended to summary")

        return jsonify({
            "status": "success",
            "sections_filled": sec,
            "fill_log": fill_log,
            "role_detected": role,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# AI CONTENT ENHANCEMENT ENGINE
# Transform minimal/weak input into recruiter-quality professional content
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/ai_enhance", methods=["POST"])
def ai_enhance():
    """
    Intelligently enhances sparse resume content into professional ATS-quality content.
    Body JSON: { sections, job_description, experience_level }
    Returns: { status, sections_enhanced, ats_estimate, enhancements_made }
    """
    try:
        d        = request.json or {}
        sec      = d.get("sections", {})
        jd       = d.get("job_description", "")
        exp_lvl  = d.get("experience_level", "mid")  # fresher | mid | senior

        enhancements = []
        jd_lower = jd.lower()

        # ══════════════════════════════════════════════════════════════════
        # ROLE DETECTION ENGINE — multi-signal, domain-adaptive
        # Reads: job_description + resume skills + experience + job_title
        # ══════════════════════════════════════════════════════════════════
        def _detect_role(sec_data, jd_text, exp_level):
            """
            Detect the user's true professional domain by scoring signals from
            their resume content and job description.  Never defaults to
            Software Engineer unless the evidence actually points there.
            """
            # Combine all resume text for signal extraction
            all_resume_text = " ".join([
                str(sec_data.get("summary", "")),
                str(sec_data.get("objective", "")),
                " ".join(sec_data.get("skills", [])),
                " ".join(sec_data.get("experience", [])),
                " ".join(sec_data.get("projects", [])),
                " ".join(sec_data.get("certifications", [])),
                str(sec_data.get("job_title", "")),
                str(sec_data.get("title", "")),
            ]).lower()

            combined = all_resume_text + " " + jd_text.lower()

            # Role signal map: role → (keywords, weight)
            # Higher-specificity keywords score more
            ROLE_SIGNALS = {
                "QA / Test Engineer": [
                    ("selenium","automation testing","test cases","bug tracking","defect","qa engineer",
                     "quality assurance","regression testing","jira testing","test plan","test script",
                     "cucumber","appium","testng","junit","manual testing","smoke testing",
                     "functional testing","test automation","sdet","quality engineer"), 2.0
                ],
                "HR / Human Resources": [
                    ("recruitment","talent acquisition","onboarding","employee relations","payroll",
                     "hr operations","hris","appraisal","performance management","hr manager",
                     "human resources","workforce","staffing","labor law","employee engagement",
                     "compensation","benefits","hr generalist","people operations","attrition"), 2.0
                ],
                "Digital Marketing": [
                    ("seo","sem","google ads","social media marketing","content marketing",
                     "email marketing","campaign","lead generation","google analytics","meta ads",
                     "influencer","ppc","cpc","ctr","marketing funnel","brand awareness",
                     "hubspot","mailchimp","conversion rate","digital marketing","growth hacking"), 2.0
                ],
                "UI/UX Designer": [
                    ("figma","wireframe","prototyping","user research","usability testing",
                     "design system","ux designer","ui designer","adobe xd","sketch",
                     "information architecture","user journey","persona","interaction design",
                     "accessibility","heuristic","low fidelity","high fidelity","mockup",
                     "visual design","product designer"), 2.0
                ],
                "Cybersecurity": [
                    ("penetration testing","vulnerability assessment","siem","soc","threat analysis",
                     "firewall","intrusion detection","ids/ips","nessus","metasploit","kali linux",
                     "ethical hacking","cybersecurity","information security","ceh","cissp","oscp",
                     "malware analysis","incident response","zero trust","network security"), 2.0
                ],
                "Data Analyst": [
                    ("sql","power bi","tableau","excel","data visualization","dashboard","reporting",
                     "data analyst","business intelligence","etl","data cleaning","pivot table",
                     "statistical analysis","kpi","metrics","google data studio","looker",
                     "data-driven","insights","forecasting","pandas","numpy","data warehouse"), 2.0
                ],
                "Data Scientist / ML Engineer": [
                    ("machine learning","deep learning","neural network","tensorflow","pytorch",
                     "scikit-learn","model training","nlp","computer vision","data science",
                     "xgboost","random forest","llm","generative ai","feature engineering",
                     "a/b testing","hypothesis testing","mlops","kubeflow","ml pipeline"), 2.0
                ],
                "DevOps / Cloud Engineer": [
                    ("kubernetes","docker","terraform","jenkins","ci/cd","github actions","aws",
                     "azure","gcp","ansible","helm","prometheus","grafana","nginx","linux admin",
                     "devops","site reliability","sre","infrastructure as code","cloudformation",
                     "bash scripting","pipeline","containerization","devsecops"), 2.0
                ],
                "Banking / Finance": [
                    ("banking","financial operations","compliance","kyc","aml","loan processing",
                     "credit analysis","treasury","risk management","financial analyst","cfa",
                     "accounting","gaap","ifrs","financial modeling","dcf","p&l","budget",
                     "audit","teller","reconciliation","investment banking","portfolio"), 2.0
                ],
                "Teacher / Educator": [
                    ("lesson plan","classroom management","curriculum","student engagement",
                     "pedagogy","teacher","educator","tutor","academic","school","k-12",
                     "e-learning","instructional design","learning objectives","assessment",
                     "differentiated instruction","education","training program","faculty",
                     "course design","learning management","lms","moodle"), 2.0
                ],
                "Nurse / Healthcare": [
                    ("patient care","clinical","nursing","healthcare","icu","wards","vitals",
                     "medication administration","ehr","emr","diagnosis","treatment","triage",
                     "patient assessment","registered nurse","rn","bls","acls","hospital",
                     "healthcare documentation","medical","clinical support","lab","pharmacy"), 2.0
                ],
                "Civil / Mechanical Engineer": [
                    ("autocad","structural","civil engineering","construction","site supervision",
                     "mechanical engineering","cad","revit","staad pro","estimation","surveying",
                     "quantity takeoff","concrete","hvac","manufacturing","production","quality control",
                     "maintenance engineer","plant","piping","welding","fabrication"), 2.0
                ],
                "Sales / Business Development": [
                    ("sales","business development","crm","lead generation","cold calling",
                     "revenue","quota","account management","client relationship","b2b","b2c",
                     "negotiation","pipeline","salesforce","deal closure","upsell","cross-sell",
                     "territory management","customer acquisition","partnerships"), 2.0
                ],
                "Product Manager": [
                    ("product roadmap","product manager","go-to-market","user stories","okr",
                     "product strategy","backlog","sprint planning","stakeholder","product owner",
                     "market research","product launch","kpi","product analytics","saas product",
                     "competitive analysis","mvp","feature prioritization","agile product"), 2.0
                ],
                "Graphic Designer": [
                    ("adobe photoshop","illustrator","indesign","graphic design","branding",
                     "typography","logo design","visual identity","print design","canva",
                     "color theory","layout design","creative design","art direction",
                     "packaging design","motion graphics","after effects","creative brief"), 2.0
                ],
                "Software Engineer": [
                    ("software engineer","backend","frontend","full stack","api development",
                     "rest api","microservices","node.js","react","angular","vue","django",
                     "flask","spring boot","java","python developer","software development",
                     "object oriented","design patterns","git","agile scrum","unit testing"), 2.0
                ],
            }

            # Score each role
            scores = {}
            for role_name, (keywords, weight) in ROLE_SIGNALS.items():
                score = sum(weight for kw in keywords if kw in combined)
                scores[role_name] = score

            best_role = max(scores, key=scores.get)
            best_score = scores[best_role]

            # If no strong signal detected, use job title as tiebreaker
            if best_score < 2.0:
                title_str = (str(sec_data.get("job_title","")) + " " +
                             str(sec_data.get("title","")) + " " +
                             jd_text).lower()
                for role_name, (keywords, _) in ROLE_SIGNALS.items():
                    if any(kw.split()[0] in title_str for kw in keywords if len(kw.split()[0]) > 4):
                        best_role = role_name
                        break

            # Fresher override: keep domain but tag freshness separately
            is_fresher = exp_level == "fresher" or any(
                x in combined for x in ["fresher","intern","graduate","entry level","no experience"]
            )

            return best_role, is_fresher

        role, is_fresher = _detect_role(sec, jd, exp_lvl)

        # ── 1. Transform weak experience bullets ────────────────────────
        WEAK_TO_STRONG = [
            # (weak_pattern, strong_replacement)
            (r"(?i)worked on", "Contributed to development of"),
            (r"(?i)helped with", "Collaborated on"),
            (r"(?i)did testing", "Executed comprehensive test plans covering unit, integration, and UAT phases"),
            (r"(?i)attended webinar", "Participated in industry-focused technical webinars, gaining expertise in emerging software methodologies and best practices"),
            (r"(?i)attended workshop", "Completed hands-on workshop training, applying learned concepts directly to academic and personal projects"),
            (r"(?i)made a project", "Designed and developed a full-stack"),
            (r"(?i)^testing$", "Knowledge of software testing methodologies including unit testing, integration testing, test case design, bug tracking, and QA processes"),
            (r"(?i)^python$", "Proficient in Python for scripting, data processing, API development, and automation workflows"),
        ]

        enhanced_exp = []
        for bullet in sec.get("experience", []):
            b = str(bullet or "").strip()
            if not b: continue
            enhanced = b
            for pattern, replacement in WEAK_TO_STRONG:
                enhanced = re.sub(pattern, replacement, enhanced, count=1)
            # Ensure bullet starts with strong action verb
            first_word = enhanced.split()[0].rstrip(".,") if enhanced.split() else ""
            exp_verb_set = {v.lower() for v in EXP_VERBS}
            if first_word.lower() not in exp_verb_set and len(first_word) > 2:
                verbs = ["Developed","Implemented","Designed","Built","Led","Optimized",
                         "Engineered","Delivered","Achieved","Automated","Collaborated on",
                         "Contributed to","Managed","Executed","Streamlined"]
                import hashlib
                idx = int(hashlib.md5(b.encode()).hexdigest(), 16) % len(verbs)
                enhanced = f"{verbs[idx]} {enhanced[0].lower()}{enhanced[1:]}"
            # Add quantification hint if no numbers present
            if not re.search(r'\d+', enhanced) and len(enhanced) < 120:
                quants = [
                    " resulting in measurable improvement in team productivity",
                    " contributing to overall project success",
                    " enhancing system reliability and performance",
                    " supporting business objectives effectively",
                ]
                import hashlib
                qi = int(hashlib.md5(b.encode()).hexdigest(), 16) % len(quants)
                enhanced = enhanced.rstrip(".") + quants[qi] + "."
            enhanced_exp.append(enhanced)
        if enhanced_exp != sec.get("experience", []):
            sec["experience"] = enhanced_exp
            enhancements.append("✓ Experience bullets transformed to achievement-oriented language")

        # ══════════════════════════════════════════════════════════════════
        # ROLE-ADAPTIVE CONTENT LIBRARY
        # All maps keyed by role returned from _detect_role()
        # Fallback = generic professional (never Software Engineer by default)
        # ══════════════════════════════════════════════════════════════════

        SUMMARIES_ENHANCED = {
            "QA / Test Engineer": (
                "Detail-oriented QA / Test Engineer with proven expertise in designing, executing, "
                "and automating test strategies across web and mobile platforms. Skilled in Selenium, "
                "TestNG, JIRA, and Agile QA processes. Consistently reduced post-release defect rates "
                "through systematic regression testing and collaborative defect management. Committed "
                "to delivering high-quality software through rigorous quality assurance practices."
            ),
            "HR / Human Resources": (
                "Results-driven HR Professional with hands-on experience in end-to-end recruitment, "
                "onboarding, employee relations, and HR operations. Adept at aligning talent strategies "
                "with business goals, managing HRIS systems, and fostering a positive workplace culture. "
                "Skilled in payroll coordination, performance appraisal cycles, and labor compliance. "
                "Passionate about building engaged, high-performing teams."
            ),
            "Digital Marketing": (
                "Data-driven Digital Marketing Specialist with expertise in SEO, SEM, social media "
                "campaigns, and content strategy. Proven track record of increasing organic traffic, "
                "reducing customer acquisition costs, and optimizing campaign ROI through Google Ads, "
                "Meta Ads, and analytics-driven decision-making. Certified in Google Analytics and "
                "HubSpot Inbound Marketing. Passionate about turning data into measurable growth."
            ),
            "UI/UX Designer": (
                "Creative and empathetic UI/UX Designer with expertise in end-to-end product design — "
                "from user research and wireframing to high-fidelity prototyping and usability testing. "
                "Proficient in Figma, Adobe XD, and design systems. Skilled in translating complex user "
                "needs into intuitive, accessible digital experiences that drive engagement and satisfaction. "
                "Passionate about human-centered design and cross-functional collaboration."
            ),
            "Cybersecurity": (
                "Motivated Cybersecurity professional with hands-on experience in vulnerability assessment, "
                "penetration testing, SOC monitoring, and incident response. Proficient in SIEM tools, "
                "network security protocols, and ethical hacking methodologies. Experienced in identifying "
                "and mitigating threats across enterprise environments. Committed to safeguarding digital "
                "assets and maintaining compliance with security frameworks."
            ),
            "Data Analyst": (
                "Analytical and detail-oriented Data Analyst with strong proficiency in SQL, Power BI, "
                "Tableau, and Excel for transforming raw data into actionable business intelligence. "
                "Experienced in building interactive dashboards, conducting statistical analyses, and "
                "delivering insights that support strategic decision-making. Skilled in ETL processes, "
                "KPI tracking, and data storytelling for both technical and non-technical stakeholders."
            ),
            "Data Scientist / ML Engineer": (
                "Innovative Data Scientist with expertise in machine learning, deep learning, and NLP. "
                "Proficient in Python, TensorFlow, PyTorch, and scikit-learn for building production-grade "
                "predictive models. Experienced in the full ML lifecycle — from data preprocessing and "
                "feature engineering to model deployment and monitoring. Passionate about applying AI "
                "to solve real-world business problems with measurable impact."
            ),
            "DevOps / Cloud Engineer": (
                "Cloud-native DevOps Engineer with expertise in CI/CD automation, container orchestration, "
                "and infrastructure as code. Proficient in AWS, Kubernetes, Docker, and Terraform. "
                "Experienced in improving deployment frequency, system reliability, and observability "
                "through Prometheus, Grafana, and ELK stacks. Strong advocate for DevSecOps culture "
                "and platform engineering best practices."
            ),
            "Banking / Finance": (
                "Detail-oriented Finance / Banking professional with strong expertise in financial "
                "operations, compliance, risk management, and client service. Experienced in KYC/AML "
                "processes, loan processing, credit analysis, and regulatory reporting. Proficient in "
                "financial modeling, Excel, and banking software. Committed to maintaining accuracy, "
                "confidentiality, and adherence to banking regulations and audit standards."
            ),
            "Teacher / Educator": (
                "Dedicated and passionate Educator with proven experience in curriculum development, "
                "lesson planning, and student-centered instruction. Skilled in adapting teaching "
                "methodologies to diverse learning styles, managing classrooms effectively, and "
                "fostering academic growth through engaging, outcome-focused content. Experienced "
                "with LMS platforms, e-learning tools, and collaborative professional development. "
                "Committed to inspiring lifelong learners."
            ),
            "Nurse / Healthcare": (
                "Compassionate and skilled Nursing professional with hands-on experience in patient "
                "care, clinical assessment, and healthcare documentation. Proficient in administering "
                "medications, monitoring vitals, and collaborating with multidisciplinary care teams "
                "to deliver high-quality, patient-centered outcomes. Trained in BLS/ACLS protocols "
                "and experienced with EHR/EMR systems. Committed to upholding clinical excellence "
                "and patient dignity."
            ),
            "Civil / Mechanical Engineer": (
                "Skilled Civil/Mechanical Engineer with hands-on experience in project design, site "
                "supervision, and quality control. Proficient in AutoCAD, STAAD Pro, and construction "
                "management practices. Experienced in coordinating multidisciplinary teams, managing "
                "project timelines, and ensuring compliance with safety and regulatory standards. "
                "Detail-oriented professional committed to delivering infrastructure projects on time "
                "and within budget."
            ),
            "Sales / Business Development": (
                "Results-oriented Sales & Business Development professional with a proven ability to "
                "generate leads, build lasting client relationships, and close deals. Skilled in CRM "
                "tools (Salesforce, HubSpot), B2B/B2C sales strategy, and pipeline management. "
                "Consistently exceeded revenue targets through consultative selling, territory "
                "management, and cross-functional collaboration. Passionate about driving business "
                "growth and delivering exceptional customer value."
            ),
            "Product Manager": (
                "Strategic Product Manager with a track record of launching successful digital products "
                "through data-driven prioritization and cross-functional leadership. Expert in Agile/Scrum, "
                "OKR frameworks, user story mapping, and go-to-market execution. Skilled in translating "
                "customer research and market signals into actionable roadmaps that deliver measurable "
                "business impact. Passionate about building products users love."
            ),
            "Graphic Designer": (
                "Creative Graphic Designer with expertise in brand identity, visual communication, and "
                "digital design. Proficient in Adobe Photoshop, Illustrator, InDesign, and Canva. "
                "Experienced in producing compelling print and digital assets — from logos and packaging "
                "to social media graphics and marketing collateral. Skilled in translating brand briefs "
                "into visually striking designs that resonate with target audiences."
            ),
            "Software Engineer": (
                "Results-driven Software Engineer with a strong foundation in full-stack development, "
                "API design, and cloud-native applications. Demonstrated ability to deliver scalable "
                "solutions that improve system performance and user experience. Passionate about clean "
                "code architecture, Agile practices, and continuous learning."
            ),
        }

        SKILLS_ENHANCED = {
            "QA / Test Engineer": [
                "Selenium WebDriver","Manual Testing","Automation Testing","TestNG","JUnit",
                "JIRA / Bugzilla","Test Case Design","Regression Testing","Smoke Testing",
                "Functional Testing","API Testing (Postman)","Agile QA","Defect Tracking",
                "SQL (for DB testing)","Cucumber / BDD","Appium (Mobile Testing)",
            ],
            "HR / Human Resources": [
                "Talent Acquisition","End-to-End Recruitment","Onboarding & Induction",
                "Employee Relations","Payroll Processing","HRIS (SAP / Workday / Zoho HR)",
                "Performance Appraisal","Labor Law Compliance","Workforce Planning",
                "Employee Engagement","Exit Interviews","HR Operations","MIS / HR Reporting",
                "Employer Branding","Conflict Resolution",
            ],
            "Digital Marketing": [
                "SEO / On-page & Off-page","Google Ads (Search & Display)","Meta Ads (Facebook/Instagram)",
                "Google Analytics 4","Content Marketing","Email Marketing","Social Media Management",
                "Lead Generation","HubSpot","Canva / Adobe Express","Keyword Research",
                "CRO (Conversion Rate Optimization)","SEM","YouTube Marketing","Marketing Funnels",
            ],
            "UI/UX Designer": [
                "Figma","Adobe XD","Wireframing","Prototyping","User Research",
                "Usability Testing","Design Systems","Information Architecture",
                "User Journey Mapping","Accessibility (WCAG)","Sketch","InVision",
                "A/B Testing","Typography & Color Theory","Responsive Design",
            ],
            "Cybersecurity": [
                "Penetration Testing","Vulnerability Assessment","SIEM (Splunk / QRadar)",
                "SOC Monitoring","Network Security","Firewall Configuration","Kali Linux",
                "Nessus / Nmap","Metasploit","Incident Response","Threat Intelligence",
                "OWASP Top 10","Identity & Access Management","Encryption / PKI","CEH / CompTIA Security+",
            ],
            "Data Analyst": [
                "SQL (Advanced)","Power BI","Tableau","Microsoft Excel (Advanced)",
                "Python (Pandas / NumPy)","Data Visualization","ETL Pipelines",
                "Dashboard Development","Statistical Analysis","KPI Tracking",
                "Google Analytics","Data Cleaning","A/B Testing","Looker / Metabase",
                "Business Intelligence",
            ],
            "Data Scientist / ML Engineer": [
                "Python","TensorFlow / PyTorch","Scikit-learn","Machine Learning",
                "Deep Learning","NLP","Feature Engineering","Model Deployment",
                "SQL","Pandas / NumPy","Jupyter Notebooks","MLOps","XGBoost / LightGBM",
                "Computer Vision","A/B Testing & Experimentation",
            ],
            "DevOps / Cloud Engineer": [
                "AWS / Azure / GCP","Kubernetes","Docker","Terraform","Jenkins",
                "GitHub Actions","CI/CD Pipelines","Linux / Bash","Ansible","Helm",
                "Prometheus / Grafana","ELK Stack","Infrastructure as Code","Nginx",
                "DevSecOps","Site Reliability Engineering",
            ],
            "Banking / Finance": [
                "Financial Operations","KYC / AML Compliance","Credit Analysis",
                "Loan Processing","Risk Management","Financial Modeling (Excel)",
                "Regulatory Reporting","Reconciliation","Banking Software (Finacle / Temenos)",
                "Portfolio Analysis","DCF Valuation","GAAP / IFRS","Treasury Operations",
                "Customer Relationship Management","Audit Support",
            ],
            "Teacher / Educator": [
                "Lesson Planning","Curriculum Development","Classroom Management",
                "Student Assessment","Differentiated Instruction","E-Learning Tools (Moodle / Canvas)",
                "Google Classroom","Bloom's Taxonomy","Formative & Summative Assessment",
                "Student Engagement Strategies","Parent Communication","Soft Skills Coaching",
                "STEM / Subject Expertise","Educational Technology","Special Needs Support",
            ],
            "Nurse / Healthcare": [
                "Patient Care & Assessment","Medication Administration","Vital Signs Monitoring",
                "EHR / EMR Systems","BLS / ACLS Certified","Clinical Documentation",
                "Wound Care","IV Cannulation","Infection Control","Patient Education",
                "Multidisciplinary Team Collaboration","Triage","ICU / Ward Care",
                "Healthcare Compliance","Compassionate Communication",
            ],
            "Civil / Mechanical Engineer": [
                "AutoCAD","STAAD Pro / ETABS","Site Supervision","Structural Design",
                "Construction Management","Bill of Quantities (BOQ)","Quality Control",
                "Project Scheduling","Safety Compliance","Soil Testing","Survey & Estimation",
                "Revit / BIM","Mechanical Design","HVAC Systems","MS Project",
            ],
            "Sales / Business Development": [
                "B2B / B2C Sales","Lead Generation","CRM (Salesforce / HubSpot)","Cold Calling",
                "Client Relationship Management","Deal Negotiation","Revenue Forecasting",
                "Territory Management","Pipeline Management","Presentation Skills",
                "Upselling & Cross-selling","Market Research","Key Account Management",
                "Sales Reporting","Business Proposals",
            ],
            "Product Manager": [
                "Product Roadmapping","Agile / Scrum","JIRA / Confluence","User Story Writing",
                "OKR & KPI Frameworks","Go-to-Market Strategy","Stakeholder Management",
                "Product Analytics","A/B Testing","Competitive Analysis","Figma (for specs)",
                "SQL (for data queries)","Customer Discovery","Backlog Prioritization","PRD Writing",
            ],
            "Graphic Designer": [
                "Adobe Photoshop","Adobe Illustrator","Adobe InDesign","Canva",
                "Brand Identity Design","Typography","Color Theory","Logo Design",
                "Print & Digital Design","Social Media Graphics","Motion Graphics (After Effects)",
                "Packaging Design","UI Assets","Creative Briefing","Visual Storytelling",
            ],
            "Software Engineer": [
                "Python","JavaScript","TypeScript","React.js","Node.js","REST APIs",
                "Microservices","Docker","AWS","PostgreSQL","MongoDB","Git/GitHub",
                "Agile/Scrum","CI/CD","Redis","Unit Testing",
            ],
        }

        PROJECTS_MAP = {
            "QA / Test Engineer": [
                "Automation Test Suite — Built end-to-end Selenium + TestNG automation framework for e-commerce portal · Reduced manual regression effort by 65%",
                "API Testing Framework — Designed REST API test suite using Postman & Newman · Integrated with CI/CD pipeline for nightly regression runs",
                "Bug Tracking Dashboard — Configured JIRA workflow for defect lifecycle management · Improved defect closure rate by 30% across 3 sprint cycles",
            ],
            "HR / Human Resources": [
                "Campus Recruitment Drive — Coordinated end-to-end hiring for 50+ positions across 8 colleges · Achieved 95% offer acceptance rate",
                "Onboarding Portal Implementation — Designed digital onboarding workflow using Zoho HR · Reduced new-hire paperwork time by 40%",
                "Employee Engagement Survey — Conducted company-wide survey (300+ employees) · Identified top 5 attrition drivers, leading to policy revisions",
            ],
            "Digital Marketing": [
                "SEO Campaign — Executed on-page and off-page SEO strategy for B2B SaaS client · Increased organic traffic by 140% in 6 months",
                "Google Ads Campaign — Managed ₹5L/month PPC budget · Improved ROAS by 3.2× through A/B ad copy testing and audience segmentation",
                "Social Media Growth Strategy — Grew brand Instagram account from 2K to 25K followers in 4 months via content calendar and influencer tie-ups",
            ],
            "UI/UX Designer": [
                "Mobile Banking App Redesign — Led UX overhaul for fintech mobile app (50K+ users) · Reduced task completion time by 35% post-launch",
                "E-Commerce Design System — Built reusable Figma component library for 3 product teams · Cut design-to-dev handoff time by 50%",
                "Usability Study — Conducted 12-participant moderated usability tests · Identified 8 critical pain points, informing next product sprint",
            ],
            "Cybersecurity": [
                "Web Application Pentest — Performed OWASP Top 10 assessment for fintech client · Discovered 4 critical vulnerabilities (SQLi, XSS, IDOR) before go-live",
                "SIEM Dashboard Setup — Deployed Splunk SIEM for network monitoring · Reduced mean time to detect (MTTD) incidents by 45%",
                "Security Awareness Training — Developed phishing simulation for 200+ employees · Reduced click-through rate on phishing emails by 70%",
            ],
            "Data Analyst": [
                "Sales Intelligence Dashboard — Built Power BI dashboard for regional sales teams · Reduced weekly reporting prep from 8 hours to 25 minutes",
                "Customer Churn Analysis — Identified churn predictors using SQL + Python · Insights adopted by retention team, reducing monthly churn by 18%",
                "Inventory Optimization Report — Analyzed 3-year supply chain data in Excel · Recommendations saved ₹12L in excess stock costs annually",
            ],
            "Data Scientist / ML Engineer": [
                "Customer Churn Prediction — XGBoost model (88% accuracy) deployed via Flask API · Integrated into CRM, reducing churn by 15% in pilot quarter",
                "NLP Sentiment Classifier — Fine-tuned BERT model on 50K product reviews · Achieved 91% F1-score for 3-class sentiment classification",
                "Sales Forecasting Engine — ARIMA + LightGBM ensemble achieving 93% accuracy on 18-month retail dataset · Reduced overstock by 22%",
            ],
            "DevOps / Cloud Engineer": [
                "CI/CD Pipeline Automation — Built GitHub Actions pipeline for 3 microservices · Reduced deployment time from 45 mins to 8 mins",
                "Kubernetes Cluster Migration — Migrated 12 services from EC2 to EKS · Improved resource utilization by 40% and cut infrastructure cost by 28%",
                "Observability Stack — Deployed Prometheus + Grafana + ELK for 15-service platform · Reduced MTTR from 2 hours to 18 minutes",
            ],
            "Banking / Finance": [
                "Loan Portfolio Analysis — Built Excel-based credit risk model for SME loan portfolio (₹50Cr) · Identified 12% of high-risk accounts proactively",
                "KYC Compliance Audit — Led internal KYC review for 5,000+ accounts · Achieved 100% regulatory compliance ahead of RBI deadline",
                "Financial Dashboard — Developed Power BI P&L dashboard for branch managers · Replaced 3 manual monthly reports, saving 6 hours/week",
            ],
            "Teacher / Educator": [
                "Digital Classroom Initiative — Integrated Google Classroom + interactive quizzes for 120 students · Improved assignment submission rate by 45%",
                "Curriculum Redesign — Rewrote Grade 9 Science curriculum aligned to NEP 2020 · Student pass percentage improved from 72% to 89%",
                "Remedial Learning Program — Designed after-school support program for 30 at-risk students · 80% achieved grade-level proficiency within one term",
            ],
            "Nurse / Healthcare": [
                "Patient Education Program — Developed post-discharge instructions for cardiac ward · Reduced 30-day readmission rate by 22% over 6 months",
                "EHR Migration Support — Assisted team migration from paper records to EHR system · Trained 15 nursing staff, achieving 100% adoption within 2 weeks",
                "Infection Control Audit — Conducted ward-level hand hygiene compliance audit · Compliance improved from 68% to 94% following targeted intervention",
            ],
            "Civil / Mechanical Engineer": [
                "Residential Complex Project — Supervised construction of 200-unit residential project · Completed 3 weeks ahead of schedule within approved budget",
                "Structural Design (AutoCAD) — Designed RCC structural drawings for 5-storey commercial building · Passed municipal approval on first submission",
                "Preventive Maintenance Plan — Developed PM schedule for 40+ machines at manufacturing plant · Reduced unplanned downtime by 35%",
            ],
            "Sales / Business Development": [
                "Enterprise Account Acquisition — Closed 3 new enterprise clients worth ₹1.8Cr ARR in Q3 through strategic outreach and product demos",
                "Territory Expansion — Launched sales operations in 2 new regions · Grew regional revenue by 62% within first 6 months",
                "CRM Implementation — Led Salesforce CRM rollout for 20-person sales team · Improved pipeline visibility and reduced deal cycle by 18%",
            ],
            "Product Manager": [
                "Mobile App Launch — Led end-to-end product launch for fintech mobile app (iOS + Android) · Acquired 15K users in first 30 days",
                "Feature Prioritization Framework — Implemented RICE scoring model for 50+ feature backlog items · Increased sprint velocity by 25%",
                "Customer Discovery Research — Conducted 40+ user interviews and synthesized findings into 3 actionable product initiatives with C-suite buy-in",
            ],
            "Graphic Designer": [
                "Brand Identity Design — Developed complete visual identity for D2C startup (logo, color palette, typography) · Used across packaging, website, and social media",
                "Marketing Campaign Assets — Designed 80+ digital creatives for Diwali campaign · CTR improved by 3.5× over previous season's assets",
                "Product Packaging Redesign — Redesigned packaging for FMCG client's 5 SKUs · Reported 18% shelf pick-up improvement post-launch",
            ],
            "Software Engineer": [
                "ResumeAI Pro — AI-powered resume optimization platform (Python/Flask, React, SQLite) · Improved user ATS scores by 40%",
                "TaskFlow API — RESTful task management API (FastAPI, PostgreSQL, Redis) · 99.9% uptime, handles 10K+ concurrent requests",
                "E-Commerce Platform — Full-stack shopping application (Node.js, React, Stripe) · 500+ products, real-time inventory management",
            ],
        }

        CERTS_MAP = {
            "QA / Test Engineer": [
                "ISTQB Foundation Level Certified Tester",
                "Selenium WebDriver with Java – Udemy",
                "API Testing with Postman – Coursera",
            ],
            "HR / Human Resources": [
                "SHRM-CP (Society for Human Resource Management – Certified Professional)",
                "Diploma in Human Resource Management – XLRI / NIPM",
                "Recruitment & Talent Acquisition – LinkedIn Learning",
            ],
            "Digital Marketing": [
                "Google Ads Search Certification",
                "HubSpot Inbound Marketing Certification",
                "Google Analytics Individual Qualification (GAIQ)",
            ],
            "UI/UX Designer": [
                "Google UX Design Professional Certificate – Coursera",
                "Figma UI UX Design Essentials – Udemy",
                "Nielsen Norman Group UX Certificate",
            ],
            "Cybersecurity": [
                "Certified Ethical Hacker (CEH) – EC-Council",
                "CompTIA Security+",
                "Google Cybersecurity Professional Certificate – Coursera",
            ],
            "Data Analyst": [
                "Google Data Analytics Professional Certificate – Coursera",
                "Microsoft Power BI Data Analyst (PL-300)",
                "Tableau Desktop Specialist",
            ],
            "Data Scientist / ML Engineer": [
                "IBM Data Science Professional Certificate – Coursera",
                "Deep Learning Specialization – DeepLearning.AI (Coursera)",
                "AWS Certified Machine Learning – Specialty",
            ],
            "DevOps / Cloud Engineer": [
                "AWS Solutions Architect – Associate",
                "Certified Kubernetes Administrator (CKA)",
                "HashiCorp Certified: Terraform Associate",
            ],
            "Banking / Finance": [
                "CFA Level I – CFA Institute",
                "Certified Financial Planner (CFP)",
                "NISM Series-V-A: Mutual Fund Distributors Certification",
            ],
            "Teacher / Educator": [
                "B.Ed (Bachelor of Education)",
                "Google Certified Educator Level 1",
                "Coursera: Learning How to Learn – McMaster University",
            ],
            "Nurse / Healthcare": [
                "BLS & ACLS Certification – American Heart Association",
                "Registered Nurse (RN) License",
                "Critical Care Nursing Certificate",
            ],
            "Civil / Mechanical Engineer": [
                "AutoCAD Certified Professional",
                "Project Management Professional (PMP) – PMI",
                "STAAD Pro Structural Analysis – Bentley Learning",
            ],
            "Sales / Business Development": [
                "Salesforce Certified Sales Cloud Consultant",
                "HubSpot Sales Software Certification",
                "Strategic Sales Management – ISB / Coursera",
            ],
            "Product Manager": [
                "Product Management Certificate – Pragmatic Institute",
                "Certified Scrum Product Owner (CSPO) – Scrum Alliance",
                "Product-Led Growth Certification – Product-Led Institute",
            ],
            "Graphic Designer": [
                "Adobe Certified Professional – Visual Design",
                "Graphic Design Specialization – CalArts (Coursera)",
                "Canva Design School Certification",
            ],
            "Software Engineer": [
                "AWS Certified Developer – Associate",
                "Meta Front-End Developer Certificate (Coursera)",
                "Full Stack Web Development – freeCodeCamp",
            ],
        }

        # Fresher-specific summary prefix
        FRESHER_PREFIX = {
            "QA / Test Engineer": "Aspiring QA Engineer and recent graduate with hands-on project experience in manual and automation testing. ",
            "HR / Human Resources": "Enthusiastic HR graduate with internship exposure to recruitment, onboarding, and HR operations. ",
            "Digital Marketing": "Creative Digital Marketing graduate with hands-on campaign projects and certified in Google Ads and Analytics. ",
            "UI/UX Designer": "Motivated UI/UX Design fresher with a strong portfolio of academic and freelance projects built in Figma. ",
            "Cybersecurity": "Cybersecurity graduate with foundational skills in ethical hacking, network security, and OWASP principles. ",
            "Data Analyst": "Detail-oriented Data Analytics fresher with project experience in SQL, Excel, Power BI, and Python for data visualization. ",
            "Software Engineer": "Motivated Computer Science graduate with hands-on full-stack project experience, eager to contribute in an Agile team. ",
        }

        # ── 2. Add missing professional sections ────────────────────────

        # Professional Summary
        base_summary = SUMMARIES_ENHANCED.get(role, (
            "Dedicated professional with strong domain expertise and a commitment to excellence. "
            "Experienced in delivering high-quality results through systematic approaches, "
            "effective communication, and continuous learning. Proven ability to collaborate "
            "across teams and contribute meaningfully to organizational goals."
        ))
        if is_fresher and role in FRESHER_PREFIX:
            base_summary = FRESHER_PREFIX[role] + base_summary

        if not sec.get("summary") or len(str(sec.get("summary","")).strip()) < 80:
            sec["summary"] = base_summary
            enhancements.append(f"✓ Professional Summary generated for {role} with ATS-optimized language")

        # Skills section
        current_skills = sec.get("skills", [])
        if len(current_skills) < 8:
            role_skills = SKILLS_ENHANCED.get(role, SKILLS_ENHANCED["Software Engineer"])
            jd_toks = set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#\-./]{1,}\b', jd_lower))
            jd_matched = [s for s in role_skills if any(t in s.lower() for t in jd_toks)]
            others = [s for s in role_skills if s not in jd_matched]
            sec["skills"] = list(dict.fromkeys(current_skills + jd_matched + others))[:16]
            enhancements.append(f"✓ Skills section expanded with {role}-specific ATS keywords ({len(sec['skills'])} total)")

        # Projects section
        if len(sec.get("projects", [])) < 2:
            base_projects = PROJECTS_MAP.get(role, PROJECTS_MAP["Software Engineer"])
            sec["projects"] = (sec.get("projects", []) + base_projects)[:4]
            enhancements.append(f"✓ Projects section populated with {role} domain examples")

        # Certifications
        if len(sec.get("certifications", [])) < 2:
            base_certs = CERTS_MAP.get(role, CERTS_MAP["Software Engineer"])
            sec["certifications"] = (sec.get("certifications", []) + base_certs)[:4]
            enhancements.append(f"✓ Certifications added for {role} career path")

        # ── 3. Estimate ATS score ────────────────────────────────────────
        score = 60
        if sec.get("summary") and len(str(sec.get("summary",""))) > 100: score += 12
        if len(sec.get("skills",[])) >= 8: score += 8
        if len(sec.get("experience",[])) >= 3: score += 10
        if sec.get("education"): score += 5
        if sec.get("projects"): score += 5
        if sec.get("certifications"): score += 5
        # Keyword match bonus
        all_text = " ".join([
            str(sec.get("summary","")),
            " ".join(sec.get("skills",[])),
            " ".join(sec.get("experience",[])),
        ]).lower()
        jd_kws = set(re.findall(r'\b[a-z][a-z0-9+#\-./]{2,}\b', jd_lower)) - STOP_WORDS
        matched = sum(1 for k in jd_kws if k in all_text)
        if jd_kws: score += min(15, int(matched/len(jd_kws)*20))
        score = min(92, score)

        return jsonify({
            "status": "success",
            "sections_enhanced": sec,
            "ats_estimate": score,
            "enhancements_made": enhancements,
            "role_detected": role,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ── USER FEEDBACK STORAGE ─────────────────────────────────────────────────────
def _init_feedback_table():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS user_feedback (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT DEFAULT 'anon',
        context    TEXT DEFAULT 'general',
        rating     INTEGER DEFAULT 0,
        comment    TEXT DEFAULT '',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    con.commit(); con.close()

_init_feedback_table()

@app.route("/save_feedback", methods=["POST"])
def save_feedback():
    """
    FIX 6: Store user feedback after resume generation/analysis.
    Body JSON: { email, rating (1-5), comment, context }
    """
    try:
        d       = request.json or {}
        email   = d.get("email", "anon")
        rating  = int(d.get("rating", 0))
        comment = d.get("comment", "").strip()[:500]
        context = d.get("context", "general")
        if not 1 <= rating <= 5:
            return jsonify({"status":"error","message":"Rating must be 1–5"}), 400
        db = get_db()
        db.execute("INSERT INTO user_feedback(user_email,context,rating,comment) VALUES(?,?,?,?)",
                   (email, context, rating, comment))
        db.commit()
        return jsonify({"status":"success","message":"Feedback saved. Thank you!"})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500


@app.route("/feedback_dashboard", methods=["GET"])
def feedback_dashboard():
    """
    FIX 6: Return feedback dashboard stats.
    Query: ?admin=1 for all records
    """
    try:
        db   = get_db()
        rows = db.execute("SELECT * FROM user_feedback ORDER BY created_at DESC LIMIT 100").fetchall()
        total = len(rows)
        avg   = round(sum(r["rating"] for r in rows) / total, 2) if total else 0
        users = len(set(r["user_email"] for r in rows))
        items = [{"id":r["id"],"email":r["user_email"],"rating":r["rating"],
                  "comment":r["comment"],"context":r["context"],"date":r["created_at"]} for r in rows]
        return jsonify({
            "status": "success",
            "stats": {"total_feedback": total, "avg_rating": avg, "unique_users": users},
            "feedback": items
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500



# ==================================================================================
# AI ENHANCE ENDPOINT - proxies to Anthropic API (server-side)
# Set ANTHROPIC_API_KEY env var to enable. Frontend also calls API directly.
# ==================================================================================
try:
    import anthropic as _anthropic_sdk
    _ANTHROPIC = True
except ImportError:
    _ANTHROPIC = False

AIE_SYSTEM_PROMPT = (
    "You are ResumeAI Pro — an intelligent, role-adaptive ATS Resume Enhancement AI. "
    "Your sole mission: enhance the user's resume for their ACTUAL job domain. "

    "CRITICAL RULES: "
    "1. DETECT the user's real domain from their resume content (job title, skills, experience, projects, certifications). "
    "2. NEVER force Software Engineer / Developer content on non-tech resumes. "
    "   — HR resume → HR content. QA resume → QA/Testing content. "
    "   — Nursing → Healthcare content. Teaching → Education content. "
    "   — Banking → Finance/Banking content. Marketing → Digital Marketing content. "
    "3. NEVER invent fake experience, skills, or qualifications not supported by the resume. "
    "4. ONLY enhance: wording, professionalism, ATS keyword density, action verbs, impact quantification. "
    "5. Preserve experience level: Fresher stays fresher-appropriate. Senior gets measurable impact. "
    "6. Use strong industry-specific action verbs and ATS-optimized terminology for the detected domain. "
    "7. Generated content must sound like a real industry professional wrote it — natural, credible, human. "

    "DOMAIN EXAMPLES (non-exhaustive): "
    "QA/Test Engineer → Selenium, TestNG, JIRA, Regression Testing, Bug Tracking, Test Cases, Automation. "
    "HR → Talent Acquisition, Onboarding, HRIS, Payroll, Employee Relations, Performance Management. "
    "Digital Marketing → SEO, Google Ads, Analytics, Content Strategy, Lead Generation, Campaign ROI. "
    "UI/UX → Figma, Wireframes, User Research, Prototyping, Usability Testing, Design Systems. "
    "Cybersecurity → VAPT, SIEM, SOC, Ethical Hacking, Incident Response, Network Security. "
    "Data Analyst → SQL, Power BI, Tableau, Dashboards, KPIs, ETL, Statistical Analysis. "
    "Banking/Finance → KYC/AML, Compliance, Credit Analysis, Financial Modeling, Reconciliation. "
    "Teacher/Educator → Lesson Planning, Curriculum, Classroom Management, Student Engagement. "
    "Nurse/Healthcare → Patient Care, Clinical Documentation, EHR, BLS/ACLS, Medication Administration. "
    "DevOps/Cloud → Kubernetes, Docker, Terraform, CI/CD, AWS/Azure, Observability. "
    "Civil/Mechanical Eng → AutoCAD, STAAD Pro, Site Supervision, BOQ, Construction Management. "

    "OUTPUT: Return ONLY valid JSON with these keys: "
    "job_domain_detected, experience_level_detected, professional_summary, "
    "enhanced_skills, enhanced_experience, enhanced_projects, "
    "ats_keywords, ats_score_estimate, resume_strengths, suggested_improvements."
)


@app.route("/ai_enhance_anthropic", methods=["POST"])
def ai_enhance_anthropic():
    """Server-side AI enhancement using Anthropic Python SDK.
    Body JSON: { resume_text, job_description, target_role }
    Requires ANTHROPIC_API_KEY environment variable.
    The browser-side AI Enhance tab calls Claude API directly (no server key needed).
    """
    try:
        d           = request.json or {}
        resume_text = d.get("resume_text", "").strip()
        jd          = d.get("job_description", "").strip()
        role        = d.get("target_role", "").strip()

        if not resume_text:
            return jsonify({"status": "error", "message": "resume_text is required"}), 400

        user_msg = (
            "INPUT RESUME CONTENT:\n" + resume_text + "\n\n"
            "INPUT JOB DESCRIPTION:\n" + (jd or "(Not provided)") + "\n\n"
            "INPUT TARGET ROLE:\n" + (role or "(Auto-detect from resume)") + "\n\n"
            "Generate a highly professional ATS-optimized enhanced resume "
            "WITHOUT changing the user domain. Return ONLY valid JSON."
        )

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key or not _ANTHROPIC:
            return jsonify({
                "status": "error",
                "message": (
                    "ANTHROPIC_API_KEY not configured. Set it as an environment variable. "
                    "The browser-side AI Enhance feature works without this server key."
                ),
                "fallback": True
            }), 503

        client  = _anthropic_sdk.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=AIE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}]
        )
        raw    = message.content[0].text if message.content else ""
        clean  = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        return jsonify({"status": "success", **result})

    except json.JSONDecodeError as e:
        return jsonify({"status": "error", "message": "JSON parse error: " + str(e)}), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("="*60)
    print("  ResumeAI Pro v5 | http://localhost:5000")
    print()
    print("  pip install flask flask-cors PyPDF2 scikit-learn")
    print("  pip install numpy reportlab pypdf python-docx openai")
    print("  Routes: /auto_fill /ai_enhance /ai_enhance_anthropic /save_feedback /feedback_dashboard /templates")
    print("  Set ANTHROPIC_API_KEY env var to enable server-side AI enhancement")
    print("  Browser AI Enhance calls Claude directly (no server key needed)")
    print("="*60)
    app.run(debug=True, port=5000)