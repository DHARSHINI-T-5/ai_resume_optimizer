"""
template_engine.py  —  ResumeAI Pro v5 Advanced Generation Engine
==================================================================
Drop-in module for app.py.

Public API
----------
  render_preview(sections, template_id, accent=None)  → HTML string
  render_pdf(sections, template_id, accent=None)       → bytes (PDF)
  generate_summary(sections)                           → str
  prepare_sections(sections)                           → dict  (auto-fills blanks)
  TEMPLATE_LAYOUTS                                     → {id: family}

Design Principles
-----------------
1. ONE HTML source for both preview and PDF (WeasyPrint path) — true 1:1 parity.
2. Gap-free rendering via Jinja2 {% if section %} blocks + CSS grid collapse.
3. 30+ templates across 8 structurally distinct layout families.
4. Graceful degradation: if WeasyPrint absent, falls back to app.py's ReportLab builders.
5. Zero regression: this module never imports from app.py; app.py calls into this.
"""

import re, os, html as _html_mod, io
from datetime import datetime

# ── Optional WeasyPrint ──────────────────────────────────────────────────────
try:
    from weasyprint import HTML as _WP_HTML
    _WP = True
except Exception:
    _WP = False

# ── Optional Jinja2 ──────────────────────────────────────────────────────────
try:
    from jinja2 import Environment, BaseLoader
    _J2 = True
    _jenv = Environment(loader=BaseLoader(), autoescape=True)
    _jenv.filters["nl2br"] = lambda v: (v or "").replace("\n", "<br>")
except Exception:
    _J2 = False

# ── Optional OpenAI ──────────────────────────────────────────────────────────
try:
    import openai as _openai
    _openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    _OAI = bool(_openai.api_key)
except Exception:
    _OAI = False

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE → LAYOUT FAMILY MAP  (30+ templates, 8 structural families)
# ═══════════════════════════════════════════════════════════════════════════════
TEMPLATE_LAYOUTS = {
    # FAMILY A – Two-Column Dark Sidebar
    "tech_modern":        "sidebar_dark",
    "fullstack_pro":      "sidebar_dark",
    "cybersec_pro":       "sidebar_dark",
    "mobile_dev":         "sidebar_dark",
    "ai_ml_engineer":     "sidebar_dark",
    "with_photo":         "sidebar_photo",

    # FAMILY B – Minimal Single-Column (max ATS)
    "tech_minimal":       "minimal",
    "biz_finance":        "minimal",
    "ats_pure":           "minimal",
    "ats_modern":         "minimal_accent",
    "legal_professional": "minimal",
    "operations_mgr":     "minimal_accent",

    # FAMILY C – Executive Centered
    "biz_executive":      "executive",
    "cto_ciso":           "executive",
    "biz_sales":          "executive",

    # FAMILY D – Vertical Accent Stripe
    "biz_marketing":      "stripe",
    "creative_media":     "stripe",
    "creative_modern":    "stripe",
    "hr_specialist":      "stripe",
    "project_manager":    "stripe",

    # FAMILY E – Education First (Fresher)
    "fresher_classic":    "fresher",
    "fresher_mba":        "fresher",
    "fresher_tech":       "fresher_tech",

    # FAMILY F – Dark Banner Header
    "data_science":       "dark_banner",
    "research_scientist": "dark_banner",
    "cloud_architect":    "dark_banner",

    # FAMILY G – Card / Narrative
    "startup_founder":    "narrative",
    "linkedin_style":     "narrative",
    "professor_academic": "academic",
    "graphic_designer":   "portfolio",
    "video_editor":       "portfolio",
    "architect_pro":      "portfolio",
    "creative_designer":  "portfolio",

    # FAMILY H – Universal Fallback
    "universal_pro":      "universal",
}
_DEFAULT_LAYOUT = "universal"

def _layout_for(tid):
    return TEMPLATE_LAYOUTS.get(tid, _DEFAULT_LAYOUT)

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _esc(t):
    return _html_mod.escape(str(t or ""))

def _contact_parts(c):
    return [x for x in [c.get("email"), c.get("phone"), c.get("linkedin"), c.get("github")] if x]

def _clean_bullet(text):
    return re.sub(r"^[\s•\-\*\d\.]+", "", str(text or "")).strip()

def _clean_list(lst, n=20):
    if not lst: return []
    return [_clean_bullet(x) for x in lst if str(x or "").strip()][:n]

def _initials(name):
    parts = str(name or "").strip().split()
    return "".join(p[0].upper() for p in parts[:2]) or "?"

def _render_j2(tmpl_str, ctx):
    if _J2:
        return _jenv.from_string(tmpl_str).render(**ctx)
    # Bare-minimum fallback if Jinja2 missing
    result = re.sub(r"\{%.*?%\}", "", tmpl_str, flags=re.DOTALL)
    for k, v in ctx.items():
        result = result.replace("{{ " + k + " }}", _esc(str(v))).replace("{{" + k + "}}", _esc(str(v)))
    return re.sub(r"\{\{.*?\}\}", "", result)

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED CSS  (injected into every layout page)
# ═══════════════════════════════════════════════════════════════════════════════

_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:'Inter','Segoe UI',Arial,sans-serif;
  background:#f0f2f5;display:flex;justify-content:center;padding:20px;
  -webkit-print-color-adjust:exact;print-color-adjust:exact
}
.a4{
  width:210mm;min-height:297mm;background:#fff;
  box-shadow:0 4px 32px rgba(0,0,0,.18);position:relative;overflow:hidden
}
ul.blist{list-style:none;padding:0}
ul.blist li{position:relative;padding-left:16px;margin-bottom:5px;
  font-size:12.5px;line-height:1.55;color:#2d2d2d}
ul.blist li::before{content:'▸';position:absolute;left:0;color:var(--accent)}
.edu-item{font-size:12.5px;margin-bottom:5px;color:#2d2d2d;line-height:1.5}
.cert-item{font-size:12px;margin-bottom:4px;color:#444}
.tag{
  display:inline-block;padding:3px 10px;border-radius:12px;
  font-size:11px;font-weight:600;
  background:var(--accent-tint);color:var(--accent);
  border:1px solid var(--accent-border);margin:2px
}
@media print{
  body{background:none;padding:0}
  .a4{box-shadow:none}
}
"""

def _page(inner_css, body_html, accent="#1e3a5f"):
    tint   = accent + "18"
    border = accent + "40"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
:root{{--accent:{accent};--accent-tint:{tint};--accent-border:{border}}}
{_BASE_CSS}
{inner_css}
</style>
</head>
<body><div class="a4">
{body_html}
</div></body></html>"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY A – SIDEBAR DARK
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_SIDEBAR_DARK = """
.a4{display:grid;grid-template-columns:168px 1fr}
.sb{background:#0f172a;padding:28px 16px;min-height:297mm;overflow:hidden}
.sb-name{font-size:17px;font-weight:700;color:var(--accent);line-height:1.25;
  margin-bottom:4px;word-break:break-word}
.sb-contact{font-size:10px;color:#94a3b8;margin-bottom:5px;
  word-break:break-all;line-height:1.4}
.sb-hr{height:1px;background:var(--accent);opacity:.35;margin:12px 0}
.sb-title{font-size:9px;font-weight:700;text-transform:uppercase;
  letter-spacing:.12em;color:var(--accent);margin-bottom:8px;padding-bottom:4px;
  border-bottom:1px solid var(--accent);opacity:.8}
.sk-tag{display:block;font-size:10.5px;margin-bottom:5px;padding:3px 8px;
  border-radius:6px;background:var(--accent-tint);color:#e2e8f0;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.main{padding:28px 26px}
.sec-title{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--accent);padding-bottom:4px;
  border-bottom:2px solid var(--accent);margin:18px 0 10px}
.summary{font-size:12.5px;color:#333;line-height:1.65;margin-bottom:4px}
"""

_TMPL_SIDEBAR_DARK = """
<div class="sb">
  <div class="sb-name">{{ name }}</div>
  {% for x in contact_items %}<div class="sb-contact">{{ x }}</div>{% endfor %}
  <div class="sb-hr"></div>
  {% if skills %}
  <div class="sb-title">Skills</div>
  {% for sk in skills %}<span class="sk-tag">{{ sk }}</span>{% endfor %}
  {% endif %}
  {% if certifications %}
  <div class="sb-title" style="margin-top:14px">Certifications</div>
  {% for ct in certifications %}<div style="font-size:10px;color:#94a3b8;margin-bottom:4px">{{ ct }}</div>{% endfor %}
  {% endif %}
</div>
<div class="main">
  {% if summary %}<div class="sec-title">Professional Summary</div><p class="summary">{{ summary }}</p>{% endif %}
  {% if experience %}<div class="sec-title">Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul>{% endif %}
  {% if projects %}<div class="sec-title">Projects</div><ul class="blist">{% for p in projects %}<li>{{ p }}</li>{% endfor %}</ul>{% endif %}
  {% if education %}<div class="sec-title">Education</div>{% for e in education %}<div class="edu-item">{{ e }}</div>{% endfor %}{% endif %}
</div>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY A2 – SIDEBAR PHOTO
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_SIDEBAR_PHOTO = _CSS_SIDEBAR_DARK + """
.photo{width:80px;height:80px;border-radius:50%;background:var(--accent-tint);
  border:3px solid var(--accent);display:flex;align-items:center;justify-content:center;
  margin:0 auto 14px;font-size:28px;font-weight:700;color:var(--accent)}
"""

_TMPL_SIDEBAR_PHOTO = """
<div class="sb">
  <div class="photo">{{ initials }}</div>
  <div class="sb-name" style="text-align:center">{{ name }}</div>
  {% for x in contact_items %}<div class="sb-contact" style="text-align:center">{{ x }}</div>{% endfor %}
  <div class="sb-hr"></div>
  {% if skills %}<div class="sb-title">Skills</div>{% for sk in skills %}<span class="sk-tag">{{ sk }}</span>{% endfor %}{% endif %}
  {% if certifications %}<div class="sb-title" style="margin-top:12px">Certifications</div>{% for ct in certifications %}<div style="font-size:10px;color:#94a3b8;margin-bottom:3px">{{ ct }}</div>{% endfor %}{% endif %}
</div>
<div class="main">
  {% if summary %}<div class="sec-title">Profile</div><p class="summary">{{ summary }}</p>{% endif %}
  {% if experience %}<div class="sec-title">Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul>{% endif %}
  {% if projects %}<div class="sec-title">Projects</div><ul class="blist">{% for p in projects %}<li>{{ p }}</li>{% endfor %}</ul>{% endif %}
  {% if education %}<div class="sec-title">Education</div>{% for e in education %}<div class="edu-item">{{ e }}</div>{% endfor %}{% endif %}
</div>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY B – MINIMAL SINGLE-COLUMN
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_MINIMAL = """
.a4{padding:36px 48px}
.hdr-name{font-size:24px;font-weight:700;color:var(--accent);margin-bottom:4px}
.hdr-contact{font-size:11px;color:#666;margin-bottom:12px}
.hdr-rule{border:none;border-top:2px solid var(--accent);margin-bottom:6px}
.sec-title{font-size:10.5px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--accent);margin:18px 0 6px;padding-bottom:3px;
  border-bottom:1.5px solid var(--accent)}
.body-text{font-size:12.5px;color:#333;line-height:1.65}
"""

_TMPL_MINIMAL = """
<div class="hdr-name">{{ name }}</div>
<div class="hdr-contact">{{ contact_line }}</div>
<hr class="hdr-rule">
{% if summary %}<div class="sec-title">Professional Summary</div><div class="body-text">{{ summary }}</div>{% endif %}
{% if skills %}<div class="sec-title">Skills</div><div>{% for sk in skills %}<span class="tag">{{ sk }}</span>{% endfor %}</div>{% endif %}
{% if experience %}<div class="sec-title">Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul>{% endif %}
{% if education %}<div class="sec-title">Education</div>{% for e in education %}<div class="edu-item">{{ e }}</div>{% endfor %}{% endif %}
{% if projects %}<div class="sec-title">Projects</div><ul class="blist">{% for p in projects %}<li>{{ p }}</li>{% endfor %}</ul>{% endif %}
{% if certifications %}<div class="sec-title">Certifications</div>{% for ct in certifications %}<div class="cert-item">{{ ct }}</div>{% endfor %}{% endif %}
"""

_CSS_MINIMAL_ACCENT = _CSS_MINIMAL + """
.a4::before{content:'';display:block;position:absolute;top:0;left:0;right:0;height:5px;background:var(--accent)}
.a4{padding-top:42px}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY C – EXECUTIVE CENTERED
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_EXECUTIVE = """
.a4{padding:44px 54px}
.exec-name{font-size:28px;font-weight:700;color:var(--accent);
  text-align:center;letter-spacing:.03em;margin-bottom:4px}
.exec-rule{border:none;border-top:4px solid var(--accent);margin:6px 0}
.exec-contact{font-size:11.5px;color:#666;text-align:center;margin-bottom:10px}
.sec-title{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:.12em;color:var(--accent);margin:20px 0 8px;padding-bottom:4px;
  border-bottom:1.5px solid var(--accent)}
.exec-summary{font-size:13px;color:#222;line-height:1.7}
.comp-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px 12px;margin-bottom:4px}
.comp-item{font-size:12px;color:#333;padding:2px 0}
.comp-item::before{content:'▪ ';color:var(--accent)}
"""

_TMPL_EXECUTIVE = """
<div class="exec-name">{{ name }}</div>
<hr class="exec-rule">
<div class="exec-contact">{{ contact_line }}</div>
{% if summary %}<div class="sec-title">Executive Summary</div><div class="exec-summary">{{ summary }}</div>{% endif %}
{% if skills %}<div class="sec-title">Core Competencies</div><div class="comp-grid">{% for sk in skills %}<div class="comp-item">{{ sk }}</div>{% endfor %}</div>{% endif %}
{% if experience %}<div class="sec-title">Professional Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul>{% endif %}
{% if education %}<div class="sec-title">Education</div>{% for e in education %}<div class="edu-item">{{ e }}</div>{% endfor %}{% endif %}
{% if certifications %}<div class="sec-title">Awards & Certifications</div>{% for ct in certifications %}<div class="cert-item">{{ ct }}</div>{% endfor %}{% endif %}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY D – VERTICAL ACCENT STRIPE
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_STRIPE = """
.a4{display:grid;grid-template-columns:10px 1fr}
.stripe-bar{background:var(--accent);min-height:297mm}
.stripe-body{padding:28px 32px}
.stripe-name{font-size:23px;font-weight:700;color:var(--accent);margin-bottom:3px}
.stripe-contact{font-size:11.5px;color:#666;margin-bottom:12px}
.stripe-hr{border:none;border-top:1px solid #e5e7eb;margin-bottom:4px}
.sec-title{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--accent);margin:16px 0 8px;padding-bottom:4px;
  border-bottom:2px solid var(--accent)}
.body-text{font-size:12.5px;color:#333;line-height:1.6}
"""

_TMPL_STRIPE = """
<div class="stripe-bar"></div>
<div class="stripe-body">
  <div class="stripe-name">{{ name }}</div>
  <div class="stripe-contact">{{ contact_line }}</div>
  <hr class="stripe-hr">
  {% if summary %}<div class="sec-title">Profile</div><div class="body-text">{{ summary }}</div>{% endif %}
  {% if skills %}<div class="sec-title">Expertise</div><div>{% for sk in skills %}<span class="tag">{{ sk }}</span>{% endfor %}</div>{% endif %}
  {% if experience %}<div class="sec-title">Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul>{% endif %}
  {% if projects %}<div class="sec-title">Projects</div><ul class="blist">{% for p in projects %}<li>{{ p }}</li>{% endfor %}</ul>{% endif %}
  {% if education %}<div class="sec-title">Education</div>{% for e in education %}<div class="edu-item">{{ e }}</div>{% endfor %}{% endif %}
  {% if certifications %}<div class="sec-title">Certifications</div>{% for ct in certifications %}<div class="cert-item">{{ ct }}</div>{% endfor %}{% endif %}
</div>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY E – FRESHER (education first)
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_FRESHER = """
.a4{padding:34px 44px}
.fsh-name{font-size:23px;font-weight:700;color:var(--accent);
  text-align:center;margin-bottom:3px}
.fsh-contact{font-size:11px;color:#666;text-align:center;margin-bottom:10px}
.fsh-rule{border:none;border-top:3px solid var(--accent);margin-bottom:6px}
.edu-box{background:var(--accent-tint);border-left:3px solid var(--accent);
  padding:8px 12px;margin-bottom:7px;border-radius:0 8px 8px 0;
  font-size:12.5px;color:#333}
.sec-title{font-size:10.5px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--accent);margin:16px 0 6px;padding-bottom:4px;
  border-bottom:2px solid var(--accent)}
.body-text{font-size:12.5px;color:#333;line-height:1.6}
"""

_TMPL_FRESHER = """
<div class="fsh-name">{{ name }}</div>
<div class="fsh-contact">{{ contact_line }}</div>
<hr class="fsh-rule">
{% if education %}<div class="sec-title">Education</div>{% for e in education %}<div class="edu-box">{{ e }}</div>{% endfor %}{% endif %}
{% if summary %}<div class="sec-title">Objective</div><div class="body-text">{{ summary }}</div>{% endif %}
{% if skills %}<div class="sec-title">Skills</div><div>{% for sk in skills %}<span class="tag">{{ sk }}</span>{% endfor %}</div>{% endif %}
{% if experience %}<div class="sec-title">Internships / Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul>{% endif %}
{% if projects %}<div class="sec-title">Academic Projects</div><ul class="blist">{% for p in projects %}<li>{{ p }}</li>{% endfor %}</ul>{% endif %}
{% if certifications %}<div class="sec-title">Achievements & Certifications</div>{% for ct in certifications %}<div class="cert-item">{{ ct }}</div>{% endfor %}{% endif %}
"""

_TMPL_FRESHER_TECH = """
<div class="fsh-name">{{ name }}</div>
<div class="fsh-contact">{{ contact_line }}</div>
<hr class="fsh-rule">
{% if skills %}<div class="sec-title">Technical Skills</div><div>{% for sk in skills %}<span class="tag">{{ sk }}</span>{% endfor %}</div>{% endif %}
{% if projects %}<div class="sec-title">Projects & Open Source</div><ul class="blist">{% for p in projects %}<li>{{ p }}</li>{% endfor %}</ul>{% endif %}
{% if experience %}<div class="sec-title">Internships / Work Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul>{% endif %}
{% if education %}<div class="sec-title">Education</div>{% for e in education %}<div class="edu-box">{{ e }}</div>{% endfor %}{% endif %}
{% if summary %}<div class="sec-title">Objective</div><div class="body-text">{{ summary }}</div>{% endif %}
{% if certifications %}<div class="sec-title">Certifications</div>{% for ct in certifications %}<div class="cert-item">{{ ct }}</div>{% endfor %}{% endif %}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY F – DARK BANNER HEADER
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_DARK_BANNER = """
.banner{background:#0f172a;padding:26px 32px;color:#fff}
.banner-name{font-size:22px;font-weight:700;color:var(--accent);margin-bottom:3px}
.banner-sub{font-size:11px;color:#94a3b8;margin-bottom:3px}
.banner-contact{font-size:10.5px;color:#64748b}
.body{padding:24px 32px}
.sec-title{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.12em;color:var(--accent);margin:18px 0 8px;padding-bottom:4px;
  border-bottom:2px solid var(--accent)}
.body-text{font-size:12.5px;color:#333;line-height:1.65}
"""

_TMPL_DARK_BANNER = """
<div class="banner">
  <div class="banner-name">{{ name }}</div>
  <div class="banner-sub">Data Scientist · ML Engineer · AI Researcher</div>
  <div class="banner-contact">{{ contact_line }}</div>
</div>
<div class="body">
  {% if summary %}<div class="sec-title">Profile</div><div class="body-text">{{ summary }}</div>{% endif %}
  {% if skills %}<div class="sec-title">Technical Stack</div><div>{% for sk in skills %}<span class="tag">{{ sk }}</span>{% endfor %}</div>{% endif %}
  {% if experience %}<div class="sec-title">Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul>{% endif %}
  {% if projects %}<div class="sec-title">Research / Projects</div><ul class="blist">{% for p in projects %}<li>{{ p }}</li>{% endfor %}</ul>{% endif %}
  {% if education %}<div class="sec-title">Education</div>{% for e in education %}<div class="edu-item">{{ e }}</div>{% endfor %}{% endif %}
  {% if certifications %}<div class="sec-title">Publications & Certifications</div>{% for ct in certifications %}<div class="cert-item">{{ ct }}</div>{% endfor %}{% endif %}
</div>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY G – NARRATIVE / CARD
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_NARRATIVE = """
.a4{padding:36px 44px}
.narr-header{background:var(--accent);padding:28px 32px;margin:-36px -44px 24px}
.narr-name{font-size:25px;font-weight:700;color:#fff;margin-bottom:3px}
.narr-contact{font-size:11px;color:rgba(255,255,255,.75)}
.card{border:1px solid #e5e7eb;border-radius:10px;padding:16px 18px;margin-bottom:14px;
  box-shadow:0 1px 4px rgba(0,0,0,.06)}
.card-title{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--accent);margin-bottom:10px}
.body-text{font-size:12.5px;color:#333;line-height:1.65}
"""

_TMPL_NARRATIVE = """
<div class="narr-header">
  <div class="narr-name">{{ name }}</div>
  <div class="narr-contact">{{ contact_line }}</div>
</div>
{% if summary %}<div class="card"><div class="card-title">About</div><div class="body-text">{{ summary }}</div></div>{% endif %}
{% if skills %}<div class="card"><div class="card-title">Skills</div>{% for sk in skills %}<span class="tag">{{ sk }}</span>{% endfor %}</div>{% endif %}
{% if experience %}<div class="card"><div class="card-title">Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul></div>{% endif %}
{% if projects %}<div class="card"><div class="card-title">Projects</div><ul class="blist">{% for p in projects %}<li>{{ p }}</li>{% endfor %}</ul></div>{% endif %}
{% if education %}<div class="card"><div class="card-title">Education</div>{% for e in education %}<div class="edu-item">{{ e }}</div>{% endfor %}</div>{% endif %}
{% if certifications %}<div class="card"><div class="card-title">Certifications</div>{% for ct in certifications %}<div class="cert-item">{{ ct }}</div>{% endfor %}</div>{% endif %}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY G2 – ACADEMIC CV
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_ACADEMIC = """
.a4{padding:38px 50px}
.acad-name{font-size:22px;font-weight:700;color:var(--accent);
  border-left:5px solid var(--accent);padding-left:14px;margin-bottom:4px}
.acad-contact{font-size:11px;color:#666;padding-left:19px;margin-bottom:14px}
.sec-title{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--accent);margin:20px 0 8px;padding-bottom:4px;
  border-bottom:2px solid var(--accent)}
.body-text{font-size:12.5px;color:#333;line-height:1.65}
"""

_TMPL_ACADEMIC = """
<div class="acad-name">{{ name }}</div>
<div class="acad-contact">{{ contact_line }}</div>
{% if summary %}<div class="sec-title">Research Statement</div><div class="body-text">{{ summary }}</div>{% endif %}
{% if certifications %}<div class="sec-title">Publications & Grants</div>{% for ct in certifications %}<div class="edu-item">{{ ct }}</div>{% endfor %}{% endif %}
{% if experience %}<div class="sec-title">Academic Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul>{% endif %}
{% if projects %}<div class="sec-title">Research Projects</div><ul class="blist">{% for p in projects %}<li>{{ p }}</li>{% endfor %}</ul>{% endif %}
{% if education %}<div class="sec-title">Education</div>{% for e in education %}<div class="edu-item">{{ e }}</div>{% endfor %}{% endif %}
{% if skills %}<div class="sec-title">Technical Skills</div><div>{% for sk in skills %}<span class="tag">{{ sk }}</span>{% endfor %}</div>{% endif %}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY G3 – PORTFOLIO / CREATIVE
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_PORTFOLIO = """
.a4{padding:34px 44px}
.port-name{font-size:26px;font-weight:700;color:var(--accent);text-align:center;margin-bottom:2px}
.port-tag{font-size:12px;color:#888;text-align:center;margin-bottom:4px}
.port-contact{font-size:11px;color:#666;text-align:center;margin-bottom:6px}
.port-rule{border:none;border-top:3px solid var(--accent);margin:6px 0 10px}
.port-link{border:1px solid var(--accent);border-radius:8px;padding:8px 16px;
  text-align:center;font-size:12px;color:var(--accent);font-weight:600;
  margin-bottom:14px;background:var(--accent-tint)}
.sec-title{font-size:10.5px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--accent);margin:16px 0 7px;padding-bottom:4px;
  border-bottom:2px solid var(--accent)}
.body-text{font-size:12.5px;color:#333;line-height:1.65}
"""

_TMPL_PORTFOLIO = """
<div class="port-name">{{ name }}</div>
<div class="port-tag">Designer · Creative Director · Visual Storyteller</div>
<div class="port-contact">{{ contact_line }}</div>
<hr class="port-rule">
{% if portfolio_link %}<div class="port-link">🔗 Portfolio: {{ portfolio_link }}</div>{% endif %}
{% if summary %}<div class="sec-title">Design Philosophy</div><div class="body-text">{{ summary }}</div>{% endif %}
{% if skills %}<div class="sec-title">Tools & Skills</div><div>{% for sk in skills %}<span class="tag">{{ sk }}</span>{% endfor %}</div>{% endif %}
{% if projects %}<div class="sec-title">Featured Projects</div><ul class="blist">{% for p in projects %}<li>{{ p }}</li>{% endfor %}</ul>{% endif %}
{% if experience %}<div class="sec-title">Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul>{% endif %}
{% if education %}<div class="sec-title">Education</div>{% for e in education %}<div class="edu-item">{{ e }}</div>{% endfor %}{% endif %}
{% if certifications %}<div class="sec-title">Awards & Recognition</div>{% for ct in certifications %}<div class="cert-item">{{ ct }}</div>{% endfor %}{% endif %}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY H – UNIVERSAL (default fallback)
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_UNIVERSAL = """
.a4{padding:32px 44px}
.uni-name{font-size:24px;font-weight:700;color:var(--accent);margin-bottom:4px}
.uni-contact{font-size:11px;color:#666;margin-bottom:10px}
.uni-rule{border:none;border-top:2px solid var(--accent);margin-bottom:6px}
.sec-title{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--accent);margin:18px 0 8px;padding-bottom:4px;
  border-bottom:2px solid var(--accent)}
.body-text{font-size:12.5px;color:#333;line-height:1.65}
"""

_TMPL_UNIVERSAL = """
<div class="uni-name">{{ name }}</div>
<div class="uni-contact">{{ contact_line }}</div>
<hr class="uni-rule">
{% if summary %}<div class="sec-title">Professional Summary</div><div class="body-text">{{ summary }}</div>{% endif %}
{% if skills %}<div class="sec-title">Skills</div><div>{% for sk in skills %}<span class="tag">{{ sk }}</span>{% endfor %}</div>{% endif %}
{% if experience %}<div class="sec-title">Experience</div><ul class="blist">{% for e in experience %}<li>{{ e }}</li>{% endfor %}</ul>{% endif %}
{% if education %}<div class="sec-title">Education</div>{% for e in education %}<div class="edu-item">{{ e }}</div>{% endfor %}{% endif %}
{% if projects %}<div class="sec-title">Projects</div><ul class="blist">{% for p in projects %}<li>{{ p }}</li>{% endfor %}</ul>{% endif %}
{% if certifications %}<div class="sec-title">Certifications</div>{% for ct in certifications %}<div class="cert-item">{{ ct }}</div>{% endfor %}{% endif %}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════
_LAYOUTS = {
    "sidebar_dark":   (_CSS_SIDEBAR_DARK,   _TMPL_SIDEBAR_DARK),
    "sidebar_photo":  (_CSS_SIDEBAR_PHOTO,  _TMPL_SIDEBAR_PHOTO),
    "minimal":        (_CSS_MINIMAL,        _TMPL_MINIMAL),
    "minimal_accent": (_CSS_MINIMAL_ACCENT, _TMPL_MINIMAL),
    "executive":      (_CSS_EXECUTIVE,      _TMPL_EXECUTIVE),
    "stripe":         (_CSS_STRIPE,         _TMPL_STRIPE),
    "fresher":        (_CSS_FRESHER,        _TMPL_FRESHER),
    "fresher_tech":   (_CSS_FRESHER,        _TMPL_FRESHER_TECH),
    "dark_banner":    (_CSS_DARK_BANNER,    _TMPL_DARK_BANNER),
    "narrative":      (_CSS_NARRATIVE,      _TMPL_NARRATIVE),
    "academic":       (_CSS_ACADEMIC,       _TMPL_ACADEMIC),
    "portfolio":      (_CSS_PORTFOLIO,      _TMPL_PORTFOLIO),
    "universal":      (_CSS_UNIVERSAL,      _TMPL_UNIVERSAL),
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_context(sections):
    """Convert raw sections dict → clean Jinja2 context. Gap-aware (empty lists = falsy)."""
    c    = sections.get("contact", {})
    name = str(c.get("name") or "").strip() or "Your Name"

    contact_items = _contact_parts(c)
    contact_line  = "  |  ".join(_esc(x) for x in contact_items)

    summary = str(sections.get("summary") or "").strip()
    skills  = _clean_list(sections.get("skills",  []), 22)
    exp     = _clean_list(sections.get("experience", []), 10)
    edu     = _clean_list(sections.get("education",  []), 5)
    proj    = _clean_list(sections.get("projects",   []), 6)
    certs   = _clean_list(sections.get("certifications", []), 6)
    portfolio_link = c.get("github") or c.get("linkedin") or ""

    return dict(
        name=name,
        contact_items=contact_items,
        contact_line=contact_line,
        initials=_initials(name),
        summary=summary,
        skills=skills,
        experience=exp,
        education=edu,
        projects=proj,
        certifications=certs,
        portfolio_link=portfolio_link,
        year=datetime.now().year,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API – render_preview
# ═══════════════════════════════════════════════════════════════════════════════

def render_preview(sections, template_id, accent=None):
    """
    Return a complete <!DOCTYPE html> string for the live iframe preview.

    Parameters
    ----------
    sections    : dict  (contact, summary, skills, experience, education, projects, certifications)
    template_id : str
    accent      : str hex color override (optional)

    Returns
    -------
    str — self-contained HTML page
    """
    if not accent:
        accent = "#1e3a5f"
    family = _layout_for(template_id)
    css, tmpl = _LAYOUTS.get(family, _LAYOUTS["universal"])
    ctx = _build_context(sections)
    try:
        body = _render_j2(tmpl, ctx)
    except Exception as e:
        body = f"<p style='color:red;padding:20px'>Render error: {_esc(str(e))}</p>"
    return _page(css, body, accent)

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API – render_pdf
# ═══════════════════════════════════════════════════════════════════════════════

_PRINT_CSS_OVERRIDE = """
@page{size:A4;margin:0}
body{background:none!important;padding:0!important}
.a4{box-shadow:none!important;width:210mm!important;min-height:297mm}
"""

def render_pdf(sections, template_id, accent=None):
    """
    Return PDF bytes.

    Priority:
    1. WeasyPrint — renders the exact same HTML as render_preview() → 1:1 parity
    2. ReportLab  — via app.py's build_pdf() (no breakage on existing installs)
    3. Plain text — absolute last resort
    """
    if not accent:
        accent = "#1e3a5f"

    # ── 1. WeasyPrint (preferred) ────────────────────────────────────────────
    if _WP:
        try:
            html = render_preview(sections, template_id, accent)
            html = html.replace("</style>", _PRINT_CSS_OVERRIDE + "\n</style>", 1)
            buf = io.BytesIO()
            _WP_HTML(string=html).write_pdf(buf)
            return buf.getvalue()
        except Exception:
            import traceback; traceback.print_exc()

    # ── 2. ReportLab direct (no circular call to app.py) ────────────────────
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rlc
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        HRFlowable, Table, TableStyle)

        buf = io.BytesIO()

        # Parse accent color safely
        try:
            hex_c = accent.lstrip("#")
            r, g, b = (int(hex_c[i:i+2], 16) / 255 for i in (0, 2, 4))
            accent_color = rlc.Color(r, g, b)
        except Exception:
            accent_color = rlc.Color(0.12, 0.23, 0.37)

        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=1.8*cm, rightMargin=1.8*cm,
            topMargin=1.5*cm, bottomMargin=1.5*cm,
        )
        base_styles = getSampleStyleSheet()

        def S(name, **kw):
            return ParagraphStyle(name, parent=base_styles["Normal"], **kw)

        style_name   = S("RLName",    fontSize=20, leading=24, textColor=accent_color, alignment=TA_CENTER, spaceAfter=2)
        style_contact= S("RLContact", fontSize=9,  leading=12, textColor=rlc.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=6)
        style_sec    = S("RLSec",     fontSize=10, leading=12, textColor=accent_color, fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=2)
        style_body   = S("RLBody",    fontSize=9,  leading=13, textColor=rlc.HexColor("#333333"), spaceAfter=2)
        style_bullet = S("RLBullet",  fontSize=9,  leading=13, textColor=rlc.HexColor("#333333"), leftIndent=10, spaceAfter=1)

        def safe(txt):
            import html as _h
            return _h.escape(str(txt or ""))

        def hr():
            return HRFlowable(width="100%", thickness=0.5, color=rlc.HexColor("#cccccc"),
                               spaceBefore=1, spaceAfter=4)

        def sec_header(title):
            return [Paragraph(safe(title.upper()), style_sec), hr()]

        def bullet_para(text):
            text = str(text or "").strip()
            if len(text) > 130:
                text = text[:130] + "…"
            return Paragraph("• " + safe(text), style_bullet)

        story = []
        ctx = _build_context(sections)

        # Name
        if ctx["name"]:
            story.append(Paragraph(safe(ctx["name"]), style_name))

        # Contact line
        if ctx["contact_line"]:
            story.append(Paragraph(safe(ctx["contact_line"]), style_contact))

        # Summary
        if ctx["summary"]:
            story += sec_header("Professional Summary")
            story.append(Paragraph(safe(ctx["summary"]), style_body))

        # Skills
        if ctx["skills"]:
            story += sec_header("Skills")
            story.append(Paragraph(safe(", ".join(ctx["skills"])), style_body))

        # Experience
        if ctx["experience"]:
            story += sec_header("Experience")
            for item in ctx["experience"]:
                story.append(bullet_para(item))

        # Education
        if ctx["education"]:
            story += sec_header("Education")
            for item in ctx["education"]:
                story.append(bullet_para(item))

        # Projects
        if ctx["projects"]:
            story += sec_header("Projects")
            for item in ctx["projects"]:
                story.append(bullet_para(item))

        # Certifications
        if ctx["certifications"]:
            story += sec_header("Certifications")
            for item in ctx["certifications"]:
                story.append(bullet_para(item))

        if not story:
            story.append(Paragraph("Resume", style_name))

        doc.build(story)
        pdf_bytes = buf.getvalue()
        if pdf_bytes[:4] == b"%PDF":
            return pdf_bytes
    except Exception:
        import traceback; traceback.print_exc()

    # ── 3. Plain-text last resort ────────────────────────────────────────────
    ctx = _build_context(sections)
    lines = [
        ctx["name"], ctx["contact_line"], "",
        "SUMMARY", ctx["summary"], "",
        "SKILLS", ", ".join(ctx["skills"]), "",
        "EXPERIENCE", *["• " + e for e in ctx["experience"]], "",
        "EDUCATION", *ctx["education"], "",
        "PROJECTS", *["• " + p for p in ctx["projects"]], "",
        "CERTIFICATIONS", *ctx["certifications"],
    ]
    return "\n".join(str(l) for l in lines).encode("utf-8")

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API – generate_summary
# ═══════════════════════════════════════════════════════════════════════════════

_ROLE_SUMMARIES = {
    "Software Engineer": (
        "Results-driven Software Engineer with hands-on experience designing scalable applications "
        "and RESTful APIs. Proficient in {top_skills}. Delivered production-grade features in Agile sprints, "
        "reduced latency by 40%, and improved test coverage from 45% to 87%. Passionate about clean architecture and CI/CD."
    ),
    "Data Scientist": (
        "Data-driven professional with expertise in machine learning, statistical analysis, and visualization. "
        "Skilled in {top_skills}. Track record of translating complex datasets into actionable insights. "
        "Experienced with end-to-end ML pipelines from ingestion to model deployment."
    ),
    "DevOps Engineer": (
        "Cloud-native DevOps Engineer specializing in infrastructure automation and CI/CD pipelines. "
        "Proficient in {top_skills}. Reduced deployment time by 60% through containerization and "
        "infrastructure-as-code. Experienced with SRE practices and observability stacks."
    ),
    "Designer": (
        "Human-centered UX/UI Designer with a portfolio of intuitive digital products. Proficient in "
        "{top_skills}. Expert at translating user research into pixel-perfect interfaces. "
        "Experience collaborating with cross-functional product teams in fast-paced environments."
    ),
    "Manager": (
        "Strategic leader with experience driving organizational growth and team performance. "
        "Skilled in {top_skills}. Proven ability to align cross-functional teams, manage P&L, "
        "and deliver projects on time and within budget. Strong communicator and mentor."
    ),
    "Finance": (
        "Detail-oriented Finance professional with expertise in financial modeling, forecasting, and "
        "investment analysis. Proficient in {top_skills}. Committed to accuracy, compliance, and "
        "delivering insights that inform executive decision-making."
    ),
    "Fresher": (
        "Motivated and detail-oriented graduate with a strong foundation in {top_skills}. "
        "Demonstrated ability to deliver quality projects and learn rapidly. Eager to contribute "
        "to a collaborative team environment and apply academic knowledge to real-world challenges."
    ),
    "Generic": (
        "Dedicated professional with expertise in {top_skills}. Proven track record of delivering "
        "results through collaboration, critical thinking, and continuous improvement. "
        "Seeking to leverage skills and experience to drive impact in a forward-thinking organization."
    ),
}

def _detect_role(sections):
    text = " ".join([
        " ".join(str(x) for x in sections.get("skills", [])),
        " ".join(str(x) for x in sections.get("experience", [])),
        str(sections.get("summary", "")),
    ]).lower()
    if any(k in text for k in ["machine learning","deep learning","pandas","tensorflow","pytorch","ml","data science"]):
        return "Data Scientist"
    if any(k in text for k in ["devops","kubernetes","terraform","jenkins","aws","ci/cd","docker","sre"]):
        return "DevOps Engineer"
    if any(k in text for k in ["figma","ux","ui design","sketch","adobe xd","designer"]):
        return "Designer"
    if any(k in text for k in ["manager","director","ceo","vp ","head of","lead"]):
        return "Manager"
    if any(k in text for k in ["finance","accounting","cfa","bloomberg","financial model"]):
        return "Finance"
    if any(k in text for k in ["python","java","javascript","react","node","flask","django","backend","frontend","api"]):
        return "Software Engineer"
    if not sections.get("experience"):
        return "Fresher"
    return "Generic"

def generate_summary(sections):
    """
    Auto-generate a professional summary.
    Tries OpenAI first; falls back to role-based template.
    Always returns a non-empty string.
    """
    skills = sections.get("skills", [])
    exp    = sections.get("experience", [])
    top_sk = ", ".join(str(s) for s in skills[:5]) if skills else "relevant technologies"

    if _OAI:
        try:
            prompt = (
                "Write a 3-4 sentence professional resume summary. "
                f"Skills: {', '.join(str(s) for s in skills[:10])}. "
                f"Experience: {'; '.join(str(e) for e in exp[:3])}. "
                "Requirements: action-oriented, ATS-friendly, no first-person pronouns. "
                "Return ONLY the summary text."
            )
            resp = _openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert resume writer. Return only the summary paragraph."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200, temperature=0.7,
            )
            result = resp.choices[0].message.content.strip()
            if len(result) > 80:
                return result
        except Exception:
            pass

    role = _detect_role(sections)
    return _ROLE_SUMMARIES.get(role, _ROLE_SUMMARIES["Generic"]).format(top_skills=top_sk)

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API – prepare_sections  (auto-fills blanks before rendering)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_sections(sections):
    """
    Pre-processes sections:
    - Auto-generates summary if missing or < 60 chars
    - Normalises lists (strips empty items)
    Returns a NEW dict (does not mutate input).
    """
    sec = dict(sections)
    summary = str(sec.get("summary") or "").strip()
    if len(summary) < 60:
        sec["summary"] = generate_summary(sec)
    return sec
