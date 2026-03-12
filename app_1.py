# app.py
import io
import re
import os
import hashlib
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_JUSTIFY

import google.generativeai as genai


# ─────────────────────────────────────────────────────────────
# 0) CONFIG
# ─────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="Policy AI Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ENV_GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY", "") or "").strip()

SIMILARITY_THRESHOLD = 0.10
TOP_K_CHUNKS = 5
SUMMARY_TOP_N_DEFAULT = 22
CHUNK_SIZE = 6

SECTION_LABELS = [
    "Goals & Objectives",
    "Key Strategies and Measures",
    "Target Groups / Stakeholders",
    "Implementation & Governance Framework",
    "Monitoring and Evaluation",
]

SCENARIO_PRESETS = [
    "Urban context (city / metropolitan area)",
    "Rural context (remote / low-resource region)",
    "Post-conflict or fragile state environment",
    "Digital-first / e-government rollout",
    "Climate-resilience adaptation scenario",
    "Youth-focused implementation",
    "Private-sector partnership model",
    "International donor-funded programme",
]

SECTION_KEYWORDS = {
    "Goals & Objectives": [
        "goal", "objective", "aim", "purpose", "vision", "mission", "intend",
        "target", "achieve", "outcome", "aspir"
    ],
    "Key Strategies and Measures": [
        "strategy", "strateg", "measure", "approach", "action", "plan",
        "initiative", "implement", "programme", "framework", "mechanism",
        "reform", "intervention", "policy"
    ],
    "Target Groups / Stakeholders": [
        "stakeholder", "community", "citizen", "youth", "women", "elderly",
        "vulnerable", "population", "beneficiar", "public", "sector", "private",
        "ngo", "civil society", "partner"
    ],
    "Implementation & Governance Framework": [
        "governance", "institution", "ministry", "department", "agency",
        "coordination", "regulation", "legal", "law", "authority", "resource",
        "budget", "fund", "capacity", "timeline"
    ],
    "Monitoring and Evaluation": [
        "monitor", "evaluat", "indicator", "assess", "review", "audit", "report",
        "data", "progress", "accountability", "transparency", "outcome", "impact", "track"
    ],
}


# ─────────────────────────────────────────────────────────────
# 1) STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero{
  background: linear-gradient(135deg,#0d1257 0%,#1a1f6b 35%,#2d3598 65%,#4f5fd4 100%);
  color:white;padding:2.1rem 2.3rem;border-radius:18px;margin-bottom:.8rem;
  box-shadow:0 10px 40px rgba(13,18,87,.35);position:relative;overflow:hidden;
}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:260px;height:260px;background:rgba(255,255,255,.04);border-radius:50%;}
.hero h1{font-size:2.1rem;font-weight:800;margin:0;letter-spacing:-.6px;}
.hero .sub{font-size:1rem;opacity:.88;margin:.4rem 0 1rem;}
.badge{display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);
  border-radius:20px;padding:3px 13px;font-size:.75rem;margin:2px 6px 2px 0;font-weight:500;}

.info-card{background:white;border-radius:14px;padding:1.2rem 1.4rem;box-shadow:0 2px 14px rgba(0,0,0,.07);
  margin-bottom:.8rem;border-left:4px solid #3f51b5;}
.info-card h3{color:#1a1f6b;font-size:1rem;font-weight:700;margin:0 0 .6rem;}
.info-card p,.info-card li{color:#37474f;font-size:.9rem;line-height:1.75;margin:0;}
.info-card ul,.info-card ol{margin:.4rem 0 0 1.1rem;padding:0;}
.info-card li{margin-bottom:.3rem;}

.obj-pill{display:inline-block;background:linear-gradient(135deg,#e8eaf6,#f5f5ff);
  border:1px solid #c5cae9;border-radius:24px;padding:5px 14px;
  font-size:.82rem;color:#283593;font-weight:500;margin:3px;}

.pipeline-box{background:#f5f6ff;border:1px solid #c5cae9;border-radius:12px;padding:1rem 1.2rem;}
.pipe-step{display:inline-flex;align-items:center;gap:6px;background:linear-gradient(135deg,#3f51b5,#5c6bc0);
  color:white;border-radius:8px;padding:6px 13px;font-size:.78rem;font-weight:600;margin:3px 0;width:100%;}
.pipe-nlp{background:linear-gradient(135deg,#1565c0,#1976d2) !important;}
.pipe-gen{background:linear-gradient(135deg,#6a1b9a,#8e24aa) !important;}
.pipe-rag{background:linear-gradient(135deg,#00695c,#00897b) !important;}
.pipe-arrow{text-align:center;color:#9fa8da;font-weight:700;font-size:1rem;margin:1px 0;}

.panel-title{font-size:1.08rem;font-weight:700;color:#1a1f6b;border-bottom:2px solid #e8eaf6;padding-bottom:.6rem;margin-bottom:1rem;}

.stat-tile{background:white;border:1px solid #e0e3f7;border-radius:12px;padding:.85rem .6rem;text-align:center;
  box-shadow:0 1px 8px rgba(0,0,0,.05);}
.stat-tile .v{font-size:1.5rem;font-weight:800;color:#1a1f6b;}
.stat-tile .l{font-size:.67rem;color:#7986cb;text-transform:uppercase;letter-spacing:.06em;margin-top:2px;}

.sec-item{padding:.4rem .7rem;margin:.3rem 0;border-left:3px solid #c5cae9;border-radius:0 6px 6px 0;background:#fafbff;
  font-size:.9rem;color:#37474f;line-height:1.65;}

.draft-scenario{background:linear-gradient(90deg,#3f51b5,#5c6bc0);color:white;border-radius:8px 8px 0 0;padding:.65rem 1rem;
  font-weight:600;font-size:.88rem;}
.draft-body{background:white;border:1px solid #c5cae9;border-top:none;border-radius:0 0 8px 8px;padding:1.1rem 1.3rem;
  font-size:.88rem;color:#37474f;line-height:1.7;margin-bottom:1rem;}

.chat-user{background:linear-gradient(135deg,#3f51b5,#5c6bc0);color:white;border-radius:16px 16px 4px 16px;padding:.7rem 1.1rem;
  margin:.5rem 0 .5rem auto;font-size:.9rem;max-width:78%;width:fit-content;box-shadow:0 2px 8px rgba(63,81,181,.3);}
.chat-bot{background:white;border:1px solid #e0e3f7;border-radius:16px 16px 16px 4px;padding:.7rem 1.1rem;margin:.5rem 0;
  font-size:.9rem;max-width:88%;color:#37474f;box-shadow:0 2px 8px rgba(0,0,0,.06);}
.src-chip{display:inline-block;background:#f3f4fd;border:1px solid #c5cae9;border-radius:6px;padding:2px 8px;font-size:.71rem;
  color:#5c6bc0;margin:2px;}

.shield-ok{display:inline-flex;align-items:center;gap:6px;background:#e8f5e9;border:1px solid #a5d6a7;border-radius:20px;padding:4px 12px;
  font-size:.78rem;color:#2e7d32;font-weight:500;}
.shield-warn{display:inline-flex;align-items:center;gap:6px;background:#fff3e0;border:1px solid #ffcc02;border-radius:20px;padding:4px 12px;
  font-size:.78rem;color:#e65100;font-weight:500;}

.step-card{background:white;border-radius:12px;padding:1rem .8rem;text-align:center;
  box-shadow:0 2px 10px rgba(0,0,0,.07);border-top:3px solid #3f51b5;}

.stButton > button{
  background:linear-gradient(135deg,#3f51b5,#5c6bc0) !important;
  color:white !important;border:none !important;border-radius:8px !important;font-weight:600 !important;
}
.stButton > button:hover{opacity:.88 !important;transform:translateY(-1px) !important;}
div[data-testid="stExpander"]{border:1px solid #e0e3f7 !important;border-radius:12px !important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 2) SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "active_page": "🏠 Home & About",
    "api_key": ENV_GEMINI_API_KEY,
    "raw_text": "",
    "clean_text": "",
    "doc_name": "Policy Document",
    "summary_sentences": [],
    "structured_summary": {},
    "text_stats": {},
    "drafts": [],         # list[(scenario, text)]
    "chat_history": [],   # list[{"role":"user/assistant","content":..., "sources":[(chunk,score)]}]
    "rag_chunks": [],
    "rag_vec": None,
    "rag_matrix": None,
    "doc_hash": "",
    "gemini_error": "",
    "top_n": SUMMARY_TOP_N_DEFAULT,
    "show_src": False,
    "auto_model": True,
    "model_choice": "models/gemini-1.5-flash",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ─────────────────────────────────────────────────────────────
# 3) GEMINI HELPERS
# ─────────────────────────────────────────────────────────────
def format_gemini_error(err: Exception) -> str:
    msg = str(err) if err else "Unknown error"
    low = msg.lower()

    if "defaultcredentialserror" in low or "no api_key" in low or "adc" in low:
        return "No API key found. Paste your key in the sidebar."

    if "api_key_invalid" in low or "api key invalid" in low or "api key expired" in low:
        return "Invalid/expired API key. Create a new key at ai.google.dev and paste it in the sidebar."

    if "429" in msg and ("quota" in low or "rate" in low):
        return "Quota / rate limit exceeded (HTTP 429). Try again later or increase quota/billing."

    if "service_disabled" in low or "has not been used in project" in low:
        return "Gemini API not enabled for this project. Enable Generative Language API in Google Cloud."

    if "api_key_service_blocked" in low or ("blocked" in low and "requests to this api" in low):
        return "API key blocked by restrictions. Remove restrictions or allow Generative Language API."

    if "permission_denied" in low or "403" in msg:
        return "Permission denied (HTTP 403). Check key restrictions and billing/quota."

    if "404" in msg and ("not found" in low or "no longer available" in low or "model" in low):
        return "Model not available (HTTP 404). Your key may not have access to this model."

    return msg


def list_available_models(api_key: str) -> list[str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        out = []
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                out.append(m.name)
        return out
    except Exception:
        return []


def pick_model(api_key: str, preferred: str = "models/gemini-1.5-flash") -> str:
    available = list_available_models(api_key)
    if not available:
        return preferred
    for cand in [preferred, "models/gemini-1.5-flash", "models/gemini-1.5-pro"]:
        if cand in available:
            return cand
    return available[0]


def get_gemini_model(api_key: str, model_name: str):
    api_key = (api_key or "").strip()
    model_name = (model_name or "").strip()
    if not api_key:
        st.session_state["gemini_error"] = "Missing API key. Paste it in the sidebar."
        return None
    try:
        genai.configure(api_key=api_key)
        st.session_state["gemini_error"] = ""
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.session_state["gemini_error"] = format_gemini_error(e)
        return None


def safe_generate(model, prompt: str) -> str:
    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        if not text:
            st.session_state["gemini_error"] = "Empty response (or blocked)."
            return ""
        st.session_state["gemini_error"] = ""
        return text
    except Exception as e:
        st.session_state["gemini_error"] = format_gemini_error(e)
        return ""


# ─────────────────────────────────────────────────────────────
# 4) PDF EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            t = (page.extract_text() or "").strip()
            if t:
                pages.append(t)
        return "\n".join(pages).strip()
    except Exception as e:
        st.session_state["gemini_error"] = f"PDF read error: {e}"
        return ""


# ─────────────────────────────────────────────────────────────
# 5) NLP PREPROCESSING + SUMMARY
# ─────────────────────────────────────────────────────────────
def clean_policy_text(text: str) -> str:
    patterns = [
        r"(?im)^page\s*\d+\s*(of\s*\d+)?\s*$",
        r"(?im)^[\s]*\d+\s*$",
        r"(?im)^(version|rev|draft|confidential)[^\n]*$",
        r"©.*?\n",
        r"\b([A-Z]{3,}\s*){4,}",
        r"[ \t]{2,}",
        r"\n{3,}",
    ]
    out = text
    for p in patterns:
        out = re.sub(p, " ", out)
    return out.strip()


def sentence_tokenize(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if len(s.strip()) > 30]


def tfidf_extractive_summary(text: str, top_n: int) -> list[str]:
    sents = sentence_tokenize(text)
    if not sents:
        return []
    n = min(int(top_n), len(sents))
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        mx = vec.fit_transform(sents)
        scores = np.array(mx.sum(axis=1)).flatten()
        idx = np.argsort(scores)[-n:]
        idx = sorted(idx)  # preserve flow
        return [sents[i] for i in idx]
    except Exception:
        return sents[:n]


def classify_sentence(sent: str) -> str:
    s = sent.lower()
    best_sec = "Key Strategies and Measures"
    best_score = 0
    for sec, kws in SECTION_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in s)
        if score > best_score:
            best_score = score
            best_sec = sec
    return best_sec


def build_structured_summary(sentences: list[str]) -> dict:
    out = {sec: [] for sec in SECTION_LABELS}
    for sent in sentences:
        out[classify_sentence(sent)].append(sent)
    return out


def get_text_stats(text: str) -> dict:
    words = len(text.split())
    sents = len(sentence_tokenize(text))
    paras = len([p for p in text.split("\n\n") if p.strip()])
    read_time = max(1, words // 200)
    return {"words": words, "sentences": sents, "paragraphs": paras, "read_time": read_time}


# ─────────────────────────────────────────────────────────────
# 6) RAG (TF-IDF)
# ─────────────────────────────────────────────────────────────
def build_chunks(text: str, chunk_size: int) -> list[str]:
    sents = sentence_tokenize(text)
    chunks = []
    for i in range(0, len(sents), chunk_size):
        c = " ".join(sents[i:i + chunk_size]).strip()
        if c:
            chunks.append(c)
    return chunks


@st.cache_data(show_spinner=False)
def build_tfidf_index(doc_hash: str, chunks: tuple[str, ...]):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mx = vec.fit_transform(list(chunks))
    return vec, mx


def retrieve_chunks(
    query: str,
    vectorizer,
    matrix,
    chunks: list[str],
    top_k: int = TOP_K_CHUNKS,
    threshold: float = SIMILARITY_THRESHOLD,
):
    sims = cosine_similarity(vectorizer.transform([query]), matrix).flatten()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [(chunks[i], float(sims[i])) for i in top_idx if float(sims[i]) >= threshold]


# ─────────────────────────────────────────────────────────────
# 7) GEMINI PROMPTS (DRAFT + BOT)
# ─────────────────────────────────────────────────────────────
def build_summary_markdown(structured: dict) -> str:
    lines = []
    for sec in SECTION_LABELS:
        sents = structured.get(sec, [])
        if sents:
            lines.append(f"## {sec}")
            lines += [f"- {s}" for s in sents]
            lines.append("")
    return "\n".join(lines).strip()


def gemini_generate_draft(model, summary_text: str, scenario: str, custom: str = "") -> str:
    extra = f"\nAdditional instructions: {custom.strip()}" if custom.strip() else ""
    prompt = f"""
You are a senior policy drafter with expertise in governance and public administration.

Using ONLY the structured policy summary below, create a detailed policy document adapted for:

SCENARIO: {scenario}

STRUCTURED POLICY SUMMARY:
{summary_text}
{extra}

REQUIREMENTS:
- Formal, professional policy language
- Do NOT invent new laws, statistics, or external facts not in the summary
- Organise under: Preamble · Objectives · Strategic Pillars · Implementation Plan · Governance & Accountability · M&E Framework
- Tailor every section to the scenario context
- Approximately 500–700 words
- End with a brief "Scenario Rationale" paragraph explaining key adaptations made
""".strip()

    out = safe_generate(model, prompt)
    if not out:
        return f"⚠️ Generation error: {st.session_state.get('gemini_error', 'Unknown error')}"
    return out


def gemini_rag_answer(model, question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks[:10])
    prompt = f"""
You are a policy assistant. Answer ONLY using the CONTEXT below.
If the answer is not in the context, reply exactly:
This information is not found in the provided policy document.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    out = safe_generate(model, prompt)
    if not out:
        return f"⚠️ Error: {st.session_state.get('gemini_error', 'Unknown error')}"
    return out if out.strip() else "This information is not found in the provided policy document."


# ─────────────────────────────────────────────────────────────
# 8) PDF EXPORT
# ─────────────────────────────────────────────────────────────
def export_to_pdf(structured: dict, drafts: list[tuple[str, str]], doc_name: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )
    styles = getSampleStyleSheet()
    T = ParagraphStyle("T", parent=styles["Title"], fontSize=20,
                       textColor=colors.HexColor("#1a1f6b"), spaceAfter=6)
    M = ParagraphStyle("M", parent=styles["Normal"], fontSize=9,
                       textColor=colors.grey, spaceAfter=14)
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14,
                        textColor=colors.HexColor("#283593"), spaceBefore=16, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11,
                        textColor=colors.HexColor("#3f51b5"), spaceBefore=10, spaceAfter=3)
    BD = ParagraphStyle("BD", parent=styles["BodyText"], fontSize=10, leading=15,
                        alignment=TA_JUSTIFY, spaceAfter=5)

    story = [
        Paragraph("🏛️ Policy AI Assistant", M),
        Paragraph("Policy Analysis Report", T),
        Paragraph(f"Document: {doc_name}  |  Generated: {datetime.now().strftime('%d %b %Y %H:%M')}", M),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3f51b5"), spaceAfter=14),
        Paragraph("Structured Policy Summary", H1),
    ]

    for section in SECTION_LABELS:
        sents = structured.get(section, [])
        if sents:
            story.append(Paragraph(section, H2))
            for s in sents:
                story.append(Paragraph(f"• {s}", BD))

    if drafts:
        for scenario, text in drafts:
            story += [
                HRFlowable(width="100%", thickness=1, color=colors.lightgrey,
                           spaceBefore=16, spaceAfter=12),
                Paragraph(f"Policy Draft — {scenario}", H1),
            ]
            for line in (text or "").split("\n"):
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 5))
                elif line.startswith("#"):
                    story.append(Paragraph(line.lstrip("# ").strip(), H2))
                else:
                    story.append(Paragraph(line, BD))

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# 9) SIDEBAR
# ─────────────────────────────────────────────────────────────
PAGES = ["🏠 Home & About", "🔬 Analyse Policy", "📊 Dashboard"]

with st.sidebar:
    st.markdown("## 🏛️ Policy AI Assistant")
    st.markdown("---")

    current_page = st.session_state.get("active_page", PAGES[0])
    page_index = PAGES.index(current_page) if current_page in PAGES else 0

    page = st.radio("", PAGES, label_visibility="collapsed", index=page_index)
    st.session_state["active_page"] = page
    st.markdown("---")

    # API KEY
    st.markdown("### 🔑 Gemini API Key")
    api_input = st.text_input(
        "",
        value=st.session_state.get("api_key", ""),
        type="password",
        placeholder="Paste your Gemini API key…",
        help="Get a key at https://ai.google.dev",
        label_visibility="collapsed",
    )
    st.session_state["api_key"] = (api_input or "").strip()

    st.markdown(
        '<span class="shield-ok">✅ API key added</span>'
        if st.session_state["api_key"]
        else '<span class="shield-warn">⚠️ API Key Required</span>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # DOCUMENT INPUT
    st.markdown("### 📂 Document Input")
    input_mode = st.radio("", ["📎 Upload PDF", "✏️ Paste Text"], horizontal=True, label_visibility="collapsed")

    if input_mode == "📎 Upload PDF":
        uploaded = st.file_uploader("", type=["pdf"], help="Drag & drop or click to browse", label_visibility="collapsed")
        if uploaded:
            raw = extract_text_from_pdf(uploaded.read())
            if raw.strip():
                st.session_state["raw_text"] = raw
                st.session_state["doc_name"] = uploaded.name
                st.success(f"✅ {uploaded.name}\n{len(raw.split()):,} words")
            else:
                st.error("⚠️ No extractable text found (scanned PDF?)")
    else:
        pasted = st.text_area("", height=200, placeholder="Paste any policy document text here…", label_visibility="collapsed")
        if pasted.strip():
            st.session_state["raw_text"] = pasted.strip()
            st.session_state["doc_name"] = "Pasted Policy Text"

    st.markdown("---")

    # OPTIONS
    st.markdown("### ⚙️ Options")
    st.session_state["top_n"] = st.slider("Top sentences (summary)", 10, 50, int(st.session_state.get("top_n", SUMMARY_TOP_N_DEFAULT)))
    st.session_state["show_src"] = st.toggle("Show RAG source chunks", bool(st.session_state.get("show_src", False)))
    st.session_state["auto_model"] = st.toggle("Auto-pick Gemini model", bool(st.session_state.get("auto_model", True)))

    if not st.session_state["auto_model"]:
        st.session_state["model_choice"] = st.selectbox(
            "Gemini model",
            ["models/gemini-1.5-flash", "models/gemini-1.5-pro"],
            index=0 if st.session_state.get("model_choice", "models/gemini-1.5-flash") == "models/gemini-1.5-flash" else 1
        )

    st.markdown("---")

    # EXPORT
    if st.session_state.get("structured_summary"):
        pdf_bytes = export_to_pdf(
            st.session_state["structured_summary"],
            st.session_state.get("drafts", []),
            st.session_state.get("doc_name", "Policy Document"),
        )
        st.download_button(
            "📄 Export Full PDF Report",
            data=pdf_bytes,
            file_name="policy_ai_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption("Policy AI Assistant v2.0\nTF-IDF NLP · Google Gemini · RAG\nBuilt with Streamlit")


# ─────────────────────────────────────────────────────────────
# 10) PAGE ▸ HOME & ABOUT
# ─────────────────────────────────────────────────────────────
if st.session_state["active_page"] == "🏠 Home & About":
    st.markdown("""
    <div class="hero">
      <h1>🏛️ Policy AI Assistant</h1>
      <p class="sub">An AI-Powered Web Application for Policy Analysis, Summarisation &amp; Scenario-Based Draft Generation</p>
      <span class="badge">🧠 TF-IDF NLP</span>
      <span class="badge">✨ Google Gemini</span>
      <span class="badge">🔍 RAG Chatbot</span>
      <span class="badge">🛡️ Hallucination Control</span>
      <span class="badge">📄 PDF Export</span>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("""
        <div class="info-card">
          <h3>📖 About This System</h3>
          <p>This project presents a <strong>Policy AI Assistant</strong> — a web-based application
          built using <strong>Streamlit</strong>, Natural Language Processing (NLP) techniques,
          and the <strong>Google Gemini API</strong>.</p><br/>
          <p>The system allows users to:</p>
          <ol>
            <li>Upload / drag-and-drop or paste <strong>any policy document</strong></li>
            <li>Automatically <strong>clean and preprocess</strong> the text</li>
            <li>Generate a <strong>structured policy summary</strong></li>
            <li>Create <strong>scenario-based policy drafts</strong></li>
            <li>Interact with a <strong>retrieval-based chatbot</strong> grounded in the document</li>
          </ol><br/>
          <p>The system integrates extractive NLP methods (TF-IDF) with Generative AI (Gemini)
          while <strong>preventing hallucination</strong> through retrieval-based grounding.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <h3>💡 What the System Can Do</h3>
          <ul>
            <li><strong>Extract key information</strong> from policy documents</li>
            <li><strong>Summarise objectives and strategies</strong> into 5 structured sections</li>
            <li><strong>Generate adapted policy drafts</strong> for new scenarios</li>
            <li><strong>Answer questions</strong> grounded strictly in policy content</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <h3>✅ System Integration</h3>
          <p>The Policy AI Assistant successfully integrates:</p>
          <ul>
            <li><strong>NLP Preprocessing</strong> — regex-based text cleaning pipeline</li>
            <li><strong>Extractive Summarisation</strong> — TF-IDF sentence scoring and classification</li>
            <li><strong>Generative AI Drafting</strong> — Gemini-powered scenario-based policy drafts</li>
            <li><strong>Retrieval-Grounded Chatbot</strong> — TF-IDF RAG with cosine similarity</li>
          </ul><br/>
          <p>The system demonstrates how Generative AI can assist in policy analysis while
          maintaining <strong>reliability and transparency</strong> through structured architecture
          and hallucination control mechanisms. This project showcases the practical application
          of Generative AI in <strong>governance and public administration contexts</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <h3>🎯 Project Objectives</h3>
          <div style="display:flex;flex-wrap:wrap;gap:7px;margin-top:.5rem;">
            <span class="obj-pill">1. Build AI-Powered Policy Assistant using Generative AI</span>
            <span class="obj-pill">2. Apply NLP Preprocessing on Policy Documents</span>
            <span class="obj-pill">3. Implement Structured Policy Summarisation</span>
            <span class="obj-pill">4. Generate Scenario-Based Policy Drafts from Summary</span>
            <span class="obj-pill">5. Implement RAG Chatbot Grounded in Policy Text</span>
            <span class="obj-pill">6. Prevent Hallucination &amp; Ensure Answer Reliability</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="info-card">
          <h3>🏗️ System Architecture Pipeline</h3>
          <div class="pipeline-box">
            <div class="pipe-step">📄 PDF Upload / Text Input</div>
            <div class="pipe-arrow">↓</div>
            <div class="pipe-step pipe-nlp">📝 Text Extraction (pypdf)</div>
            <div class="pipe-arrow">↓</div>
            <div class="pipe-step pipe-nlp">🧹 NLP Cleaning (Regex Preprocessing)</div>
            <div class="pipe-arrow">↓</div>
            <div class="pipe-step pipe-nlp">📊 TF-IDF Extractive Summarisation</div>
            <div class="pipe-arrow">↓</div>
            <div class="pipe-step">📋 Structured Policy Summary</div>
            <div class="pipe-arrow">↓</div>
            <div style="display:flex;gap:6px;">
              <div class="pipe-step pipe-gen" style="width:48%;">✍️ Scenario Draft Generation (Gemini)</div>
              <div class="pipe-step pipe-rag" style="width:48%;">💬 RAG Chatbot (TF-IDF + Gemini)</div>
            </div>
          </div>
          <p style="font-size:.8rem;color:#546e7a;margin-top:.8rem;">
            <strong>NLP tasks</strong> → handled locally (TF-IDF, cosine similarity)<br/>
            <strong>Generative tasks</strong> → handled by Gemini API<br/>
            This ensures reliability and clear modular design.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <h3>🛡️ Hallucination Prevention</h3>
          <ul>
            <li>Drafts generated <strong>only from structured summary</strong> — not raw PDF</li>
            <li>Chatbot answers grounded <strong>only in retrieved chunks</strong></li>
            <li>Similarity threshold → returns <em>"Not found"</em> when confidence is low</li>
            <li>Explicit Gemini system instructions enforce content constraints</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <h3>🔧 Technologies Used</h3>
          <ul>
            <li><strong>Web App:</strong> Streamlit</li>
            <li><strong>PDF Extraction:</strong> pypdf</li>
            <li><strong>NLP / Summarisation:</strong> scikit-learn (TF-IDF)</li>
            <li><strong>Similarity:</strong> Cosine Similarity</li>
            <li><strong>Generative AI:</strong> Google Gemini API</li>
            <li><strong>PDF Export:</strong> ReportLab</li>
            <li><strong>Environment:</strong> python-dotenv</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🚀 How to Use")
    steps = [
        ("1️⃣", "Upload Document", "Drag & drop a PDF or paste text in the sidebar"),
        ("2️⃣", "Analyse Policy", "Click Analyse — text is cleaned, scored, and summarised"),
        ("3️⃣", "Review Summary", "Explore the 5-section structured policy summary"),
        ("4️⃣", "Generate Drafts", "Pick a scenario and generate AI-adapted policy drafts"),
        ("5️⃣", "Ask Questions", "Use the RAG chatbot to query your policy document"),
    ]
    for col, (icon, title, desc) in zip(st.columns(5), steps):
        col.markdown(f"""
        <div class="step-card">
          <div style="font-size:1.7rem;">{icon}</div>
          <div style="font-weight:700;color:#1a1f6b;font-size:.86rem;margin:.4rem 0 .3rem;">{title}</div>
          <div style="font-size:.77rem;color:#546e7a;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Upload a PDF or paste policy text in the sidebar, then go to **🔬 Analyse Policy** to begin.")


# ─────────────────────────────────────────────────────────────
# 11) PAGE ▸ ANALYSE POLICY
# ─────────────────────────────────────────────────────────────
elif st.session_state["active_page"] == "🔬 Analyse Policy":
    st.markdown("""
    <div class="hero" style="padding:1.6rem 2rem 1.4rem;">
      <h1 style="font-size:1.7rem;">🔬 Policy Analysis Workspace</h1>
      <p class="sub">Left Panel: Summarisation &nbsp;·&nbsp; Right Panel: Scenario-Based Draft Generation &nbsp;·&nbsp; Bottom: RAG Chatbot</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state["raw_text"].strip():
        st.warning("👈 Please upload a PDF or paste policy text in the sidebar first.")
        st.stop()

    b1, b2 = st.columns([2, 5])
    with b1:
        run = st.button("🔍 Analyse Policy Document", use_container_width=True)
    with b2:
        st.caption(f"📄 **{st.session_state['doc_name']}** · {len(st.session_state['raw_text'].split()):,} words loaded")

    if run:
        prog = st.progress(0, "Initialising…")
        prog.progress(15, "Cleaning text…")
        clean = clean_policy_text(st.session_state["raw_text"])

        prog.progress(35, "TF-IDF summarisation…")
        sents = tfidf_extractive_summary(clean, top_n=st.session_state["top_n"])

        prog.progress(60, "Structuring into 5 sections…")
        structured = build_structured_summary(sents)

        prog.progress(78, "Building RAG index…")
        chunks = build_chunks(clean, CHUNK_SIZE)
        dh = hashlib.md5(clean.encode("utf-8")).hexdigest()
        vec, mx = build_tfidf_index(dh, tuple(chunks))

        prog.progress(100, "Done")
        st.session_state.update({
            "clean_text": clean,
            "summary_sentences": sents,
            "structured_summary": structured,
            "text_stats": get_text_stats(clean),
            "rag_chunks": chunks,
            "rag_vec": vec,
            "rag_matrix": mx,
            "doc_hash": dh,
        })
        prog.empty()
        st.success("✅ Analysis complete!")
        st.rerun()

    if not st.session_state["structured_summary"]:
        st.info("Click **Analyse Policy Document** above to begin.")
        st.stop()

    # Stats bar
    s = st.session_state["text_stats"]
    for col, (val, lbl) in zip(st.columns(6), [
        (s["words"], "Words"),
        (s["sentences"], "Sentences"),
        (s["paragraphs"], "Paragraphs"),
        (s["read_time"], "Min Read"),
        (len(st.session_state["rag_chunks"]), "RAG Chunks"),
        (sum(len(v) for v in st.session_state["structured_summary"].values()), "Key Sentences"),
    ]):
        col.markdown(f'<div class="stat-tile"><div class="v">{val:,}</div><div class="l">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    left_col, right_col = st.columns([1, 1], gap="large")

    # LEFT: Summary
    with left_col:
        st.markdown('<div class="panel-title">📋 Policy Summarisation</div>', unsafe_allow_html=True)
        st.caption("TF-IDF extractive summarisation → auto-classified into 5 policy sections")

        tab1, tab2, tab3 = st.tabs(["📌 Structured View", "📝 All Sentences", "🔑 Keywords"])

        with tab1:
            icons = {
                "Goals & Objectives": "🎯",
                "Key Strategies and Measures": "⚙️",
                "Target Groups / Stakeholders": "👥",
                "Implementation & Governance Framework": "🏛️",
                "Monitoring and Evaluation": "📊"
            }
            for sec in SECTION_LABELS:
                ss = st.session_state["structured_summary"].get(sec, [])
                with st.expander(f"{icons.get(sec,'📌')} {sec} ({len(ss)})", expanded=bool(ss)):
                    if not ss:
                        st.caption("No sentences classified to this section.")
                    for t in ss:
                        st.markdown(f'<div class="sec-item">• {t}</div>', unsafe_allow_html=True)

        with tab2:
            all_s = st.session_state["summary_sentences"]
            st.caption(f"{len(all_s)} key sentences extracted")
            for i, t in enumerate(all_s, 1):
                st.markdown(
                    f'<div class="sec-item" style="border-color:#e0e3f7;">'
                    f'<span style="color:#7986cb;font-weight:700;">{i}.</span> {t}</div>',
                    unsafe_allow_html=True
                )

        with tab3:
            try:
                kv = TfidfVectorizer(stop_words="english", max_features=20, ngram_range=(1, 2))
                kv.fit([st.session_state["clean_text"]])
                kws = sorted(kv.vocabulary_.keys())
                kc1, kc2 = st.columns(2)
                for col, group in zip([kc1, kc2], [kws[:10], kws[10:]]):
                    for kw in group:
                        col.markdown(f'<span class="obj-pill" style="margin:2px;display:inline-block;">{kw}</span>',
                                     unsafe_allow_html=True)
            except Exception:
                st.caption("Keywords unavailable.")

        with st.expander("ℹ️ How NLP Summarisation Works"):
            st.markdown("""
**Step 1 – Text Cleaning (Regex Preprocessing)**
Removes page numbers, version headers, copyright notices, standalone numeric lines, acronym blocks, excess whitespace.

**Step 2 – Sentence Tokenisation**
Splits cleaned text into sentences (min 30 chars).

**Step 3 – TF-IDF Scoring**
`TfidfVectorizer` (bi-gram, English stop words) scores each sentence by combined term weight.

**Step 4 – Top-N Selection**
Highest-scoring sentences kept in document order for readability.

**Step 5 – Section Classification**
Each sentence matched to 1 of 5 policy sections via keyword presence scoring.

> **Why TF-IDF?** Lightweight · No external models · No install errors · Academically explainable · Stable & reproducible
            """)

    # RIGHT: Draft Generation
    with right_col:
        st.markdown('<div class="panel-title">✍️ Scenario-Based Policy Draft Generation</div>', unsafe_allow_html=True)
        st.caption("Drafts are generated strictly from the structured summary — not the raw PDF")

        api_key = st.session_state.get("api_key", "").strip()
        model_name = pick_model(api_key) if st.session_state.get("auto_model", True) else st.session_state.get("model_choice", "models/gemini-1.5-flash")
        model = get_gemini_model(api_key, model_name)

        if not model:
            st.warning(f"⚠️ {st.session_state.get('gemini_error','Enter a valid Gemini API key in the sidebar.')}")
        else:
            summary_text = build_summary_markdown(st.session_state["structured_summary"])

            mode = st.radio("", ["🎛️ Preset Scenarios", "✏️ Custom Scenarios"], horizontal=True, label_visibility="collapsed")
            if "Preset" in mode:
                sc1 = st.selectbox("Scenario A", SCENARIO_PRESETS, index=0)
                sc2 = st.selectbox("Scenario B", SCENARIO_PRESETS, index=1)
            else:
                sc1 = st.text_input("Scenario A", placeholder="e.g. Small island developing state")
                sc2 = st.text_input("Scenario B", placeholder="e.g. Post-pandemic economic recovery")

            custom_instr = st.text_area("Additional instructions (optional)",
                                        placeholder="e.g. Align with SDG goals; focus on digital infrastructure",
                                        height=70)

            g1, g2, g3 = st.columns(3)
            gen_a = g1.button("⚡ Draft A", use_container_width=True, key="draft_a_btn")
            gen_b = g2.button("⚡ Draft B", use_container_width=True, key="draft_b_btn")
            gen_both = g3.button("🚀 Both", use_container_width=True, key="draft_both_btn")

            def save_draft(scenario: str, text: str):
                idx = [i for i, (sname, _) in enumerate(st.session_state["drafts"]) if sname == scenario]
                if idx:
                    st.session_state["drafts"][idx[0]] = (scenario, text)
                else:
                    st.session_state["drafts"].append((scenario, text))

            if (gen_a or gen_both) and sc1:
                with st.spinner(f"Generating draft for: {sc1}…"):
                    save_draft(sc1, gemini_generate_draft(model, summary_text, sc1, custom_instr))

            if (gen_b or gen_both) and sc2:
                with st.spinner(f"Generating draft for: {sc2}…"):
                    save_draft(sc2, gemini_generate_draft(model, summary_text, sc2, custom_instr))

            st.markdown('<div style="margin:.5rem 0;"><span class="shield-ok">🛡️ Grounded in Summary Only — Hallucination Controlled</span></div>',
                        unsafe_allow_html=True)

            if st.session_state["drafts"]:
                for scenario, draft_text in st.session_state["drafts"]:
                    safe_html = (draft_text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                    st.markdown(f'<div class="draft-scenario">📄 {scenario}</div><div class="draft-body">{safe_html}</div>',
                                unsafe_allow_html=True)

                    d1, _ = st.columns([1, 3])
                    d1.download_button(
                        "⬇️ Download",
                        data=draft_text or "",
                        file_name=f"draft_{scenario[:25].replace(' ','_')}.txt",
                        mime="text/plain",
                        key=f"dl_{scenario[:12]}_{len(draft_text or '')}",
                        use_container_width=True
                    )

        with st.expander("ℹ️ How Draft Generation Works"):
            st.markdown("""
**Generation Process:**
1. The 5-section structured summary is formatted into a focused prompt context
2. Gemini receives summary + scenario as generation constraints
3. Model drafts a full policy document **using only summary content**
4. Prompt rules enforce: no new laws, no invented data, formal tone

**Why generate from summary only (not raw PDF)?**
- Ensures logical policy transformation
- Reduces hallucination risk
- Creates clear traceability between source and output
- Aligns with structured architecture design principles

**Scenario Adaptation:** Governance structures, resource constraints, stakeholder dynamics, and implementation approaches all adapt to the chosen scenario context.
            """)

    # ─────────────────────────────────────────────────────────────
    # RAG CHATBOT
    # ─────────────────────────────────────────────────────────────
    st.markdown("---")
    ch1, ch2 = st.columns([3, 1])

    with ch1:
        st.markdown("## 💬 Policy RAG Chatbot")
        st.markdown(
            '<span class="shield-ok">🛡️ Answers grounded strictly in policy document</span>'
            '<span style="font-size:.81rem;color:#546e7a;margin-left:.8rem;">'
            'Returns "Not found" when similarity score is below threshold</span>',
            unsafe_allow_html=True
        )

    with ch2:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_btn"):
            st.session_state["chat_history"] = []
            st.rerun()

    show_src = bool(st.session_state.get("show_src", False))

    st.markdown("**Quick questions:**")
    qcols = st.columns(5)
    quick_qs = [
        "What are the main goals?",
        "Who are the key stakeholders?",
        "How will this be monitored?",
        "What is the implementation plan?",
        "What strategies are proposed?",
    ]
    for i, (qc, qq) in enumerate(zip(qcols, quick_qs)):
        if qc.button(qq, use_container_width=True, key=f"qq_btn_{i}"):
            st.session_state["_qquick"] = qq

    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if show_src and msg.get("sources"):
                for src, sc in msg["sources"]:
                    st.markdown(
                        f'<span class="src-chip">sim={sc:.2f}</span> '
                        f'<small style="color:#78909c">{src[:110]}…</small>',
                        unsafe_allow_html=True,
                    )

    user_q = st.chat_input("Ask anything about the policy document…")
    if "_qquick" in st.session_state:
        user_q = st.session_state.pop("_qquick")

    if user_q:
        st.session_state["chat_history"].append({"role": "user", "content": user_q})

        vec = st.session_state.get("rag_vec", None)
        mx = st.session_state.get("rag_matrix", None)
        chs = st.session_state.get("rag_chunks", None)

        if vec is None or mx is None or not chs:
            ans, sources = "⚠️ Please upload/paste and then click **Analyse Policy Document** first.", []
        else:
            api_key = st.session_state.get("api_key", "").strip()
            model_name = pick_model(api_key) if st.session_state.get("auto_model", True) else st.session_state.get("model_choice", "models/gemini-1.5-flash")
            bot_model = get_gemini_model(api_key, model_name)

            results = retrieve_chunks(user_q, vec, mx, chs)

            if not results:
                ans, sources = "This information is not found in the provided policy document.", []
            elif not bot_model:
                ans, sources = f"⚠️ {st.session_state.get('gemini_error', 'Gemini error')}", results
            else:
                ans = gemini_rag_answer(bot_model, user_q, [c for c, _ in results])
                sources = results

        st.session_state["chat_history"].append({"role": "assistant", "content": ans, "sources": sources})
        st.rerun()

    with st.expander("ℹ️ How the RAG Chatbot Works"):
        st.markdown("""
**Retrieval-Augmented Generation (RAG) Process:**
1. Document split into sentence chunks (6 sentences each)
2. Chunks indexed using TF-IDF bi-gram vectorisation
3. User query vectorised with the same vocabulary
4. Cosine similarity computed across all chunks
5. Top-K relevant chunks (above similarity threshold 0.10) retrieved
6. Gemini generates an answer using **only** those chunks as context

**Hallucination Prevention:**
- If max similarity < 0.10 → returns *"This information is not found in the provided policy document."*
- Gemini explicitly instructed to answer from retrieved excerpts only
- No model training knowledge injected into answers

> **Why TF-IDF over FAISS/Embeddings?** No heavy dependencies, no installation errors, no CUDA requirement — reliable, fast, and academically explainable.
        """)


# ─────────────────────────────────────────────────────────────
# 12) PAGE ▸ DASHBOARD
# ─────────────────────────────────────────────────────────────
elif "Dashboard" in st.session_state["active_page"]:

    st.markdown("""
    <div class="hero" style="padding:1.6rem 2rem 1.4rem;">
      <h1 style="font-size:1.7rem;">📊 Project Dashboard</h1>
      <p class="sub">System Overview · Objectives · Architecture · Design Decisions · Features</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="border-left:4px solid #00897b;background:linear-gradient(135deg,#e0f2f1,#fff);">
      <h3 style="color:#00695c;">✅ System Integration Summary</h3>
      <p>The <strong>Policy AI Assistant</strong> successfully integrates:</p>
      <ul>
        <li><strong>NLP Preprocessing</strong> — Regex-based text cleaning pipeline</li>
        <li><strong>Extractive Summarisation</strong> — TF-IDF sentence scoring and section classification</li>
        <li><strong>Generative AI Drafting</strong> — Gemini-powered scenario-based policy drafts</li>
        <li><strong>Retrieval-Grounded Chatbot</strong> — TF-IDF RAG with cosine similarity thresholding</li>
      </ul><br/>
      <p>The system demonstrates how Generative AI can assist in policy analysis while maintaining
      <strong>reliability and transparency</strong> through structured architecture and hallucination
      control mechanisms. This project showcases the practical application of Generative AI in
      <strong>governance and public administration contexts</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns(2, gap="large")

    with d1:
        st.markdown("""
        <div class="info-card">
          <h3>🎯 Project Objectives</h3>
          <ol style="color:#37474f;font-size:.9rem;line-height:1.8;padding-left:1.2rem;">
            <li>To build an AI-powered policy assistant using Generative AI</li>
            <li>To apply NLP preprocessing techniques on policy documents</li>
            <li>To implement structured policy summarisation</li>
            <li>To generate scenario-based policy drafts derived strictly from the summary</li>
            <li>To implement a Retrieval-Augmented Generation (RAG) chatbot grounded in policy text</li>
            <li>To prevent hallucination and ensure answer reliability</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <h3>💡 Key Capabilities</h3>
          <ul>
            <li>Extract key information from policy documents</li>
            <li>Summarise objectives and strategies into structured sections</li>
            <li>Generate adapted policy drafts for new scenarios</li>
            <li>Answer questions grounded strictly in policy content</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card" style="border-left:4px solid #e53935;">
          <h3>⚠️ Challenges &amp; Solutions</h3>
          <ul>
            <li><strong>Embedding install errors</strong> → Replaced with TF-IDF</li>
            <li><strong>FAISS dependency issues</strong> → Used cosine similarity locally</li>
            <li><strong>Hallucinated chatbot answers</strong> → Similarity threshold + strict prompts</li>
            <li><strong>Draft accuracy</strong> → Enforced summary-only generation pipeline</li>
            <li><strong>Scanned PDFs</strong> → User alert with clear error message</li>
            <li><strong>Local transformer fallback</strong> → Removed; Gemini-only approach</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown("""
        <div class="info-card">
          <h3>🔧 Technologies &amp; Components</h3>
          <table style="width:100%;font-size:.87rem;border-collapse:collapse;">
            <tr style="background:#e8eaf6;"><th style="padding:6px 10px;text-align:left;">Component</th><th style="padding:6px 10px;text-align:left;">Technology</th></tr>
            <tr><td style="padding:5px 10px;color:#546e7a;">Web Application</td><td style="padding:5px 10px;font-weight:600;color:#1a1f6b;">Streamlit</td></tr>
            <tr style="background:#fafbff;"><td style="padding:5px 10px;color:#546e7a;">PDF Extraction</td><td style="padding:5px 10px;font-weight:600;color:#1a1f6b;">pypdf</td></tr>
            <tr><td style="padding:5px 10px;color:#546e7a;">NLP / Summarisation</td><td style="padding:5px 10px;font-weight:600;color:#1a1f6b;">scikit-learn (TF-IDF)</td></tr>
            <tr style="background:#fafbff;"><td style="padding:5px 10px;color:#546e7a;">Similarity Matching</td><td style="padding:5px 10px;font-weight:600;color:#1a1f6b;">Cosine Similarity</td></tr>
            <tr><td style="padding:5px 10px;color:#546e7a;">Generative AI</td><td style="padding:5px 10px;font-weight:600;color:#1a1f6b;">Google Gemini API</td></tr>
            <tr style="background:#fafbff;"><td style="padding:5px 10px;color:#546e7a;">PDF Export</td><td style="padding:5px 10px;font-weight:600;color:#1a1f6b;">ReportLab</td></tr>
            <tr><td style="padding:5px 10px;color:#546e7a;">Environment Variables</td><td style="padding:5px 10px;font-weight:600;color:#1a1f6b;">python-dotenv</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <h3>🧩 Key Design Decisions</h3>
          <p><strong>Why TF-IDF instead of Embeddings?</strong></p>
          <ul>
            <li>No heavy dependencies or CUDA requirements</li>
            <li>No installation errors (FAISS, transformers avoided)</li>
            <li>Suitable for academic project scope</li>
            <li>Easier to explain in viva / presentations</li>
          </ul><br/>
          <p><strong>Why Gemini Only for generation?</strong></p>
          <ul>
            <li>High-quality, coherent text generation</li>
            <li>Stable, well-documented API</li>
            <li>Controlled via explicit system prompt constraints</li>
            <li>Easy integration with Python</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
          <h3>🏗️ Architecture Layers</h3>
          <div style="margin-top:.5rem;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:.5rem;">
              <span style="background:#1565c0;color:white;padding:4px 11px;border-radius:6px;font-size:.77rem;font-weight:600;white-space:nowrap;">NLP Layer (Local)</span>
              <span style="font-size:.82rem;color:#546e7a;">TF-IDF · Cosine Similarity · Regex Preprocessing</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:.5rem;">
              <span style="background:#6a1b9a;color:white;padding:4px 11px;border-radius:6px;font-size:.77rem;font-weight:600;white-space:nowrap;">Generative Layer (API)</span>
              <span style="font-size:.82rem;color:#546e7a;">Gemini · Scenario Drafts · Structured Output</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="background:#00695c;color:white;padding:4px 11px;border-radius:6px;font-size:.77rem;font-weight:600;white-space:nowrap;">RAG Layer (Hybrid)</span>
              <span style="font-size:.82rem;color:#546e7a;">TF-IDF Retrieval + Gemini Generation</span>
            </div>
          </div>
          <p style="font-size:.78rem;color:#78909c;margin-top:.8rem;">
            Separation ensures reliability, modularity, and easy academic explanation.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="margin-top:.3rem;">
      <h3>✅ Features Implemented</h3>
      <div style="display:flex;flex-wrap:wrap;gap:7px;margin-top:.6rem;">
        <span class="obj-pill">✔ Policy PDF Upload / Drag &amp; Drop</span>
        <span class="obj-pill">✔ Text Paste Input</span>
        <span class="obj-pill">✔ NLP Text Preprocessing (Regex)</span>
        <span class="obj-pill">✔ TF-IDF Extractive Summarisation</span>
        <span class="obj-pill">✔ 5-Section Structured Summary</span>
        <span class="obj-pill">✔ Keyword Extraction Display</span>
        <span class="obj-pill">✔ Scenario Draft Generation (2 Scenarios)</span>
        <span class="obj-pill">✔ 8 Preset + Custom Scenarios</span>
        <span class="obj-pill">✔ RAG Chatbot Grounded in Policy</span>
        <span class="obj-pill">✔ Hallucination Prevention (Threshold)</span>
        <span class="obj-pill">✔ Source Chunk Display Toggle</span>
        <span class="obj-pill">✔ PDF Export (Summary + Drafts)</span>
        <span class="obj-pill">✔ Session State Management</span>
        <span class="obj-pill">✔ Document Statistics Dashboard</span>
        <span class="obj-pill">✔ Progress Indicators</span>
        <span class="obj-pill">✔ Quick Question Chips</span>
        <span class="obj-pill">✔ Individual Draft Download</span>
      </div>
    </div>
    """, unsafe_allow_html=True)