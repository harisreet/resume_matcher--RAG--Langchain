"""
streamlit_app.py — AI Resume Matcher | HR Dashboard
Run: streamlit run frontend/streamlit_app.py
"""

import io
import time
import requests
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Resume Matcher | HR Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium dark theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2e 0%, #0a1628 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #63b3ed;
}

/* ── Headings ── */
h1 { color: #63b3ed !important; font-weight: 800 !important; }
h2 { color: #90cdf4 !important; font-weight: 700 !important; }
h3 { color: #bee3f8 !important; font-weight: 600 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 12px 16px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3182ce, #2b6cb0);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 15px;
    padding: 10px 28px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(49,130,206,0.35);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4299e1, #3182ce);
    box-shadow: 0 6px 20px rgba(49,130,206,0.55);
    transform: translateY(-1px);
}

/* ── Text area & inputs ── */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.2) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(99,179,237,0.05) !important;
    border: 2px dashed rgba(99,179,237,0.3) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(99,179,237,0.6) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(99,179,237,0.06);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(99,179,237,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #90cdf4;
    font-weight: 500;
    padding: 8px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2b6cb0, #2c5282) !important;
    color: white !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(99,179,237,0.08) !important;
    border-radius: 10px !important;
    color: #90cdf4 !important;
    font-weight: 600 !important;
}

/* ── Divider ── */
hr { border-color: rgba(99,179,237,0.15) !important; }

/* ── Candidate card ── */
.candidate-card {
    background: linear-gradient(135deg, rgba(13,27,46,0.95), rgba(10,22,40,0.95));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.candidate-card:hover {
    border-color: rgba(99,179,237,0.5);
    box-shadow: 0 8px 32px rgba(49,130,206,0.15);
}

/* ── Score badges ── */
.score-excellent { color: #68d391; font-weight: 800; font-size: 22px; }
.score-strong    { color: #63b3ed; font-weight: 800; font-size: 22px; }
.score-medium    { color: #f6e05e; font-weight: 800; font-size: 22px; }
.score-weak      { color: #fc8181; font-weight: 800; font-size: 22px; }

/* ── Skill chips ── */
.skill-chip-matched {
    display: inline-block;
    background: rgba(104,211,145,0.15);
    color: #68d391;
    border: 1px solid rgba(104,211,145,0.35);
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px;
    font-size: 13px;
    font-weight: 500;
}
.skill-chip-missing {
    display: inline-block;
    background: rgba(252,129,129,0.12);
    color: #fc8181;
    border: 1px solid rgba(252,129,129,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px;
    font-size: 13px;
    font-weight: 500;
}

/* ── Rank badge ── */
.rank-badge {
    background: linear-gradient(135deg, #3182ce, #2c5282);
    color: white;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 16px;
    box-shadow: 0 3px 10px rgba(49,130,206,0.4);
}

/* ── QA answer box ── */
.qa-answer-box {
    background: rgba(49,130,206,0.08);
    border: 1px solid rgba(99,179,237,0.25);
    border-left: 4px solid #63b3ed;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 12px;
    white-space: pre-wrap;
    line-height: 1.7;
}

/* ── Toast / info box ── */
.stAlert { border-radius: 10px !important; }

/* ── Dataframe ── */
.dataframe { color: #e2e8f0 !important; }

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #3182ce, #63b3ed) !important;
    border-radius: 4px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "match_results": [],
        "jd_text": "",
        "qa_answer": "",
        "qa_sources": [],
        "uploaded_files": [],
        "api_online": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ──────────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────────

def check_api() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def fetch_candidates() -> list:
    try:
        r = requests.get(f"{API_BASE}/candidates", timeout=5)
        return r.json().get("resumes", [])
    except Exception:
        return []


def upload_resumes(files) -> dict | None:
    try:
        file_tuples = [("files", (f.name, f.read(), "application/pdf")) for f in files]
        r = requests.post(f"{API_BASE}/upload-resumes", files=file_tuples, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def run_match(jd: str, top_k: int = 5) -> list:
    try:
        r = requests.post(
            f"{API_BASE}/match",
            json={"job_description": jd, "top_k": top_k},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("candidates", [])
    except Exception as e:
        st.error(f"Match failed: {e}")
        return []


def run_qa(question: str, files: list) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/qa",
            json={"question": question, "shortlisted_files": files},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Q&A failed: {e}")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────

def score_class(score: int) -> str:
    if score >= 85:
        return "score-excellent"
    elif score >= 70:
        return "score-strong"
    elif score >= 50:
        return "score-medium"
    return "score-weak"


def score_label(score: int) -> str:
    if score >= 85:
        return "🟢 Excellent"
    elif score >= 70:
        return "🔵 Strong"
    elif score >= 50:
        return "🟡 Medium"
    return "🔴 Weak"


def render_candidate_card(c: dict, idx: int):
    score = c.get("match_score", 0)
    matched = c.get("matched_skills", [])
    missing = c.get("missing_skills", [])
    evidence = c.get("evidence", "")
    preview = c.get("raw_text_preview", "")
    resume_file = c.get("resume_file", "unknown.pdf")

    matched_html = "".join(f'<span class="skill-chip-matched">✓ {s}</span>' for s in matched) or "<em style='color:#718096'>None listed</em>"
    missing_html = "".join(f'<span class="skill-chip-missing">✗ {s}</span>' for s in missing) or "<em style='color:#718096'>None</em>"

    st.markdown(f"""
    <div class="candidate-card">
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:14px;">
            <div class="rank-badge">#{c.get('rank', idx+1)}</div>
            <div>
                <div style="font-size:17px; font-weight:700; color:#e2e8f0;">📄 {resume_file}</div>
                <div style="font-size:13px; color:#718096; margin-top:2px;">{score_label(score)}</div>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <span class="{score_class(score)}">{score}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Score bar
    st.progress(score / 100)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Matched Skills**")
        st.markdown(matched_html, unsafe_allow_html=True)
    with col2:
        st.markdown("**❌ Missing Skills**")
        st.markdown(missing_html, unsafe_allow_html=True)

    if evidence:
        st.markdown(f"<div style='color:#a0aec0; font-size:13px; margin-top:10px; line-height:1.6;'>💡 {evidence}</div>", unsafe_allow_html=True)

    if preview:
        with st.expander("📖 Resume Preview"):
            st.markdown(f"<div style='color:#a0aec0; font-size:13px; white-space:pre-wrap;'>{preview}…</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:8px 0;'>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size: 48px;">🎯</div>
        <div style="font-size: 22px; font-weight: 800; color: #63b3ed; line-height: 1.2;">
            AI Resume<br>Matcher
        </div>
        <div style="font-size: 12px; color: #4a5568; margin-top: 6px;">
            Powered by LangChain + RAG
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # API status
    api_ok = check_api()
    st.session_state.api_online = api_ok
    if api_ok:
        st.markdown("🟢 **Backend Online**")
    else:
        st.markdown("🔴 **Backend Offline** — start FastAPI first")

    st.markdown("---")

    # Upload resumes
    st.markdown("### 📂 Upload Resumes")
    uploaded = st.file_uploader(
        "Drop PDF resumes here",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploader",
        label_visibility="collapsed",
    )

    if uploaded and st.button("⬆️ Upload to Server", use_container_width=True):
        if not api_ok:
            st.error("Backend is offline.")
        else:
            with st.spinner("Uploading…"):
                result = upload_resumes(uploaded)
            if result:
                st.success(result.get("message", "Uploaded!"))

    st.markdown("---")

    # Existing resumes
    st.markdown("### 📋 Uploaded Resumes")
    candidates = fetch_candidates()
    if candidates:
        for f in candidates:
            st.markdown(f"<div style='padding:6px 0; color:#90cdf4; font-size:13px;'>📄 {f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#4a5568; font-size:12px; margin-top:8px;'>Total: {len(candidates)} resume(s)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#4a5568; font-size:13px;'>No resumes uploaded yet.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='color:#4a5568; font-size:11px; text-align:center;'>LangChain · ChromaDB · Gemini 2.5 Flash</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style="margin-bottom: 4px;">🎯 AI Resume Screening Dashboard</h1>
<p style="color: #718096; font-size: 15px; margin-bottom: 24px;">
  Upload resumes → Enter a Job Description → Instantly rank candidates with AI match scores
</p>
""", unsafe_allow_html=True)

# KPI row
total = len(fetch_candidates())
matched = len(st.session_state.match_results)
top_score = max((c.get("match_score", 0) for c in st.session_state.match_results), default=0)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("📂 Resumes Uploaded", total)
kpi2.metric("🏆 Candidates Ranked", matched)
kpi3.metric("🎯 Top Match Score", f"{top_score}%" if top_score else "—")
kpi4.metric("🤖 AI Model", "Gemini 2.5 Flash")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_match, tab_board, tab_qa = st.tabs([
    "📋  Match Candidates",
    "🏆  Leaderboard",
    "💬  Candidate Q&A",
])


# ── TAB 1: Match ──────────────────────────────────────────────────────────────
with tab_match:
    st.markdown("### 📝 Job Description")
    jd = st.text_area(
        "Paste the Job Description here",
        height=200,
        placeholder="e.g. We are looking for a Senior Python Engineer with experience in FastAPI, PostgreSQL, Docker, and cloud deployments…",
        key="jd_input",
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([3, 1])
    with col_b:
        top_k = st.selectbox("Top candidates", [3, 5, 10], index=1)

    with col_a:
        run_btn = st.button("🚀 Run AI Match", use_container_width=True, type="primary")

    if run_btn:
        if not st.session_state.api_online:
            st.error("Backend is offline. Please start FastAPI first.")
        elif not jd.strip():
            st.warning("Please enter a Job Description.")
        elif total == 0:
            st.warning("No resumes uploaded. Please upload PDFs first.")
        else:
            st.session_state.jd_text = jd
            with st.spinner("🤖 Analysing resumes with AI… This may take 30-60 seconds."):
                results = run_match(jd, top_k=top_k)
            st.session_state.match_results = results
            if results:
                st.success(f"✅ Ranked {len(results)} candidate(s) successfully!")
            else:
                st.error("No results returned. Check the backend logs.")

    # Results
    if st.session_state.match_results:
        st.markdown("---")
        st.markdown(f"### 🏆 Ranked Candidates — {len(st.session_state.match_results)} Results")
        for i, c in enumerate(st.session_state.match_results):
            render_candidate_card(c, i)


# ── TAB 2: Leaderboard ────────────────────────────────────────────────────────
with tab_board:
    st.markdown("### 🏆 Candidate Leaderboard")

    if not st.session_state.match_results:
        st.info("Run a match first to see the leaderboard.")
    else:
        rows = []
        for c in st.session_state.match_results:
            rows.append({
                "Rank": c.get("rank"),
                "Resume File": c.get("resume_file"),
                "Match Score (%)": c.get("match_score"),
                "Matched Skills": ", ".join(c.get("matched_skills", [])),
                "Missing Skills": ", ".join(c.get("missing_skills", [])),
                "Verdict": score_label(c.get("match_score", 0)),
            })

        df = pd.DataFrame(rows)

        # Colour-coded score column
        def highlight_score(val):
            if val >= 85:
                return "background-color: rgba(104,211,145,0.15); color: #68d391; font-weight: bold;"
            elif val >= 70:
                return "background-color: rgba(99,179,237,0.15); color: #63b3ed; font-weight: bold;"
            elif val >= 50:
                return "background-color: rgba(246,224,94,0.12); color: #f6e05e; font-weight: bold;"
            return "background-color: rgba(252,129,129,0.12); color: #fc8181; font-weight: bold;"

        # Use map for pandas 2.1.0+ and fallback to applymap for older versions
        if hasattr(df.style, "map"):
            styled = df.style.map(highlight_score, subset=["Match Score (%)"])
        else:
            styled = df.style.applymap(highlight_score, subset=["Match Score (%)"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv,
            file_name="candidate_leaderboard.csv",
            mime="text/csv",
            use_container_width=False,
        )

        # Score distribution mini bar chart
        st.markdown("---")
        st.markdown("### 📊 Score Distribution")
        chart_df = df[["Resume File", "Match Score (%)"]].set_index("Resume File")
        st.bar_chart(chart_df)


# ── TAB 3: Q&A ────────────────────────────────────────────────────────────────
with tab_qa:
    st.markdown("### 💬 Ask About Shortlisted Candidates")
    st.markdown("<div style='color:#718096; font-size:14px; margin-bottom:16px;'>Select candidates from the match results and ask any question about them.</div>", unsafe_allow_html=True)

    if not st.session_state.match_results:
        st.info("Run a match first to shortlist candidates.")
    else:
        all_files = [c.get("resume_file") for c in st.session_state.match_results]
        selected = st.multiselect(
            "Shortlist candidates to query",
            options=all_files,
            default=all_files[:2],
            key="qa_select",
        )

        question = st.text_input(
            "Ask a question about the selected candidates",
            placeholder="e.g. Who has stronger experience in Python? Who should I interview first?",
            key="qa_question",
        )

        ask_btn = st.button("🔍 Ask AI", use_container_width=False, type="primary")

        if ask_btn:
            if not question.strip():
                st.warning("Please enter a question.")
            elif not selected:
                st.warning("Please select at least one candidate.")
            elif not st.session_state.api_online:
                st.error("Backend is offline.")
            else:
                with st.spinner("🤖 Querying candidates…"):
                    result = run_qa(question, selected)

                st.session_state.qa_answer = result.get("answer", "")
                st.session_state.qa_sources = result.get("sources", [])

        if st.session_state.qa_answer:
            st.markdown("#### 🤖 AI Answer")
            st.markdown(
                f'<div class="qa-answer-box">{st.session_state.qa_answer}</div>',
                unsafe_allow_html=True,
            )
            if st.session_state.qa_sources:
                st.markdown(
                    f"<div style='color:#4a5568; font-size:12px; margin-top:8px;'>📎 Sources: {', '.join(st.session_state.qa_sources)}</div>",
                    unsafe_allow_html=True,
                )
