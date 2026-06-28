"""
app.py  —  RIE Streamlit UI  (clean version)

What it shows:
  1. Upload JD
  2. Upload candidate pool (.jsonl)
  3. Click Run
  4. Get ranked list with score + reasoning per candidate

No internal signals exposed. No must-have / disqualifier panels.

Run:
    streamlit run app.py

Place alongside: jd_parser.py, scorer.py, reasoner.py, config.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import types
from collections import defaultdict
from datetime import date

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="RIE — Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Config mock (used when config.py is absent) ───────────────────────────────
if "config" not in sys.modules:
    _C = types.ModuleType("config")
    _C.TODAY                    = date.today()
    _C.CANDIDATES_FILE          = "candidates.jsonl"
    _C.EMBEDDINGS_FILE          = "models/candidate_embeddings.npy"
    _C.CANDIDATE_IDS_FILE       = "models/candidate_ids.json"
    _C.SUBMISSION_FILE          = "outputs/submission.csv"
    _C.EMBEDDING_MODEL          = "BAAI/bge-small-en-v1.5"
    _C.EMBEDDING_DIM            = 384
    _C.EMBEDDING_BATCH          = 256
    _C.BM25_TOP_K               = 5000
    _C.SEMANTIC_TOP_K           = 5000
    _C.RRF_TOP_K                = 2000
    _C.RRF_K_CONSTANT           = 60
    _C.FINAL_TOP_N              = 100
    _C.W_SKILL                  = 0.40
    _C.W_CAREER                 = 0.35
    _C.W_LOCATION               = 0.25
    _C.SALARY_MIN_MAX_RATIO     = 1.1
    _C.SKILL_DURATION_YOE_RATIO = 1.2
    _C.HONEYPOT_FLAGS_NEEDED    = 2
    _C.MAX_INACTIVE_DAYS_FULL   = 365
    _C.MAX_INACTIVE_DAYS_HEAVY  = 180
    _C.MAX_INACTIVE_DAYS_LIGHT  = 90
    _C.NOTICE_MULTIPLIERS       = [(30, 1.08), (60, 1.03), (90, 0.95)]
    _C.MUST_HAVE_SKILLS         = []
    _C.NICE_TO_HAVE_SKILLS      = []
    sys.modules["config"] = _C

# ── Pipeline imports ──────────────────────────────────────────────────────────
try:
    from jd_parser import parse_jd
    from scorer   import score_candidate, detect_honeypot, is_hard_disqualified
    from reasoner import generate_reasoning
    PIPELINE_OK = True
except ImportError as e:
    PIPELINE_OK    = False
    PIPELINE_ERROR = str(e)

# ── BM25 helpers (inlined from rank.py — avoids argparse conflict) ────────────
def _bm25_candidate_tokens(c: dict) -> list[str]:
    tokens: list[str] = []
    p      = c["profile"]
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    for w in p["current_title"].lower().split():
        tokens.extend([w] * 3)
    tokens.extend(p.get("current_industry", "").lower().replace("/", " ").split())
    for s in skills:
        name      = s["name"].lower().replace(" ", "_").replace("-", "_")
        dur_boost = min(s.get("duration_months", 0) // 12, 3)
        end_boost = min(s.get("endorsements", 0) // 15, 2)
        tokens.extend([name] * (1 + dur_boost + end_boost))
    for r in career[:2]:
        desc = r.get("description", "")
        if desc:
            tokens.extend(desc.lower().split()[:80])
        tokens.extend(r["title"].lower().split())
    summary = p.get("summary", "")
    if summary:
        tokens.extend(summary.lower().split()[:60])
    return [t for t in tokens if len(t) > 2]


def _bm25_jd_tokens(jd: dict) -> list[str]:
    tokens: list[str] = []
    for t in jd.get("must_have_terms", []):
        tokens.extend([t.lower().replace(" ","_").replace("-","_")] * 3)
    for t in jd.get("nice_have_terms", []):
        tokens.append(t.lower().replace(" ","_").replace("-","_"))
    for b in jd.get("must_have_skills", []):
        tokens.extend(b.lower().split()[:20])
    return [t for t in tokens if len(t) > 2]


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for rl in ranked_lists:
        for rank, cid in enumerate(rl, start=1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def _to_csv(rows: list[dict]) -> bytes:
    lines = ["candidate_id,rank,score,reasoning"]
    for r in rows:
        lines.append(
            f'{r["candidate_id"]},{r["rank"]},{r["score"]:.6f},'
            f'"{r["reasoning"].replace(chr(34), chr(39))}"'
        )
    return "\n".join(lines).encode("utf-8")


# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0d10 !important;
    color: #e2ddd6 !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

/* ── Upload cards ── */
.upload-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin: 2rem 0;
}
.upload-card {
    background: #13131a;
    border: 1.5px dashed #2c2b38;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.upload-card:hover { border-color: #5b4fe8; }
.upload-card-icon { font-size: 2rem; margin-bottom: 8px; }
.upload-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2ddd6;
    margin-bottom: 4px;
}
.upload-card-sub {
    font-size: 0.78rem;
    color: #5c5970;
}
.upload-done {
    border-color: #3d9e6a !important;
    border-style: solid !important;
}
.upload-done .upload-card-title { color: #45c97e; }

/* ── Hero ── */
.hero {
    padding: 3.5rem 0 2rem;
    text-align: center;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #5b4fe8;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: #f0ede8;
    margin-bottom: 12px;
}
.hero-sub {
    font-size: 1rem;
    color: #5c5970;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Run button ── */
.stButton > button {
    width: 100%;
    background: #5b4fe8 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.01em;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #6d62ef !important; }
.stButton > button:disabled {
    background: #1e1d28 !important;
    color: #3a3850 !important;
}

/* ── Stats strip ── */
.stats-strip {
    display: flex;
    gap: 12px;
    margin: 2rem 0 1.5rem;
}
.stat-pill {
    background: #13131a;
    border: 1px solid #2c2b38;
    border-radius: 8px;
    padding: 10px 18px;
    flex: 1;
    text-align: center;
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #5b4fe8;
    line-height: 1;
}
.stat-val.g { color: #45c97e; }
.stat-val.r { color: #e85d5d; }
.stat-val.a { color: #e8a145; }
.stat-lbl {
    font-size: 11px;
    color: #5c5970;
    margin-top: 3px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Candidate list ── */
.cand-row {
    display: grid;
    grid-template-columns: 48px 1fr auto;
    gap: 16px;
    align-items: start;
    background: #13131a;
    border: 1px solid #1e1d28;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}
.cand-row:hover { border-color: #2c2b38; }
.cand-rank-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #2c2b38;
    padding-top: 2px;
    text-align: center;
}
.cand-rank-num.top3 { color: #5b4fe8; }
.cand-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2ddd6;
    margin-bottom: 2px;
}
.cand-meta {
    font-size: 0.8rem;
    color: #5c5970;
    margin-bottom: 8px;
}
.cand-reason {
    font-size: 0.83rem;
    color: #9b97a8;
    line-height: 1.55;
    border-left: 2px solid #2c2b38;
    padding-left: 10px;
    margin-top: 6px;
}
.score-badge {
    background: #1a1927;
    border: 1px solid #2c2b38;
    border-radius: 8px;
    padding: 8px 12px;
    text-align: center;
    min-width: 58px;
}
.score-big {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #5b4fe8;
    line-height: 1;
}
.score-lbl {
    font-size: 10px;
    color: #5c5970;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}

/* ── Progress steps ── */
.steps-wrap { padding: 1rem 0 0.5rem; }
.step-line {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    font-size: 0.85rem;
    color: #3a3850;
}
.step-line.done  { color: #45c97e; }
.step-line.active { color: #e2ddd6; }
.dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #2c2b38; flex-shrink: 0;
}
.dot.done   { background: #45c97e; }
.dot.active { background: #5b4fe8; }
.step-t {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #45c97e;
}

/* ── Download ── */
[data-testid="stDownloadButton"] button {
    background: #13131a !important;
    border: 1px solid #2c2b38 !important;
    color: #e2ddd6 !important;
    border-radius: 8px !important;
    width: 100%;
    font-size: 0.88rem !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #5b4fe8 !important;
    color: #5b4fe8 !important;
}

/* ── File uploader cleanup ── */
[data-testid="stFileUploader"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] section {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}
.stFileUploader label { display: none !important; }

/* ── Divider ── */
hr { border-color: #1e1d28 !important; margin: 2rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline guard ────────────────────────────────────────────────────────────
if not PIPELINE_OK:
    st.error(
        f"**Pipeline modules not found:** {PIPELINE_ERROR}\n\n"
        "Place `jd_parser.py`, `scorer.py`, `reasoner.py` in the same folder as `app.py`."
    )
    st.stop()


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("results", None),
    ("stats",   None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Recruiter Intelligence Engine</div>
  <div class="hero-title">Find the right candidates,<br>not just the matching keywords.</div>
  <div class="hero-sub">Upload a job description and a candidate pool. Get a ranked shortlist with a reason for every pick.</div>
</div>
""", unsafe_allow_html=True)


# ── Upload section ────────────────────────────────────────────────────────────
col_jd, col_cand = st.columns(2, gap="large")

with col_jd:
    jd_file = st.file_uploader(
        "jd", type=["docx", "md", "txt"],
        label_visibility="collapsed",
        key="jd_upload",
    )
    done_cls = "upload-done" if jd_file else ""
    icon     = "✅" if jd_file else "📄"
    title    = jd_file.name if jd_file else "Job Description"
    sub      = "Ready" if jd_file else ".docx · .md · .txt"
    st.markdown(f"""
    <div class="upload-card {done_cls}">
      <div class="upload-card-icon">{icon}</div>
      <div class="upload-card-title">{title}</div>
      <div class="upload-card-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

with col_cand:
    cand_file = st.file_uploader(
        "candidates", type=["jsonl", "json"],
        label_visibility="collapsed",
        key="cand_upload",
    )
    done_cls = "upload-done" if cand_file else ""
    icon     = "✅" if cand_file else "👥"
    title    = cand_file.name if cand_file else "Candidate Pool"
    sub      = "Ready" if cand_file else ".jsonl"
    st.markdown(f"""
    <div class="upload-card {done_cls}">
      <div class="upload-card-icon">{icon}</div>
      <div class="upload-card-title">{title}</div>
      <div class="upload-card-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Run button ────────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run = st.button(
        "Rank Candidates",
        disabled=(jd_file is None or cand_file is None),
    )


# ── Pipeline execution ────────────────────────────────────────────────────────
if run:
    st.session_state.results = None
    st.session_state.stats   = None

    t_global = time.time()

    # Write uploads to temp files
    jd_file.seek(0)
    cand_file.seek(0)

    with tempfile.NamedTemporaryFile(
        suffix=f".{jd_file.name.rsplit('.', 1)[-1]}", delete=False
    ) as jd_tmp:
        jd_tmp.write(jd_file.read())
        jd_path = jd_tmp.name

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as c_tmp:
        c_tmp.write(cand_file.read())
        cand_path = c_tmp.name

    # ── Progress renderer ─────────────────────────────────────────────────────
    STEP_LABELS = [
        "Reading job description",
        "Loading candidates",
        "Retrieving top matches",
        "Scoring candidates",
        "Writing reasons",
    ]
    done_steps: list[tuple[str, float]] = []
    progress_ph = st.empty()

    def _render(active: str = "") -> None:
        html = '<div class="steps-wrap">'
        for s in STEP_LABELS:
            is_done   = any(s == d[0] for d in done_steps)
            is_active = s == active
            row_cls   = "done" if is_done else ("active" if is_active else "")
            dot_cls   = "done" if is_done else ("active" if is_active else "")
            t_html    = ""
            if is_done:
                elapsed = next(d[1] for d in done_steps if d[0] == s)
                t_html  = f'<span class="step-t">{elapsed:.1f}s</span>'
            html += (
                f'<div class="step-line {row_cls}">'
                f'<div class="dot {dot_cls}"></div>{s}{t_html}</div>'
            )
        html += "</div>"
        progress_ph.markdown(html, unsafe_allow_html=True)

    # ── Step 1: Parse JD ──────────────────────────────────────────────────────
    _render("Reading job description")
    t0 = time.time()
    jd = parse_jd(jd_path)
    os.unlink(jd_path)
    done_steps.append(("Reading job description", time.time() - t0))
    _render("Loading candidates")

    # ── Step 2: Load + filter candidates ──────────────────────────────────────
    t1 = time.time()
    all_cands: dict[str, dict] = {}
    corpus:    list[list[str]] = []
    ids:       list[str]       = []
    n_total = n_hp = n_disq = 0

    with open(cand_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1

            is_hp, _ = detect_honeypot(c)
            if is_hp:
                n_hp += 1
                continue

            disq, _ = is_hard_disqualified(c, jd)
            if disq:
                n_disq += 1
                continue

            cid = c["candidate_id"]
            all_cands[cid] = c
            corpus.append(_bm25_candidate_tokens(c))
            ids.append(cid)

    os.unlink(cand_path)
    done_steps.append(("Loading candidates", time.time() - t1))
    _render("Retrieving top matches")

    # ── Step 3: BM25 + RRF ────────────────────────────────────────────────────
    t2 = time.time()
    bm25_ranked: list[str] = []
    if corpus:
        from rank_bm25 import BM25Okapi
        jd_tokens   = _bm25_jd_tokens(jd)
        bm25        = BM25Okapi(corpus)
        scores      = bm25.get_scores(jd_tokens) if jd_tokens else np.zeros(len(corpus))
        top_k       = min(5000, len(ids))
        top_idx     = np.argsort(scores)[::-1][:top_k]
        bm25_ranked = [ids[i] for i in top_idx]

    rrf         = _rrf([bm25_ranked])
    pool_ids    = [cid for cid, _ in rrf[:2000] if cid in all_cands]
    done_steps.append(("Retrieving top matches", time.time() - t2))
    _render("Scoring candidates")

    # ── Step 4: Score ─────────────────────────────────────────────────────────
    t3 = time.time()
    scored = []
    for cid in pool_ids:
        r = score_candidate(all_cands[cid], jd)
        r["_c"] = all_cands[cid]
        scored.append(r)

    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    top100 = scored[:100]
    done_steps.append(("Scoring candidates", time.time() - t3))
    _render("Writing reasons")

    # ── Step 5: Reasoning ─────────────────────────────────────────────────────
    t4 = time.time()
    output: list[dict] = []
    for rank, r in enumerate(top100, start=1):
        c         = r.pop("_c")
        reasoning = generate_reasoning(c, r, rank)
        output.append({
            "candidate_id": r["candidate_id"],
            "rank":         rank,
            "score":        r["final_score"],
            "reasoning":    reasoning,
            "_c":           c,
        })

    done_steps.append(("Writing reasons", time.time() - t4))
    _render()

    st.session_state.results = output
    st.session_state.stats   = {
        "total":      n_total,
        "honeypots":  n_hp,
        "disq":       n_disq,
        "eligible":   len(all_cands),
        "elapsed":    time.time() - t_global,
    }


# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results
    stats   = st.session_state.stats

    st.markdown("<hr>", unsafe_allow_html=True)

    # Stats strip
    st.markdown(f"""
    <div class="stats-strip">
      <div class="stat-pill">
        <div class="stat-val">{stats['total']:,}</div>
        <div class="stat-lbl">Candidates</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val g">{stats['eligible']:,}</div>
        <div class="stat-lbl">Eligible</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val r">{stats['honeypots']:,}</div>
        <div class="stat-lbl">Filtered out</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val">{len(results)}</div>
        <div class="stat-lbl">Shortlisted</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val a">{stats['elapsed']:.1f}s</div>
        <div class="stat-lbl">Runtime</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Download
    st.download_button(
        "⬇  Download submission.csv",
        data=_to_csv([
            {"candidate_id": r["candidate_id"],
             "rank":         r["rank"],
             "score":        r["score"],
             "reasoning":    r["reasoning"]}
            for r in results
        ]),
        file_name="submission.csv",
        mime="text/csv",
    )

    st.markdown(f"### Top {len(results)} candidates")

    # Candidate rows
    for row in results:
        c   = row["_c"]
        p   = c["profile"]
        yoe = p["years_of_experience"]
        pct = int(row["score"] * 100)
        top_cls = "top3" if row["rank"] <= 3 else ""

        st.markdown(f"""
        <div class="cand-row">
          <div class="cand-rank-num {top_cls}">#{row['rank']}</div>
          <div>
            <div class="cand-title">{p['current_title']} · {p['current_company']}</div>
            <div class="cand-meta">{p['location']} · {yoe:.1f} yrs experience · {row['candidate_id']}</div>
            <div class="cand-reason">{row['reasoning']}</div>
          </div>
          <div class="score-badge">
            <div class="score-big">{pct}</div>
            <div class="score-lbl">score</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── Empty state ───────────────────────────────────────────────────────────────
elif not run and not st.session_state.results:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0 4rem;color:#3a3850">
      <div style="font-size:0.85rem;letter-spacing:0.1em;text-transform:uppercase;
                  font-family:'JetBrains Mono',monospace">
        Upload both files above to get started
      </div>
    </div>
    """, unsafe_allow_html=True)