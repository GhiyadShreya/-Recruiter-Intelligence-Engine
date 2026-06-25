"""
RIE — Central configuration.
"""
import os
from datetime import date

# ── Runtime reference date ──────────────────────────────────────────────────
# Defaults to the real current date so inactivity/notice calculations stay
# correct as time passes. Override via env var (e.g. for reproducing a past
# run or testing) by setting RIE_TODAY="YYYY-MM-DD".
_today_override = os.environ.get("RIE_TODAY")
TODAY = date.fromisoformat(_today_override) if _today_override else date.today()

# ── File paths ───────────────────────────────────────────────────────────────
CANDIDATES_FILE    = "candidates.jsonl"
EMBEDDINGS_FILE    = "models/candidate_embeddings.npy"
CANDIDATE_IDS_FILE = "models/candidate_ids.json"
SUBMISSION_FILE    = "outputs/submission.csv"

# ── Embedding model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384
EMBEDDING_BATCH = 64

# ── Retrieval settings ───────────────────────────────────────────────────────
BM25_TOP_K    = 5000
SEMANTIC_TOP_K = 5000
RRF_TOP_K     = 2000
RRF_K_CONSTANT = 60
FINAL_TOP_N   = 100

# ── Scoring weights (must sum to 1.0) ────────────────────────────────────────
W_SKILL    = 0.55
W_CAREER   = 0.25
W_LOCATION = 0.20

# ── Behavioral multiplier thresholds ─────────────────────────────────────────
MAX_INACTIVE_DAYS_FULL  = 180
MAX_INACTIVE_DAYS_HEAVY = 90
MAX_INACTIVE_DAYS_LIGHT = 30
NOTICE_MULTIPLIERS = [(30, 1.00), (60, 0.92), (90, 0.82), (180, 0.70)]

# ── Honeypot detection ────────────────────────────────────────────────────────
HONEYPOT_FLAGS_NEEDED    = 2
SKILL_DURATION_YOE_RATIO = 1.5
SALARY_MIN_MAX_RATIO     = 1.05
