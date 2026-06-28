---
title: RIE Candidate Ranker
emoji: 🎯
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# RIE — Recruiter Intelligence Engine

Upload a job description and a candidate pool. Get a ranked shortlist with a reason for every pick.

Multi-signal ranking: BM25 retrieval, trust-weighted skill scoring, honeypot detection, fresher-aware career scoring. No GPU, no API calls.

**Demo:** Upload `job_description.docx` and `sample_candidates.jsonl` to see the pipeline run.

---

## How it works

The pipeline runs in two phases:

### Phase 1 — Pre-computation (run once, not time-limited)

```bash
python precompute_embeddings.py --jd job_description.docx --candidates candidates.jsonl
```

- Parses the JD into structured signals: must-have terms, nice-to-have terms, disqualifier tokens, experience range, preferred locations
- Embeds the JD and all 100K candidates using a local sentence-transformer model (no network call — model must be cached)
- Saves embeddings to `models/` for the ranking step to load

### Phase 2 — Ranking (must complete in < 5 minutes)

```bash
python rank.py --candidates candidates.jsonl --out submission.csv
```

Full pipeline:
1. Load cached JD parse and embeddings
2. Stream candidates — filter honeypots and hard-disqualified profiles
3. BM25 retrieval → top 5,000 candidates
4. FAISS semantic retrieval → top 5,000 candidates
5. Reciprocal Rank Fusion → pool of 2,000
6. Multi-signal feature scoring on the 2,000-candidate pool
7. Sort, take top 100, generate per-candidate reasoning, write CSV

---

## Scoring signals

Each candidate receives scores across five components:

| Component | Weight | What it measures |
|---|---|---|
| Skill score | 40% | JD term matches, weighted by endorsements + duration |
| Career score | 35% | YOE fit, role depth, title alignment |
| Location score | 25% | Preferred city match, relocation willingness |
| Behavioral multiplier | ×modifier | Engagement, recency, notice period, GitHub activity |
| Honeypot penalty | hard filter | Structurally impossible profiles removed before scoring |

Career scoring is fresher-aware: when `experience_min = 0`, scoring uses a linear scale rather than a cliff. When a candidate has fewer than 2 career roles, education tier blends into the career score as a substitute signal.

---

## Honeypot detection

Six checks run on every candidate before scoring:

- Salary range inverted (min > max)
- Skill duration exceeds total career months
- Last-active date before signup date
- Negative role duration
- PhD listed before Bachelor in chronological order
- Expert-level proficiency with 0 months duration

Any two flags triggers removal. Honeypots are excluded before BM25 indexing and FAISS search.

---

## Setup

```bash
pip install -r requirements.txt
```

**Required files before running:**
```
job_description.docx      # provided in hackathon bundle
candidates.jsonl          # unzip from candidates.jsonl.gz
```

**Pre-compute embeddings (once):**
```bash
python precompute_embeddings.py \
    --jd job_description.docx \
    --candidates candidates.jsonl
```

**Produce submission CSV:**
```bash
python rank.py \
    --candidates candidates.jsonl \
    --out submission.csv
```

**Validate submission:**
```bash
python validate_submission.py submission.csv
```

---

## File structure

```
├── jd_parser.py               JD parsing — section detection, term extraction
├── precompute_embeddings.py   One-time embedding pre-computation
├── rank.py                    Main ranking pipeline (5-min budget)
├── scorer.py                  Multi-signal feature scorer
├── reasoner.py                Per-candidate reasoning generator
├── config.py                  Weights, thresholds, file paths
├── models/                    Pre-computed artifacts (generated, not committed)
│   ├── jd_parsed.json
│   ├── jd_embedding.npy
│   ├── candidate_embeddings.npy
│   └── candidate_ids.json
├── requirements.txt
└── submission_metadata.yaml
```

---

## Compute constraints

| Constraint | Value |
|---|---|
| Runtime (ranking step) | < 5 minutes |
| RAM | ≤ 16 GB |
| GPU | Not used |
| Network during ranking | Not used |
| LLM API calls | None |

Pre-computation (embeddings) runs offline and is not subject to the 5-minute limit.

---

## Requirements

```
python-docx
sentence-transformers
faiss-cpu
rank-bm25
numpy
scikit-learn
tqdm
```