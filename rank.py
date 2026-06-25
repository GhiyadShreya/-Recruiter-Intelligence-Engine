"""
RIE — Main Ranking Script  v2
==============================
Runs the full pipeline in < 5 min on CPU, 16 GB RAM, no network.

Pipeline:
  0. Load JD (cached jd_parsed.json or parse file fresh)
  1. Stream candidates — honeypot filter + hard disqualification
  2. BM25 retrieval → top BM25_TOP_K
  3. Semantic FAISS retrieval → top SEMANTIC_TOP_K  [if embeddings exist]
  4. RRF fusion → top RRF_TOP_K
  5. Feature scoring on fusion pool
  6. Top-100 selection → reasoning → CSV

"""

import argparse
import json
import os
import time
from collections import defaultdict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import config as C
from jd_parser import parse_jd
from scorer import score_candidate, detect_honeypot, is_hard_disqualified
from reasoner import generate_reasoning


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_exp(exp_min, exp_max) -> str:
    if exp_min is None and exp_max is None:
        return "unconstrained"
    if exp_min is None:
        return f"up to {exp_max} yrs"
    if exp_max is None:
        return f"{exp_min}+ yrs"
    return f"{exp_min}–{exp_max} yrs"


# ── JD loader ─────────────────────────────────────────────────────────────────

def load_jd(jd_path: str) -> dict:
    
    cached = "models/jd_parsed.json"
    if os.path.exists(cached):
        with open(cached) as f:
            jd = json.load(f)
        print(f"  Loaded cached JD from {cached}")
        # Back-fill jd_type if loaded from older cache (jd_parser < v3.1)
        if "jd_type" not in jd:
            from jd_parser import _classify_jd_type
            jd["jd_type"] = _classify_jd_type(jd.get("experience_min"), jd.get("experience_max"))
        return jd

    if os.path.exists(jd_path):
        print(f"  Parsing JD from {jd_path} ...")
        return parse_jd(jd_path)

    raise RuntimeError(
        f"JD not found: neither '{cached}' nor '{jd_path}' exists.\n"
        f"Run: python precompute_embeddings.py --jd <path_to_jd> --candidates <path_to_candidates>"
    )


# ── BM25 tokenisers ───────────────────────────────────────────────────────────

def build_bm25_tokens(candidate: dict) -> list[str]:
    tokens: list[str] = []
    p      = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    for word in p["current_title"].lower().split():
        tokens.extend([word] * 3)

    tokens.extend(p["current_industry"].lower().replace("/", " ").split())

    for s in skills:
        sname     = s["name"].lower().replace(" ", "_").replace("-", "_")
        dur_boost = min(s.get("duration_months", 0) // 12, 3)
        end_boost = min(s.get("endorsements", 0) // 15, 2)
        tokens.extend([sname] * (1 + dur_boost + end_boost))

    for r in career[:4]:
        desc = r.get("description", "")
        if desc:
            tokens.extend(desc.lower().split()[:80])
        for word in r["title"].lower().split():
            tokens.append(word)

    summary = p.get("summary", "")
    if summary:
        tokens.extend(summary.lower().split()[:60])

    return [t for t in tokens if len(t) > 2]


def build_jd_bm25_tokens(jd: dict) -> list[str]:
    tokens: list[str] = []
    for term in jd.get("must_have_terms", []):
        t = term.lower().replace(" ", "_").replace("-", "_")
        tokens.extend([t] * 3)
    for term in jd.get("nice_have_terms", []):
        t = term.lower().replace(" ", "_").replace("-", "_")
        tokens.append(t)
    for bullet in jd.get("must_have_skills", []):
        tokens.extend(bullet.lower().split()[:20])
    return [t for t in tokens if len(t) > 2]


# ── RRF ──────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = C.RRF_K_CONSTANT,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, cid in enumerate(ranked_list, start=1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── CSV writer ────────────────────────────────────────────────────────────────

def write_csv(results: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        for r in results:
            reasoning = r["reasoning"].replace('"', "'")
            f.write(f'{r["candidate_id"]},{r["rank"]},{r["score"]:.6f},"{reasoning}"\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RIE Candidate Ranker")
    parser.add_argument("--jd",             default="job_description.docx")
    parser.add_argument("--candidates",     default=C.CANDIDATES_FILE)
    parser.add_argument("--embeddings",     default=C.EMBEDDINGS_FILE)
    parser.add_argument("--ids",            default=C.CANDIDATE_IDS_FILE)
    parser.add_argument("--out",            default=C.SUBMISSION_FILE)
    parser.add_argument("--skip-retrieval", action="store_true",
                        help="BM25-only mode — faster but lower recall")
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs("outputs", exist_ok=True)

    # ── STEP 0: Load JD ───────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 0: Loading JD")
    jd = load_jd(args.jd)   # BUG-1 FIX: raises on missing
    print(f"  Must-have terms  : {jd.get('must_have_terms', [])}")
    print(f"  Experience range : {_fmt_exp(jd.get('experience_min'), jd.get('experience_max'))}")  # BUG-2 FIX
    print(f"  JD type          : {jd.get('jd_type', 'unspecified')}")
    print(f"  Preferred locs   : {jd.get('preferred_locations', [])}")

    jd_bm25_tokens = build_jd_bm25_tokens(jd)

    # ── STEP 1: Stream candidates, filter, build BM25 ─────────────────────────
    print("\nSTEP 1: Loading + filtering candidates")
    t1 = time.time()

    all_candidates: dict[str, dict] = {}
    bm25_corpus:    list[list[str]] = []
    bm25_ids:       list[str]       = []
    n_total = n_honeypot = n_disqualified = 0

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading+filtering", unit="cand"):
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            n_total += 1

            is_hp, _ = detect_honeypot(c)
            if is_hp:
                n_honeypot += 1
                continue

            disq, _ = is_hard_disqualified(c, jd)
            if disq:
                n_disqualified += 1
                continue

            cid = c["candidate_id"]
            all_candidates[cid] = c
            bm25_corpus.append(build_bm25_tokens(c))
            bm25_ids.append(cid)

    print(f"  Total: {n_total:,} | Honeypots: {n_honeypot:,} | "
          f"Disqualified: {n_disqualified:,} | Eligible: {len(all_candidates):,}")
    print(f"  Step 1 done in {time.time()-t1:.1f}s")

    # ── STEP 2: BM25 retrieval ────────────────────────────────────────────────
    print("\nSTEP 2: BM25 retrieval")
    t2 = time.time()
    bm25        = BM25Okapi(bm25_corpus)
    bm25_scores = bm25.get_scores(jd_bm25_tokens)
    bm25_top    = np.argsort(bm25_scores)[::-1][:C.BM25_TOP_K]
    bm25_ranked = [bm25_ids[i] for i in bm25_top]
    print(f"  {len(bm25_ranked):,} retrieved in {time.time()-t2:.1f}s")

    # ── STEP 3: Semantic retrieval ────────────────────────────────────────────
    if (not args.skip_retrieval
            and os.path.exists(args.embeddings)
            and os.path.exists(args.ids)):
        print("\nSTEP 3: Semantic retrieval (FAISS)")
        t3 = time.time()

        all_embeddings = np.load(args.embeddings)
        with open(args.ids) as f:
            all_ids = json.load(f)
        jd_embedding = np.load("models/jd_embedding.npy")

        eligible_set = set(all_candidates.keys())
        mask         = [i for i, cid in enumerate(all_ids) if cid in eligible_set]
        eligible_emb = all_embeddings[mask].astype(np.float32)
        eligible_ids = [all_ids[i] for i in mask]

        index = faiss.IndexFlatIP(C.EMBEDDING_DIM)
        index.add(eligible_emb)
        _, I         = index.search(jd_embedding, C.SEMANTIC_TOP_K)
        semantic_ranked = [eligible_ids[i] for i in I[0] if i < len(eligible_ids)]
        print(f"  {len(semantic_ranked):,} retrieved in {time.time()-t3:.1f}s")
    else:
        if not args.skip_retrieval:
            print("\nSTEP 3: Embeddings not found — BM25-only mode")
        semantic_ranked = bm25_ranked[:C.SEMANTIC_TOP_K]

    # ── STEP 4: RRF fusion ────────────────────────────────────────────────────
    print("\nSTEP 4: RRF fusion")
    t4 = time.time()
    rrf_results  = reciprocal_rank_fusion([bm25_ranked, semantic_ranked])
    rrf_pool_ids = [cid for cid, _ in rrf_results[:C.RRF_TOP_K] if cid in all_candidates]
    print(f"  Pool: {len(rrf_pool_ids):,} in {time.time()-t4:.1f}s")

    # ── STEP 5: Feature scoring ───────────────────────────────────────────────
    print("\nSTEP 5: Feature scoring")
    t5 = time.time()
    scored = []
    for cid in tqdm(rrf_pool_ids, desc="Scoring", unit="cand"):
        result              = score_candidate(all_candidates[cid], jd)
        result["_candidate"] = all_candidates[cid]
        scored.append(result)
    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    top100 = scored[:C.FINAL_TOP_N]
    print(f"  Scored {len(scored):,} in {time.time()-t5:.1f}s")

    # ── STEP 6: Reasoning + CSV ───────────────────────────────────────────────
    print("\nSTEP 6: Generating reasoning + writing CSV")
    t6 = time.time()

    output_rows = []
    for rank, result in enumerate(top100, start=1):
        # BUG-3 FIX: extract _candidate BEFORE mutation, not after sort
        c = result.pop("_candidate")
        reasoning = generate_reasoning(c, result, rank)
        output_rows.append({
            "candidate_id": result["candidate_id"],
            "rank":         rank,
            "score":        result["final_score"],
            "reasoning":    reasoning,
        })

    write_csv(output_rows, args.out)
    print(f"  Written in {time.time()-t6:.1f}s")

    total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"DONE — {total:.1f}s")
    if output_rows:
        print(f"Rank 1 : {output_rows[0]['candidate_id']}  (score={output_rows[0]['score']:.4f})")
    print(f"Output : {args.out}")
    print(f"Validate: python validate_submission.py {args.out}")


if __name__ == "__main__":
    main()
