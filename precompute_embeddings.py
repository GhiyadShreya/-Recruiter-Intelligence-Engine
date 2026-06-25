"""
RIE — Pre-computation  v2
==========================
Run ONCE before rank.py.

Usage:
    python precompute_embeddings.py --jd job_description.docx \
                                    --candidates candidates.jsonl

Outputs → models/:
    candidate_embeddings.npy    float32 [N, 384]
    candidate_ids.json          list[str]
    jd_embedding.npy            float32 [1, 384]
    jd_parsed.json              full parsed JD dict
"""

from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config as C
from jd_parser import parse_jd


def _skill_credibility(skill: dict) -> float:
    end_norm = min(skill.get("endorsements", 0), 50) / 50.0
    dur_norm = min(skill.get("duration_months", 0), 48) / 48.0
    return 0.50 * end_norm + 0.50 * dur_norm


def build_candidate_text(candidate: dict) -> str:
    p      = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    edu    = candidate.get("education", [])
    certs  = candidate.get("certifications", [])
    sig    = candidate["redrob_signals"]

    parts: list[str] = []

    parts.append(f"{p['current_title']} at {p['current_company']}.")

    if p.get("headline"):
        parts.append(p["headline"])

    if p.get("summary"):
        parts.append(p["summary"][:400])

    top_skills = sorted(skills, key=_skill_credibility, reverse=True)[:12]
    if top_skills:
        parts.append("Skills: " + ", ".join(s["name"] for s in top_skills) + ".")

    for i, r in enumerate(career[:4]):
        desc  = r.get("description", "")
        limit = 300 if i < 2 else 150
        if desc:
            parts.append(
                f"{r['title']} at {r['company']} ({r.get('industry', '')}): {desc[:limit]}"
            )
        else:
            parts.append(f"{r['title']} at {r['company']}.")

    edu_parts: list[str] = []
    for e in edu:
        field = e.get("field_of_study", "")
        tier  = e.get("tier", "")
        deg   = e.get("degree", "")
        if field:
            token = f"{deg} in {field}" if deg else field
            if tier:
                token += f" ({tier})"
            edu_parts.append(token)
    if edu_parts:
        parts.append("Education: " + "; ".join(edu_parts) + ".")

    country  = p.get("country", "")
    location = p.get("location", "")
    if country or location:
        geo = ", ".join(x for x in [location, country] if x)
        parts.append(f"Location: {geo}.")

    cert_names = [c.get("name", "") for c in certs if c.get("name")]
    if cert_names:
        parts.append("Certifications: " + ", ".join(cert_names[:4]) + ".")

    github = sig.get("github_activity_score", -1)
    if github >= 40:
        parts.append("Active open-source contributor with high GitHub activity.")
    elif github >= 20:
        parts.append("Moderate GitHub activity and open-source presence.")
    elif github > 0:
        parts.append("Some GitHub activity.")

    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="RIE — Pre-compute embeddings")
    parser.add_argument("--jd",         default="job_description.docx")
    parser.add_argument("--candidates", default=C.CANDIDATES_FILE)
    parser.add_argument("--batch-size", type=int, default=C.EMBEDDING_BATCH)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    t0 = time.time()

    # ── Step 1: Parse + save JD ───────────────────────────────────────────────
    print(f"Parsing JD: {args.jd}")
    if not os.path.exists(args.jd):
        raise FileNotFoundError(
            f"JD file not found: {args.jd}\n"
            f"Pass the correct path with --jd path/to/job_description.docx"
        )

    jd = parse_jd(args.jd)

    exp_min = jd.get("experience_min")
    exp_max = jd.get("experience_max")
    if exp_min is not None and exp_max is not None:
        exp_str = f"{exp_min}-{exp_max} years"
    elif exp_min is not None:
        exp_str = f"{exp_min}+ years"
    else:
        exp_str = "not specified"

    print(f"  Experience range   : {exp_str}")
    print(f"  JD type            : {jd.get('jd_type', 'unknown')}")
    print(f"  Must-have terms    : {jd.get('must_have_terms', [])}")
    print(f"  Nice-have terms    : {jd.get('nice_have_terms', [])}")
    print(f"  Preferred locations: {jd.get('preferred_locations', [])}")
    print(f"  Disqualifiers      : {len(jd.get('disqualifiers', []))} bullets")
    # BUG-8 FIX: safe-get optional key
    disq_terms = jd.get("disqualifier_terms", [])
    if disq_terms:
        print(f"  Disqualifier terms : {disq_terms}")
    print(f"  Embedding text     : {len(jd.get('embedding_text', ''))} chars")

    with open("models/jd_parsed.json", "w", encoding="utf-8") as f:
        json.dump(jd, f, indent=2, ensure_ascii=False)
    print("  Saved → models/jd_parsed.json")

    # ── Step 2: Load embedding model ──────────────────────────────────────────
    print(f"\nLoading model: {C.EMBEDDING_MODEL}")
    model = SentenceTransformer(C.EMBEDDING_MODEL)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Step 3: Embed JD ──────────────────────────────────────────────────────
    embedding_text = jd.get("embedding_text") or jd.get("raw_text", "")
    if not embedding_text:
        raise ValueError("JD has no embeddable text — check parse_jd output.")

    print("\nEmbedding JD ...")
    jd_embedding = model.encode(
        [embedding_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    np.save("models/jd_embedding.npy", jd_embedding.astype(np.float32))
    print("  Saved → models/jd_embedding.npy")

    # ── Step 4: Stream + embed candidates ────────────────────────────────────
    print(f"\nReading candidates from {args.candidates} ...")
    candidate_ids:   list[str] = []
    candidate_texts: list[str] = []
    n_empty = 0

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading", unit="cand"):
            line = line.strip()
            if not line:
                continue
            c   = json.loads(line)
            txt = build_candidate_text(c)
            if not txt.strip():
                n_empty += 1
                continue
            candidate_ids.append(c["candidate_id"])
            candidate_texts.append(txt)

    total = len(candidate_ids)
    print(f"  {total:,} candidates" + (f" ({n_empty} skipped — empty text)" if n_empty else ""))

    print(f"\nEncoding {total:,} candidates (batch={args.batch_size}) ...")
    t_enc = time.time()
    embeddings = model.encode(
        candidate_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t_enc
    print(f"  Encoded in {elapsed:.1f}s  ({total/max(elapsed,1):.0f} cand/s)")

    np.save(C.EMBEDDINGS_FILE, embeddings.astype(np.float32))
    with open(C.CANDIDATE_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)

    print(f"\nSaved:")
    print(f"  {C.EMBEDDINGS_FILE}  shape={embeddings.shape}")
    print(f"  {C.CANDIDATE_IDS_FILE}  {total:,} IDs")
    print(f"\nTotal pre-computation time: {time.time()-t0:.1f}s")
    print("Done.  Now run:  python rank.py --jd job_description.docx")


if __name__ == "__main__":
    main()
