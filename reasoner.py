"""
RIE — Reasoning generator.
Produces specific, data-backed 1-2 sentence reasoning for each ranked candidate.
Stage 4 checks: specific facts, JD connection, honest concerns, no hallucination,
variation across rows, rank consistency.
"""
from __future__ import annotations
from datetime import date, datetime
import config as C


def _parse_date(s):
    if not s: return None
    try: return datetime.strptime(s, "%Y-%m-%d").date()
    except: return None

def _days_inactive(sig):
    la = _parse_date(sig.get("last_active_date"))
    if not la: return 999
    return (C.TODAY - la).days


def generate_reasoning(candidate: dict, score_result: dict, rank: int) -> str:
    p   = candidate["profile"]
    sig = candidate["redrob_signals"]
    career = candidate.get("career_history", [])

    cid   = candidate["candidate_id"]
    title = p["current_title"]
    company = p["current_company"]
    yoe   = p["years_of_experience"]
    loc   = p["location"]

    matched_must = score_result.get("matched_must", [])
    matched_nice = score_result.get("matched_nice", [])
    notice = sig.get("notice_period_days", 90)
    resp   = sig.get("recruiter_response_rate", 0.5)
    otw    = sig.get("open_to_work_flag", False)
    github = sig.get("github_activity_score", -1)
    days_inactive = _days_inactive(sig)
    preferred_locs = score_result.get("preferred_locations", [])

    # ── Strength sentence ─────────────────────────────────────────────────────
    strength_parts = []
    strength_parts.append(f"{yoe:.1f}-year {title} at {company}")

    top_skills = (matched_must[:3] if matched_must else matched_nice[:2])
    if top_skills:
        strength_parts.append(f"with hands-on {', '.join(top_skills)}")

    if career:
        pr = career[0]
        strength_parts.append(f"(built at {pr['company']}, {pr['industry']})")

    availability_positives = []
    if otw: availability_positives.append("open to work")
    if notice <= 30: availability_positives.append("sub-30-day notice")
    elif notice <= 60: availability_positives.append(f"{notice}-day notice")
    if resp >= 0.70: availability_positives.append(f"{resp*100:.0f}% recruiter response rate")
    if github >= 30: availability_positives.append(f"github score {github:.0f}")

    in_preferred = False
    if preferred_locs:
        in_preferred = any(city.lower() in loc.lower() for city in preferred_locs)
        if in_preferred:
            availability_positives.append(f"based in {loc.split(',')[0]}")

    sentence1 = " ".join(strength_parts) + (
        f"; {', '.join(availability_positives[:2])}" if availability_positives else ""
    ) + "."

    # ── Concern / qualification sentence ─────────────────────────────────────
    concerns = []

    if days_inactive > 180: concerns.append(f"inactive on platform for {days_inactive} days")
    elif days_inactive > 90: concerns.append(f"last active {days_inactive} days ago")
    if notice > 90: concerns.append(f"{notice}-day notice period is above JD preference")
    if resp < 0.30: concerns.append(f"low recruiter response rate ({resp*100:.0f}%)")
    if not matched_must and matched_nice: concerns.append("no direct must-have skill matches")

    if preferred_locs:
        if not in_preferred and not sig.get("willing_to_relocate", False):
            concerns.append(f"based outside preferred locations ({loc.split(',')[0].strip()}), not willing to relocate")
        elif not in_preferred and sig.get("willing_to_relocate", False):
             concerns.append(f"based in {loc.split(',')[0].strip()}, relocation required")

    skill_score = score_result.get("skill_score", 0)
    career_score = score_result.get("career_score", 0)
    if skill_score < 0.25 and career_score > 0.5:
        concerns.append("career trajectory suggests fit but skill listing does not explicitly cover JD must-haves")

    if rank > 50 and not concerns:
        concerns.append("ranked lower due to competitive pool; profile meets minimum bar but not standout on key signals")

    if concerns:
        return f"{sentence1} Concern: {'; '.join(concerns[:2])}."
    else:
        return sentence1