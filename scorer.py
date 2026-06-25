"""
scorer.py  —  RIE Candidate Scorer  v2.

Computes all features and final score for a single candidate dict.

"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import config as C


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_date(s: Optional[str]):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


# ── Experience range — safe None handling ─────────────────────────────────────

def _safe_exp(jd: dict) -> tuple[Optional[float], Optional[float]]:
    
    lo = jd.get("experience_min")
    hi = jd.get("experience_max")
    if lo is None and hi is None:
        return None, None
    lo = float(lo) if lo is not None else 0.0
    hi = float(hi) if hi is not None else lo + 8.0
    return lo, hi


# ── YOE scoring — fresher-aware ───────────────────────────────────────────────

def _compute_yoe_score(yoe: float, exp_min: Optional[float], exp_max: Optional[float]) -> float:
    
    if exp_min is None:
        return 1.0

    if exp_min == 0.0:
        # Fresher / junior JD
        cap = exp_max if exp_max and exp_max > 0 else 2.0
        if yoe <= cap:
            return min(yoe / cap, 1.0) if cap > 0 else 1.0
        else:
            overshoot = yoe - cap
            return max(1.0 - overshoot * 0.08, 0.50)

    # Standard tiered
    hi = exp_max if exp_max is not None else exp_min + 6.0
    if exp_min <= yoe <= hi:
        return 1.00
    if (exp_min - 1 <= yoe < exp_min) or (hi < yoe <= hi + 3):
        return 0.75
    if (exp_min - 2 <= yoe < exp_min - 1) or (hi + 3 < yoe <= hi + 6):
        return 0.50
    return 0.25


# ── Disqualifier term extraction ──────────────────────────────────────────────

def _extract_disqualifier_terms(disqualifier_bullets: list[str]) -> list[str]:
    
    terms: set[str] = set()

    _role_re = re.compile(
        r'\b(consultant|analyst|manager|director|vp|executive|architect|'
        r'intern|fresher|student|contractor|freelancer|coach|trainer|'
        r'advisor|researcher|lecturer|professor|teacher|recruiter)\b',
        re.I,
    )
    _negated_re = re.compile(
        r'\b(?:without|never|no)\s+([a-z][a-z\s]{2,20}?)'
        r'(?:\s+experience|\s+skills?|\s+background|[,.]|$)',
        re.I,
    )

    for bullet in disqualifier_bullets:
        for m in _role_re.finditer(bullet):
            terms.add(m.group(1).lower())
        for m in _negated_re.finditer(bullet):
            terms.add(m.group(1).strip().lower())
        for m in re.finditer(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', bullet):
            t = m.group(1).lower()
            if len(t) > 3 and t not in {
                'things', 'that', 'who', 'have', 'when',
                'this', 'they', 'people', 'person',
            }:
                terms.add(t)

    return list(terms)


def _get_disq_terms(jd: dict) -> list[str]:
    """
    Use pre-extracted disqualifier_terms if present (from jd_parser v2),
    otherwise extract from raw bullets now.
    """
    pre = jd.get("disqualifier_terms", [])
    if pre:
        return pre
    return _extract_disqualifier_terms(jd.get("disqualifiers", []))


# ── Skill matching — word-boundary regex ──────────────────────────────────────

def _skill_match_score(skill_name: str, keyword_list: list[str]) -> bool:
    
    sn = skill_name.lower()
    for kw in keyword_list:
        kw = kw.lower().strip()
        if not kw:
            continue
        if len(kw) <= 3:
            if re.search(r'\b' + re.escape(kw) + r'\b', sn):
                return True
        else:
            if re.search(r'\b' + re.escape(kw), sn):
                return True
    return False


# ── Honeypot detection ────────────────────────────────────────────────────────

def detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """
    Detects structurally impossible candidate profiles (planted fake records).

    Checks:
      1. Salary range inverted (min > max)
      2. Skill duration exceeds total career months by large margin
      3. Last-active date before signup date
      4. Negative role duration
      5. PhD listed before Bachelor in chronological order
      6. FIX-10: Expert/advanced skill with 0 months duration

    Returns (is_honeypot, evidence_flags).
    Requires C.HONEYPOT_FLAGS_NEEDED flags to trigger (default 2).
    """
    flags: list[str] = []
    p      = candidate["profile"]
    sig    = candidate["redrob_signals"]
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    yoe_months = p["years_of_experience"] * 12

    # 1. Salary inverted
    sal = sig.get("expected_salary_range_inr_lpa", {})
    if sal and sal.get("min", 0) > sal.get("max", 999) * C.SALARY_MIN_MAX_RATIO:
        flags.append(f"salary_inverted(min={sal['min']},max={sal['max']})")

    # 2. Skill duration impossible
    for s in skills:
        dm = s.get("duration_months", 0)
        if dm > yoe_months * C.SKILL_DURATION_YOE_RATIO + 24:
            flags.append(
                f"skill_duration_impossible"
                f"({s['name']}:{dm}mo > yoe:{yoe_months:.0f}mo)"
            )
            break

    # 3. Active before signup
    la = sig.get("last_active_date", "")
    su = sig.get("signup_date", "")
    if la and su and la < su:
        flags.append("active_before_signup")

    # 4. Negative duration
    for r in career:
        if r.get("duration_months", 0) < 0:
            flags.append(f"negative_duration({r['company']})")
            break

    # 5. PhD before Bachelor
    edu = sorted(
        candidate.get("education", []),
        key=lambda e: e.get("start_year", 9999),
    )
    degrees = [e.get("degree", "").lower() for e in edu]
    for i, d in enumerate(degrees):
        if any(x in d for x in ["ph.d", "phd", "doctor"]):
            rest = degrees[i + 1:]
            if any(
                any(x in r for x in ["b.e", "b.tech", "bachelor", "b.sc"])
                for r in rest
            ):
                flags.append("phd_before_bachelor")
                break

    # 6. FIX-10: Expert with 0-month duration
    for s in skills:
        if (
            s.get("duration_months", 1) == 0
            and s.get("proficiency", "") in ("expert", "advanced")
        ):
            flags.append(f"expert_zero_duration({s['name']})")
            break

    return len(flags) >= C.HONEYPOT_FLAGS_NEEDED, flags


# ── Hard disqualification ─────────────────────────────────────────────────────

def is_hard_disqualified(
    candidate: dict,
    jd: Optional[dict] = None,
) -> tuple[bool, str]:
    
    if not jd:
        return False, ""

    disq_terms = _get_disq_terms(jd)
    if not disq_terms:
        return False, ""

    p      = candidate["profile"]
    career = candidate.get("career_history", [])
    title  = p["current_title"].lower()

    for term in disq_terms:
        if re.search(r'\b' + re.escape(term) + r'\b', title):
            return True, f"title_matches_disqualifier:{term}"

    all_titles = [r["title"].lower() for r in career]
    if all_titles and all(
        any(re.search(r'\b' + re.escape(t) + r'\b', role) for t in disq_terms)
        for role in all_titles
    ):
        return True, "entire_career_in_disqualified_domain"

    return False, ""


# ── Skill scoring ─────────────────────────────────────────────────────────────

def compute_skill_score(
    candidate: dict,
    must_have_skills: Optional[list[str]] = None,
    nice_to_have_skills: Optional[list[str]] = None,
) -> tuple[float, list[str], list[str]]:
    """
    Scores how well the candidate's skills match JD requirements.

    Trust per skill: base 0.40, up to 1.0 with endorsement and duration.
    Normalisation caps:
      endorsements:   50  (p95 of real data is ~20; 50 is generous ceiling)
      duration_months: 48  (4 years)

    FRESHER NOTE:
      A fresher with endorsements=10, duration=10 gets:
        trust = 0.40 + 0.30*(10/50) + 0.30*(10/48) = 0.40 + 0.06 + 0.063 = 0.52
      A veteran with endorsements=40, duration=36 gets:
        trust = 0.40 + 0.30*(40/50) + 0.30*(36/48) = 0.40 + 0.24 + 0.225 = 0.87
      Both score meaningfully — freshers are not zeroed out, just appropriately
      lower when they genuinely have less verified experience with a skill.


    """
    skills = candidate.get("skills", [])
    must_have_skills    = must_have_skills    or []
    nice_to_have_skills = nice_to_have_skills or []

    matched_must: list[str] = []
    matched_nice: list[str] = []
    raw = 0.0

    for s in skills:
        sname          = s["name"]
        end_norm       = min(s.get("endorsements", 0), 50) / 50.0
        dur_norm       = min(s.get("duration_months", 0), 48) / 48.0
        trust          = 0.40 + 0.30 * end_norm + 0.30 * dur_norm

        if _skill_match_score(sname, must_have_skills):
            raw += trust * 2.0
            matched_must.append(sname)
        elif _skill_match_score(sname, nice_to_have_skills):
            raw += trust * 0.8
            matched_nice.append(sname)

    # Normalise: 6 perfect must-have matches = 1.0
    score = min(raw / 12.0, 1.0)
    return score, matched_must, matched_nice


# ── Career scoring ────────────────────────────────────────────────────────────

def _education_tier_score(candidate: dict) -> float:
    """
    Converts best education tier to a 0-1 score.
    Used as a substitute signal for freshers with thin career history.

    Tier mapping (from real data: tier_1 through tier_4):
      tier_1 → 1.00  (IIT, IIM, top national)
      tier_2 → 0.75
      tier_3 → 0.50
      tier_4 → 0.25
      unknown → 0.40 (neutral, not penalised)

    Not used as primary signal for experienced candidates — education tier
    becomes less predictive of capability after ~3 years of work.
    """
    tier_map = {"tier_1": 1.00, "tier_2": 0.75, "tier_3": 0.50, "tier_4": 0.25}
    edu = candidate.get("education", [])
    if not edu:
        return 0.40

    best = max(
        (tier_map.get(e.get("tier", ""), 0.40) for e in edu),
        default=0.40,
    )
    return best


def compute_career_score(candidate: dict, jd: dict) -> tuple[float, dict]:
    """
    Scores career trajectory fit against the JD.

    Components:
      yoe_score     (30%): Experience range fit.  Fresher-aware (FIX-9).
      product_score (45%): Depth and duration of relevant roles.
                           FIX-4: denominator scales with career length.
      title_score   (25%): How well recent job titles align with JD terms.
                           Extended to career[:3] with decaying weights.

    FRESHER ADJUSTMENT:
      When a candidate has fewer than 2 career roles, education tier
      substitutes for a portion of the product_score signal.  This is
      the architecturally correct place for it — education is a proxy
      for career depth, not a proxy for skills.

      Blending formula when n_roles < 2:
        product_score = 0.60 * role_signal + 0.40 * edu_tier_score
      This means a fresher with 0 roles and tier_1 education gets
      product_score ≈ 0.40, not 0.0.  A fresher with 1 role and tier_2
      gets product_score ≈ 0.60*(role_signal) + 0.40*0.75.
    """
    p      = candidate["profile"]
    career = candidate.get("career_history", [])
    yoe    = p["years_of_experience"]

    exp_min, exp_max = _safe_exp(jd)
    yoe_score = _compute_yoe_score(yoe, exp_min, exp_max)

    # ── Product / role score ──────────────────────────────────────────────────
    disq_terms = _get_disq_terms(jd)

    product_raw = 0.0
    n_counted   = 0
    for r in career:
        role_title = r["title"].lower()
        if any(
            re.search(r'\b' + re.escape(t) + r'\b', role_title)
            for t in disq_terms
        ):
            continue
        dur_yrs    = r.get("duration_months", 0) / 12.0
        dur_weight = min(dur_yrs / 3.0, 1.0)      # 3 years = full weight
        product_raw += 0.5 + 0.5 * dur_weight
        n_counted  += 1

    # FIX-4: denominator scales with expected career depth from JD range
    if exp_min is None:
        expected_roles = max(n_counted, 2)
    else:
        hi = exp_max if exp_max is not None else exp_min + 4.0
        expected_roles = max(round((exp_min + hi) / 4.0), 2)

    role_signal   = min(product_raw / expected_roles, 1.0)
    edu_tier      = _education_tier_score(candidate)

    # FRESHER ADJUSTMENT: blend education tier into product signal
    if n_counted < 2:
        blend = 1.0 - (0.40 if n_counted == 0 else 0.20)
        product_score = blend * role_signal + (1.0 - blend) * edu_tier
    else:
        product_score = role_signal

    # ── Title alignment score ─────────────────────────────────────────────────
    must_terms = jd.get("must_have_terms", [])
    nice_terms = jd.get("nice_have_terms", [])

    title_score = 0.0
    for r, w in zip(career[:3], [1.0, 0.6, 0.3]):
        t = r["title"].lower()
        if must_terms and any(
            re.search(r'\b' + re.escape(kw), t) for kw in must_terms
        ):
            title_score += 1.0 * w
        elif nice_terms and any(
            re.search(r'\b' + re.escape(kw), t) for kw in nice_terms
        ):
            title_score += 0.5 * w
    title_score = min(title_score, 1.0)

    career_score = (
        0.30 * yoe_score
        + 0.45 * product_score
        + 0.25 * title_score
    )

    breakdown = {
        "yoe_score":     round(yoe_score, 3),
        "product_score": round(product_score, 3),
        "role_signal":   round(role_signal, 3),
        "edu_tier":      round(edu_tier, 3),
        "title_score":   round(title_score, 3),
        "n_roles":       n_counted,
        "exp_min":       exp_min,
        "exp_max":       exp_max,
    }
    return round(career_score, 4), breakdown


# ── Location scoring ──────────────────────────────────────────────────────────

def compute_location_score(
    candidate: dict,
    preferred_cities: Optional[list[str]] = None,
) -> float:
    """
    FIX-7: Word-boundary city matching prevents "impune" matching "pune".

    Returns:
      1.00 — candidate is in a preferred city
      0.75 — not in preferred city but willing to relocate
      0.55 — not in preferred city, not willing to relocate
      1.00 — if JD has no preferred cities (no geographic constraint)
    """
    if not preferred_cities:
        return 1.0

    p        = candidate["profile"]
    sig      = candidate["redrob_signals"]
    location = p["location"].lower()
    relocate = sig.get("willing_to_relocate", False)

    in_preferred = any(
        re.search(r'\b' + re.escape(city.lower()) + r'\b', location)
        for city in preferred_cities
    )

    if in_preferred:
        return 1.0
    elif relocate:
        return 0.75
    else:
        return 0.55


# ── Behavioral multiplier ─────────────────────────────────────────────────────

def compute_behavioral_multiplier(candidate: dict) -> tuple[float, dict]:
    """
    Multiplier applied to the weighted raw score to modulate for availability
    and engagement signals.

    Philosophy:
      - Starts at 1.0
      - Positive signals push up to 1.15 ceiling
      - Negative signals pull down to 0.55 floor
      - Behavioral signals MODULATE fit; they do not OVERRIDE it


    FRESHER NOTE:
      Freshers structurally have fewer recruiter interactions and lower
      interview completion rates because they've had fewer opportunities,
      not because they're unresponsive.  The multiplier uses absolute
      thresholds — a fresher with 0.50 response rate gets the neutral
      path (no penalty), same as a veteran.  The only fresher-specific
      risk is very low interview completion rate which is penalised
      uniformly across all YOE bands.  This is correct: if a fresher
      repeatedly abandons interview processes, that's a real signal.
    """
    sig   = candidate["redrob_signals"]
    bm    = 1.0
    notes: dict = {}

    # open_to_work: positive signal only
    # FIX-5: passive gets 0.95 not 0.88
    if sig.get("open_to_work_flag", False):
        bm *= 1.08
        notes["open_to_work"] = True
    else:
        bm *= 0.95
        notes["open_to_work"] = False

    # Activity recency
    last_active = _parse_date(sig.get("last_active_date"))
    if last_active:
        days_inactive = (C.TODAY - last_active).days
        notes["days_inactive"] = days_inactive
        if days_inactive > C.MAX_INACTIVE_DAYS_FULL:
            bm *= 0.60          # was 0.50 — less aggressive
        elif days_inactive > C.MAX_INACTIVE_DAYS_HEAVY:
            bm *= 0.80          # was 0.75
        elif days_inactive > C.MAX_INACTIVE_DAYS_LIGHT:
            bm *= 0.92          # was 0.90

    # Recruiter response rate
    resp = sig.get("recruiter_response_rate", 0.5)
    notes["resp_rate"] = resp
    if resp < 0.20:
        bm *= 0.78              # was 0.70
    elif resp < 0.40:
        bm *= 0.90              # was 0.85
    elif resp > 0.70:
        bm *= 1.05

    # Notice period
    notice = sig.get("notice_period_days", 90)
    notes["notice_days"] = notice
    applied = False
    for threshold, mult in C.NOTICE_MULTIPLIERS:
        if notice <= threshold:
            bm *= mult
            applied = True
            break
    if not applied:
        bm *= 0.72              # was 0.65 — long notice is common in India

    # Interview completion rate
    icr = sig.get("interview_completion_rate", 0.7)
    if icr < 0.50:
        bm *= 0.88
    elif icr > 0.85:
        bm *= 1.03

    # GitHub activity — positive only, never penalised for absence
    github = sig.get("github_activity_score", -1)
    if github >= 40:
        bm *= 1.06
    elif github >= 20:
        bm *= 1.02
    # github == -1 (no data) or github < 20: no adjustment

    # FIX-6: Raised floor
    bm = max(0.55, min(bm, 1.15))
    return round(bm, 4), notes


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_candidate(candidate: dict, jd: Optional[dict] = None) -> dict:
    """
    Compute all features and final score for one candidate.

    Final score = min((W_SKILL*skill + W_CAREER*career + W_LOCATION*location) * bm, 1.0)

    Returns a result dict with all intermediate scores for debugging
    and for the reasoner to use when generating per-candidate reasoning.
    """
    jd = jd or {}

    _empty = {
        "candidate_id":          candidate["candidate_id"],
        "final_score":           0.0,
        "is_honeypot":           False,
        "honeypot_flags":        [],
        "disqualified":          False,
        "disqualify_reason":     "",
        "skill_score":           0.0,
        "career_score":          0.0,
        "location_score":        0.0,
        "behavioral_multiplier": 0.0,
        "matched_must":          [],
        "matched_nice":          [],
        "career_breakdown":      {},
        "behavioral_notes":      {},
        "preferred_locations":   jd.get("preferred_locations", []),
    }

    # Honeypot check — hard exit
    is_hp, hp_flags = detect_honeypot(candidate)
    if is_hp:
        return {**_empty, "is_honeypot": True, "honeypot_flags": hp_flags}

    # Hard disqualification — hard exit
    disq, disq_reason = is_hard_disqualified(candidate, jd)
    if disq:
        return {**_empty, "disqualified": True, "disqualify_reason": disq_reason}

    # Feature scores
    skill_score, matched_must, matched_nice = compute_skill_score(
        candidate,
        jd.get("must_have_terms") or [],
        jd.get("nice_have_terms") or [],
    )
    career_score, career_bd = compute_career_score(candidate, jd)
    location_score          = compute_location_score(
        candidate, jd.get("preferred_locations")
    )
    bm, bm_notes = compute_behavioral_multiplier(candidate)

    raw         = (
        C.W_SKILL    * skill_score
        + C.W_CAREER   * career_score
        + C.W_LOCATION * location_score
    )
    final_score = round(min(raw * bm, 1.0), 6)

    return {
        "candidate_id":          candidate["candidate_id"],
        "final_score":           final_score,
        "is_honeypot":           False,
        "honeypot_flags":        [],
        "disqualified":          False,
        "disqualify_reason":     "",
        "skill_score":           round(skill_score, 4),
        "career_score":          round(career_score, 4),
        "location_score":        round(location_score, 4),
        "behavioral_multiplier": bm,
        "matched_must":          matched_must,
        "matched_nice":          matched_nice,
        "career_breakdown":      career_bd,
        "behavioral_notes":      bm_notes,
        "preferred_locations":   jd.get("preferred_locations", []),
    }