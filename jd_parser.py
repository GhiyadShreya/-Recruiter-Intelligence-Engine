"""
RIE — Job Description Parser.

Reads a .docx or .md or .txt JD file and extracts:
  - must_have_skills   : list[str]
  - nice_to_have_skills: list[str]
  - disqualifiers      : list[str]
  - experience_range   : tuple[int, int]
  - preferred_locations: list[str]
  - notice_preference  : str
  - raw_text           : str   (full JD for embedding)
  - embedding_text     : str   (condensed, signal-heavy text for semantic search)

Everything is derived from the file at runtime.
"""

from __future__ import annotations
import re
import os
from pathlib import Path
from typing import Optional


# ── Section heading patterns ─────────────────────────────────────────────────
# The parser identifies sections by these marker phrases (case-insensitive).
# Works for both the Redrob JD and most standard JD formats.

MUST_HAVE_SECTION_MARKERS = [
    "things you absolutely need",
    "must have", "must-have", "requirements",
    "you need", "required skills", "what we need",
]

NICE_TO_HAVE_SECTION_MARKERS = [
    "things we'd like you to have",
    "nice to have", "nice-to-have", "preferred",
    "bonus", "good to have", "we'd like",
    "would like you to have",
]

DISQUALIFIER_SECTION_MARKERS = [
    "things we explicitly do not want",
    "we do not want", "disqualifier", "not a fit",
    "explicitly do not", "we won't", "deal breaker",
]

LOCATION_SECTION_MARKERS = [
    "on location", "location:", "where you",
    "office", "based in",
]

IDEAL_CANDIDATE_SECTION_MARKERS = [
    "ideal candidate", "ideal profile", "ideal person", "ideal fit",
    "who you are", "who we're looking for", "the person we're looking for",
    "what success looks like", "you are someone who",
]


# ── File readers ──────────────────────────────────────────────────────────────

def _read_docx(path: str) -> list[str]:
    """Returns list of non-empty paragraph strings."""
    try:
        import docx
        doc = docx.Document(path)
        return [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")


def _read_txt_or_md(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_jd_paragraphs(jd_path: str) -> list[str]:
    ext = Path(jd_path).suffix.lower()
    if ext == ".docx":
        return _read_docx(jd_path)
    elif ext in (".md", ".txt", ""):
        return _read_txt_or_md(jd_path)
    else:
        # Try docx first, fall back to text
        try:
            return _read_docx(jd_path)
        except Exception:
            return _read_txt_or_md(jd_path)


# ── Section detection ─────────────────────────────────────────────────────────

def _is_header_shaped(line: str) -> bool:
    """
    A line can only plausibly BE a section header if it's short and doesn't
    read like a full sentence. This stops marker words that legitimately
    appear inside content (e.g. "Office in San Francisco, CA." contains
    "office", but is a sentence, not a header) from being misread as a new
    section boundary.
    """
    stripped = line.strip()
    if len(stripped) > 60:
        return False
    # A line ending in a sentence-final period followed by more text, or
    # containing multiple sentences, reads as content, not a header.
    if stripped.count(".") > 0 and not stripped.endswith(":"):
        return False
    return True


def _is_section_header(line: str, markers: list[str]) -> bool:
    if not _is_header_shaped(line):
        return False
    ll = line.lower()
    return any(m in ll for m in markers)


def _extract_section_bullets(paragraphs: list[str], markers: list[str]) -> list[str]:
    """
    Find the section matching `markers`, then collect all bullet/line items
    until the next recognisable section header.

    Handles two JD shapes:
      1. A header line ("On Location") followed by separate bullet lines.
      2. An inline "Label: value" line ("Location: Austin, TX | ...") where
         the header and the content are the same line — in which case the
         text after the colon is captured as content, not discarded.
    """
    
    ALL_SECTION_MARKERS = (
        MUST_HAVE_SECTION_MARKERS
        + NICE_TO_HAVE_SECTION_MARKERS
        + DISQUALIFIER_SECTION_MARKERS
        + LOCATION_SECTION_MARKERS
        + IDEAL_CANDIDATE_SECTION_MARKERS
        + ["responsibilities", "about the role", "about the team",
           "compensation", "salary", "benefits", "how to apply",
           "about us", "about the company", "perks"]
    )

    in_section = False
    items: list[str] = []

    for para in paragraphs:
        ll = para.lower()

        if _is_section_header(para, markers):
            in_section = True
            # If this header line also carries inline content after a colon
            # (e.g. "Location: Austin, TX | Employment type: Full-time"),
            # keep that content rather than discarding the whole line.
            if ":" in para:
                after_colon = para.split(":", 1)[1].strip()
                if after_colon:
                    items.append(after_colon)
            continue

        if in_section:
            # Stop when we hit any other known section header (but not on
            # the very first line we collect, in case it happens to share
            # vocabulary with another marker).
            if items and _is_header_shaped(para) and any(m in ll for m in ALL_SECTION_MARKERS):
                break
            # Stop on obvious new section headers (short lines ending in a
            # header-like character, e.g. "Compensation:")
            if len(para) < 60 and para.endswith((":", "—", "-")) and not para.startswith(("-", "•", "*")):
                if items:  # only break if we've already collected something
                    break
            items.append(para)

    # Clean up: strip bullet chars, skip very short lines
    cleaned = []
    for item in items:
        item = item.lstrip("-•*·▪ ").strip()
        if len(item) > 15:
            cleaned.append(item)
    return cleaned


# ── Experience range extractor ────────────────────────────────────────────────

# Range pattern: "5-9 years", "5–9 years", "6 to 8 years"
_EXPERIENCE_RANGE_PATTERN = re.compile(
    r'(\d+)\s*(?:[–\-—]|to)\s*(\d+)\s*\+?\s*years?', re.IGNORECASE
)
# Single-bound pattern: "5+ years", "at least 5 years", "minimum 5 years"
_EXPERIENCE_MIN_ONLY_PATTERN = re.compile(
    r'(?:(\d+)\s*\+\s*years?)|(?:(?:at least|minimum(?:\s+of)?)\s+(\d+)\s*years?)',
    re.IGNORECASE,
)


def _extract_experience_range(paragraphs: list[str]) -> tuple[Optional[int], Optional[int]]:
    """
    
    Returns (min_years, max_years), or (None, None) if no experience
    requirement is stated. A generalized parser should not invent a range
    (e.g. defaulting to (4, 12)) for JDs that don't mention years of
    experience at all — that silently biases scoring toward a mid-senior
    band. Downstream scoring should treat (None, None) as "no experience
    constraint" rather than substituting a guessed range.
    """
    for para in paragraphs:
        m = _EXPERIENCE_RANGE_PATTERN.search(para)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if 0 < lo < hi < 30:
                return lo, hi

    for para in paragraphs:
        m = _EXPERIENCE_MIN_ONLY_PATTERN.search(para)
        if m:
            lo = int(m.group(1) or m.group(2))
            if 0 < lo < 30:
                return lo, None

    return None, None


# ── JD type classifier ────────────────────────────────────────────────────────
# rank.py expects a `jd_type` field (and a `_classify_jd_type` function to
# back-fill it for older cached JDs). Classification is based purely on the
# experience range — never on domain/title — so it stays generalized across
# any JD, and degrades to "unspecified" rather than guessing when there's no
# experience signal at all (e.g. after the Concern-1 fix, exp_min/exp_max can
# now legitimately be None).
_JD_TYPE_BANDS = [
    (0, 2, "entry"),
    (2, 5, "mid"),
    (5, 9, "senior"),
    (9, 100, "staff_plus"),
]


def _classify_jd_type(exp_min: Optional[int], exp_max: Optional[int]) -> str:
    """
    Classify seniority band from the experience range alone. Returns
    "unspecified" when no experience requirement was extracted — this is
    intentionally NOT defaulted to "senior" or any other band, since doing
    so would reintroduce the same kind of silent bias as the old (4, 12)
    experience default.
    """
    if exp_min is None and exp_max is None:
        return "unspecified"
    # Use whichever bound is available; prefer the midpoint when both exist.
    if exp_min is not None and exp_max is not None:
        reference = (exp_min + exp_max) / 2
    else:
        reference = exp_min if exp_min is not None else exp_max

    for lo, hi, label in _JD_TYPE_BANDS:
        if lo <= reference < hi:
            return label
    return "staff_plus"


# ── Location extractor ────────────────────────────────────────────────────────

_NON_LOCATION_FRAGMENTS = [
    "full-time", "full time", "part-time", "part time", "contract",
    "employment type", "job type", "internship", "temporary", "freelance",
    "permanent", "fixed-term", "fixed term",
]

# Legitimate work-arrangement signals that are location-adjacent but aren't
# place names — kept distinct from city/region names but still valid output.
_WORK_ARRANGEMENT_TERMS = ["remote", "hybrid", "on-site", "onsite", "in-office"]

# Words that commonly precede/contain real place names in JD text. Matching
# is structural, but constrained to avoid false positives like "Jira, Figma"
# (two capitalized nouns separated by a comma are not necessarily a place).
# We only accept "City, XX" when XX is a US state code, or "City, Country"
# when the second token is a recognizable country/region name.
_US_STATE_CODES = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in","ia",
    "ks","ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv","nh","nj",
    "nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn","tx","ut","vt",
    "va","wa","wv","wi","wy","dc",
}
_COMMON_COUNTRIES = {
    "usa", "us", "uk", "canada", "india", "germany", "france", "spain",
    "italy", "australia", "singapore", "ireland", "netherlands", "poland",
    "mexico", "brazil", "japan", "china", "philippines", "ukraine",
    "indonesia", "vietnam", "pakistan", "bangladesh", "nigeria", "egypt",
    "south africa", "united states", "united kingdom",
}

_PLACE_PATTERN = re.compile(
    r'\b([A-Z][a-zA-Z]+(?:[\s\-][A-Z][a-zA-Z]+)*),\s*([A-Z]{2}|[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)\b'
)
_PLACE_PHRASE_PATTERN = re.compile(
    r'\b(?:based in|located in|office in|offices in|hub in|onsite in|on-site in)\s+'
    r'([A-Z][a-zA-Z]+(?:[\s\-][A-Z][a-zA-Z]+)*(?:,\s*[A-Z][a-zA-Z]+)?)',
)


def _is_non_location_fragment(text: str) -> bool:
    tl = text.lower()
    return any(f in tl for f in _NON_LOCATION_FRAGMENTS)


def _extract_places_from_text(text: str) -> set[str]:
    """
    Run structural place-detection (City+State, City+Country, "based in X")
    over a chunk of text. Shared by both the section-bullet pass and the
    whole-document pass so a sentence like "This role is based in Singapore
    for candidates outside the US." yields just "singapore", not the entire
    sentence.
    """
    found: set[str] = set()

    for m in _PLACE_PATTERN.finditer(text):
        city, region = m.group(1).strip(), m.group(2).strip()
        region_l = region.lower()
        if _is_non_location_fragment(city) or _is_non_location_fragment(region):
            continue
        if region_l in _US_STATE_CODES or region_l in _COMMON_COUNTRIES:
            found.add(f"{city.lower()}, {region_l}")

    for m in _PLACE_PHRASE_PATTERN.finditer(text):
        place = m.group(1).strip()
        if not _is_non_location_fragment(place):
            found.add(place.lower())

    return found


def _extract_locations(paragraphs: list[str]) -> list[str]:
    locations: set[str] = set()

    # Pass 1: structural bullets inside a location-flagged section. Split on
    # slash/pipe/ampersand (multi-city separators) — NOT comma, since
    # "City, ST"/"City, Country" needs to stay intact. Each fragment is run
    # through the same place-detection used document-wide, plus a narrow
    # fallback for bare city names and work-arrangement terms.
    loc_bullets = _extract_section_bullets(paragraphs, LOCATION_SECTION_MARKERS)
    for item in loc_bullets:
        structural_hits = _extract_places_from_text(item)
        locations |= structural_hits

        parts = [p.strip() for p in re.split(r'[/|&]', item) if p.strip()]
        for part in parts:
            pl = part.lower()
            if _is_non_location_fragment(part):
                continue
            if pl in _WORK_ARRANGEMENT_TERMS:
                locations.add(pl)
                continue
            # Skip fragments already captured structurally above (e.g. the
            # "New York, NY" piece of a slash-separated list).
            if any(pl in hit or hit in pl for hit in structural_hits):
                continue
            word_count = len(part.split())
            if part[:1].isupper() and "." not in part and word_count <= 4:
                locations.add(pl)

    # Pass 2: whole-document scan, regardless of section, for "City, ST" /
    # "City, Country" and "based in X" / "office in X" phrasing.
    full_text = " ".join(paragraphs)
    locations |= _extract_places_from_text(full_text)

    # Pass 3: remote/hybrid/on-site signals anywhere in the doc, even with no
    # dedicated location section.
    full_text_lower = full_text.lower()
    for term in _WORK_ARRANGEMENT_TERMS:
        if re.search(rf'\b{re.escape(term)}\b', full_text_lower):
            locations.add(term)

    return sorted(locations)

    # Pass 2: whole-document scan for "City, ST"/"City, Country" patterns and
    # "based in X" / "office in X" phrasing, regardless of section. Only
    # accept matches where the second token is a known state code or country
    # name, so arbitrary "Capitalized, Capitalized" pairs (e.g. product
    # names like "Jira, Figma") don't get treated as locations.
    full_text = " ".join(paragraphs)
    for m in _PLACE_PATTERN.finditer(full_text):
        city, region = m.group(1).strip(), m.group(2).strip()
        region_l = region.lower()
        if _is_non_location_fragment(city) or _is_non_location_fragment(region):
            continue
        if region_l in _US_STATE_CODES or region_l in _COMMON_COUNTRIES:
            locations.add(f"{city.lower()}, {region_l}")

    for m in _PLACE_PHRASE_PATTERN.finditer(full_text):
        place = m.group(1).strip()
        if not _is_non_location_fragment(place):
            locations.add(place.lower())

    # Pass 3: remote/hybrid/on-site signals anywhere in the doc, even with no
    # dedicated location section.
    full_text_lower = full_text.lower()
    for term in _WORK_ARRANGEMENT_TERMS:
        if re.search(rf'\b{re.escape(term)}\b', full_text_lower):
            locations.add(term)

    return sorted(locations)


# ── Title extractor ───────────────────────────────────────────────────────────

_BOILERPLATE_MARKERS = [
    "equal opportunity", "eeo", "all qualified applicants", "without regard to",
    "race, color", "disability", "veteran status", "background check",
    "we are an equal", "diversity and inclusion", "reasonable accommodation",
    "benefits include", "health insurance", "401(k)", "401k", "pto",
    "paid time off", "about us", "about the company", "founded in",
    "headquartered", "we are a", "is a leading", "our mission", "our company",
]

_TITLE_HINT_WORDS = [
    "engineer", "manager", "director", "lead", "specialist", "analyst",
    "scientist", "developer", "designer", "consultant", "associate",
    "executive", "representative", "coordinator", "administrator",
    "architect", "officer", "intern", "head of", "vp ", "vice president",
]


def _looks_like_boilerplate(line: str) -> bool:
    ll = line.lower()
    return any(m in ll for m in _BOILERPLATE_MARKERS)


def _extract_title(paragraphs: list[str]) -> str:
    """
    Best-effort job title extraction. Looks at the first several paragraphs
    and picks the most title-like candidate rather than assuming the JD
    opens with the title.
    """
    candidates = paragraphs[:10]

    # Pass 1: short, punctuation-free lines containing a common title word.
    for para in candidates:
        if _looks_like_boilerplate(para):
            continue
        if len(para) <= 80 and not para.rstrip().endswith((".", "!", "?")):
            if any(w in para.lower() for w in _TITLE_HINT_WORDS):
                return para.strip()

    # Pass 2: any short, non-boilerplate, non-sentence line near the top.
    for para in candidates:
        if _looks_like_boilerplate(para):
            continue
        if len(para) <= 80 and not para.rstrip().endswith((".", "!", "?")):
            return para.strip()

    # Fallback: first non-boilerplate paragraph, even if it's a full sentence.
    for para in candidates:
        if not _looks_like_boilerplate(para):
            return para.strip()

    # Last resort: literally the first paragraph, if any.
    return paragraphs[0].strip() if paragraphs else ""


# ── Notice period extractor ───────────────────────────────────────────────────

_NOTICE_DURATION_PATTERN = re.compile(
    r'\b\d+\s*[-–]?\s*(day|days|week|weeks|month|months)\b', re.IGNORECASE
)


def _extract_notice_preference(paragraphs: list[str]) -> str:
    for para in paragraphs:
        ll = para.lower()
        if "notice period" in ll:
            return para.strip()
        if "notice" in ll and _NOTICE_DURATION_PATTERN.search(para):
            return para.strip()
    return ""


# ── Skill term extractor from free text ──────────────────────────────────────

_SKILL_CUE_PATTERN = re.compile(
    r'(?:experience (?:with|in|using)|proficiency in|proficient (?:with|in)|'
    r'knowledge of|familiarity with|skilled in|background in|expertise in|'
    r'fluency in|working knowledge of|such as|including)\s+'
    r'([^.;:]+?)'
    r'(?=\s+(?:for|to|so|which|that|in order|so that)\b|[.;:]|$)',
    re.IGNORECASE,
)

# Words too generic to ever be useful as a "skill", filtered out of cue-list
# extraction (English connectors/fillers, not domain terms).
_SKILL_STOPWORDS = {
    "the", "and", "you", "we", "or", "a", "an", "with", "in", "of", "to",
    "for", "is", "are", "etc", "such", "as", "related", "similar", "tools",
    "technologies", "platforms", "systems", "other",
    "experience", "proficiency", "proficient", "knowledge", "familiarity",
    "skilled", "background", "expertise", "fluency", "working",
}


def _extract_skill_terms(text_items: list[str]) -> list[str]:
    terms: set[str] = set()

   
    cap_pattern = re.compile(r'\b([A-Z][a-zA-Z0-9.\+#]+(?:\s+[A-Z][a-zA-Z0-9.\+#]+)*)\b')
    for item in text_items:
        for match in cap_pattern.finditer(item):
            term = match.group(1).lower()
            if len(term) > 2 and term not in _SKILL_STOPWORDS:
                terms.add(term)

   
    _INNER_CUE = re.compile(r'\b(?:such as|including)\s+', re.IGNORECASE)
    for item in text_items:
        for cue_match in _SKILL_CUE_PATTERN.finditer(item):
            list_text = cue_match.group(1)
           
            inner = _INNER_CUE.search(list_text)
            if inner:
                list_text = list_text[inner.end():]
            # Split the list on commas, "and"/"or", and slashes.
            pieces = re.split(r',|\bor\b|\band\b|/', list_text, flags=re.IGNORECASE)
            for piece in pieces:
                term = piece.strip().strip(".").lower()
               
                if not term or term in _SKILL_STOPWORDS:
                    continue
                word_count = len(term.split())
                if 1 <= word_count <= 3 and 2 < len(term) <= 30:
                    terms.add(term)

    return sorted(terms)


# ── Main parse function ───────────────────────────────────────────────────────

def parse_jd(jd_path: str) -> dict:
    """
    Parse a JD file and return a structured dict.

    Returns:
    {
        "title":               str,         # best-effort detected role title
        "must_have_skills":    list[str],   # raw bullet text from must-have section
        "nice_to_have_skills": list[str],   # raw bullet text from nice-to-have section
        "disqualifiers":       list[str],   # raw bullet text from disqualifiers section
        "must_have_terms":     list[str],   # tech terms extracted from must-have bullets
        "nice_have_terms":     list[str],   # tech terms extracted from nice-to-have bullets
        "experience_min":      int,
        "experience_max":      int,
        "preferred_locations": list[str],
        "notice_preference":   str,
        "raw_text":            str,         # full JD concatenated
        "embedding_text":      str,         # condensed text for embedding
    }
    """
    paragraphs = read_jd_paragraphs(jd_path)
    raw_text   = "\n".join(paragraphs)

    must_have_bullets    = _extract_section_bullets(paragraphs, MUST_HAVE_SECTION_MARKERS)
    nice_to_have_bullets = _extract_section_bullets(paragraphs, NICE_TO_HAVE_SECTION_MARKERS)
    disqualifier_bullets = _extract_section_bullets(paragraphs, DISQUALIFIER_SECTION_MARKERS)

    must_have_terms  = _extract_skill_terms(must_have_bullets)
    nice_have_terms  = _extract_skill_terms(nice_to_have_bullets)
    exp_min, exp_max = _extract_experience_range(paragraphs)
    jd_type          = _classify_jd_type(exp_min, exp_max)
    locations        = _extract_locations(paragraphs)
    notice_pref      = _extract_notice_preference(paragraphs)

    # ── Build embedding text ──────────────────────────────────────────────────
    # This is what gets embedded and compared against candidate embeddings.
    # Structured to be dense with signal, not padded with boilerplate.
    embedding_parts = []

    # Role title — best-effort detection, not just paragraphs[0]
    title = _extract_title(paragraphs)
    if title:
        embedding_parts.append(title)

    # Must-haves (highest weight — repeat key terms)
    if must_have_bullets:
        embedding_parts.append("Required skills and experience:")
        embedding_parts.extend(must_have_bullets[:6])
        # Repeat extracted terms for weight
        if must_have_terms:
            embedding_parts.append("Key terms: " + ", ".join(must_have_terms))

    # Nice-to-haves
    if nice_to_have_bullets:
        embedding_parts.append("Preferred skills:")
        embedding_parts.extend(nice_to_have_bullets[:4])


    if exp_min is not None and exp_max is not None:
        embedding_parts.append(f"Experience: {exp_min}-{exp_max} years.")
    elif exp_min is not None:
        embedding_parts.append(f"Experience: {exp_min}+ years.")
    elif exp_max is not None:
        embedding_parts.append(f"Experience: up to {exp_max} years.")

    # Location
    if locations:
        embedding_parts.append(f"Location preference: {', '.join(l.title() for l in locations)}.")

    # Ideal candidate description — generalized markers, no JD-specific phrasing
    for idx, para in enumerate(paragraphs):
        if any(m in para.lower() for m in IDEAL_CANDIDATE_SECTION_MARKERS):
            # Grab next few lines
            for p in paragraphs[idx:idx + 5]:
                if len(p) > 20:
                    embedding_parts.append(p)
            break

    embedding_text = " ".join(embedding_parts)

    return {
        "title":               title,
        "jd_type":             jd_type,
        "must_have_skills":    must_have_bullets,
        "nice_to_have_skills": nice_to_have_bullets,
        "disqualifiers":       disqualifier_bullets,
        "must_have_terms":     must_have_terms,
        "nice_have_terms":     nice_have_terms,
        "experience_min":      exp_min,
        "experience_max":      exp_max,
        "preferred_locations": locations,
        "notice_preference":   notice_pref,
        "raw_text":            raw_text,
        "embedding_text":      embedding_text,
    }


# ── CLI: inspect parsed output ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    jd_path = sys.argv[1] if len(sys.argv) > 1 else "job_description.docx"
    if not os.path.exists(jd_path):
        print(f"File not found: {jd_path}")
        sys.exit(1)

    result = parse_jd(jd_path)

    print("=" * 60)
    print("PARSED JD")
    print("=" * 60)
    print(f"\nDetected title: {result['title']}")
    print(f"Experience range: {result['experience_min']}–{result['experience_max']} years")
    print(f"Preferred locations: {result['preferred_locations']}")
    print(f"Notice preference: {result['notice_preference']}")

    print(f"\nMust-have skills ({len(result['must_have_skills'])} bullets):")
    for b in result["must_have_skills"]:
        print(f"  • {b[:100]}")

    print(f"\nExtracted must-have tech terms:")
    print(f"  {result['must_have_terms']}")

    print(f"\nNice-to-have skills ({len(result['nice_to_have_skills'])} bullets):")
    for b in result["nice_to_have_skills"]:
        print(f"  • {b[:100]}")

    print(f"\nDisqualifiers ({len(result['disqualifiers'])} bullets):")
    for b in result["disqualifiers"]:
        print(f"  • {b[:100]}")

    print(f"\nEmbedding text ({len(result['embedding_text'])} chars):")
    print(f"  {result['embedding_text'][:600]}...")