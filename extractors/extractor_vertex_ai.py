"""
LegaTech Tunisia Legal Extractor — v4 (Vertex AI)
==================================================
Two-phase pipeline:
  Phase 1 : PDF text → LLM → document metadata + list of raw articles
  Phase 2 : Each raw article individually → LLM → all 62 enriched fields

API backend: Google Vertex AI (Gemini)
  Auth     : Service account JSON via GOOGLE_APPLICATION_CREDENTIALS
  Project  : PROJECT_ID env var
  Location : LOCATION env var
  Model    : MODEL env var
"""

import json
import re
import time
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import os
import fitz  # pymupdf

# ── optional .env support ──────────────────────────────────────────────────
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Vertex AI credentials ─────────────────────────────────────────────────────
# Set these in your .env file or as environment variables:
#   GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
#   PROJECT_ID=your_project_id
#   LOCATION=us-central1
#   MODEL=gemini-2.5-flash
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

PROJECT_ID = os.getenv("PROJECT_ID", "")
LOCATION   = os.getenv("LOCATION", "us-central1")
MODEL      = os.getenv("MODEL", "gemini-2.5-flash")

# ── Tuning ────────────────────────────────────────────────────────────────────
PDF_PARSE_CHUNK_CHARS = 20000
PDF_PARSE_MAX_TOKENS  = 32768
ENRICH_MAX_TOKENS     = 32768
RATE_LIMIT_DELAY      = 2.0   # seconds between normal requests
RETRY_ATTEMPTS        = 3     # attempts before giving up
RETRY_BASE_DELAY      = 2     # seconds (doubles on each retry)


# ─────────────────────────────────────────────────────────────────────────────
# VERTEX AI CLIENT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig

class VertexCaller:
    """
    Thin wrapper around the Vertex AI Gemini API.
    Single model — retries with exponential back-off on failure.
    """

    def __init__(self):
        if not PROJECT_ID:
            raise RuntimeError(
                "PROJECT_ID not set.\n"
                "  Add to your .env:  PROJECT_ID=your_project_id"
            )
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS not set.\n"
                "  Add to your .env:  GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json"
            )

        self._client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
            http_options=HttpOptions(api_version="v1")
        )

        print(f"🔑 Vertex AI credentials loaded")
        print(f"🤖 Model: {MODEL}")
        print(f"🌐 Project: {PROJECT_ID} | Location: {LOCATION}")

    @property
    def current_model(self):
        return MODEL

    def call(self, system, user, max_tokens, label=""):
        """
        Call Vertex AI Gemini.
        Returns the assistant message text.
        Raises RuntimeError after RETRY_ATTEMPTS failed attempts.
        """
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            attempt_label = f"{label} [{MODEL} a{attempt}]"
            try:
                # Combine system + user into a single prompt
                # (Vertex AI Gemini supports system instructions via config)
                full_prompt = f"{system}\n\n{user}"

                response = self._client.models.generate_content(
                    model=MODEL,
                    contents=full_prompt,
                    config=GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.2,
                    )
                )

                text = response.text or ""

                # Log token usage if available
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    tokens_in  = getattr(response.usage_metadata, "prompt_token_count", "?")
                    tokens_out = getattr(response.usage_metadata, "candidates_token_count", "?")
                    print(f"   📊 [{attempt_label}] {tokens_in} in / {tokens_out} out")

                # Check finish reason
                if response.candidates:
                    finish = str(response.candidates[0].finish_reason)
                    if "MAX_TOKENS" in finish.upper():
                        print(f"   ⚠️  Output truncated (token limit hit)")

                return text  # success

            except Exception as e:
                err_str = str(e)
                # Rate limit / quota exceeded
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    print(f"   🚦 [{attempt_label}] Rate-limited: {e}")
                else:
                    print(f"   ❌ [{attempt_label}] API error: {e}")

                if attempt < RETRY_ATTEMPTS:
                    wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(f"   ⏳ waiting {wait}s before retry...")
                    time.sleep(wait)

        raise RuntimeError(f"[{label}] Vertex AI call failed after {RETRY_ATTEMPTS} attempts.")


# singleton caller (created lazily when first needed)
_rotator = None

def get_rotator():
    global _rotator
    if _rotator is None:
        _rotator = VertexCaller()
    return _rotator


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Tunisian legal analyst and professional bilingual translator (French <-> Arabic).

ABSOLUTE RULES -- never break these:
- Return ONLY valid MINIFIED JSON -- no whitespace, no indentation, no newlines
- NO EXPLANATIONS, NO REASONING, NO THOUGHT PROCESS -- just the JSON
- DO NOT include any text before or after the JSON
- NEVER use null -- use "" for missing strings, [] for missing arrays
- Always close every bracket and brace
- Booleans: true or false (no quotes)
- Dates: ISO 8601 format YYYY-MM-DDT00:00:00Z
- status values: ACTIVE | AMENDED | REPEALED | SUSPENDED
- business_impact values: LOW | MEDIUM | HIGH
- article_type values: ARTICLE | DECREE | NOMINATION | MINUTES | CORRECTION | GOV_ORDER | CIRCULAR | REGULATORY | PENAL | PROCEDURAL | DEFINITIONAL | TRANSITIONAL | ABROGATION
- ambiguity_level values: LOW | MEDIUM | HIGH
- relation_types values: REFERENCES | AMENDS | REPEALS | IMPLEMENTS | SUPERSEDES
- entity_types values: ORGANIZATION | INSTITUTION | COURT | MINISTRY | PERSON
- source_name values: JORT | BCT | OTHER
- institution rules: Loi -> "Assemblee des Representants du Peuple" | Decret -> "Presidence de la Republique" | Circulaire BCT -> "Banque Centrale de Tunisie"
"""


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 PROMPT
# ─────────────────────────────────────────────────────────────────────────────

PDF_PARSE_PROMPT = """You are reading a chunk of a Tunisian official legal document.
Your task: identify and INDEX every important legal unit -- do NOT copy the full text.

A legal unit is ANY of:
  - decree/decision by a minister
  - nomination or appointment
  - meeting minutes
  - erratum/correction
  - governmental order
  - article inside a law
  - Any other official legal text

For EACH unit return ONE object. doc_metadata: fill what you find, "" if absent.

RETURN MINIFIED JSON:
{"doc_metadata":{"law_type":"","law_number":"","law_title_arabic":"","institution":"","source_name":"JORT","source_date":"","effective_date":"","publication_date":"","year":0},"units":[{"unit_type":"DECREE or NOMINATION or ARTICLE or MINUTES or CORRECTION or GOV_ORDER","unit_title":"full Arabic title of this unit (first line/heading)","unit_number":"sequential or article number","anchor":"EXACT first 80 characters of this unit as they appear in the text"}]}

CRITICAL RULES:
- anchor must be the EXACT first 80 characters of the unit copied verbatim from the text
- unit_title is the heading/title line of the unit
- Do NOT include unit_text -- we extract the full text in a separate step
- Skip table of contents lines (lines ending with page numbers after dots .......)
- Each distinct legal unit = one object in units array
- If the same unit appears in TOC and again as content, use the content version for the anchor

DOCUMENT CHUNK:
{pdf_text}

RETURN ONLY THE MINIFIED JSON. NO MARKDOWN. NO EXPLANATIONS."""


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 PROMPT
# ─────────────────────────────────────────────────────────────────────────────

ENRICH_PROMPT = """Enrich this Tunisian legal article into a complete JSON object with ALL fields below.

DOCUMENT METADATA (already extracted -- use these values directly, do not re-derive):
{doc_metadata}

RAW ARTICLE:
{article_data}

FIELD DEFINITIONS:

-- IDENTITY --
  id            : Generate a new UUID4 string
  jurisdiction  : Always "TUNISIA"
  year          : Integer year from doc_metadata
  chapter       : From raw article data, "" if absent
  section       : From raw article data, "" if absent
  article_number: From raw article data

-- STATIC (write "COMPUTED" -- Python replaces these) --
  content_hash_sha256, content_combined, chapter_normalized,
  preceding_article_id, following_article_id, last_checked, next_check
  version: 1, graph_level: 1
  repeal_date: "", superseded_by_id: "", supersedes_id: "", source_url: "", source_number: ""

-- FROM DOC METADATA (copy directly) --
  law_type, law_number, institution, institution_primary, institution_secondary,
  institutions (array), title_french, title_arabic, source_name, source_date,
  effective_date, publication_date
  parent_document_id: "tn-{law_type_slug}-{law_number}"

-- LLM ANALYSIS --

  content_french : FULL verbatim French text of the article. NEVER summarize or truncate.

  content_arabic : Complete Arabic translation. ONLY Arabic script -- no Latin/French characters.

  summary        : EXACTLY 40-60 French words. Must describe what this article does and who it affects.

  summary_french : exact copy of summary
  summary_arabic : Arabic translation of summary -- 40-60 Arabic words

  search_content : title_french + article_number + summary + top keywords

  keywords       : Array of 5-7 French keywords. Meaningful legal/business terms only.
                   Never generic words like "loi", "article", "tunisie".

  legal_concepts : Array of 3-8 legal doctrines in French. E.g. ["bonne foi", "force majeure"]

  legal_domains  : Array of 1-3 French domains. Never empty.
                   Valid: Droit Civil | Droit Commercial | Droit Fiscal | Droit Administratif |
                   Droit du Travail | Droit Penal | Droit de la Sante | Droit Agricole | Droit Bancaire

  business_impact  : LOW | MEDIUM | HIGH
  target_audience  : Array of affected parties in French. Never empty.
  status           : ACTIVE | AMENDED | REPEALED | SUSPENDED

  article_type   : ARTICLE | DECREE | NOMINATION | MINUTES | CORRECTION | GOV_ORDER |
                   CIRCULAR | REGULATORY | PENAL | PROCEDURAL | DEFINITIONAL | TRANSITIONAL | ABROGATION
  article_order  : Integer sequence number of this unit in the document.

  ambiguity_level : LOW | MEDIUM | HIGH

  has_obligations : true only if article uses obligation language
  has_penalties   : true only if article mentions fines, imprisonment, sanctions
  has_deadlines   : true only if article specifies a time limit or deadline
  has_exceptions  : true only if article uses exception language
  is_abrogation   : true only if article explicitly abrogates another text
  is_transitional : true only if article contains transitional provisions

  related_laws    : ONLY laws EXPLICITLY cited in the article text. [] if none.
  relation_target_ids : IDs of explicitly referenced laws. [] if none.
  relation_types      : Parallel to relation_target_ids.

  entity_names : ONLY specific named entities explicitly in the article. [] if none.
  entity_types : Parallel to entity_names.
  entity_ids   : Slug for each entity.

  community_id      : Thematic slug e.g. "droit-du-travail"
  community_label   : 5-7 word French description
  community_summary : One French sentence describing the thematic cluster

RETURN ONLY A VALID MINIFIED JSON OBJECT. NO EXPLANATIONS. NO REASONING. JUST THE JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages = []
    doc = fitz.open(pdf_path)
    print(f"📄 PDF loaded: {len(doc)} pages")
    for page in doc:
        text = page.get_text()
        if text and text.strip():
            pages.append(text)
    doc.close()
    full_text = "\n\n".join(pages)
    print(f"📝 Extracted {len(full_text):,} characters from {len(pages)} pages")
    return full_text, pages


def extract_text_from_md(md_path):
    path = Path(md_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown not found: {md_path}")
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content, [content]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE-GROUP CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def is_toc_page(text):
    dot_lines = sum(1 for line in text.split('\n') if line.count('.') > 5)
    return dot_lines >= 3


def group_pages_into_chunks(pages, max_chars=PDF_PARSE_CHUNK_CHARS, skip_toc_pages=True):
    content_pages = []
    for i, page in enumerate(pages, 1):
        if skip_toc_pages and is_toc_page(page):
            first_line = next(
                (l.strip() for l in page.split('\n') if len(l.strip()) > 10), "(empty)")
            print(f"   🗂️  Page {i} skipped (TOC): {first_line[:80]}")
        else:
            content_pages.append(page)

    skipped = len(pages) - len(content_pages)
    if skip_toc_pages and skipped > 0:
        print(f"   📋 {skipped} TOC page(s) skipped | {len(content_pages)} content page(s) kept")
    else:
        print(f"   📄 {len(content_pages)} page(s) kept")

    chunks, current = [], ""
    for page in content_pages:
        if len(current) + len(page) > max_chars and current:
            chunks.append(current.strip())
            current = page
        else:
            current += ("\n\n" if current else "") + page
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING & REPAIR
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_safely(raw, label=""):
    if not raw:
        return None
    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        if label:
            print(f"⚠️  [{label}] JSON error at pos {e.pos}: {e.msg}")

    try:
        candidate  = cleaned[:cleaned.rfind('}') + 1] if '}' in cleaned else cleaned
        opens_sq   = candidate.count('[') - candidate.count(']')
        opens_curl = candidate.count('{') - candidate.count('}')
        candidate += ']' * max(0, opens_sq) + '}' * max(0, opens_curl)
        result = json.loads(candidate)
        if label:
            print(f"✅ [{label}] JSON repaired (brackets)")
        return result
    except json.JSONDecodeError:
        pass

    try:
        fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
        fixed = re.sub(r'(?<=[^\\])\n', ' ', fixed)
        fixed = re.sub(r'(?<=[^\\])\t', ' ', fixed)
        result = json.loads(fixed)
        if label:
            print(f"✅ [{label}] JSON repaired (syntax)")
        return result
    except json.JSONDecodeError:
        pass

    if label:
        print(f"❌ [{label}] all repair attempts failed")
        print(f"   First 300: {raw[:300]}")
        print(f"   Last  200: {raw[-200:]}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ARABIC CONTAMINATION CHECK
# ─────────────────────────────────────────────────────────────────────────────

def has_arabic_contamination(text):
    if not text:
        return False
    total         = len(text)
    arabic_chars  = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin_letters = sum(1 for c in text if 'a' <= c <= 'z' or 'A' <= c <= 'Z')
    if arabic_chars > total * 0.7:
        if latin_letters > arabic_chars * 0.05:
            return True
    return False


def clean_arabic_text(text):
    if not text:
        return ""
    cleaned = re.sub(r'[a-zA-Z]{2,}', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# UNIT SPLITTING (DECREES CONTAINING ARTICLES)
# ─────────────────────────────────────────────────────────────────────────────

def split_decree_into_units(unit):
    text = unit.get("article_text", "")
    if not text:
        return [unit]

    pattern = r'(الفصل\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|الواحد\sوعشرون|الوحيد|\d+))'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return [unit]

    first_match = matches[0]
    preamble = text[:first_match.start()].strip()
    decree_unit = unit.copy()
    decree_unit["article_text"] = preamble
    decree_unit["unit_type"] = "DECREE"
    decree_unit["unit_number"] = ""

    result = [decree_unit]

    ordinal_map = {
        "الأول": "1", "الثاني": "2", "الثالث": "3", "الرابع": "4", "الخامس": "5",
        "السادس": "6", "السابع": "7", "الثامن": "8", "التاسع": "9", "العاشر": "10",
        "الواحد وعشرون": "21", "الوحيد": "1"
    }

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        article_text = text[start:end].strip()
        heading = match.group(1)
        number_match = re.search(r'\d+', heading)
        if number_match:
            article_number = number_match.group()
        else:
            for word, num in ordinal_map.items():
                if word in heading:
                    article_number = num
                    break
            else:
                article_number = str(i+1)

        article_unit = unit.copy()
        article_unit["article_text"] = article_text
        article_unit["unit_type"] = "ARTICLE"
        article_unit["unit_number"] = article_number
        article_unit["unit_title"] = heading
        article_unit["article_number"] = article_number
        result.append(article_unit)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — extract metadata + raw article list
# ─────────────────────────────────────────────────────────────────────────────

EMPTY_DOC_METADATA = {
    "law_type": "", "law_number": "", "law_title_french": "",
    "law_title_arabic": "", "institution": "", "source_name": "JORT",
    "source_date": "", "effective_date": "", "publication_date": "", "year": 0
}


def phase1b_locate_units(index, full_text):
    """Phase 1b: pure Python, zero API calls. Slices text using anchors."""
    located = []
    for unit in index:
        anchor = unit.get("anchor", "").strip()
        if not anchor:
            continue
        pos = full_text.find(anchor)
        if pos == -1:
            pos = full_text.find(anchor[:40].strip())
        if pos == -1:
            print(f"   ⚠️  Anchor not found: {anchor[:60]!r}")
            continue
        unit["_start"] = pos
        located.append(unit)

    located.sort(key=lambda u: u["_start"])

    enriched = []
    for i, unit in enumerate(located):
        start = unit["_start"]
        end   = located[i + 1]["_start"] if i + 1 < len(located) else len(full_text)
        unit["article_text"]   = full_text[start:end].strip()
        unit["article_number"] = str(unit.get("unit_number", i + 1))
        unit.setdefault("unit_type",  "ARTICLE")
        unit.setdefault("unit_title", "")
        unit.setdefault("chapter",    "")
        unit.setdefault("section",    "")
        unit.pop("_start", None)
        enriched.append(unit)

    final_units = []
    for unit in enriched:
        text = unit.get("article_text", "")
        if "الفصل" in text:
            split_units = split_decree_into_units(unit)
            final_units.extend(split_units)
        else:
            final_units.append(unit)

    return final_units


def phase1_extract(pdf_pages, skip_toc_pages=True):
    print("\n── Phase 1: Extracting all legal units ──")
    rotator   = get_rotator()
    full_text = "\n\n".join(pdf_pages)
    chunks    = group_pages_into_chunks(pdf_pages, skip_toc_pages=skip_toc_pages)
    doc_meta  = dict(EMPTY_DOC_METADATA)
    raw_index = []
    seen_anchors = set()

    print(f"📦 {len(chunks)} content chunk(s) after skipping TOC pages")

    for i, chunk in enumerate(chunks, 1):
        label = f"parse {i}/{len(chunks)}"
        print(f"\n   [{label}] {len(chunk):,} chars...")

        prompt = PDF_PARSE_PROMPT.replace("{pdf_text}", chunk)
        try:
            raw = rotator.call(SYSTEM_PROMPT, prompt, PDF_PARSE_MAX_TOKENS, label)
        except RuntimeError as e:
            raise RuntimeError(f"Phase 1 failed on chunk {i}/{len(chunks)}: {e}") from e

        parsed = parse_json_safely(raw, label)
        if not parsed:
            raise RuntimeError(
                f"Phase 1 failed on chunk {i}/{len(chunks)}: "
                f"LLM returned non-JSON.\nRaw (first 300): {(raw or '')[:300]}"
            )

        chunk_meta = parsed.get("doc_metadata", {})
        if isinstance(chunk_meta, dict):
            for key, val in chunk_meta.items():
                if key in doc_meta and not doc_meta[key] and val:
                    doc_meta[key] = val

        units = parsed.get("units") or parsed.get("articles") or []
        added = 0
        for unit in units:
            anchor = unit.get("anchor", "").strip()[:80]
            if not anchor or anchor in seen_anchors:
                continue
            seen_anchors.add(anchor)
            unit["anchor"] = anchor
            raw_index.append(unit)
            added += 1
        print(f"   ✅ {added} new units indexed | total: {len(raw_index)}")

        if i < len(chunks):
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\n── Phase 1b: Locating unit texts in document ──")
    print(f"   Full text: {len(full_text):,} chars | {len(raw_index)} anchors")

    all_raw   = phase1b_locate_units(raw_index, full_text)
    not_found = len(raw_index) - len(all_raw)
    if not_found:
        print(f"   ⚠️  {not_found} anchor(s) not found in text")
    print(f"\n✅ Phase 1 done: {len(all_raw)} units located | metadata: {doc_meta}")
    return doc_meta, all_raw


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — enrich one article
# ─────────────────────────────────────────────────────────────────────────────

def phase2_enrich(raw_article, doc_meta, idx, total):
    """Phase 2: enrich one raw article into all 62 fields."""
    rotator = get_rotator()
    label   = f"enrich {idx}/{total} art.{raw_article.get('article_number','?')}"

    prompt = (ENRICH_PROMPT
              .replace("{doc_metadata}", json.dumps(doc_meta,    ensure_ascii=False))
              .replace("{article_data}", json.dumps(raw_article, ensure_ascii=False)))

    last_error = ""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            raw = rotator.call(SYSTEM_PROMPT, prompt, ENRICH_MAX_TOKENS,
                               f"{label} attempt {attempt}")
        except RuntimeError as e:
            last_error = str(e)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY * attempt)
            continue

        parsed = parse_json_safely(raw, label)
        if not parsed:
            last_error = f"non-JSON response (first 300): {(raw or '')[:300]}"
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY * attempt)
            continue

        if has_arabic_contamination(parsed.get("content_arabic", "")):
            last_error = "content_arabic contaminated with non-Arabic characters"
            print(f"   ⚠️  [{label}] {last_error} — retrying...")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY * attempt)
            continue

        return parsed

    raise RuntimeError(f"Phase 2 failed on {label}: {last_error}")


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def compute_static_fields(article):
    french   = article.get("content_french", "")
    arabic   = article.get("content_arabic", "")
    combined = f"{french} {arabic}".strip()

    article["content_hash_sha256"] = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    article["content_combined"]    = combined

    chapter = article.get("chapter") or ""
    if chapter:
        norm = chapter.lower()
        for src, dst in [('àáâãäå','a'),('èéêë','e'),('ìíîï','i'),
                         ('òóôõö','o'),('ùúûü','u'),('ç','c')]:
            for ch in src:
                norm = norm.replace(ch, dst)
        norm = re.sub(r'[^a-z0-9]+', '_', norm).strip('_')
        article["chapter_normalized"] = norm
    else:
        article["chapter_normalized"] = ""

    law_type_slug = re.sub(r'[^a-z0-9]', '-',
                           article.get("law_type", "loi").lower()).strip('-')
    law_number    = re.sub(r'[^a-z0-9]', '-',
                           article.get("law_number", "").lower()).strip('-')
    parent_id     = f"tn-{law_type_slug}-{law_number}"
    article["parent_document_id"] = parent_id

    raw_num   = str(article.get("article_number", "0"))
    num_match = re.search(r'\d+', raw_num)
    n = int(num_match.group()) if num_match else 0
    article["article_order"]        = n
    article["preceding_article_id"] = f"{parent_id}-art-{n-1}" if n > 1 else ""
    article["following_article_id"] = f"{parent_id}-art-{n+1}"

    article.setdefault("version",     1)
    article.setdefault("graph_level", 1)

    for f in ("repeal_date", "superseded_by_id", "supersedes_id",
              "source_url", "source_number", "source_date",
              "publication_date", "effective_date", "institution_secondary"):
        if f not in article or article[f] in ("", "COMPUTED"):
            article[f] = None

    article["embedding_text"] = "\n".join(filter(None, [
        article.get("chapter", ""),
        article.get("summary_french", "") or article.get("summary", ""),
        article.get("summary_arabic", ""),
        article.get("content_french", ""),
        article.get("content_arabic", ""),
        " | ".join(article.get("keywords", [])),
    ]))

    now    = datetime.now(timezone.utc)
    future = now + timedelta(days=182)
    article["last_checked"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    article["next_check"]   = future.strftime("%Y-%m-%dT%H:%M:%SZ")

    names = article.get("entity_names", [])
    if isinstance(names, list) and names:
        article["entity_ids"] = [
            "tn-org-" + re.sub(r'[^a-z0-9]+', '-', n.lower()).strip('-')
            for n in names
        ]

    
    article["id"] = str(uuid.uuid4())
    article["jurisdiction"] = "TUNISIA"
    return article


def sanitize_article(article):
    string_fields = [
        "id","jurisdiction","law_type","law_number","institution",
        "institution_primary","title_french","title_arabic",
        "content_french","content_arabic","summary","summary_french","summary_arabic",
        "search_content","embedding_text","business_impact","status","article_type",
        "ambiguity_level","community_id","community_label","community_summary",
        "source_name","chapter_normalized","parent_document_id",
        "preceding_article_id","following_article_id","content_hash_sha256",
        "content_combined","last_checked","next_check","article_number",
        "chapter","section"
    ]
    nullable_fields = [
        "source_date","effective_date","publication_date","repeal_date",
        "institution_secondary","superseded_by_id","supersedes_id",
        "source_url","source_number"
    ]
    array_fields = [
        "institutions","keywords","legal_concepts","legal_domains","target_audience",
        "related_laws","relation_target_ids","relation_types",
        "entity_names","entity_types","entity_ids"
    ]
    int_fields  = ["year","version","graph_level","article_order"]
    bool_fields = ["has_obligations","has_penalties","has_deadlines",
                   "has_exceptions","is_abrogation","is_transitional"]

    for k in string_fields:
        if article.get(k) is None or article.get(k) == "COMPUTED":
            article[k] = ""

    for k in nullable_fields:
        if article.get(k) in ("", "COMPUTED"):
            article[k] = None

    for k in array_fields:
        if not isinstance(article.get(k), list):
            article[k] = []

    for k in int_fields:
        try:
            article[k] = int(article[k]) if article.get(k) not in (None,"","COMPUTED") else 0
        except (ValueError, TypeError):
            article[k] = 0

    for k in bool_fields:
        val = article.get(k)
        if not isinstance(val, bool):
            article[k] = str(val).lower() in ("true","1","yes") if val else False

    VALID_ARTICLE_TYPES = {
        "ARTICLE","DECREE","NOMINATION","MINUTES","CORRECTION",
        "GOV_ORDER","CIRCULAR","REGULATORY","PENAL","PROCEDURAL",
        "DEFINITIONAL","TRANSITIONAL","ABROGATION"
    }
    if article.get("article_type","").upper() not in VALID_ARTICLE_TYPES:
        article["article_type"] = "ARTICLE"

    unit_title = article.pop("unit_title", None)
    if unit_title and not article.get("title_french"):
        article["title_french"] = unit_title

    if not article["keywords"]:
        text  = article.get("content_french","") or article.get("summary","")
        words = re.findall(r'\b[a-zA-Zeèêëàâùûüîïôçœæ]{5,}\b', text)
        stop  = {"dans","pour","avec","sont","cette","leur","aussi","dont","plus",
                 "article","tunisie","droit","lequel","laquelle","selon","toute"}
        kws   = list(dict.fromkeys(w.lower() for w in words if w.lower() not in stop))
        article["keywords"] = kws[:7] if kws else ["droit-du-travail"]

    article["keywords"] = [
        k for k in article["keywords"]
        if not re.search(r'[\u0600-\u06FF]', k)
    ]
    if not article["keywords"]:
        article["keywords"] = ["droit", "tunisie", "loi"]

    if not article.get("legal_concepts"):
        article["legal_concepts"] = []
    article["legal_concepts"] = [
        c for c in article["legal_concepts"]
        if not re.search(r'[\u0600-\u06FF]', c)
    ]

    if not article["legal_domains"]:
        cl = (article.get("content_french","") + " " + article.get("summary","")).lower()
        domains = []
        if any(w in cl for w in ["travail","salarie","employe","contrat de travail","salaire"]):
            domains.append("Droit du Travail")
        if any(w in cl for w in ["commerce","societe","entreprise","commercial","capital"]):
            domains.append("Droit Commercial")
        if any(w in cl for w in ["fiscal","impot","taxe","contribution","declaration"]):
            domains.append("Droit Fiscal")
        if any(w in cl for w in ["penal","infraction","amende","emprisonnement","sanction"]):
            domains.append("Droit Penal")
        if any(w in cl for w in ["administratif","ministere","administration","decret"]):
            domains.append("Droit Administratif")
        article["legal_domains"] = domains if domains else ["Droit Administratif"]

    if not article["target_audience"]:
        article["target_audience"] = ["Entreprises", "Particuliers"]

    if not article.get("summary"):
        french = article.get("content_french","")
        article["summary"]        = french[:300] + "..." if len(french) > 300 else french
        article["summary_french"] = article["summary"]

    for field in ["content_arabic", "summary_arabic", "title_arabic"]:
        if article.get(field):
            article[field] = clean_arabic_text(article[field])

    return article


# ─────────────────────────────────────────────────────────────────────────────
# DEDUP + SORT + GROUP
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(articles):
    seen, unique = set(), []
    for a in articles:
        law_id      = f"{a.get('law_type','')}|{a.get('law_number','')}"
        fingerprint = (a.get('content_french','')[:200] + a.get('content_arabic','')[:200])
        key = (law_id, fingerprint)
        if key not in seen:
            seen.add(key)
            unique.append(a)
    removed = len(articles) - len(unique)
    if removed:
        print(f"🔁 Removed {removed} duplicates")
    return unique


def sort_articles(articles):
    return sorted(articles, key=lambda a: a.get("article_order", 0))


def group_articles_under_decrees(units):
    decree_types = ('DECREE', 'GOV_ORDER')
    decrees = [u for u in units if u.get('article_type') in decree_types
               or u.get('unit_type') in decree_types]
    grouped, used = [], set()
    for d in decrees:
        law_num, src_date = d.get('law_number',''), d.get('source_date','')
        children = [u for u in units
                    if u.get('article_type') == 'ARTICLE'
                    and u.get('law_number') == law_num
                    and u.get('source_date') == src_date]
        for c in children:
            used.add(id(c))
        if children:
            d['articles'] = children
        grouped.append(d)
    for u in units:
        if id(u) not in used and u.get('article_type') not in decree_types:
            grouped.append(u)
    return grouped


# ─────────────────────────────────────────────────────────────────────────────
# INCREMENTAL SAVE + RESUME
# ─────────────────────────────────────────────────────────────────────────────

def _save_incremental(output_path, articles, hierarchical=False):
    if hierarchical:
        articles = group_articles_under_decrees(articles)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def _load_existing_results(output_path):
    if not Path(output_path).exists():
        return [], set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        done_keys = set()
        for a in existing:
            num         = str(a.get("article_number", ""))
            law_num     = str(a.get("law_number", ""))
            fingerprint = a.get("content_french","")[:200] + a.get("content_arabic","")[:200]
            done_keys.add((law_num, num, fingerprint))
        print(f"   ♻️  Loaded {len(existing)} already-extracted articles from {output_path}")
        return existing, done_keys
    except Exception as e:
        print(f"   ⚠️  Could not load existing output ({e}) — starting fresh")
        return [], set()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_extraction(input_path, output_path=None, enrich_limit=None,
                   dry_run=False, test_mode=False, hierarchical=False):
    print("\n" + "=" * 60)
    print("🇹🇳 LegaTech Tunisia Extractor v4 — Vertex AI Edition")
    print("=" * 60)

    if not PROJECT_ID:
        raise RuntimeError(
            "Vertex AI credentials not set.\n"
            "  Add to your .env:  PROJECT_ID=your_project_id\n"
            "                     GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json"
        )

    if output_path is None:
        stem        = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}_extracted.json")

    if test_mode and not enrich_limit:
        enrich_limit = 1

    if dry_run:
        print("\n🔬 DRY-RUN MODE — no API calls will be made")
    if test_mode:
        print("\n🧪 TEST MODE — only 1 article will be enriched")

    # Step 1: Read input
    print("\n📖 Step 1: Reading input...")
    if input_path.lower().endswith('.md'):
        _full_text, pdf_pages = extract_text_from_md(input_path)
        skip_toc_pages = False
    elif input_path.lower().endswith('.pdf'):
        _full_text, pdf_pages = extract_text_from_pdf(input_path)
        skip_toc_pages = True
    else:
        raise ValueError(f"Unsupported file type: {input_path}. Only .pdf and .md are supported.")

    # Step 2: Phase 1
    print("\n📑 Step 2: Phase 1 — extracting all legal units...")
    if dry_run:
        doc_meta = dict(EMPTY_DOC_METADATA)
        raw_arts = []
        print("   [dry-run] skipping Phase 1 API calls")
    else:
        doc_meta, raw_arts = phase1_extract(pdf_pages, skip_toc_pages=skip_toc_pages)

    if not dry_run and not raw_arts:
        raise ValueError("No legal units found in input file")

    print(f"\n📋 Metadata: law_type={doc_meta.get('law_type')} | "
          f"law_number={doc_meta.get('law_number')} | "
          f"source_date={doc_meta.get('source_date')}")

    # Step 3: Phase 2
    enriched, done_keys = [], set()
    to_enrich = raw_arts[:enrich_limit] if enrich_limit else raw_arts
    total     = len(to_enrich)
    law_num   = doc_meta.get("law_number", "")
    success = skipped = 0

    print(f"\n🤖 Step 3: Phase 2 — enriching {total} unit(s) "
          f"({len(done_keys)} already done)...")

    for i, raw in enumerate(to_enrich, 1):
        unit_label  = raw.get("unit_title") or raw.get("unit_type") or "unit"
        art_num     = str(raw.get("article_number", ""))
        fingerprint = raw.get("article_text", "")[:200]
        done_key    = (law_num, art_num, fingerprint)

        print(f"\n── Unit {i}/{total} [{raw.get('unit_type','?')}] "
              f"{str(unit_label)[:60]} ──")

        if dry_run:
            print(f"   ⏭️  [dry-run] would enrich: "
                  f"{str(raw.get('article_text',''))[:120]}...")
            skipped += 1
            continue

        result = phase2_enrich(raw, doc_meta, i, total)
        result = sanitize_article(result)
        result = compute_static_fields(result)
        enriched.append(result)
        done_keys.add(done_key)
        success += 1

        try:
            _save_incremental(output_path, enriched, hierarchical)
        except Exception as e:
            print(f"   ⚠️  save error: {e}")

        if i < total:
            time.sleep(RATE_LIMIT_DELAY)

    # Step 4: Final save
    print(f"\n📊 Total: {success} enriched | {skipped} skipped")
    print("\n⚙️  Step 4: Sorting and saving...")
    final = sort_articles(enriched)
    ids = [a["id"] for a in final]
    if len(ids) != len(set(ids)):
        print(f"⚠️  WARNING: {len(ids) - len(set(ids))} duplicate IDs detected — this should not happen")
    _save_incremental(output_path, final, hierarchical)
    print(f"✅ Final count: {len(final)} units")
    print(f"\n💾 Saved -> {output_path}")
    print("=" * 60 + "\n")
    return final


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractor_vertex.py <file> [output.json] "
              "[--limit N] [--dry-run] [--test] [--hierarchical]")
        print()
        print("  <file>          .pdf or .md input file")
        print("  --dry-run       Parse only, skip all API calls (free)")
        print("  --test          Enrich 1 article only (minimal cost)")
        print("  --hierarchical  Group articles under parent decrees")
        print()
        print("Environment variables (.env):")
        print("  GOOGLE_APPLICATION_CREDENTIALS   Path to service account JSON")
        print("  PROJECT_ID                        GCP project ID")
        print("  LOCATION                          Vertex AI region (e.g. us-central1)")
        print("  MODEL                             Gemini model (e.g. gemini-2.5-flash)")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file, limit = None, None
    dry_run = test_mode = hierarchical = False
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1]); i += 2
        elif args[i] == "--dry-run":
            dry_run = True; i += 1
        elif args[i] == "--test":
            test_mode = True; i += 1
        elif args[i] == "--hierarchical":
            hierarchical = True; i += 1
        elif not output_file and not args[i].startswith("--"):
            output_file = args[i]; i += 1
        else:
            i += 1

    results = run_extraction(
        input_file, output_file,
        enrich_limit=limit,
        dry_run=dry_run,
        test_mode=test_mode,
        hierarchical=hierarchical,
    )

    print(f"✅ Done — {len(results)} articles extracted")
    if results:
        s = results[0]
        print(f"\nSample — Article {s.get('article_number')}:")
        print(f"  title_french:   {s.get('title_french','')[:70]}")
        print(f"  content_french: {s.get('content_french','')[:150]}...")
        print(f"  summary:        {s.get('summary','')[:150]}")
        print(f"  keywords:       {s.get('keywords')}")
        print(f"  source_date:    {s.get('source_date')}")
        print(f"  effective_date: {s.get('effective_date')}")
        print(f"  related_laws:   {s.get('related_laws')}")
        print(f"  entity_names:   {s.get('entity_names')}")
        print(f"  model used:     {MODEL}")