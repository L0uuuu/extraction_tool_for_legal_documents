"""
LegaTech Tunisia Legal Extractor — v3 (Cloudflare Workers AI)
==============================================================
Two-phase pipeline:
  Phase 1 : PDF text → LLM → document metadata + list of raw articles
  Phase 2 : Each raw article individually → LLM → all 62 enriched fields

API backend: Cloudflare Workers AI
  Endpoint : https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1
  Model    : @cf/openai/gpt-oss-20b
  Auth     : Bearer token via CF_AUTH_TOKEN env var
  Account  : CF_ACCOUNT_ID env var
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
from openai import OpenAI
import fitz  # pymupdf

# ── optional .env support ──────────────────────
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# ── Cloudflare credentials ───────────────────
# Set these in your .env file or as environment variables.
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_AUTH_TOKEN = os.getenv("CF_AUTH_TOKEN", "")

# ── Model ─────────────────────────────────────
CF_MODEL = "@cf/openai/gpt-oss-20b"

PDF_PARSE_CHUNK_CHARS = 8000
PDF_PARSE_MAX_TOKENS  = 8192
ENRICH_MAX_TOKENS     = 8192
RATE_LIMIT_DELAY      = 3.0
RETRY_ATTEMPTS        = 3
RETRY_BASE_DELAY      = 2


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Tunisian legal analyst and professional bilingual translator (French ↔ Arabic).

ABSOLUTE RULES — never break these:
- Return ONLY valid MINIFIED JSON — no whitespace, no indentation, no newlines
- NO EXPLANATIONS, NO REASONING, NO THOUGHT PROCESS — just the JSON
- DO NOT include any text before or after the JSON
- NEVER use null — use "" for missing strings, [] for missing arrays
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
- institution rules: Loi → "Assemblée des Représentants du Peuple" | Décret → "Présidence de la République" | Circulaire BCT → "Banque Centrale de Tunisie"
"""


# ─────────────────────────────────────────────
# PHASE 1 PROMPT — document metadata + raw articles
# ─────────────────────────────────────────────

PDF_PARSE_PROMPT = """You are reading a chunk of a Tunisian official legal document.
Your task: identify and INDEX every important legal unit — do NOT copy the full text.

A legal unit is ANY of:
  - قرار    (decree/decision by a minister)
  - بمقتضى قرار / تسمية / تكليف  (nomination or appointment)
  - محضر جلسة  (meeting minutes)
  - إصلاح خطأ  (erratum/correction)
  - أمر حكومي  (governmental order)
  - فصل / مادة  (article inside a law)
  - Any other official legal text

For EACH unit return ONE object. doc_metadata: fill what you find, "" if absent.

RETURN MINIFIED JSON:
{"doc_metadata":{"law_type":"","law_number":"","law_title_arabic":"","institution":"","source_name":"JORT","source_date":"","effective_date":"","publication_date":"","year":0},"units":[{"unit_type":"DECREE or NOMINATION or ARTICLE or MINUTES or CORRECTION or GOV_ORDER","unit_title":"full Arabic title of this unit (first line/heading)","unit_number":"sequential or article number","anchor":"EXACT first 80 characters of this unit as they appear in the text"}]}

CRITICAL RULES:
- anchor must be the EXACT first 80 characters of the unit copied verbatim from the text
- unit_title is the heading/title line of the unit (e.g. "قرار من وزير الصحة مؤرخ في 2 فيفري 2026 يتعلق...")
- Do NOT include unit_text — we extract the full text in a separate step
- Skip table of contents lines (lines ending with page numbers after dots .......)
- Each distinct legal unit = one object in units array
- If the same unit appears in TOC and again as content, use the content version for the anchor

DOCUMENT CHUNK:
{pdf_text}

RETURN ONLY THE MINIFIED JSON. NO MARKDOWN. NO EXPLANATIONS."""


# ─────────────────────────────────────────────
# PHASE 2 PROMPT — enrich one article (all 62 fields)
# ─────────────────────────────────────────────

ENRICH_PROMPT = """Enrich this Tunisian legal article into a complete JSON object with ALL fields below.

DOCUMENT METADATA (already extracted — use these values directly, do not re-derive):
{doc_metadata}

RAW ARTICLE:
{article_data}

═══════════════════════════════════════════
FIELD DEFINITIONS
═══════════════════════════════════════════

── IDENTITY (use values from doc_metadata above) ──
  id           : Generate a new UUID4 string
  jurisdiction : Always "TUNISIA"
  year         : Integer year from doc_metadata
  chapter      : From raw article data, "" if absent
  section      : From raw article data, "" if absent
  article_number: From raw article data (e.g. "1", "6", "14 quater")

── STATIC (write "COMPUTED" — Python replaces these) ──
  content_hash_sha256  : "COMPUTED"
  content_combined     : "COMPUTED"
  chapter_normalized   : "COMPUTED"
  preceding_article_id : "COMPUTED"
  following_article_id : "COMPUTED"
  last_checked         : "COMPUTED"
  next_check           : "COMPUTED"
  version              : 1
  graph_level          : 1
  repeal_date          : ""
  superseded_by_id     : ""
  supersedes_id        : ""
  source_url           : ""
  source_number        : ""

── FROM DOC METADATA (copy from doc_metadata above) ──
  law_type         : copy from doc_metadata.law_type
  law_number       : copy from doc_metadata.law_number
  institution      : copy from doc_metadata.institution
  institution_primary   : first institution name
  institution_secondary : second institution if multiple, else ""
  institutions          : array e.g. ["Assemblée des Représentants du Peuple"]
  title_french     : copy from doc_metadata.law_title_french
  title_arabic     : copy from doc_metadata.law_title_arabic
  source_name      : copy from doc_metadata.source_name
  source_date      : copy from doc_metadata.source_date
  effective_date   : copy from doc_metadata.effective_date
  publication_date : copy from doc_metadata.publication_date
  parent_document_id : "tn-{law_type_slug}-{law_number}" e.g. "tn-loi-66-27"

── LLM ANALYSIS (analyze the article text and generate) ──

  content_french : FULL verbatim French text of the article — copy exactly from article_text.
                   NEVER summarize. NEVER truncate. If article_text is already French, copy it as-is.

  content_arabic : Complete Arabic translation of content_french.
                   STRICT RULE: use ONLY Arabic script — absolutely no French, Latin, or Cyrillic
                   characters mixed in. Every word must be Arabic.

  summary        : EXACTLY 40-60 French words describing what this article does and who it affects.
                   COUNT your words — if under 40, add more detail. If over 60, trim.
                   MUST be in French. NEVER write a title. NEVER use Arabic here.
                   Example of good summary: "Cet article établit les conditions d'application du
                   code du travail aux catégories de travailleurs exerçant dans des entreprises
                   industrielles ou commerciales, incluant les gérants, directeurs et représentants
                   de commerce dont la rémunération est liée aux résultats."

  summary_french : exact copy of summary
  summary_arabic : Arabic translation of summary — 40-60 Arabic words

  search_content : Combined search string: title_french + article_number + summary + top keywords in French and Arabic

  keywords       : Array of 5-7 keywords. STRICT RULES:
                   - ALWAYS in French — never Arabic, never mixed language
                   - Extract from actual article content — meaningful legal/business terms
                   - No generic words like "loi", "article", "tunisie"
                   - Good examples: ["contrat de travail", "convention collective", "salaire minimum"]

  legal_concepts : Array of 3-8 legal doctrines or principles present in this article.
                   DIFFERENT from keywords — these are the legal concepts at play, not search terms.
                   Always in French. Never empty if the article has legal substance.
                   Examples: ["droit de rétention", "bonne foi", "prescription acquisitive",
                              "nullité relative", "force majeure", "responsabilité contractuelle"]

  legal_domains  : Array of 1-3 French legal domains. NEVER empty.
                   Valid: Droit Civil | Droit Commercial | Droit Fiscal | Droit Administratif |
                   Droit du Travail | Droit Pénal | Droit de la Santé | Droit Agricole | Droit Bancaire

  business_impact  : LOW | MEDIUM | HIGH
  target_audience  : Array of affected parties in French. NEVER empty. E.g. ["Entreprises", "Salariés"]
  status           : ACTIVE | AMENDED | REPEALED | SUSPENDED

  article_type   : The type of this legal unit. Use the unit_type from the raw data if present, otherwise infer:
                   ARTICLE     — a numbered فصل inside a decree or law
                   DECREE      — a full قرار (decision/decree) by a minister or authority
                   NOMINATION  — a بمقتضى قرار appointing or assigning a person
                   MINUTES     — محضر جلسة meeting minutes
                   CORRECTION  — إصلاح خطأ erratum / correction
                   GOV_ORDER   — أمر حكومي governmental order
                   CIRCULAR    — منشور / دورية circular
                   REGULATORY | PENAL | PROCEDURAL | DEFINITIONAL | TRANSITIONAL | ABROGATION — for plain articles
  article_order  : Integer sequence number of this unit within the document (e.g. 1, 2, 3...).

  ambiguity_level : LOW | MEDIUM | HIGH

  has_obligations : true only if article explicitly uses: "doit", "est tenu de", "ne doit pas", "il incombe de", "est obligé de"
  has_penalties   : true only if article explicitly mentions fines, imprisonment, sanctions, or withdrawal of license
  has_deadlines   : true only if article specifies a time limit, deadline, or transition period
  has_exceptions  : true only if article uses: "sauf", "sous réserve de", "à l'exception de", "hormis", "toutefois"
  is_abrogation   : true only if article uses: "est abrogé", "sont abrogées", "demeure abrogé"
  is_transitional : true only if article contains explicitly transitional or temporary provisions

  related_laws    : STRICT RULE — only laws EXPLICITLY cited by name or number in the article text.
                    If no law is explicitly cited, return []. NEVER invent or suggest related laws.
                    Good example: article says "...en application du décret du 4 août 1936..." → ["Décret du 4 août 1936"]
                    Bad example: article is about labour law → do NOT add "Code des Sociétés Commerciales"

  relation_target_ids : IDs of explicitly referenced laws e.g. ["tn-decret-1936-08-04"]. [] if none.
  relation_types      : Parallel to relation_target_ids. Values: REFERENCES | AMENDS | REPEALS | IMPLEMENTS | SUPERSEDES

  entity_names : STRICT RULE — only SPECIFIC named legal entities explicitly mentioned in the article.
                 Examples of valid entries: "Ministère des Affaires Sociales", "Tribunal de Tunis", "BCT"
                 NEVER include generic roles: "l'employeur", "le salarié", "les travailleurs", "les entreprises"
                 If no specific named entity appears, return [].
  entity_types : Parallel to entity_names. Values: ORGANIZATION | INSTITUTION | COURT | MINISTRY | PERSON
  entity_ids   : Slug for each entity e.g. ["tn-org-ministere-affaires-sociales"]

  community_id      : Thematic slug e.g. "droit-du-travail" | "droit-fiscal" | "droit-bancaire"
  community_label   : 5-7 word French description e.g. "Droit du travail et relations salariales"
  community_summary : One French sentence describing the thematic cluster

═══════════════════════════════════════════
RETURN ONLY A VALID MINIFIED JSON OBJECT. 
NO EXPLANATIONS. NO REASONING. NO THOUGHT PROCESS. JUST THE JSON.
DO NOT include any text before or after the JSON.
═══════════════════════════════════════════"""


# ─────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> tuple[str, list[str]]:
    """
    Extract text from PDF using pymupdf (fitz).
    Returns (full_text, pages) where pages is a list of per-page strings.
    pymupdf handles Arabic RTL and Unicode correctly — no character reversal issues.
    Per-page list is used by the gazette splitter.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[str] = []
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


# ─────────────────────────────────────────────
# PAGE-GROUP CHUNKING
# ─────────────────────────────────────────────

def is_toc_page(text: str) -> bool:
    """
    Return True if this page is a table of contents.
    TOC pages have many dot-lines like: "قرار من وزير ........ 218"
    Content pages have actual article/decree text.
    """
    dot_lines = sum(1 for line in text.split('\n') if line.count('.') > 5)
    return dot_lines >= 3


def group_pages_into_chunks(pages: list[str],
                            max_chars: int = PDF_PARSE_CHUNK_CHARS) -> list[str]:
    """
    Group pages into chunks for Phase 1.
    - Skips TOC pages entirely (no useful content for extraction)
    - Groups consecutive pages up to max_chars per chunk
    - Never splits a page across two chunks
    """
    content_pages = []
    for i, page in enumerate(pages, 1):
        if is_toc_page(page):
            # Show first meaningful line so user knows what was skipped
            first_line = next(
                (l.strip() for l in page.split('\n') if len(l.strip()) > 10),
                "(empty)"
            )
            print(f"   🗂️  Page {i} skipped (TOC): {first_line[:80]}")
        else:
            content_pages.append(page)

    skipped = len(pages) - len(content_pages)
    print(f"   📋 {skipped} TOC page(s) skipped | {len(content_pages)} content page(s) kept")

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


# ─────────────────────────────────────────────
# JSON PARSING & REPAIR
# ─────────────────────────────────────────────

def parse_json_safely(raw: str, label: str = "") -> Optional[dict]:
    if not raw:
        return None

    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```\s*$',          '', cleaned)
    cleaned = cleaned.strip()

    # Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        if label:
            print(f"⚠️  [{label}] JSON error at pos {e.pos}: {e.msg}")

    # Repair 1: fix unclosed brackets
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

    # Repair 2: common syntax mistakes
    try:
        fixed = re.sub(r',\s*([}\]])',      r'\1',  cleaned)  # trailing commas
        fixed = re.sub(r'(?<=[^\\])\n',     ' ',    fixed)    # bare newlines
        fixed = re.sub(r'(?<=[^\\])\t',     ' ',    fixed)    # bare tabs
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


# ─────────────────────────────────────────────
# REGEX FALLBACK — article extraction
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# OPENAI-COMPATIBLE CLIENT FOR CLOUDFLARE
# ─────────────────────────────────────────────

def _make_client() -> OpenAI:
    return OpenAI(
        api_key  = CF_AUTH_TOKEN,
        base_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1",
    )


# ─────────────────────────────────────────────
# CLOUDFLARE API WRAPPER
# ─────────────────────────────────────────────

def call_cf(client, system: str, user: str,
            max_tokens: int, label: str = "") -> str:
    """
    Call Cloudflare Workers AI via the OpenAI-compatible client.

    Uses: POST /v1/chat/completions
    Model: CF_MODEL (@cf/openai/gpt-oss-20b)

    On failure: raises RuntimeError with reason — caller decides what to do.
    Retries up to RETRY_ATTEMPTS times with exponential backoff.
    """
    cf_client = _make_client()

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = cf_client.chat.completions.create(
                model    = CF_MODEL,
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens      = max_tokens,
                temperature     = 0.2,
                response_format = {"type": "json_object"},
            )
            text          = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            tokens_in     = response.usage.prompt_tokens     if response.usage else "?"
            tokens_out    = response.usage.completion_tokens if response.usage else "?"

            if finish_reason == "length":
                print(f"   ⚠️  [{label}] output truncated (token limit hit)")

            print(f"   📊 [{label}] {tokens_in} in / {tokens_out} out "
                  f"| finish: {finish_reason} | model: {CF_MODEL}")

            return text

        except Exception as e:
            print(f"   ❌ [{label}] attempt {attempt}/{RETRY_ATTEMPTS}: {e}")
            if attempt < RETRY_ATTEMPTS:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"   ⏳ retry in {delay}s...")
                time.sleep(delay)

    raise RuntimeError(f"[{label}] API call failed after {RETRY_ATTEMPTS} attempts")

# ─────────────────────────────────────────────
# ARABIC CONTAMINATION CHECK
# ─────────────────────────────────────────────

def has_arabic_contamination(text: str) -> bool:
    """
    Check if Arabic text contains non-Arabic contamination.
    Allows:
    - Numbers (0-9)
    - Common punctuation
    - Known acronyms (JORT, BCT, etc.)
    - Short Latin terms that might be part of legal citations
    """
    if not text:
        return False
    
    # Remove numbers and common punctuation first
    cleaned = re.sub(r'[0-9\s\-–—:;.,!?()\[\]{}"\']+', ' ', text)
    
    # Known acceptable Latin terms in Tunisian legal documents
    acceptable_latin = {'JORT', 'BCT', 'CF', 'art', 'Art', 'ART', 'Code', 'Loi', 'Décret'}
    
    # Split into words and check each
    words = cleaned.split()
    for word in words:
        # If word contains Latin characters
        if re.search(r'[a-zA-Z]', word):
            # Check if it's a known acceptable term
            if word not in acceptable_latin and not any(term in word for term in acceptable_latin):
                # Check if it's a long Latin sequence (more than 3 chars)
                if len(re.findall(r'[a-zA-Z]', word)) > 3:
                    return True
    
    return False

# ─────────────────────────────────────────────
# PHASE 1 — extract metadata + raw article list
# ─────────────────────────────────────────────

EMPTY_DOC_METADATA = {
    "law_type": "", "law_number": "", "law_title_french": "",
    "law_title_arabic": "", "institution": "", "source_name": "JORT",
    "source_date": "", "effective_date": "", "publication_date": "", "year": 0
}

# ─────────────────────────────────────────────
# SECTION BOUNDARY DETECTION
# ─────────────────────────────────────────────

# Headers that mark a new major section in a Tunisian legal document.
# When one of these is found, the article numbering resets — so we start
# a new section and allow article numbers to repeat.
SECTION_MARKERS = [
    r'DISPOSITIONS\s+GENERALES',
    r'TITRE\s+(?:I|II|III|IV|V|VI|VII|VIII|IX|X|PREMIER)',
    r'LIVRE\s+(?:I|II|III|IV|V|PREMIER)',
    r'PARTIE\s+(?:I|II|III|IV|PREMIERE)',
]

def split_text_into_sections(pdf_text: str) -> list[tuple[str, str]]:
    """
    Split PDF text into sections at major structural boundaries.
    Returns list of (section_label, section_text) tuples.

    This solves the duplicate article number problem: a document like the
    Code du Travail has articles 1-4 in the promulgation law, then articles
    1-N again in the code body. By splitting into sections we can track
    (section, article_number) as a unique key.
    """
    combined_pattern = re.compile(
        '|'.join(f'({m})' for m in SECTION_MARKERS),
        re.IGNORECASE
    )

    splits = list(combined_pattern.finditer(pdf_text))

    if not splits:
        # No section markers found — treat whole text as one section
        return [('section-1', pdf_text)]

    sections = []

    # Text before first marker = section 0 (preamble/promulgation)
    preamble = pdf_text[:splits[0].start()].strip()
    if preamble:
        sections.append(('promulgation', preamble))

    for i, match in enumerate(splits):
        start = match.start()
        end   = splits[i+1].start() if i+1 < len(splits) else len(pdf_text)
        label = re.sub(r'\s+', '-', match.group(0).strip().lower())
        sections.append((label, pdf_text[start:end].strip()))

    print(f"   📂 Document split into {len(sections)} sections: {[s[0] for s in sections]}")
    return sections



def phase1b_locate_units(index: list[dict], full_text: str) -> list[dict]:
    """
    Phase 1b — pure Python, zero API calls.

    Uses the anchors returned by Phase 1a to locate each unit's start position
    in the full document text. End of unit N = start of unit N+1.
    Returns the same list with `article_text` populated.
    """
    located = []

    # Find start position for each unit using its anchor
    for unit in index:
        anchor = unit.get("anchor", "").strip()
        if not anchor:
            continue
        pos = full_text.find(anchor)
        if pos == -1:
            # Try shorter anchor (first 40 chars) in case of minor whitespace differences
            short = anchor[:40].strip()
            pos   = full_text.find(short)
        if pos == -1:
            print(f"   ⚠️  Anchor not found: {anchor[:60]!r}")
            continue
        unit["_start"] = pos
        located.append(unit)

    # Sort by position in document
    located.sort(key=lambda u: u["_start"])

    # Assign full text: start of this unit → start of next unit
    enriched = []
    for i, unit in enumerate(located):
        start = unit["_start"]
        end   = located[i + 1]["_start"] if i + 1 < len(located) else len(full_text)
        text  = full_text[start:end].strip()

        unit["article_text"]   = text
        unit["article_number"] = str(unit.get("unit_number", i + 1))
        unit.setdefault("unit_type",  "ARTICLE")
        unit.setdefault("unit_title", "")
        unit.setdefault("chapter",    "")
        unit.setdefault("section",    "")
        unit.pop("_start", None)
        enriched.append(unit)

    return enriched


def phase1_extract(client, pdf_pages: list[str]) -> tuple[dict, list[dict]]:
    """
    Two-sub-step Phase 1:

    1a) LLM reads each page-group chunk and returns a LIGHTWEIGHT INDEX:
        unit_type, unit_title, unit_number, anchor (first 80 chars).
        Output is tiny → never hits token limit.

    1b) Python uses anchors to slice the full text into per-unit verbatim blocks.
        Next anchor start = end of current unit.  Zero API calls.

    Returns: (doc_metadata, list of raw unit dicts with article_text populated)
    """
    print("\n── Phase 1: Extracting all legal units ──")

    full_text = "\n\n".join(pdf_pages)
    chunks    = group_pages_into_chunks(pdf_pages)
    doc_meta  = dict(EMPTY_DOC_METADATA)
    raw_index: list[dict] = []
    seen_anchors: set     = set()

    print(f"📦 {len(chunks)} content chunk(s) after skipping TOC pages")

    # ── 1a: LLM builds lightweight index ─────────────────────────────────────
    for i, chunk in enumerate(chunks, 1):
        label = f"parse {i}/{len(chunks)}"
        print(f"\n   [{label}] {len(chunk):,} chars...")

        prompt = PDF_PARSE_PROMPT.replace("{pdf_text}", chunk)
        try:
            raw = call_cf(None, SYSTEM_PROMPT, prompt, PDF_PARSE_MAX_TOKENS, label)
        except RuntimeError as e:
            raise RuntimeError(f"Phase 1 failed on chunk {i}/{len(chunks)}: {e}") from e

        parsed = parse_json_safely(raw, label)
        if not parsed:
            raise RuntimeError(
                f"Phase 1 failed on chunk {i}/{len(chunks)}: "
                f"LLM returned non-JSON.\nRaw (first 300): {(raw or '')[:300]}"
            )

        # Merge doc_metadata
        chunk_meta = parsed.get("doc_metadata", {})
        if isinstance(chunk_meta, dict):
            for key, val in chunk_meta.items():
                if key in doc_meta and not doc_meta[key] and val:
                    doc_meta[key] = val

        # Collect index entries — dedup by anchor
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

    # ── 1b: Python slices full text at anchor positions ───────────────────────
    all_raw = phase1b_locate_units(raw_index, full_text)

    located   = len(all_raw)
    not_found = len(raw_index) - located
    if not_found:
        print(f"   ⚠️  {not_found} anchor(s) not found in text")
    print(f"\n✅ Phase 1 done: {located} units located | metadata: {doc_meta}")
    return doc_meta, all_raw



# ─────────────────────────────────────────────
# PHASE 2 — enrich one article
# ─────────────────────────────────────────────

def phase2_enrich(client, raw_article: dict, doc_meta: dict,
                  idx: int, total: int) -> Optional[dict]:
    """
    Phase 2: enrich one raw article into all 62 fields.
    Passes doc_metadata so the LLM doesn't need to re-derive dates/law info.
    Retries if content_arabic is contaminated.
    """
    label = f"enrich {idx}/{total} art.{raw_article.get('article_number','?')}"

    prompt = (ENRICH_PROMPT
              .replace("{doc_metadata}", json.dumps(doc_meta,    ensure_ascii=False))
              .replace("{article_data}", json.dumps(raw_article, ensure_ascii=False)))

    last_error = ""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            raw = call_cf(None, SYSTEM_PROMPT, prompt, ENRICH_MAX_TOKENS,
                          f"{label} attempt {attempt}")
        except RuntimeError as e:
            last_error = str(e)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY * attempt)
            continue

        parsed = parse_json_safely(raw, label)
        if not parsed:
            last_error = f"non-JSON response (first 300 chars): {(raw or '')[:300]}"
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY * attempt)
            continue

        # Check Arabic contamination
        arabic = parsed.get("content_arabic", "")
        if has_arabic_contamination(arabic):
            last_error = "content_arabic contaminated with non-Arabic characters"
            print(f"   ⚠️  [{label}] {last_error} — retrying...")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY * attempt)
            continue

        return parsed

    raise RuntimeError(f"Phase 2 failed on {label}: {last_error}")


# ─────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────

def compute_static_fields(article: dict) -> dict:
    """Python owns all static fields — recompute regardless of what LLM returned."""
    french   = article.get("content_french", "")
    arabic   = article.get("content_arabic", "")
    combined = f"{french} {arabic}".strip()

    article["content_hash_sha256"] = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    article["content_combined"]    = combined

    # Chapter normalization
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

    # Parent ID and navigation
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

    # Nullable fields — keep as None if not set, never force to ""
    for f in ("repeal_date", "superseded_by_id", "supersedes_id",
              "source_url", "source_number", "source_date",
              "publication_date", "effective_date", "institution_secondary"):
        if f not in article or article[f] == "" or article[f] == "COMPUTED":
            article[f] = None

    # embedding_text — pre-built blob for vector embedding models
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

    # Entity IDs
    names = article.get("entity_names", [])
    if isinstance(names, list) and names:
        article["entity_ids"] = [
            "tn-org-" + re.sub(r'[^a-z0-9]+', '-', n.lower()).strip('-')
            for n in names
        ]

    if not article.get("id") or article.get("id") == "COMPUTED":
        article["id"] = str(uuid.uuid4())
    article["jurisdiction"] = "TUNISIA"

    return article


def sanitize_article(article: dict) -> dict:
    """Enforce correct types on all 62 fields and apply fallbacks."""
    # Fields that must be strings (None/COMPUTED → "")
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
    # Fields that are None when unknown — never forced to ""
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

    # Nullable fields: "" and "COMPUTED" → None, but keep actual values
    for k in nullable_fields:
        v = article.get(k)
        if v == "" or v == "COMPUTED":
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

    # ── Fallbacks for critical fields ──

    # Carry over unit_type and unit_title from Phase 1 raw data if LLM missed them
    VALID_ARTICLE_TYPES = {
        "ARTICLE","DECREE","NOMINATION","MINUTES","CORRECTION",
        "GOV_ORDER","CIRCULAR","REGULATORY","PENAL","PROCEDURAL",
        "DEFINITIONAL","TRANSITIONAL","ABROGATION"
    }
    if article.get("article_type","").upper() not in VALID_ARTICLE_TYPES:
        article["article_type"] = "ARTICLE"  # safe default

    # unit_title — store as title_french if not already set and we have one
    unit_title = article.pop("unit_title", None)
    if unit_title and not article.get("title_french"):
        article["title_french"] = unit_title

    # Keywords: always in French, never empty
    if not article["keywords"]:
        text  = article.get("content_french","") or article.get("summary","")
        words = re.findall(r'\b[a-zA-Zéèêëàâùûüîïôçœæ]{5,}\b', text)
        stop  = {"dans","pour","avec","sont","cette","leur","aussi","dont","plus",
                 "article","tunisie","droit","lequel","laquelle","selon","toute"}
        kws   = list(dict.fromkeys(w.lower() for w in words if w.lower() not in stop))
        article["keywords"] = kws[:7] if kws else ["droit-du-travail"]

    # Remove any Arabic from keywords
    article["keywords"] = [
        k for k in article["keywords"]
        if not re.search(r'[\u0600-\u06FF]', k)
    ]
    if not article["keywords"]:
        article["keywords"] = ["droit", "tunisie", "loi"]

    # legal_concepts fallback — if LLM left it empty, keep as []
    if not article.get("legal_concepts"):
        article["legal_concepts"] = []
    # Remove any Arabic that leaked into legal_concepts
    article["legal_concepts"] = [
        c for c in article["legal_concepts"]
        if not re.search(r'[؀-ۿ]', c)
    ]

    # Legal domains fallback
    if not article["legal_domains"]:
        cl = (article.get("content_french","") + " " + article.get("summary","")).lower()
        domains = []
        if any(w in cl for w in ["travail","salarié","employé","contrat de travail","salaire"]):
            domains.append("Droit du Travail")
        if any(w in cl for w in ["commerce","société","entreprise","commercial","capital"]):
            domains.append("Droit Commercial")
        if any(w in cl for w in ["fiscal","impôt","taxe","contribution","déclaration"]):
            domains.append("Droit Fiscal")
        if any(w in cl for w in ["pénal","infraction","amende","emprisonnement","sanction"]):
            domains.append("Droit Pénal")
        if any(w in cl for w in ["administratif","ministère","administration","décret"]):
            domains.append("Droit Administratif")
        article["legal_domains"] = domains if domains else ["Droit Administratif"]

    # Target audience fallback
    if not article["target_audience"]:
        article["target_audience"] = ["Entreprises", "Particuliers"]

    # Summary fallback
    if not article.get("summary"):
        french = article.get("content_french","")
        article["summary"]        = french[:300] + "..." if len(french) > 300 else french
        article["summary_french"] = article["summary"]

    return article


# ─────────────────────────────────────────────
# DEDUP + SORT
# ─────────────────────────────────────────────

def deduplicate(articles: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for a in articles:
        key = (a.get("article_number",""), a.get("law_number",""), a.get("law_type",""))
        if key not in seen:
            seen.add(key)
            unique.append(a)
    removed = len(articles) - len(unique)
    if removed:
        print(f"🔁 Removed {removed} duplicates")
    return unique


def sort_articles(articles: list[dict]) -> list[dict]:
    return sorted(articles, key=lambda a: a.get("article_order", 0))


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def _load_existing_results(output_path: str) -> tuple[list[dict], set]:
    """
    Feature 1 — Resume support.
    Load already-extracted articles from an existing output file.
    Returns (list_of_articles, set_of_done_keys).
    A done_key is (law_number, article_number, content_fingerprint).
    """
    if not Path(output_path).exists():
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        done_keys = set()
        for a in existing:
            num         = str(a.get("article_number", ""))
            law_num     = str(a.get("law_number", ""))
            fingerprint = str(a.get("content_french", ""))[:50]
            done_keys.add((law_num, num, fingerprint))
        print(f"   ♻️  Loaded {len(existing)} already-extracted articles from {output_path}")
        return existing, done_keys
    except Exception as e:
        print(f"   ⚠️  Could not load existing output ({e}) — starting fresh")
        return [], set()


def _save_incremental(output_path: str, articles: list[dict]) -> None:
    """Write current results to disk after every article — crash-safe."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def run_extraction(pdf_path: str, output_path: Optional[str] = None,
                   enrich_limit: Optional[int] = None,
                   dry_run: bool = False,
                   test_mode: bool = False) -> list[dict]:
    """
    Full 2-phase extraction pipeline.

    Features:
      - Resume: skips articles already present in output_path (Feature 1)
      - Model rotation: switches model on rate/token limit (Feature 2)
      - Key rotation: switches API key when all models exhausted (Feature 3)
      - Incremental save: writes output after every article — crash-safe

    Args:
        pdf_path:     Path to PDF file
        output_path:  Output JSON path (default: same dir as PDF)
        enrich_limit: Only enrich first N articles — useful for testing
    """
    print("\n" + "="*60)
    print("🇹🇳 LegaTech Tunisia Extractor v3")
    print("="*60)

    if not CF_AUTH_TOKEN:
        raise RuntimeError(
            "No Cloudflare auth token found.\n"
            "  Windows: $env:CF_AUTH_TOKEN='your_token'\n"
            "  Linux/Mac: export CF_AUTH_TOKEN='your_token'"
        )
    if not CF_ACCOUNT_ID:
        raise RuntimeError(
            "CF_ACCOUNT_ID not set. Add it to your .env or environment:\n"
            "  CF_ACCOUNT_ID=your_account_id_here"
        )
    print(f"🔑 Auth token loaded | model: {CF_MODEL}")
    print(f"🌐 Endpoint: https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1")
    if dry_run:
        print("\n🔬 DRY-RUN MODE — no API calls will be made (free)")
    if test_mode:
        print("\n🧪 TEST MODE — only 1 article will be enriched (minimal cost)")

    # Resolve output path early so we can check for existing results
    if output_path is None:
        stem        = Path(pdf_path).stem
        output_path = str(Path(pdf_path).parent / f"{stem}_extracted.json")

    if test_mode and not enrich_limit:
        enrich_limit = 1

    # Step 1: Read PDF
    print("\n📖 Step 1: Reading PDF...")
    pdf_text, pdf_pages = extract_text_from_pdf(pdf_path)

    # Step 2: Phase 1 — LLM extracts all legal units from page-grouped chunks
    print("\n📑 Step 2: Phase 1 — extracting all legal units...")
    if dry_run:
        print("   ⏭️  [dry-run] skipping Phase 1 LLM — no API call")
        doc_meta = {"law_type":"","law_number":"","law_title_french":"",
                    "law_title_arabic":"","institution":"","source_name":"JORT",
                    "source_date":"","effective_date":"","publication_date":"","year":0}
        raw_arts = [{"article_number": "1", "article_text": pdf_text[:500],
                     "unit_type": "ARTICLE", "unit_title": "", "chapter": "", "section": ""}]
    else:
        doc_meta, raw_arts = phase1_extract(None, pdf_pages)

    if not raw_arts:
        raise ValueError("No legal units found in PDF")

    print(f"\n📋 Metadata: law_type={doc_meta.get('law_type')} | "
          f"law_number={doc_meta.get('law_number')} | "
          f"source_date={doc_meta.get('source_date')}")

    # Step 3: Phase 2 — enrich each unit into full 64-field format
    enriched, done_keys = _load_existing_results(output_path)
    to_enrich = raw_arts[:enrich_limit] if enrich_limit else raw_arts
    total     = len(to_enrich)
    law_num   = doc_meta.get("law_number", "")
    success = skipped = 0

    print(f"\n🤖 Step 3: Phase 2 — enriching {total} unit(s) ({len(done_keys)} already done)...")

    for i, raw in enumerate(to_enrich, 1):
        unit_label  = raw.get("unit_title") or raw.get("unit_type") or "unit"
        art_num     = str(raw.get("article_number", ""))
        fingerprint = str(raw.get("article_text", ""))[:80]
        done_key    = (law_num, art_num, fingerprint)

        if done_key in done_keys:
            print(f"\n── Unit {i}/{total} — ⏭️  already done")
            skipped += 1
            continue

        print(f"\n── Unit {i}/{total} [{raw.get('unit_type','?')}] {unit_label[:60]} ──")

        if dry_run:
            print(f"   ⏭️  [dry-run] would enrich: {str(raw.get('article_text',''))[:120]}...")
            skipped += 1
            continue

        result = phase2_enrich(None, raw, doc_meta, i, total)
        result = sanitize_article(result)
        result = compute_static_fields(result)
        enriched.append(result)
        done_keys.add(done_key)
        success += 1

        try:
            _save_incremental(output_path, sort_articles(enriched))
        except Exception as e:
            print(f"   ⚠️  save error: {e}")

        if i < total:
            time.sleep(RATE_LIMIT_DELAY)

    # Step 4: Final sort and save
    print(f"\n📊 Total: {success} enriched | {skipped} skipped")
    print("\n⚙️  Step 4: Sorting and saving...")
    final = sort_articles(enriched)
    _save_incremental(output_path, final)
    print(f"✅ Final count: {len(final)} units")
    print(f"\n💾 Saved → {output_path}")
    print("="*60 + "\n")
    return final


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <pdf> [output.json] [--limit N] [--dry-run] [--test]")
        print("  --dry-run  Parse PDF, skip all API calls (free)")
        print("  --test     Enrich 1 article only (minimal cost)")
        sys.exit(1)

    pdf_file, output_file, limit = sys.argv[1], None, None
    dry_run = test_mode = False
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--limit" and i+1 < len(args):
            limit = int(args[i+1]); i += 2
        elif args[i] == "--dry-run":
            dry_run = True; i += 1
        elif args[i] == "--test":
            test_mode = True; i += 1
        elif not output_file and not args[i].startswith("--"):
            output_file = args[i]; i += 1
        else:
            i += 1

    results = run_extraction(pdf_file, output_file, enrich_limit=limit,
                             dry_run=dry_run, test_mode=test_mode)

    print(f"✅ Done — {len(results)} articles")
    if results:
        s = results[0]
        print(f"\nSample — Article {s.get('article_number')}:")
        print(f"  title_french:    {s.get('title_french','')[:70]}")
        print(f"  content_french:  {s.get('content_french','')[:150]}...")
        print(f"  summary:         {s.get('summary','')[:150]}")
        print(f"  keywords:        {s.get('keywords')}")
        print(f"  source_date:     {s.get('source_date')}")
        print(f"  effective_date:  {s.get('effective_date')}")
        print(f"  related_laws:    {s.get('related_laws')}")
        print(f"  entity_names:    {s.get('entity_names')}")