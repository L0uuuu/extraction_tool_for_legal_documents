"""
LegaTech Tunisia Legal Extractor — v3
======================================
Two-phase pipeline:
  Phase 1 : PDF text → LLM → document metadata + list of raw articles
  Phase 2 : Each raw article individually → LLM → all 62 enriched fields

Fixes applied vs v2:
  - Phase 1 now extracts document-level metadata (source_date, effective_date,
    publication_date, source_name, law_number, law_type, title) from the PDF
    header/preamble and passes it to every article in Phase 2
  - summary enforced 40-60 words in French
  - related_laws only from explicit citations in text — no hallucination
  - keywords always in French
  - entity_names only specific named entities, not generic roles
  - content_arabic contamination detection + re-request
  - source_date / effective_date / publication_date extracted in Phase 1
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
import pdfplumber
from groq import Groq

# ── optional .env support ──────────────────────
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# ── API keys (Feature 3: key rotation) ───────
# Add as many keys as you have. At minimum set GROQ_API_KEY.
# You can also set GROQ_API_KEY_2, GROQ_API_KEY_3, etc.
GROQ_API_KEYS: list[str] = [
    k for k in [
        os.getenv("GROQ_API_KEY",   ""),
        os.getenv("GROQ_API_KEY_2", ""),
    ] if k  # drop empty strings
]

# ── Model rotation list (Feature 2) ──────────
# Tried in order when a model hits its rate/token limit.
# Reset to index 0 whenever the API key rotates.
GROQ_MODELS: list[str] = [
    "moonshotai/kimi-k2-instruct-0905",   # primary  — best Arabic, 8192 out
    "openai/gpt-oss-120b",        # fallback 1
    "llama-3.3-70b-versatile",    # fallback 2 
    "openai/gpt-oss-20b",         # fallback 3
    "qwen/qwen3-32b",             # fallback 3
]

# ── Rotation state (mutable at runtime) ──────
_current_key_idx   = 0
_current_model_idx = 0

PDF_PARSE_CHUNK_CHARS = 8000
PDF_PARSE_MAX_TOKENS  = 8192
ENRICH_MAX_TOKENS     = 8192
RATE_LIMIT_DELAY      = 3.0
RETRY_ATTEMPTS        = 3
RETRY_BASE_DELAY      = 2
RATE_LIMIT_WAIT       = 62    # seconds to wait after exhausting all models on a key


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Tunisian legal analyst and professional bilingual translator (French ↔ Arabic).

ABSOLUTE RULES — never break these:
- Return ONLY valid JSON — no markdown, no code blocks, no explanations
- NEVER use null — use "" for missing strings, [] for missing arrays
- Keep all strings on one line — no literal newlines inside JSON strings
- Always close every bracket and brace
- Booleans: true or false (no quotes)
- Dates: ISO 8601 format YYYY-MM-DDT00:00:00Z
- status values: ACTIVE | AMENDED | REPEALED | SUSPENDED
- business_impact values: LOW | MEDIUM | HIGH
- article_type values: REGULATORY | PENAL | PROCEDURAL | DEFINITIONAL | TRANSITIONAL | ABROGATION
- ambiguity_level values: LOW | MEDIUM | HIGH
- relation_types values: REFERENCES | AMENDS | REPEALS | IMPLEMENTS | SUPERSEDES
- entity_types values: ORGANIZATION | INSTITUTION | COURT | MINISTRY | PERSON
- source_name values: JORT | BCT | OTHER
- institution rules: Loi → "Assemblée des Représentants du Peuple" | Décret → "Présidence de la République" | Circulaire BCT → "Banque Centrale de Tunisie"
"""


# ─────────────────────────────────────────────
# PHASE 1 PROMPT — document metadata + raw articles
# ─────────────────────────────────────────────

PDF_PARSE_PROMPT = """Analyze this Tunisian legal document text and return a JSON object with two things:
1. Document-level metadata found in the header, preamble, or any part of the text
2. A complete list of every article found in the text

RETURN THIS EXACT STRUCTURE:
{{
  "doc_metadata": {{
    "law_type"        : "Loi or Décret or Arrêté etc — infer from text",
    "law_number"      : "number only e.g. 66-27 — no prefix like Loi n°",
    "law_title_french": "official French title of the law",
    "law_title_arabic": "official Arabic title — translate if needed",
    "institution"     : "issuing authority",
    "source_name"     : "JORT or BCT or OTHER",
    "source_date"     : "ISO 8601 date of publication e.g. 1966-04-28T00:00:00Z — search for dates like 'du 28 avril 1966', 'le 1 mai 1966' — empty string if not found",
    "effective_date"  : "ISO 8601 date the law enters into force — search for 'entrera en vigueur le', 'applicable à partir du' — empty string if not found",
    "publication_date": "ISO 8601 date published in official journal (JORT) — empty string if not found",
    "year"            : 0
  }},
  "articles": [
    {{
      "article_number": "1",
      "article_text"  : "complete verbatim text of the article — do NOT summarize",
      "chapter"       : "chapter title or empty string",
      "section"       : "section title or empty string"
    }}
  ]
}}

RULES:
- article_text must be the FULL verbatim text — never summarize or truncate
- Extract EVERY article you can find — look for: Article X, Art. X, المادة X
- For doc_metadata: scan the entire text including headers, preambles, signatures
- For dates: look for patterns like "du 28 avril 1966", "le 1er mai 1966", "J.O.R.T. n°35 du..."
- If a field is not found, use "" — NEVER invent values

DOCUMENT TEXT:
{pdf_text}

RETURN ONLY THE JSON OBJECT. NO MARKDOWN. NO EXPLANATIONS."""


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

  legal_domains  : Array of 1-3 French legal domains. NEVER empty.
                   Valid: Droit Civil | Droit Commercial | Droit Fiscal | Droit Administratif |
                   Droit du Travail | Droit Pénal | Droit de la Santé | Droit Agricole | Droit Bancaire

  business_impact  : LOW | MEDIUM | HIGH
  target_audience  : Array of affected parties in French. NEVER empty. E.g. ["Entreprises", "Salariés"]
  status           : ACTIVE | AMENDED | REPEALED | SUSPENDED

  article_type   : REGULATORY | PENAL | PROCEDURAL | DEFINITIONAL | TRANSITIONAL | ABROGATION
  article_order  : Integer version of article_number (e.g. 6). For "14 quater" use 14.

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
RETURN ONLY A VALID JSON OBJECT. NO MARKDOWN. NO EXPLANATIONS.
═══════════════════════════════════════════"""


# ─────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        print(f"📄 PDF loaded: {len(pdf.pages)} pages")
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(text)

    full_text = "\n\n".join(pages_text)
    print(f"📝 Extracted {len(full_text):,} characters")
    return full_text


# ─────────────────────────────────────────────
# CHUNKING FOR PHASE 1
# ─────────────────────────────────────────────

def split_for_parsing(text: str, max_chars: int = PDF_PARSE_CHUNK_CHARS) -> list[str]:
    """Split PDF text at paragraph boundaries for Phase 1."""
    if len(text) <= max_chars:
        return [text]

    chunks, current = [], ""
    paragraphs = re.split(r'\n{2,}', text)

    for para in paragraphs:
        if len(para) > max_chars:
            # Para too big — split by lines
            for line in para.split('\n'):
                if len(current) + len(line) + 1 > max_chars:
                    if current.strip():
                        chunks.append(current.strip())
                    current = line
                else:
                    current += ("\n" if current else "") + line
            continue

        if len(current) + len(para) + 2 > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = para
        else:
            current += ("\n\n" if current else "") + para

    if current.strip():
        chunks.append(current.strip())

    print(f"📦 Split into {len(chunks)} parsing chunks")
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

def extract_articles_with_regex(text: str) -> list[dict]:
    """Fallback when Phase 1 LLM fails on a chunk."""
    pattern = re.compile(
        r'(?:Article|ARTICLE|Art\.?|المادة)\s+(\d+(?:\s*(?:bis|ter|quater|quinquies|-\d+))?)',
        re.IGNORECASE
    )
    matches = list(pattern.finditer(text))

    if not matches:
        # Try numbered paragraphs
        para_pattern = re.compile(r'(?:^|\n)\s*(\d+)[\.)\-]\s+(.+)', re.MULTILINE)
        para_matches = list(para_pattern.finditer(text))
        if para_matches:
            articles = []
            for i, m in enumerate(para_matches):
                end = para_matches[i+1].start() if i+1 < len(para_matches) else len(text)
                articles.append({
                    "article_number": m.group(1),
                    "article_text":   text[m.start():end].strip(),
                    "chapter": "", "section": ""
                })
            print(f"📄 Regex (paragraphs): {len(articles)} items")
            return articles
        return [{"article_number": "1", "article_text": text, "chapter": "", "section": ""}]

    articles = []
    for i, match in enumerate(matches):
        start = match.start()
        end   = matches[i+1].start() if i+1 < len(matches) else len(text)
        articles.append({
            "article_number": match.group(1).strip(),
            "article_text":   text[start:end].strip(),
            "chapter": "", "section": ""
        })

    print(f"📄 Regex fallback: {len(articles)} articles")
    return articles


# ─────────────────────────────────────────────
# GROQ CLIENT FACTORY + ROTATION HELPERS
# ─────────────────────────────────────────────

def _current_key() -> str:
    return GROQ_API_KEYS[_current_key_idx]

def _current_model() -> str:
    return GROQ_MODELS[_current_model_idx]

def _make_client() -> Groq:
    return Groq(api_key=_current_key())

def _rotate_model(label: str) -> bool:
    """
    Try the next model in the list.
    Returns True if a new model is available, False if all models exhausted.
    """
    global _current_model_idx
    if _current_model_idx + 1 < len(GROQ_MODELS):
        _current_model_idx += 1
        print(f"   🔄 [{label}] switching model → {_current_model()}")
        return True
    return False

def _rotate_key(label: str) -> bool:
    """
    Try the next API key in the list and reset model index.
    Returns True if a new key is available, False if all keys exhausted.
    """
    global _current_key_idx, _current_model_idx
    if _current_key_idx + 1 < len(GROQ_API_KEYS):
        _current_key_idx   += 1
        _current_model_idx  = 0  # reset to best model on fresh key
        print(f"   🔑 [{label}] switching API key → key #{_current_key_idx + 1} | model reset to {_current_model()}")
        return True
    return False

def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg

def _is_token_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("json_validate_failed" in msg and "max completion tokens" in msg) or            "max_tokens" in msg


# ─────────────────────────────────────────────
# GROQ API WRAPPER  (with model + key rotation)
# ─────────────────────────────────────────────

def call_groq(client, system: str, user: str,
              max_tokens: int, label: str = "") -> Optional[str]:
    """
    Call the Groq API with automatic model and key rotation on failure.

    On 429 (rate limit):
      1. Try next model on same key
      2. If all models exhausted → try next API key (resets model list)
      3. If all keys exhausted → wait RATE_LIMIT_WAIT seconds and retry from top
    On token limit error:
      → Try next model (smaller models sometimes handle output differently)
    On other errors:
      → Standard exponential backoff up to RETRY_ATTEMPTS times
    """
    global _current_key_idx, _current_model_idx

    # We allow unlimited outer loops through the rotation; inner attempts per model = RETRY_ATTEMPTS
    max_outer = (len(GROQ_MODELS) * len(GROQ_API_KEYS)) + 1

    for outer in range(max_outer):
        # Rebuild client if key may have changed since caller created it
        active_client = _make_client()
        active_model  = _current_model()

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                response = active_client.chat.completions.create(
                    model=active_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user}
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )
                resp_content  = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                usage         = response.usage

                if finish_reason == "length":
                    print(f"   ⚠️  [{label}] token limit hit — will attempt partial parse")

                print(f"   📊 [{label}] {usage.prompt_tokens} in / {usage.completion_tokens} out "
                      f"| finish: {finish_reason} | model: {active_model}")
                return resp_content

            except Exception as e:
                print(f"   ❌ [{label}] attempt {attempt}/{RETRY_ATTEMPTS}: {e}")

                if _is_rate_limit_error(e):
                    # Don't burn retries on rate limits — rotate immediately
                    print(f"   ⚡ [{label}] rate limit detected")
                    if _rotate_model(label):
                        active_model  = _current_model()
                        active_client = _make_client()
                        time.sleep(2)
                        break  # break inner loop, restart outer with new model
                    elif _rotate_key(label):
                        active_client = _make_client()
                        active_model  = _current_model()
                        time.sleep(2)
                        break
                    else:
                        # All models on all keys exhausted — wait and reset
                        print(f"   😴 [{label}] all keys/models rate-limited — waiting {RATE_LIMIT_WAIT}s...")
                        time.sleep(RATE_LIMIT_WAIT)
                        _current_key_idx   = 0
                        _current_model_idx = 0
                        active_client = _make_client()
                        active_model  = _current_model()
                        break

                elif _is_token_limit_error(e):
                    print(f"   📏 [{label}] token limit error — trying next model")
                    if _rotate_model(label):
                        active_model  = _current_model()
                        active_client = _make_client()
                        break
                    # else fall through to normal retry

                else:
                    # Generic error — exponential backoff
                    if attempt < RETRY_ATTEMPTS:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        print(f"   ⏳ retry in {delay}s...")
                        time.sleep(delay)

        else:
            # inner loop exhausted without success or rotation
            print(f"   💥 [{label}] failed after {RETRY_ATTEMPTS} attempts on model {active_model}")
            return None

    print(f"   💥 [{label}] exhausted all rotation options")
    return None


# ─────────────────────────────────────────────
# ARABIC CONTAMINATION CHECK
# ─────────────────────────────────────────────

def has_arabic_contamination(text: str) -> bool:
    """
    Returns True if content_arabic contains non-Arabic sequences
    (Latin, Cyrillic, etc.) — indicates a bad translation.
    Allows short sequences like numbers and punctuation.
    """
    if not text:
        return False
    # Flag if 4+ consecutive Latin or Cyrillic characters appear
    return bool(re.search(r'[a-zA-Z\u0400-\u04FF]{4,}', text))


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


def phase1_extract(client, pdf_text: str) -> tuple[dict, list[dict]]:
    """
    Phase 1: send PDF chunks to LLM, collect:
      - doc_metadata (merged across chunks, first non-empty value wins)
      - raw articles list

    DEDUPLICATION STRATEGY:
      A Tunisian legal document often has TWO sets of articles with the same
      numbers. For example, the Code du Travail has:
        - Articles 1-4 in the promulgation law (pages 1-7)
        - Articles 1-N in the code body itself (pages 9+)
      
      Old approach (dedup by article_number alone) dropped the second set.
      New approach: dedup by (article_number, first_50_chars_of_text).
      Same number + same text = true duplicate (overlap between chunks).
      Same number + different text = different article, keep both.

    Returns: (doc_metadata dict, list of raw article dicts)
    """
    print("\n── Phase 1: Identifying articles and document metadata ──")

    chunks    = split_for_parsing(pdf_text)
    doc_meta  = dict(EMPTY_DOC_METADATA)
    all_raw   = []
    # Key = (article_number, first_50_chars_of_text) — catches real duplicates
    # (same chunk overlap) while allowing same-numbered articles with different content
    seen_keys = set()

    print(f"📦 Split into {len(chunks)} parsing chunks")

    for i, chunk in enumerate(chunks, 1):
        label  = f"parse {i}/{len(chunks)}"
        print(f"\n   [{label}] {len(chunk):,} chars...")

        prompt = PDF_PARSE_PROMPT.replace("{pdf_text}", chunk)
        raw    = call_groq(None, SYSTEM_PROMPT, prompt, PDF_PARSE_MAX_TOKENS, label)
        parsed = parse_json_safely(raw, label) if raw else None

        if parsed:
            # Merge doc_metadata — first non-empty value wins
            chunk_meta = parsed.get("doc_metadata", {})
            if isinstance(chunk_meta, dict):
                for key, val in chunk_meta.items():
                    if key in doc_meta and not doc_meta[key] and val:
                        doc_meta[key] = val

            # Collect articles — dedup by (number, content_fingerprint)
            articles = parsed.get("articles", [])
            if isinstance(articles, list):
                added = 0
                for art in articles:
                    num         = str(art.get("article_number", "")).strip()
                    text        = str(art.get("article_text", "")).strip()
                    fingerprint = text[:50]  # first 50 chars = enough to distinguish different articles
                    key         = (num, fingerprint)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    all_raw.append(art)
                    added += 1
                print(f"   ✅ {added} new articles | meta so far: {doc_meta.get('law_number','?')} | total: {len(all_raw)}")
        else:
            # LLM failed — regex fallback
            print(f"   ⚠️  LLM failed — regex fallback for articles")
            regex_arts = extract_articles_with_regex(chunk)
            for art in regex_arts:
                num         = str(art.get("article_number", "")).strip()
                text        = str(art.get("article_text", "")).strip()
                fingerprint = text[:50]
                key         = (num, fingerprint)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_raw.append(art)

        if i < len(chunks):
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\n✅ Phase 1 done: {len(all_raw)} articles | metadata: {doc_meta}")
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

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        raw    = call_groq(None, SYSTEM_PROMPT, prompt, ENRICH_MAX_TOKENS,
                           f"{label} attempt {attempt}")
        parsed = parse_json_safely(raw, label) if raw else None

        if not parsed:
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY * attempt)
            continue

        # Check Arabic contamination
        arabic = parsed.get("content_arabic", "")
        if has_arabic_contamination(arabic):
            print(f"   ⚠️  [{label}] content_arabic contaminated — retrying...")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY * attempt)
            continue

        return parsed

    return None


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

    for f in ("repeal_date","superseded_by_id","supersedes_id","source_url","source_number"):
        article[f] = ""

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
    string_fields = [
        "id","jurisdiction","law_type","law_number","institution",
        "institution_primary","institution_secondary","title_french","title_arabic",
        "content_french","content_arabic","summary","summary_french","summary_arabic",
        "search_content","business_impact","status","article_type","ambiguity_level",
        "community_id","community_label","community_summary","source_name","source_date",
        "effective_date","publication_date","chapter_normalized","parent_document_id",
        "preceding_article_id","following_article_id","content_hash_sha256",
        "content_combined","repeal_date","superseded_by_id","supersedes_id",
        "source_url","source_number","last_checked","next_check","article_number",
        "chapter","section"
    ]
    array_fields = [
        "institutions","keywords","legal_domains","target_audience","related_laws",
        "relation_target_ids","relation_types","entity_names","entity_types","entity_ids"
    ]
    int_fields  = ["year","version","graph_level","article_order"]
    bool_fields = ["has_obligations","has_penalties","has_deadlines",
                   "has_exceptions","is_abrogation","is_transitional"]

    for k in string_fields:
        if article.get(k) is None or article.get(k) == "COMPUTED":
            article[k] = ""

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
                   enrich_limit: Optional[int] = None) -> list[dict]:
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

    if not GROQ_API_KEYS:
        raise RuntimeError(
            "No GROQ API key found. Set at least one:\n"
            "  Windows: $env:GROQ_API_KEY='gsk_...'\n"
            "  Linux/Mac: export GROQ_API_KEY='gsk_...'"
        )
    print(f"🔑 {len(GROQ_API_KEYS)} API key(s) loaded | {len(GROQ_MODELS)} models available")

    # Resolve output path early so we can check for existing results
    if output_path is None:
        stem        = Path(pdf_path).stem
        output_path = str(Path(pdf_path).parent / f"{stem}_extracted.json")

    # Step 1: Read PDF
    print("\n📖 Step 1: Reading PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)

    # Step 2: Phase 1
    print("\n📑 Step 2: Phase 1 — extracting document metadata + article list...")
    doc_meta, raw_arts = phase1_extract(None, pdf_text)  # client not needed — built inside call_groq

    if not raw_arts:
        raise ValueError("No articles found in PDF")

    print(f"\n📋 Document metadata extracted:")
    print(f"   law_type:         {doc_meta.get('law_type')}")
    print(f"   law_number:       {doc_meta.get('law_number')}")
    print(f"   source_date:      {doc_meta.get('source_date')}")
    print(f"   effective_date:   {doc_meta.get('effective_date')}")
    print(f"   publication_date: {doc_meta.get('publication_date')}")
    print(f"   source_name:      {doc_meta.get('source_name')}")

    # Step 3: Phase 2 — with resume support
    to_enrich = raw_arts[:enrich_limit] if enrich_limit else raw_arts
    total     = len(to_enrich)

    # Feature 1: load already-done articles
    existing_results, done_keys = _load_existing_results(output_path)
    enriched = list(existing_results)  # start with what we already have

    skipped, success, failed = 0, 0, 0
    law_num = doc_meta.get("law_number", "")

    print(f"\n🤖 Step 3: Phase 2 — enriching {total} articles ({len(done_keys)} already done)...")

    for i, raw in enumerate(to_enrich, 1):
        art_num     = str(raw.get("article_number", ""))
        fingerprint = str(raw.get("article_text", ""))[:50]
        done_key    = (law_num, art_num, fingerprint)

        # Feature 1: skip if already extracted
        if done_key in done_keys:
            print(f"\n── Article {i}/{total} (#{art_num}) — ⏭️  already extracted, skipping")
            skipped += 1
            continue

        print(f"\n── Article {i}/{total} (#{art_num}) ──")

        result = phase2_enrich(None, raw, doc_meta, i, total)

        if result:
            # Post-process immediately before saving
            try:
                result = sanitize_article(result)
                result = compute_static_fields(result)
            except Exception as e:
                print(f"   ⚠️  post-process error: {e}")
            enriched.append(result)
            done_keys.add(done_key)
            success += 1
        else:
            print(f"   ⚠️  failed — using raw fallback")
            fallback = {
                "article_number":   raw.get("article_number", ""),
                "content_french":   raw.get("article_text", ""),
                "chapter":          raw.get("chapter", ""),
                "section":          raw.get("section", ""),
                "title_french":     doc_meta.get("law_title_french", ""),
                "title_arabic":     doc_meta.get("law_title_arabic", ""),
                "law_number":       doc_meta.get("law_number", ""),
                "law_type":         doc_meta.get("law_type", ""),
                "source_date":      doc_meta.get("source_date", ""),
                "effective_date":   doc_meta.get("effective_date", ""),
                "publication_date": doc_meta.get("publication_date", ""),
                "source_name":      doc_meta.get("source_name", "JORT"),
                "year":             doc_meta.get("year", 0),
            }
            try:
                fallback = sanitize_article(fallback)
                fallback = compute_static_fields(fallback)
            except Exception:
                pass
            enriched.append(fallback)
            done_keys.add(done_key)
            failed += 1

        # Incremental save after every article — crash-safe
        _save_incremental(output_path, sort_articles(enriched))

        if i < total:
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\n📊 Phase 2: {success} enriched | {skipped} skipped | {failed} fallbacks")

    # Step 4: Final sort and save
    print("\n⚙️  Step 4: Sorting and saving...")
    final = sort_articles(enriched)
    _save_incremental(output_path, final)
    print(f"✅ Final count: {len(final)} articles")
    print(f"\n💾 Saved → {output_path}")
    print("="*60 + "\n")
    return final


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <pdf> [output.json] [--limit N]")
        sys.exit(1)

    pdf_file, output_file, limit = sys.argv[1], None, None
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--limit" and i+1 < len(args):
            limit = int(args[i+1]); i += 2
        elif not output_file and not args[i].startswith("--"):
            output_file = args[i]; i += 1
        else:
            i += 1

    results = run_extraction(pdf_file, output_file, enrich_limit=limit)

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