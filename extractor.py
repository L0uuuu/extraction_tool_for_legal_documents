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

GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL            = "openai/gpt-oss-120b"

PDF_PARSE_CHUNK_CHARS = 8000   # Phase 1 chunk size
PDF_PARSE_MAX_TOKENS  = 8192  # raised — Phase 1 copies article texts, needs headroom
ENRICH_MAX_TOKENS     = 8192  # raised — content_french + Arabic translation needs space
RATE_LIMIT_DELAY      = 3.0    # increased — avoid 429 rate limit errors
RETRY_ATTEMPTS        = 3
RETRY_BASE_DELAY      = 2


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
# GROQ API WRAPPER
# ─────────────────────────────────────────────

def call_groq(client: Groq, system: str, user: str,
              max_tokens: int, label: str = "") -> Optional[str]:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            content       = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            usage         = response.usage

            if finish_reason == "length":
                print(f"   ⚠️  [{label}] token limit hit — will attempt partial parse")

            print(f"   📊 [{label}] {usage.prompt_tokens} in / {usage.completion_tokens} out "
                  f"| finish: {finish_reason}")
            return content

        except Exception as e:
            print(f"   ❌ [{label}] attempt {attempt}/{RETRY_ATTEMPTS}: {e}")
            if attempt < RETRY_ATTEMPTS:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"   ⏳ retry in {delay}s...")
                time.sleep(delay)

    print(f"   💥 [{label}] failed after {RETRY_ATTEMPTS} attempts")
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

def phase1_extract(client: Groq, pdf_text: str) -> tuple[dict, list[dict]]:
    """
    Phase 1: send PDF chunks to LLM, collect:
      - doc_metadata (merged across chunks, first non-empty value wins)
      - raw articles list (deduplicated by article_number)

    Returns: (doc_metadata dict, list of raw article dicts)
    """
    print("\n── Phase 1: Identifying articles and document metadata ──")

    chunks     = split_for_parsing(pdf_text)
    doc_meta   = dict(EMPTY_DOC_METADATA)  # will be filled from first chunk that has data
    all_raw    = []
    seen_nums  = set()

    for i, chunk in enumerate(chunks, 1):
        label  = f"parse {i}/{len(chunks)}"
        print(f"\n   [{label}] {len(chunk):,} chars...")

        prompt = PDF_PARSE_PROMPT.format(pdf_text=chunk)
        raw    = call_groq(client, SYSTEM_PROMPT, prompt, PDF_PARSE_MAX_TOKENS, label)
        parsed = parse_json_safely(raw, label) if raw else None

        if parsed:
            # Merge doc_metadata — first non-empty value wins for each field
            chunk_meta = parsed.get("doc_metadata", {})
            if isinstance(chunk_meta, dict):
                for key, val in chunk_meta.items():
                    if key in doc_meta and not doc_meta[key] and val:
                        doc_meta[key] = val

            # Collect articles
            articles = parsed.get("articles", [])
            if isinstance(articles, list):
                added = 0
                for art in articles:
                    num = str(art.get("article_number", "")).strip()
                    if num in seen_nums:
                        continue
                    if num:
                        seen_nums.add(num)
                    all_raw.append(art)
                    added += 1
                print(f"   ✅ {added} new articles | meta so far: {doc_meta.get('law_number','?')}")
        else:
            # LLM failed — regex fallback for articles, metadata stays as-is
            print(f"   ⚠️  LLM failed — regex fallback for articles")
            regex_arts = extract_articles_with_regex(chunk)
            for art in regex_arts:
                num = str(art.get("article_number", "")).strip()
                if num in seen_nums:
                    continue
                if num:
                    seen_nums.add(num)
                all_raw.append(art)

        if i < len(chunks):
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\n✅ Phase 1 done: {len(all_raw)} articles | metadata: {doc_meta}")
    return doc_meta, all_raw


# ─────────────────────────────────────────────
# PHASE 2 — enrich one article
# ─────────────────────────────────────────────

def phase2_enrich(client: Groq, raw_article: dict, doc_meta: dict,
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
        raw    = call_groq(client, SYSTEM_PROMPT, prompt, ENRICH_MAX_TOKENS,
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

def run_extraction(pdf_path: str, output_path: Optional[str] = None,
                   enrich_limit: Optional[int] = None) -> list[dict]:
    """
    Full 2-phase extraction pipeline.

    Args:
        pdf_path:     Path to PDF file
        output_path:  Output JSON path (default: same dir as PDF)
        enrich_limit: Only enrich first N articles — useful for testing

    Returns:
        List of enriched article dicts (all 62 fields)
    """
    print("\n" + "="*60)
    print("🇹🇳 LegaTech Tunisia Extractor v3")
    print("="*60)

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set. Export it as an environment variable:\n"
            "  Windows: $env:GROQ_API_KEY='gsk_...'\n"
            "  Linux/Mac: export GROQ_API_KEY='gsk_...'"
        )

    # Step 1: Read PDF
    print("\n📖 Step 1: Reading PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)

    # Step 2: Phase 1
    print("\n📑 Step 2: Phase 1 — extracting document metadata + article list...")
    client              = Groq(api_key=GROQ_API_KEY)
    doc_meta, raw_arts  = phase1_extract(client, pdf_text)

    if not raw_arts:
        raise ValueError("No articles found in PDF")

    print(f"\n📋 Document metadata extracted:")
    print(f"   law_type:         {doc_meta.get('law_type')}")
    print(f"   law_number:       {doc_meta.get('law_number')}")
    print(f"   source_date:      {doc_meta.get('source_date')}")
    print(f"   effective_date:   {doc_meta.get('effective_date')}")
    print(f"   publication_date: {doc_meta.get('publication_date')}")
    print(f"   source_name:      {doc_meta.get('source_name')}")

    # Step 3: Phase 2
    to_enrich = raw_arts[:enrich_limit] if enrich_limit else raw_arts
    total     = len(to_enrich)
    print(f"\n🤖 Step 3: Phase 2 — enriching {total} articles...")

    enriched, success, failed = [], 0, 0

    for i, raw in enumerate(to_enrich, 1):
        print(f"\n── Article {i}/{total} (#{raw.get('article_number','?')}) ──")

        result = phase2_enrich(client, raw, doc_meta, i, total)

        if result:
            enriched.append(result)
            success += 1
        else:
            print(f"   ⚠️  failed — using raw fallback")
            enriched.append({
                "article_number": raw.get("article_number",""),
                "content_french": raw.get("article_text",""),
                "chapter":        raw.get("chapter",""),
                "section":        raw.get("section",""),
                "title_french":   doc_meta.get("law_title_french",""),
                "title_arabic":   doc_meta.get("law_title_arabic",""),
                "law_number":     doc_meta.get("law_number",""),
                "law_type":       doc_meta.get("law_type",""),
                "source_date":    doc_meta.get("source_date",""),
                "effective_date": doc_meta.get("effective_date",""),
                "publication_date": doc_meta.get("publication_date",""),
                "source_name":    doc_meta.get("source_name","JORT"),
                "year":           doc_meta.get("year", 0),
            })
            failed += 1

        if i < total:
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\n📊 Phase 2: {success} enriched, {failed} fallbacks")

    # Step 4: Post-process
    print("\n⚙️  Step 4: Post-processing...")
    processed = []
    for article in enriched:
        try:
            article = sanitize_article(article)
            article = compute_static_fields(article)
            processed.append(article)
        except Exception as e:
            print(f"   ⚠️  post-process error: {e}")

    # processed = deduplicate(processed)
    processed = sort_articles(processed)
    print(f"✅ Final count: {len(processed)} articles")

    # Step 5: Save
    if output_path is None:
        stem        = Path(pdf_path).stem
        output_path = str(Path(pdf_path).parent / f"{stem}_extracted.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Saved → {output_path}")
    print("="*60 + "\n")
    return processed


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