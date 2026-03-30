#!/usr/bin/env python3
"""
Batch Folder Processor — Legal PDF Extractor (Azure OpenAI)
Converts PDFs to structured JSON using Azure OpenAI
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://modelsetafakna.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY","")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

MAX_INPUT_TOKENS = 100000
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_OUTPUT_TOKENS", "32000"))
CHARS_PER_TOKEN = 4
MAX_CHUNK_CHARS = min(MAX_INPUT_TOKENS * CHARS_PER_TOKEN, 200000)
API_MAX_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = """You are an expert UAE legal document analyzer. Extract legal documents into structured JSON format following UAE legal taxonomy.

CRITICAL JSON RULES:
- Return ONLY valid JSON, no markdown, no code blocks
- Escape all special characters in strings (quotes, newlines, tabs)
- Use \\" for quotes inside strings
- Keep all text in single lines (no literal newlines in strings)
- NEVER truncate mid-string or mid-object
- If running out of space, complete current article and stop cleanly
- Always close all JSON brackets and braces properly"""

EXTRACTION_PROMPT = """Extract this UAE legal document into structured JSON format.

**STRICT REQUIREMENTS:**
1. Return ONLY the JSON object (no markdown, no ```json blocks)
2. Extract articles completely - NEVER truncate mid-article
3. If you sense you're running out of tokens, complete the current article and close JSON properly
4. Better to extract fewer complete articles than many incomplete ones
5. Escape all special characters properly

**JSON STRUCTURE:**
{
  "doc_id": "string",
  "jurisdiction": "FEDERAL or EMIRATE_DUBAI or FREE_ZONE",
  "law_type": "Federal Law or Emiri Decree or Cabinet Decision",
  "law_number": "string",
  "title_english": "string",
  "title_arabic": "string",
  "year": 2024,
  "status": "ACTIVE or AMENDED or REPEALED",
  "articles": [
    {
      "article_id": "string",
      "article_number": "Article X",
      "content_english": "concise article summary - max 2-3 sentences",
      "content_arabic": "arabic summary",
      "keywords": ["keyword1", "keyword2"],
      "legal_domains": ["Commercial", "Labour"],
      "business_impact": "HIGH or MEDIUM or LOW",
      "target_audience": ["Companies"],
      "sharia_reference": true,
      "free_zone_impact": "DIFC or ADGM or N/A",
      "search_content": "concise search-optimized text",
      "jurisdiction": "FEDERAL",
      "law_type": "Federal Law",
      "law_number": "string",
      "title_english": "string",
      "title_arabic": "string",
      "year": 2024,
      "status": "ACTIVE"
    }
  ]
}

**DOCUMENT TEXT:**
{document_text}

**CRITICAL:** Return ONLY valid JSON. Complete all articles fully or stop at a complete article boundary."""

TUNISIA_SYSTEM_PROMPT = """You are an expert Tunisian legal document analyzer and professional translator fluent in French and Arabic. Extract legal articles into a rich structured JSON format covering ALL metadata fields needed for vector embedding, graph databases, AI training, and full-text search.

CRITICAL JSON RULES:
- Return ONLY a single valid JSON object (no markdown, no code blocks, no array wrapper)
- Escape all special characters in strings (quotes, newlines, tabs)
- Use \\" for quotes inside strings
- Keep all text in single lines (no literal newlines in strings)
- NEVER truncate mid-string or mid-object — complete cleanly or stop at a field boundary
- Always close all JSON brackets and braces properly
- NEVER use null for string fields — use "" instead
- NEVER use null for array fields — use [] instead
- ALWAYS extract keywords (minimum 5-7 per article) — NEVER empty
- ALWAYS populate legal_domains with French names — NEVER empty
- ALWAYS translate French content to Arabic (title_arabic, content_arabic, summary_arabic)
- ALWAYS generate embedding_text combining title + summary (FR+AR) + content (FR+AR) + legal_concepts
- Use status values: "ACTIVE", "ABROGATED", "MODIFIED", "PENDING"
- Use business_impact values: "HIGH", "MEDIUM", "LOW"
- Use article_type values: "REGULATORY", "DEFINITIONAL", "PROCEDURAL", "TRANSITIONAL", "PENAL", "ABROGATION"
- Use ambiguity_level values: "LOW", "MEDIUM", "HIGH"
- Boolean fields (has_obligations, has_penalties, has_deadlines, has_exceptions, is_abrogation, is_transitional): always true or false
- All fields listed in the JSON TEMPLATE below MUST be present in the output"""

TUNISIA_PDF_PARSING_PROMPT = """Extract individual articles from this Tunisian legal document PDF text. Identify each article and return them as a JSON object with an "articles" array.

**STRICT REQUIREMENTS:**
1. Return ONLY a JSON object (no markdown, no ```json blocks)
2. The JSON must have an "articles" key containing an array
3. Each article should be a complete object
4. Identify article boundaries (look for "Article", "Art.", "المادة", etc.)
5. Extract article number, content, and any available metadata

**ARTICLE BOUNDARY RULES:**
1. An article STARTS when you see: "Article", "Art.", "المادة", or "الفصل" followed by a number
2. An article ENDS when you encounter:
   - The next article marker (e.g., "Article 2", "Art. 3", "المادة 4", "الفصل 5")
   - A signature line (e.g., "Le Ministre", "Le Président")
   - A date line (e.g., "Fait à Tunis, le...")
   - The end of the document
3. If you see a range like "Articles 1-4" or "Art. 1 à 4", treat EACH number as a separate article
4. Do NOT combine multiple articles into one even if they share a heading

**JSON STRUCTURE:**
{
  "articles": [
    {
      "article": "full article text in French",
      "article_number": "Article 1",
      "titre": "law title if available",
      "loi": "law number if available",
      "chapitre": "chapter if available"
    },
    ...
  ]
}

**PDF TEXT:**
{pdf_text}

**CRITICAL:** Return ONLY a valid JSON object with an "articles" array. Extract all articles you can identify."""

TUNISIA_EXTRACTION_PROMPT = """EXTRACT TUNISIAN LEGAL ARTICLE INTO COMPLETE JSON — ALL FIELDS REQUIRED

INSTRUCTIONS:
Analyze the article data below and produce a single JSON object containing EVERY field in the template.
Do NOT omit any field. Use "" for missing strings and [] for missing arrays.

═══════════════════════════════════════════════════════
JSON TEMPLATE  (fill every field — never skip any)
═══════════════════════════════════════════════════════
{
  "jurisdiction": "TUNISIA",
  "institution": "",
  "law_type": "",
  "law_number": "",
  "year": "",
  "status": "",

  "title_french": "",
  "title_arabic": "",

  "chapter": "",
  "chapter_normalized": "",
  "section": "",

  "article_number": "",
  "article_order": 0,
  "article_type": "",

  "content_french": "",
  "content_arabic": "",
  "content_combined": "",

  "summary": "",
  "summary_french": "",
  "summary_arabic": "",

  "search_content": "",
  "embedding_text": "",

  "keywords": [],
  "legal_domains": [],
  "legal_concepts": [],
  "business_impact": "",
  "target_audience": [],
  "related_laws": [],

  "ambiguity_level": "",
  "has_obligations": false,
  "has_penalties": false,
  "has_deadlines": false,
  "has_exceptions": false,
  "is_abrogation": false,
  "is_transitional": false,

  "institution_primary": "",
  "institution_secondary": "",
  "institutions": [],

  "source_name": "JORT",
  "source_number": "",
  "source_url": "",
  "source_date": "",
  "publication_date": "",
  "effective_date": "",

  "relation_target_ids": [],
  "relation_types": [],

  "entity_names": [],
  "entity_types": [],
  "entity_ids": [],

  "community_id": "",
  "community_label": "",
  "community_summary": "",

  "parent_document_id": "",
  "preceding_article_id": "",
  "following_article_id": "",
  "graph_level": 1,

  "version": 1,
  "repeal_date": "",
  "superseded_by_id": "",
  "supersedes_id": "",

  "last_checked": "",
  "next_check": ""
}

═══════════════════════════════════════════════════════
FIELD-BY-FIELD INSTRUCTIONS
═══════════════════════════════════════════════════════

IDENTITY & CLASSIFICATION
- jurisdiction          : Always "TUNISIA"
- institution           : Issuing body (e.g. "Présidence de la République", "Assemblée des Représentants du Peuple", "Ministère des Finances")
- law_type              : "Loi" | "Décret" | "Arrêté" | "Loi de Finances" | "Acte de Société" | "Décret-loi" etc.
- law_number            : Official number, e.g. "2023-97" or registration number like "QN°7892"
- year                  : 4-digit year extracted from dates in content, e.g. "2023"
- status                : "ACTIVE" | "ABROGATED" | "MODIFIED" | "PENDING"

TITLES (bilingual — BOTH required)
- title_french          : Full official French title of the law/decree
- title_arabic          : MUST translate title_french to Arabic using proper legal terminology

STRUCTURE
- chapter               : Chapter title/number if present, else ""
- chapter_normalized    : Normalized chapter label, e.g. "Chapitre 1 - Dispositions générales", else ""
- section               : Section title/number if present, else ""
- article_number        : e.g. "Article 1", "Article premier"
- article_order         : Integer position of article in the document (1, 2, 3 …)
- article_type          : "REGULATORY" | "DEFINITIONAL" | "PROCEDURAL" | "TRANSITIONAL" | "PENAL" | "ABROGATION"

CONTENT (bilingual — BOTH required)
- content_french        : Full French text of the article — REQUIRED
- content_arabic        : MUST translate content_french to Arabic — REQUIRED
- content_combined      : content_french + " " + content_arabic (concatenated)

SUMMARIES (all three required)
- summary               : 2-3 sentence French summary of the article's purpose
- summary_french        : Same as summary (duplicate for schema symmetry)
- summary_arabic        : MUST translate summary to Arabic

SEARCH & EMBEDDING
- search_content        : Key terms in French and Arabic (law number, title keywords, domain terms)
- embedding_text        : Full concatenation: title_french + "\\n" + summary_french + "\\n" + summary_arabic + "\\n" + content_french + "\\n" + content_arabic + "\\n" + legal_concepts joined by " | "

ENRICHMENT (all arrays NEVER empty)
- keywords              : 5-7 relevant legal/business keywords extracted from content — NEVER []
- legal_domains         : 1-3 French domain names: "Droit Commercial" | "Droit Fiscal" | "Droit Administratif" | "Droit du Travail" | "Droit Civil" | "Droit de l'urbanisme" | "Droit des collectivités locales" etc. — NEVER []
- legal_concepts        : 3-5 core legal concepts, e.g. ["gestion par objectifs", "coordination administrative"]
- business_impact       : "HIGH" | "MEDIUM" | "LOW"
- target_audience       : French labels, e.g. ["Entreprises", "Investisseurs", "Particuliers", "Collectivités locales"] — at least 1
- related_laws          : Full citations of referenced laws/articles, e.g. ["Loi n° 2018-29 du 9 mai 2018"]

LEGAL FLAGS (boolean — infer from content)
- ambiguity_level       : "LOW" | "MEDIUM" | "HIGH"
- has_obligations       : true if article imposes obligations
- has_penalties         : true if article mentions fines/sanctions/penalties
- has_deadlines         : true if article mentions deadlines/dates
- has_exceptions        : true if article contains exceptions/derogations
- is_abrogation         : true if article abrogates another law
- is_transitional       : true if article is a transitional/final provision

INSTITUTIONS
- institution_primary   : Main issuing institution
- institution_secondary : Secondary institution if any, else ""
- institutions          : Array of all institutions mentioned

SOURCE
- source_name           : "JORT" (Journal Officiel de la République Tunisienne)
- source_number         : JORT issue number if available, else ""
- source_url            : URL if available, else ""
- source_date           : ISO date of publication, e.g. "2023-02-06T00:00:00Z"
- publication_date      : Same as source_date
- effective_date        : Date the law takes effect (same as publication_date if not specified)

RELATIONS (graph edges)
- relation_target_ids   : IDs of related documents, e.g. ["tn-loi-2018-29"]
- relation_types        : Corresponding relation type per entry: "REFERENCES" | "AMENDS" | "REPEALS" | "IMPLEMENTS"
- entity_names          : Named entities (ministries, organizations) mentioned
- entity_types          : Type per entity: "MINISTRY" | "ORGANIZATION" | "PERSON" | "PLACE"
- entity_ids            : Slugified IDs, e.g. ["tn-org-ministere-de-l-equipement"]

GRAPH / COMMUNITY
- community_id          : Short slug grouping related laws, e.g. "urbanisme-collectivites"
- community_label       : Human label, e.g. "Urbanisme et Collectivités Locales"
- community_summary     : 1 sentence describing the community
- parent_document_id    : ID of the parent law/decree, e.g. "tn-decret-2023-97"
- preceding_article_id  : ID of the previous article, or ""
- following_article_id  : ID of the next article, or ""
- graph_level           : Depth in document hierarchy (1 = top-level article)

VERSIONING
- version               : Always 1 for new extraction
- repeal_date           : ISO date if repealed, else ""
- superseded_by_id      : ID of superseding document, else ""
- supersedes_id         : ID of document this supersedes, else ""
- last_checked          : Today's ISO date
- next_check            : ISO date 6 months from today

═══════════════════════════════════════════════════════
ARTICLE DATA TO PROCESS:
{article_data}
═══════════════════════════════════════════════════════

RETURN ONLY A SINGLE VALID JSON OBJECT. NO MARKDOWN. NO EXPLANATIONS. NO ARRAY WRAPPER."""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    """Extract text from PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text, len(text)


def call_azure_openai(system_prompt: str, user_prompt: str, max_tokens: int = API_MAX_OUTPUT_TOKENS) -> str:
    """Call Azure OpenAI API"""
    try:
        estimated_tokens = len(system_prompt + user_prompt) // CHARS_PER_TOKEN
        print(f"    📊 Estimated prompt tokens: ~{estimated_tokens} / {MAX_INPUT_TOKENS}")

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.95,
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        content = choice.message.content or ""

        if finish_reason == "length":
            print(f"    ⚠️  Token limit hit (finish_reason=length) — will attempt JSON repair")
        elif finish_reason == "content_filter":
            print(f"    ⚠️  Stopped by content filter")

        usage = response.usage
        print(f"    📊 Usage: prompt={usage.prompt_tokens} / completion={usage.completion_tokens} | finish={finish_reason}")

        return content

    except Exception as error:
        print(f"    ❌ Azure OpenAI error: {error}")
        if "429" in str(error):
            print("    ⚠️  Rate limit or quota exceeded. Increasing delay may help.")
        raise


def parse_json_safely(json_string: str, label: str) -> Dict:
    """Parse JSON safely with repair attempts"""
    try:
        cleaned = json_string.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"```json\n?", "", cleaned)
            cleaned = re.sub(r"```\n?", "", cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError as error:
        print(f"    ⚠️  [{label}] JSON parse error: {error}")

        try:
            fixed = json_string.strip()
            fixed = re.sub(r"```json\n?", "", fixed)
            fixed = re.sub(r"```\n?", "", fixed)

            last_brace = fixed.rfind("}")
            if last_brace == -1:
                raise ValueError("No closing brace found")

            # Try to close the JSON properly
            closed = False
            for i in range(last_brace, min(len(fixed), last_brace + 100)):
                if fixed[i] == "]":
                    fixed = fixed[:i + 1] + "}"
                    closed = True
                    break

            if not closed:
                fixed = fixed[:last_brace + 1] + "]}"

            parsed = json.loads(fixed)
            print(f"    ✅ [{label}] JSON repaired successfully")
            return parsed

        except Exception as repair_error:
            print(f"    ❌ [{label}] repair failed: {repair_error}")
            raise ValueError(f"JSON parsing failed after repair: {error}")


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split text into chunks at article boundaries"""
    article_pattern = r'(?=(?:Art\.?|ARTICLE|Article|المادة|الفصل)\s+(?:\d+|[٠-٩]+))'
    matches = list(re.finditer(article_pattern, text))

    if not matches:
        return split_by_paragraphs(text, max_chars)

    print(f"    📑 Found {len(matches)} article boundaries")
    chunks = []
    current_chunk = ""

    for i, match in enumerate(matches):
        article_start = match.start()
        article_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_text = text[article_start:article_end]

        if len(article_text) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.extend(split_by_paragraphs(article_text, max_chars))
            continue

        if len(current_chunk) + len(article_text) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = article_text
        else:
            current_chunk += article_text

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) > 50]


def split_by_paragraphs(text: str, max_chars: int) -> List[str]:
    """Split text by paragraphs"""
    chunks = []
    paragraphs = re.split(r'\n\s*\n', text)
    current_chunk = ""

    for para in paragraphs:
        if len(para) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            sentences = re.split(r'[.!?]\s+', para)
            for sent in sentences:
                if len(current_chunk) + len(sent) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent
                else:
                    if current_chunk:
                        current_chunk += ". " + sent
                    else:
                        current_chunk = sent

        elif len(current_chunk) + len(para) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def extract_chunk(chunk_text: str, chunk_number: int, total_chunks: int, retries: int = 3) -> Dict:
    """Extract legal data from a single chunk"""
    prompt = EXTRACTION_PROMPT.replace("{document_text}", chunk_text)

    for attempt in range(1, retries + 1):
        try:
            print(f"    🤖 Chunk {chunk_number}/{total_chunks} (attempt {attempt}/{retries})…")
            raw = call_azure_openai(SYSTEM_PROMPT, prompt)
            parsed = parse_json_safely(raw, f"chunk-{chunk_number}")

            if "articles" not in parsed or not isinstance(parsed["articles"], list):
                raise ValueError("Missing articles array")
            if len(parsed["articles"]) == 0:
                raise ValueError("No articles extracted")

            print(f"    ✅ Chunk {chunk_number}: {len(parsed['articles'])} articles")
            return parsed

        except Exception as error:
            print(f"    ❌ Chunk {chunk_number} attempt {attempt}: {error}")
            if attempt == retries:
                raise
            time.sleep(2 * (2 ** (attempt - 1)))

    raise RuntimeError(f"Failed to extract chunk {chunk_number}")


def merge_chunk_results(chunk_results: List[Dict]) -> Dict:
    """Merge results from multiple chunks"""
    if not chunk_results:
        return None
    if len(chunk_results) == 1:
        return chunk_results[0]

    merged = chunk_results[0].copy()
    merged["articles"] = []
    seen = set()

    for chunk in chunk_results:
        for article in chunk.get("articles", []):
            key = article.get("article_number") or article.get("article_id") or json.dumps(article, sort_keys=True)
            if key not in seen:
                seen.add(key)
                merged["articles"].append(article)

    # Sort articles by number
    def get_article_num(a):
        match = re.search(r"(\d+)", a.get("article_number", ""))
        return int(match.group(1)) if match else 0

    merged["articles"].sort(key=get_article_num)

    return merged


def extract_legal_data_with_chunking(document_text: str) -> Dict:
    """Extract legal data with automatic chunking"""
    print(f"    📄 Document size: {len(document_text)} chars (~{len(document_text) // CHARS_PER_TOKEN} tokens)")

    if len(document_text) <= MAX_CHUNK_CHARS:
        print("    📦 Single-chunk processing…")
        return extract_chunk(document_text, 1, 1)

    print("    ✂️  Splitting into chunks…")
    chunks = split_into_chunks(document_text)
    chunk_sizes = [f"{len(c)//1000}KB" for c in chunks]
    print(f"    📦 {len(chunks)} chunks: {', '.join(chunk_sizes)}")

    chunk_results = []
    fail_count = 0

    for i, chunk in enumerate(chunks):
        try:
            result = extract_chunk(chunk, i + 1, len(chunks))
            chunk_results.append(result)
        except Exception as error:
            fail_count += 1
            print(f"    ⚠️  Chunk {i + 1} permanently failed: {error}")
            if fail_count > len(chunks) * 0.5:
                raise RuntimeError(f"Too many chunk failures: {fail_count}/{len(chunks)}")

        if i < len(chunks) - 1:
            time.sleep(1)

    if not chunk_results:
        raise RuntimeError("All chunks failed")

    merged = merge_chunk_results(chunk_results)
    print(f"    ✅ Extraction complete: {len(merged.get('articles', []))} articles")
    return merged


# ============================================================================
# TUNISIA-SPECIFIC FUNCTIONS
# ============================================================================

def ensure_keywords_populated(extracted_article: Dict, original_article_data: Dict) -> Dict:
    """Ensure all required fields are populated"""
    content = extracted_article.get("content_french", original_article_data.get("article", ""))

    # Ensure keywords
    keywords = extracted_article.get("keywords", [])
    if len(keywords) < 3:
        # Extract important terms
        important_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        for term in important_terms:
            if len(term) > 3 and term not in ["Article", "Société", "Sociétés", "Constitution", "République", "Tunisienne", "Tunisie"]:
                if not any(k.lower() == term.lower() for k in keywords):
                    keywords.append(term)

        # Extract domain terms
        domain_terms = re.findall(r'\b(?:commerce|distribution|souscription|bénéfices|capital|société|entreprise|loi|décret|arrêté|assemblée|générale|administrateur|commissaire|comptes)\w*\b', content, re.I)
        for term in domain_terms:
            if not any(k.lower() == term.lower() for k in keywords):
                keywords.append(term)

        # Extract phrases
        phrases = re.findall(r'\b(?:siège social|assemblée générale|établissement secondaire|code des sociétés|états financiers|commissaire aux|conseil d\'administration|société anonyme)\b', content, re.I)
        for phrase in phrases:
            if not any(k.lower() == phrase.lower() for k in keywords):
                keywords.append(phrase)

        extracted_article["keywords"] = keywords[:7]

    if not extracted_article.get("keywords"):
        extracted_article["keywords"] = ["article", "juridique", "tunisie"]

    # Ensure legal_domains
    if not extracted_article.get("legal_domains"):
        content_lower = content.lower()
        domains = []
        if any(term in content_lower for term in ["commerce", "société", "entreprise"]):
            domains.append("Droit Commercial")
        if any(term in content_lower for term in ["travail", "employé", "salarié"]):
            domains.append("Droit du Travail")
        if any(term in content_lower for term in ["fiscal", "impôt", "taxe"]):
            domains.append("Droit Fiscal")
        if any(term in content_lower for term in ["administratif", "ministère", "décret"]):
            domains.append("Droit Administratif")
        if any(term in content_lower for term in ["civil", "contrat", "obligation"]):
            domains.append("Droit Civil")
        if any(term in content_lower for term in ["urbain", "aménagement", "habitat"]):
            domains.append("Droit de l'urbanisme")
        if "collectivit" in content_lower:
            domains.append("Droit des collectivités locales")

        extracted_article["legal_domains"] = domains if domains else ["Droit Commercial"]

    # Ensure legal_concepts
    if not extracted_article.get("legal_concepts"):
        extracted_article["legal_concepts"] = extracted_article["keywords"][:3]

    # Ensure content_combined
    if not extracted_article.get("content_combined"):
        extracted_article["content_combined"] = (
            (extracted_article.get("content_french") or "") + " " +
            (extracted_article.get("content_arabic") or "")
        ).strip()

    # Ensure summaries
    if not extracted_article.get("summary_french"):
        extracted_article["summary_french"] = extracted_article.get("summary", "")
    if not extracted_article.get("summary_arabic"):
        extracted_article["summary_arabic"] = ""

    # Ensure embedding_text
    if not extracted_article.get("embedding_text"):
        concepts = " | ".join(extracted_article.get("legal_concepts", []))
        parts = [
            extracted_article.get("title_french", ""),
            extracted_article.get("summary_french", ""),
            extracted_article.get("summary_arabic", ""),
            extracted_article.get("content_french", ""),
            extracted_article.get("content_arabic", ""),
            concepts,
        ]
        extracted_article["embedding_text"] = "\n".join(p for p in parts if p)

    # Ensure search_content
    if not extracted_article.get("search_content"):
        extracted_article["search_content"] = " ".join([
            extracted_article.get("law_number", ""),
            extracted_article.get("title_french", ""),
            " ".join(extracted_article.get("keywords", [])),
        ]).strip()

    # Ensure boolean fields
    bool_fields = ["has_obligations", "has_penalties", "has_deadlines", "has_exceptions", "is_abrogation", "is_transitional"]
    for field in bool_fields:
        if not isinstance(extracted_article.get(field), bool):
            extracted_article[field] = False

    # Ensure default string fields
    default_strings = {
        "article_type": "REGULATORY",
        "ambiguity_level": "LOW",
        "business_impact": "MEDIUM",
        "source_name": "JORT",
        "institution_primary": extracted_article.get("institution", ""),
    }
    for field, default in default_strings.items():
        if not extracted_article.get(field):
            extracted_article[field] = default

    if extracted_article.get("graph_level") is None:
        extracted_article["graph_level"] = 1
    if extracted_article.get("version") is None:
        extracted_article["version"] = 1
    if extracted_article.get("article_order") is None:
        extracted_article["article_order"] = 0

    # Ensure array fields
    array_fields = ["target_audience", "related_laws", "institutions", "relation_target_ids",
                    "relation_types", "entity_names", "entity_types", "entity_ids"]
    for field in array_fields:
        if not isinstance(extracted_article.get(field), list):
            extracted_article[field] = []

    # Remove None values
    for key, value in list(extracted_article.items()):
        if value is None:
            extracted_article[key] = [] if isinstance(extracted_article.get(key), list) else ""

    return extracted_article


def build_fallback(article_data: Dict, error_message: str = "") -> Dict:
    """Build a fallback article structure when extraction fails"""
    now = datetime.now()
    six_months = now + timedelta(days=180)

    iso_now = now.strftime("%Y-%m-%dT00:00:00Z")
    iso_next = six_months.strftime("%Y-%m-%dT00:00:00Z")

    law_number_match = re.search(r"[\d-]+", article_data.get("loi", ""))
    year_match = re.search(r"\d{4}", article_data.get("loi", ""))

    article_num_match = re.search(r"Art\.?\s*\d+", article_data.get("article", ""), re.I)

    result = {
        "jurisdiction": "TUNISIA",
        "institution": "",
        "law_type": "",
        "law_number": law_number_match.group(0) if law_number_match else "",
        "year": year_match.group(0) if year_match else "",
        "status": "ACTIVE",
        "title_french": article_data.get("titre", ""),
        "title_arabic": "",
        "chapter": article_data.get("chapitre", ""),
        "chapter_normalized": "",
        "section": "",
        "article_number": article_num_match.group(0) if article_num_match else "",
        "article_order": 0,
        "article_type": "REGULATORY",
        "content_french": article_data.get("article", ""),
        "content_arabic": "",
        "content_combined": article_data.get("article", ""),
        "summary": "",
        "summary_french": "",
        "summary_arabic": "",
        "search_content": article_data.get("article", ""),
        "embedding_text": article_data.get("article", ""),
        "keywords": [],
        "legal_domains": [],
        "legal_concepts": [],
        "business_impact": "MEDIUM",
        "target_audience": [],
        "related_laws": [],
        "ambiguity_level": "LOW",
        "has_obligations": False,
        "has_penalties": False,
        "has_deadlines": False,
        "has_exceptions": False,
        "is_abrogation": False,
        "is_transitional": False,
        "institution_primary": "",
        "institution_secondary": "",
        "institutions": [],
        "source_name": "JORT",
        "source_number": "",
        "source_url": "",
        "source_date": "",
        "publication_date": "",
        "effective_date": "",
        "relation_target_ids": [],
        "relation_types": [],
        "entity_names": [],
        "entity_types": [],
        "entity_ids": [],
        "community_id": "",
        "community_label": "",
        "community_summary": "",
        "parent_document_id": "",
        "preceding_article_id": "",
        "following_article_id": "",
        "graph_level": 1,
        "version": 1,
        "repeal_date": "",
        "superseded_by_id": "",
        "supersedes_id": "",
        "last_checked": iso_now,
        "next_check": iso_next,
    }

    if error_message:
        result["_extraction_error"] = error_message

    return result


def extract_articles_with_regex(text: str) -> List[Dict]:
    """Extract articles using regex fallback"""
    articles = []
    pattern = r'(?:Article|Art\.?|المادة|الفصل)\s+(\d+)[\s\S]*?(?=(?:Article|Art\.?|المادة|الفصل)\s+\d+|$)'
    for match in re.finditer(pattern, text, re.I):
        articles.append({
            "article": match.group(0).strip(),
            "article_number": f"Article {match.group(1)}",
            "titre": "",
            "loi": "",
            "chapitre": ""
        })
    print(f"    📄 Regex fallback: {len(articles)} articles")
    return articles


def extract_articles_from_pdf_text(pdf_text: str, retries: int = 3) -> List[Dict]:
    """Extract articles from PDF text using AI"""
    limited_text = pdf_text[:100000]
    prompt = TUNISIA_PDF_PARSING_PROMPT.replace("{pdf_text}", limited_text)

    for attempt in range(1, retries + 1):
        try:
            print(f"    🤖 Parsing PDF text (attempt {attempt}/{retries})…")
            raw = call_azure_openai(TUNISIA_SYSTEM_PROMPT, prompt)

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r"\[[\s\S]*\]", raw)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    raise ValueError("Could not parse articles array")

            articles = parsed if isinstance(parsed, list) else parsed.get("articles", [])
            print(f"    ✅ Extracted {len(articles)} articles from PDF")
            return articles

        except Exception as error:
            print(f"    ❌ Attempt {attempt}: {error}")
            if attempt == retries:
                return extract_articles_with_regex(pdf_text)
            time.sleep(2 * (2 ** (attempt - 1)))

    return extract_articles_with_regex(pdf_text)


def extract_tunisia_article(article_data: Dict, retries: int = 3) -> Dict:
    """Extract a single Tunisia article"""
    prompt = TUNISIA_EXTRACTION_PROMPT.replace("{article_data}", json.dumps(article_data))

    for attempt in range(1, retries + 1):
        try:
            preview = article_data.get("article", "")[:80]
            print(f"    🤖 Tunisia article (attempt {attempt}/{retries}): {preview}…")
            raw = call_azure_openai(TUNISIA_SYSTEM_PROMPT, prompt)
            parsed = parse_json_safely(raw, "Tunisia")
            result = ensure_keywords_populated(parsed, article_data)
            print(f"    ✅ Tunisia article extracted")
            return result

        except Exception as error:
            print(f"    ❌ Attempt {attempt}: {error}")
            if attempt == retries:
                return ensure_keywords_populated(build_fallback(article_data, str(error)), article_data)
            time.sleep(2 * (2 ** (attempt - 1)))

    return ensure_keywords_populated(build_fallback(article_data), article_data)


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_pdf_uae(pdf_path: str, filename: str) -> Dict:
    """Process PDF for UAE mode"""
    print(f"  📄 Parsing PDF…")
    text, text_len = extract_text_from_pdf(pdf_path)
    print(f"  📄 Pages: (extracted), {text_len} chars")

    structured_data = extract_legal_data_with_chunking(text)

    return {
        "success": True,
        "filename": filename,
        "raw_text_length": text_len,
        "structured_data": structured_data,
        "articles_count": len(structured_data.get("articles", [])),
        "processing_mode": "chunked" if text_len > MAX_CHUNK_CHARS else "single",
        "processed_at": datetime.now().isoformat()
    }


def process_pdf_tunisia(pdf_path: str, filename: str, delay_ms: int = 5000) -> List[Dict]:
    """Process PDF for Tunisia mode - returns articles array directly"""
    print(f"  📄 Parsing PDF…")
    text, text_len = extract_text_from_pdf(pdf_path)
    print(f"  📄 Pages: (extracted), {text_len} chars")

    articles = extract_articles_from_pdf_text(text)
    if not articles:
        raise RuntimeError("No articles found in PDF")

    print(f"  🇹🇳 Processing {len(articles)} articles…")
    extracted = []

    for i, article in enumerate(articles):
        print(f"  📋 Article {i + 1}/{len(articles)} — {article.get('loi', '')}")
        try:
            extracted.append(extract_tunisia_article(article))
        except Exception as error:
            extracted.append(ensure_keywords_populated(build_fallback(article, str(error)), article))

        if i < len(articles) - 1:
            print(f"  ⏳ Waiting {delay_ms}ms before next article…")
            time.sleep(delay_ms / 1000)

    return extracted


# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================

def load_checkpoint(checkpoint_file: str) -> Dict:
    """Load checkpoint from file"""
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"processed": [], "failed": [], "started_at": datetime.now().isoformat()}


def save_checkpoint(checkpoint_file: str, checkpoint: Dict):
    """Save checkpoint to file"""
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch PDF Legal Extractor")
    parser.add_argument("--input", default="./Law 2026", help="Input directory with PDF files")
    parser.add_argument("--output", default="./output", help="Output directory for JSON files")
    parser.add_argument("--mode", default="tunisia", choices=["uae", "tunisia"], help="Processing mode")
    parser.add_argument("--delay", type=int, default=5000, help="Delay between files in milliseconds")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and reprocess all files")

    args = parser.parse_args()

    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    MODE = args.mode
    DELAY_MS = args.delay
    RESET_CHECKPOINT = args.reset

    CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, ".checkpoint.json")

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         BATCH FOLDER PROCESSOR — Legal PDF Extractor (Azure OpenAI)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

  📁 Input  : {INPUT_DIR}
  📁 Output : {OUTPUT_DIR}
  ⚙️  Mode   : {MODE.upper()}
  ⏱️  Delay  : {DELAY_MS}ms between files
  🔄 Reset  : {'YES (checkpoint cleared)' if RESET_CHECKPOINT else 'NO (resumes from last run)'}
""")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Validate input folder
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Cannot read input folder: {INPUT_DIR}")
        print(f"   Create the folder and place your PDF files in it.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    pdf_files.sort()

    if not pdf_files:
        print(f"❌ No PDF files found in: {INPUT_DIR}")
        sys.exit(1)

    print(f"📂 Found {len(pdf_files)} PDF file(s):\n")
    for i, f in enumerate(pdf_files):
        print(f"   {i + 1}. {f}")
    print()

    # Load or reset checkpoint
    if RESET_CHECKPOINT:
        checkpoint = {"processed": [], "failed": [], "started_at": datetime.now().isoformat()}
        save_checkpoint(CHECKPOINT_FILE, checkpoint)
        print("🔄 Checkpoint reset — will process all files from scratch.\n")
    else:
        checkpoint = load_checkpoint(CHECKPOINT_FILE)
        if checkpoint.get("processed"):
            print(f"🔖 Checkpoint loaded — {len(checkpoint['processed'])} file(s) already done, resuming…\n")

    processed_set = set(checkpoint.get("processed", []))
    todo = [f for f in pdf_files if f not in processed_set]

    if not todo:
        print("🎉 All files already processed! Use --reset to reprocess.\n")
        sys.exit(0)

    print(f"📋 {len(todo)} file(s) to process ({len(pdf_files) - len(todo)} skipped via checkpoint)\n")

    success_count = 0
    fail_count = 0

    for i, filename in enumerate(todo):
        pdf_path = os.path.join(INPUT_DIR, filename)
        output_name = filename.replace(".pdf", "") + ".json"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        print(f"\n{'-' * 78}")
        print(f"[{i + 1}/{len(todo)}] 📄 {filename}")
        print(f"{'-' * 78}")

        try:
            if MODE == "tunisia":
                result = process_pdf_tunisia(pdf_path, filename, DELAY_MS)
            else:
                result = process_pdf_uae(pdf_path, filename)

            # Save result JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  💾 Saved → {output_path}")

            # Update checkpoint
            checkpoint["processed"].append(filename)
            if filename in checkpoint.get("failed", []):
                checkpoint["failed"] = [f for f in checkpoint["failed"] if f != filename]
            save_checkpoint(CHECKPOINT_FILE, checkpoint)

            success_count += 1
            article_count = len(result) if isinstance(result, list) else result.get("articles_count", "?")
            print(f"  ✅ Done ({article_count} articles extracted)")

        except Exception as error:
            print(f"  ❌ FAILED: {error}")

            if "failed" not in checkpoint:
                checkpoint["failed"] = []
            if filename not in checkpoint["failed"]:
                checkpoint["failed"].append(filename)
            save_checkpoint(CHECKPOINT_FILE, checkpoint)

            fail_count += 1

            # Save error record
            err_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", "") + ".error.json")
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump({
                    "success": False,
                    "filename": filename,
                    "error": str(error),
                    "failed_at": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)

        # Delay between files
        if i < len(todo) - 1:
            print(f"\n  ⏳ Waiting {DELAY_MS}ms before next file…")
            time.sleep(DELAY_MS / 1000)

    # Final summary
    print(f"\n{'═' * 78}")
    print(f"🏁 BATCH COMPLETE")
    print(f"{'═' * 78}")
    print(f"   ✅ Success : {success_count}")
    print(f"   ❌ Failed  : {fail_count}")
    print(f"   📁 Output  : {OUTPUT_DIR}")

    if checkpoint.get("failed"):
        print(f"\n   Failed files (will retry on next run):")
        for f in checkpoint["failed"]:
            print(f"      • {f}")
    print()


if __name__ == "__main__":
    main()