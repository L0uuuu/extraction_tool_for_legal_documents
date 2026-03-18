"""
fix_columns_groq.py — Fix two-column RTL layout using Groq LLM
===============================================================
Reads the OCR .txt file, sends each page to Groq, and asks it
to reorder the text so the RIGHT column comes first (Arabic RTL).

SETUP:
  pip install groq
  Get free API key at: https://console.groq.com/
  Set it: $env:GROQ_API_KEY='your_key'  (PowerShell)
          or add to .env: GROQ_API_KEY=your_key

USAGE:
  python fix_columns_groq.py <ocr_txt_file>
  python fix_columns_groq.py ocr_output/1991/JORT_001_ocr.txt

OUTPUT:
  ocr_output/1991/JORT_001_ocr_fixed.txt
"""

import os
import sys
import re
import time
import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Groq settings ─────────────────────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"   # fast + smart, free tier
MAX_TOKENS   = 8192
TEMPERATURE  = 0.0                          # deterministic — we want exact reordering
RETRY_DELAY  = 5                            # seconds between retries on rate limit


SYSTEM_PROMPT = """You are an expert in Arabic document layout and text processing.
Your only job is to reorder OCR-extracted text from two-column Arabic documents.

RULES — follow them exactly:
- Arabic documents are read RIGHT to LEFT, so the RIGHT column must come FIRST
- The input text was extracted left-to-right by OCR, so columns are in WRONG order
- Identify the two column flows by looking at the content (each decree/article has its own logical flow)
- Output the RIGHT column content first, then the LEFT column content
- DO NOT translate, summarize, correct, or modify ANY word — only reorder
- DO NOT add any explanation, commentary, or headers
- Output ONLY the reordered text, nothing else
- Preserve all original Arabic text exactly as-is including any Markdown formatting"""


REORDER_PROMPT = """Below is one page of OCR text extracted from a JORT (Tunisian Official Gazette).
It is a two-column Arabic page but the OCR read LEFT column first instead of RIGHT column first.

Reorder the text so the RIGHT column comes first, then the LEFT column.
Each column contains one or more complete decrees/orders (أمر/قرار) or article sections (الفصل).
Use the logical flow of the Arabic text to identify which paragraphs belong to which column.

OUTPUT ONLY THE REORDERED TEXT. NO EXPLANATIONS.

PAGE TEXT:
{page_text}"""


# ══════════════════════════════════════════════════════════════════════════════
# GROQ API CALL
# ══════════════════════════════════════════════════════════════════════════════

def call_groq(client, page_text: str, page_label: str) -> str:
    """Send one page to Groq and get reordered text back."""
    prompt = REORDER_PROMPT.format(page_text=page_text)

    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens  = MAX_TOKENS,
                temperature = TEMPERATURE,
            )
            result = response.choices[0].message.content or ""
            tokens_in  = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens
            print(f"      ✅ {tokens_in} in / {tokens_out} out")
            return result.strip()

        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                print(f"      ⏳ Rate limit hit, waiting {RETRY_DELAY}s... (attempt {attempt}/3)")
                time.sleep(RETRY_DELAY)
            else:
                print(f"      ❌ Error on attempt {attempt}/3: {e}")
                if attempt == 3:
                    print(f"      ⚠️  Keeping original text for {page_label}")
                    return page_text  # fallback: return unchanged
                time.sleep(2)

    return page_text  # fallback


# ══════════════════════════════════════════════════════════════════════════════
# PAGE SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def split_into_pages(text: str) -> list[tuple[str, str]]:
    """Split text into (page_header, page_content) tuples."""
    pattern = re.compile(
        r'(---\s*\n###\s*Page\s*\d+\s*\n---)',
        re.IGNORECASE
    )
    parts = pattern.split(text)
    pages = []

    # Text before first page marker (header/preamble)
    if parts and not pattern.match(parts[0].strip()):
        if parts[0].strip():
            pages.append(("PREAMBLE", parts[0]))
        parts = parts[1:]

    i = 0
    while i < len(parts) - 1:
        if pattern.match(parts[i].strip()):
            header  = parts[i].strip()
            content = parts[i + 1] if i + 1 < len(parts) else ""
            pages.append((header, content))
            i += 2
        else:
            i += 1

    if not pages:
        pages = [("PAGE 1", text)]

    return pages


def is_single_column(page_content: str) -> bool:
    """
    Heuristic: skip Groq call for pages that are clearly single-column
    (e.g. table of contents, cover page, announcement pages).
    Saves API calls and cost.
    """
    lines = [l for l in page_content.split('\n') if l.strip()]
    if len(lines) < 5:
        return True

    # If there are clear table markers it's a table page — skip
    pipe_lines = sum(1 for l in lines if l.count('|') >= 2)
    if pipe_lines / len(lines) > 0.3:
        return True

    # If total text is very short, probably a cover/title page
    if len(page_content.strip()) < 300:
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# RESOLVE OUTPUT PATH
# ══════════════════════════════════════════════════════════════════════════════

def resolve_output_path(input_path: str) -> str:
    """Save _fixed.txt in the same folder as the input _ocr.txt."""
    p = Path(input_path)
    # Remove _ocr suffix if present, add _fixed
    stem = p.stem
    if stem.endswith("_ocr"):
        stem = stem[:-4]
    return str(p.parent / f"{stem}_ocr_fixed.txt")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def fix_columns(input_path: str, output_path: str = None) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set.\n"
            "  1. Get free key at https://console.groq.com/\n"
            "  2. PowerShell: $env:GROQ_API_KEY='your_key'\n"
            "  3. Or add to .env: GROQ_API_KEY=your_key"
        )

    client = Groq(api_key=api_key)

    if output_path is None:
        output_path = resolve_output_path(input_path)

    print(f"\n{'='*60}")
    print(f"🔧 Column Fixer — Groq ({GROQ_MODEL})")
    print(f"{'='*60}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Split into pages
    pages = split_into_pages(raw_text)
    print(f"📄 Found {len(pages)} pages\n")

    fixed_pages = []
    skipped = 0
    fixed = 0

    for i, (header, content) in enumerate(pages, 1):
        label = header.replace('\n', ' ').strip()[:50]
        print(f"  [{i}/{len(pages)}] {label}")

        if is_single_column(content):
            print(f"      → single column / short page, skipping Groq")
            fixed_pages.append(f"{header}\n{content}")
            skipped += 1
            continue

        print(f"      → sending to Groq for column reorder...")
        reordered = call_groq(client, content.strip(), label)
        fixed_pages.append(f"{header}\n{reordered}")
        fixed += 1

        # Small delay to stay within free tier rate limits
        time.sleep(1)

    # Rejoin all pages
    final_text = '\n\n'.join(fixed_pages)

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# OCR EXTRACTED TEXT — COLUMNS FIXED BY GROQ\n")
        f.write(f"# Source: {Path(input_path).name}\n")
        f.write(f"# Model: {GROQ_MODEL}\n")
        f.write(f"# Pages fixed: {fixed} | skipped: {skipped}\n")
        f.write(f"{'='*60}\n\n")
        f.write(final_text)

    print(f"\n{'='*60}")
    print(f"✅ Done! {fixed} pages fixed | {skipped} pages skipped")
    print(f"💾 Saved → {output_path}")
    print(f"{'='*60}\n")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix two-column RTL layout in JORT OCR output using Groq"
    )
    parser.add_argument("input", help="Path to OCR .txt file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (default: <input>_fixed.txt)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ File not found: {args.input}")
        sys.exit(1)

    fix_columns(args.input, args.output)