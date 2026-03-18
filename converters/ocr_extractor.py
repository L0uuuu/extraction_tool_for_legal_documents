"""
JORT OCR Extractor — Mistral OCR + Tesseract fallback
======================================================
Saves output to a separate ocr_output/ folder, mirroring the PDF folder structure.

  pdfs/1991/JORT_001.pdf  →  ocr_output/1991/JORT_001_ocr.txt

USAGE:
  python ocr_extractor.py <pdf_file>
  python ocr_extractor.py <pdf_file> --engine tesseract
  python ocr_extractor.py <pdf_file> --pages 1-3
  python ocr_extractor.py <pdf_file> --output my_out.txt
"""

import os
import sys
import re
import base64
import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 1 — MISTRAL OCR
# ══════════════════════════════════════════════════════════════════════════════

def extract_with_mistral(pdf_path: str, page_range: tuple = None) -> str:
    try:
        from mistralai.client import Mistral
    except ImportError:
        raise ImportError("Run: pip install --upgrade mistralai")

    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY not set.\n"
            "  PowerShell: $env:MISTRAL_API_KEY='your_key'\n"
            "  Or add to .env file: MISTRAL_API_KEY=your_key"
        )

    client = Mistral(api_key=api_key)

    print("📤 Uploading PDF to Mistral OCR...")
    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

    print("🔍 Running Mistral OCR...")
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_b64}",
        },
        include_image_base64=False,
    )

    pages_text = []
    total_pages = len(response.pages)

    for page in response.pages:
        page_num = page.index + 1
        if page_range:
            start, end = page_range
            if not (start <= page_num <= end):
                continue
        print(f"   ✅ Page {page_num}/{total_pages}: {len(page.markdown)} chars")
        pages_text.append(f"<!-- PAGE {page_num} -->\n{page.markdown}")

    full_text = "\n\n".join(pages_text)
    print(f"📝 Total: {len(full_text):,} chars from {len(pages_text)} pages")
    return full_text


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 2 — TESSERACT OCR
# ══════════════════════════════════════════════════════════════════════════════

def extract_with_tesseract(pdf_path: str, page_range: tuple = None) -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        raise ImportError("Run: pip install pdf2image pillow pytesseract")

    print("🖼️  Converting PDF to images (300 DPI)...")
    images = convert_from_path(pdf_path, dpi=300)
    total = len(images)
    print(f"📄 {total} pages loaded")

    pages_text = []
    for i, image in enumerate(images):
        page_num = i + 1
        if page_range:
            start, end = page_range
            if not (start <= page_num <= end):
                continue
        print(f"   🔍 OCR page {page_num}/{total}...", end=" ")
        text = pytesseract.image_to_string(image, lang="ara+fra", config="--psm 6")
        print(f"{len(text.strip())} chars")
        if text.strip():
            pages_text.append(f"<!-- PAGE {page_num} -->\n{text}")

    full_text = "\n\n".join(pages_text)
    print(f"📝 Total: {len(full_text):,} chars")
    return full_text


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def postprocess_text(raw_text: str, engine: str) -> str:
    text = raw_text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r'^\s*[|_.]{3,}\s*$', '', text, flags=re.MULTILINE)

    if engine == "mistral":
        text = re.sub(r'<!--\s*PAGE\s*(\d+)\s*-->', r'\n\n---\n### Page \1\n---\n', text)

    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT — separate folder from input
# ══════════════════════════════════════════════════════════════════════════════

def resolve_output_path(pdf_path: str) -> str:
    """
    Always saves to a separate ocr_output/ folder at the project root,
    mirroring the PDF subfolder structure.

    Example:
      Input:  C:/project/pdfs/1991/JORT_001.pdf
      Output: C:/project/ocr_output/1991/JORT_001_ocr.txt
    """
    pdf = Path(pdf_path).resolve()
    stem = pdf.stem

    # Find the root folder that contains the PDF tree
    # Strategy: walk up from the PDF until we find a folder whose name
    # suggests it's an input container (pdfs, pdf, input, documents, data)
    INPUT_FOLDER_NAMES = {"pdfs", "pdf", "input", "inputs", "documents", "data", "raw"}

    project_root = None
    input_folder = None

    for parent in pdf.parents:
        if parent.name.lower() in INPUT_FOLDER_NAMES:
            project_root = parent.parent   # e.g. C:/project/
            input_folder = parent          # e.g. C:/project/pdfs/
            break

    # Fallback: use current working directory as project root
    if project_root is None:
        project_root = Path.cwd()
        input_folder = pdf.parent

    # Mirror the subfolder structure
    try:
        relative_subfolders = pdf.parent.relative_to(input_folder)
    except ValueError:
        relative_subfolders = Path(pdf.parent.name)

    output_dir = project_root / "ocr_output" / relative_subfolders
    output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir / f"{stem}_ocr.txt")


def save_output(text: str, pdf_path: str, output_path: str = None) -> str:
    """Save extracted text. Output always goes to ocr_output/ folder."""
    if output_path is None:
        output_path = resolve_output_path(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# OCR EXTRACTED TEXT\n")
        f.write(f"# Source: {Path(pdf_path).name}\n")
        f.write("# Encoding: UTF-8\n")
        f.write("# Tables: Markdown format (| col | col |)\n")
        f.write("# Page breaks: ### Page N\n")
        f.write(f"{'='*60}\n\n")
        f.write(text)

    print(f"\n💾 Saved → {output_path}")
    print(f"   {len(text):,} characters | {len(text.splitlines()):,} lines")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_page_range(s: str):
    parts = s.split('-')
    if len(parts) == 1:
        n = int(parts[0])
        return (n, n)
    return (int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(description="OCR extractor for JORT Arabic/French PDFs")
    parser.add_argument("pdf", help="Path to the scanned PDF file")
    parser.add_argument("--engine", choices=["mistral", "tesseract"], default="mistral")
    parser.add_argument("--output", "-o", default=None, help="Custom output path")
    parser.add_argument("--pages", default=None, help="Page range e.g. '1-5' or '3'")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"❌ File not found: {args.pdf}")
        sys.exit(1)

    page_range = parse_page_range(args.pages) if args.pages else None

    print(f"\n{'='*60}")
    print(f"📄 JORT OCR Extractor")
    print(f"{'='*60}")
    print(f"  PDF:    {args.pdf}")
    print(f"  Engine: {args.engine.upper()}")
    print(f"  Pages:  {f'{page_range[0]}-{page_range[1]}' if page_range else 'ALL'}")
    if not args.output:
        print(f"  Output: {resolve_output_path(args.pdf)}")
    print(f"{'='*60}\n")

    if args.engine == "mistral":
        raw_text = extract_with_mistral(args.pdf, page_range)
    else:
        raw_text = extract_with_tesseract(args.pdf, page_range)

    print("\n⚙️  Post-processing...")
    clean_text = postprocess_text(raw_text, args.engine)

    out_path = save_output(clean_text, args.pdf, args.output)

    print(f"\n✅ Done!")
    print(f"\n📋 Preview (first 500 chars):")
    print("-" * 40)
    print(clean_text[:500])
    print("-" * 40)


if __name__ == "__main__":
    main()