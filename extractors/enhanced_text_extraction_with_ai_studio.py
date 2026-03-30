import os
import fitz
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# 1. Extract raw text from PDF
def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return text

# 2. Send to Gemini and get clean markdown back
def to_markdown(raw_text: str) -> str:
    prompt = f"""You are given raw text extracted from an Arabic legal PDF of the Tunisian Official Gazette (JORT).  
The text has OCR‑like issues: split words, mixed columns, stray page numbers, and page headers/footers.

**Your task:**  
1. Remove **all** non‑legal content:
   - Table of contents (whether numbered or bulleted)  
   - Page headers, footers, and page numbers  
   - Any line that looks like "صفحــة 218" or "عــدد 15" (except when it is part of the actual decree title)  
   - Any administrative metadata (like the digital signature block)  
   - Any repeated page‑span lines like "صفحــة 220"  

2. Keep **only** the official legal acts:
   - Decrees (قرار), nominations (تسمية), appointments, erratum, etc.  
   - Their full content, including articles (الفصل) and tables (جداول).  

3. Output the result as **clean markdown**:
   - Use appropriate headings for each legal act.  
   - Format tables exactly as they appear.  
   - Do **not** summarize, translate, or add any commentary.  
   - Keep all Arabic text exactly as it is, fixing only obvious broken words (e.g., "تكلّف" should not be split as "تكل ـف").

4. Preserve the **exact wording** – no paraphrasing.

RAW TEXT:
{raw_text}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"max_output_tokens": 65536}
    )
    return response.text

# 3. Run
if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "test_pdf.pdf"

    print("Extracting text...")
    raw = extract_text(pdf)

    print("Sending to Gemini...")
    md = to_markdown(raw)

    out = pdf.replace(".pdf", "_cleaned.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Saved to {out}")