# Journal officiel des lois, décrets, décisions et avis
# this tool is for scraping the JORT website and downloading all the issues
# for a given period defined by START_YEAR and END_YEAR.
# Files are saved under:
#   pdfs/Journal_Officiel_Lois_Decrets_Decisions_Avis/
#       2024/
#       2025/
#       2026/
#       ...

from playwright.sync_api import sync_playwright
from datetime import date, timedelta
import os
import time
import re

# ============================================================
#  CONFIGURATION — change these two values to set the period
# ============================================================
START_YEAR = 1991   # first year to download (January 1st)
END_YEAR   = 1995   # last year to download  (December 31st, or today if current year)
# ============================================================

START_URL = "http://www.iort.gov.tn/WD120AWP/WD120Awp.exe/CONNECT/SITEIORT"
BASE_DIR  = "pdfs/Journal_Officiel_Lois_Decrets_Decisions_Avis"

start_date = date(START_YEAR, 1, 1)

# If END_YEAR is the current year, stop at today; otherwise go to Dec 31
if END_YEAR >= date.today().year:
    end_date = date.today()
else:
    end_date = date(END_YEAR, 12, 31)

# Pre-create one subfolder per year in the range
for y in range(START_YEAR, END_YEAR + 1):
    os.makedirs(os.path.join(BASE_DIR, str(y)), exist_ok=True)

print(f"📆 Period  : {start_date} → {end_date}")
print(f"📂 Output  : {BASE_DIR}/<year>/\n")

def parse_issue(text):
    match = re.search(r'(\d+)\s+بتاريخ\s+(\d{2}/\d{2}/\d{4})', text)
    if match:
        issue = match.group(1).zfill(3)
        d, m, y = match.group(2).split('/')
        return issue, f"{y}-{m}-{d}"
    return None, None

def fill_date(page, date_str):
    page.evaluate(f"""
        var inp = document.getElementById('A4');
        inp.focus();
        inp.value = '{date_str}';
        inp.dispatchEvent(new Event('input', {{bubbles: true}}));
        inp.dispatchEvent(new Event('change', {{bubbles: true}}));
        inp.blur();
        inp.dispatchEvent(new Event('blur', {{bubbles: true}}));
    """)
    page.wait_for_timeout(300)

def go_to_search(page):
    page.goto(START_URL, wait_until="networkidle", timeout=30000)
    page.click('a[name="A8"]')
    page.wait_for_load_state("networkidle")
    page.wait_for_selector('input#A4', timeout=15000)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()

    total_downloaded = 0
    total_skipped = 0
    total_no_result = 0

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%d/%m/%Y")

        # Year subfolder for this date
        year_dir = os.path.join(BASE_DIR, str(current.year))

        try:
            go_to_search(page)
        except Exception as e:
            print(f"  📅 {date_str} → ❌ Navigation error: {e}")
            current += timedelta(days=1)
            continue

        fill_date(page, date_str)

        try:
            page.click('img[name="z_A40_IMG"]')
            page.wait_for_load_state("networkidle", timeout=15000)
            page.wait_for_timeout(800)
        except Exception as e:
            print(f"  📅 {date_str} → ❌ Submit error: {e}")
            current += timedelta(days=1)
            continue

        try:
            result_el   = page.query_selector('a[name="A8"]')
            download_el = page.query_selector('a[name="A15"]')

            if not result_el or not download_el:
                print(f"  📅 {date_str} → no issue")
                total_no_result += 1
                current += timedelta(days=1)
                continue

            result_text = result_el.inner_text().strip()
            issue_num, date_iso = parse_issue(result_text)

            if not issue_num:
                print(f"  📅 {date_str} → no issue")
                total_no_result += 1
                current += timedelta(days=1)
                continue

            if date_iso != current.strftime("%Y-%m-%d"):
                print(f"  📅 {date_str} → no issue (closest: {date_iso})")
                total_no_result += 1
                current += timedelta(days=1)
                continue

        except Exception as e:
            print(f"  📅 {date_str} → ❌ Parse error: {e}")
            current += timedelta(days=1)
            continue

        filename = f"JORT_{issue_num}_{date_iso}.pdf"
        filepath = os.path.join(year_dir, filename)

        if os.path.exists(filepath):
            print(f"  📅 {date_str} → ⏩ {filename}")
            total_skipped += 1
            current += timedelta(days=1)
            continue

        try:
            with page.expect_download(timeout=30000) as dl_info:
                page.click('a[name="A15"]')
            dl = dl_info.value
            dl.save_as(filepath)
            print(f"  📅 {date_str} → ✅ {filename}")
            total_downloaded += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"  📅 {date_str} → ❌ Download error: {e}")

        current += timedelta(days=1)

    browser.close()
    print(f"""
{'='*50}
✅ Done!
  Period     : {start_date} → {end_date}
  Downloaded : {total_downloaded}
  Skipped    : {total_skipped}
  No issue   : {total_no_result}
  Saved to   : ./{BASE_DIR}/<year>/
{'='*50}
""")