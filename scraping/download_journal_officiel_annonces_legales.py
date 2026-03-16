from playwright.sync_api import sync_playwright
import os
import re
import time

OUTPUT_DIR = "pdfs/Journal_Officiel_Annonces_Legales"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_URL = "http://www.iort.gov.tn/WD120AWP/WD120Awp.exe/CONNECT/SITEIORT"
YEAR = "2026"

def parse_date(date_str):
    d, m, y = date_str.split('/')
    return f"{y}-{m}-{d}"

def navigate_to_table(page):
    print("🌐 Loading homepage...")
    page.goto(START_URL, wait_until="networkidle", timeout=30000)
    print("🔗 Clicking M8...")
    page.click('a[name="M8"]')
    page.wait_for_load_state("networkidle")
    print("🔍 Clicking search (A10)...")
    page.wait_for_selector('a[name="A10"]', timeout=15000)
    page.click('a[name="A10"]')
    page.wait_for_load_state("networkidle")
    print("🖱️  Hovering over A20...")
    page.wait_for_selector('a[name="A20"]', timeout=15000)
    page.hover('a[name="A20"]')
    page.wait_for_timeout(500)
    print("📋 Clicking A4 (سنة الرائد)...")
    page.wait_for_selector('a[name="A4"]', timeout=15000)
    page.click('a[name="A4"]')
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(500)
    print(f"📅 Selecting year {YEAR}...")
    page.wait_for_selector('select#A19', timeout=15000)
    page.select_option('select#A19', label=YEAR)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1000)
    print("✅ Table loaded\n")

def get_rows(page):
    """Get all issue rows: returns list of (issue_num, date_str, row_index)."""
    rows = []
    for tr in page.query_selector_all('tr[id^="A5_"]'):
        links = tr.query_selector_all('a')
        if len(links) >= 2:
            issue_num = links[0].inner_text().strip()
            date_str  = links[1].inner_text().strip()
            if re.match(r'\d{3}', issue_num) and '/' in date_str:
                rows.append((issue_num, date_str, links[1]))  # click the date link
    return rows

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()

    navigate_to_table(page)

    total_downloaded = 0
    total_skipped = 0
    page_num = 1

    while True:
        print(f"📄 Table page {page_num}...")
        rows = get_rows(page)
        print(f"   Found {len(rows)} issues")

        if not rows:
            print("   No rows found, stopping.")
            break

        for issue_num, date_str, date_link in rows:
            date_iso = parse_date(date_str)
            filename = f"JORT_Annonces_{issue_num}_{date_iso}.pdf"
            filepath = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(filepath):
                print(f"  📋 {issue_num} ({date_str}) → ⏩ already exists")
                total_skipped += 1
                continue

            # Click the date link — directly triggers download
            try:
                with page.expect_download(timeout=30000) as dl_info:
                    date_link.click()
                dl = dl_info.value
                dl.save_as(filepath)
                print(f"  📋 {issue_num} ({date_str}) → ✅ {filename}")
                total_downloaded += 1
                time.sleep(0.5)
            except Exception as e:
                print(f"  📋 {issue_num} ({date_str}) → ❌ {e}")

        # Find ">" next page button
        next_link = None
        for a in page.query_selector_all('a[href*="SCROLLTABLE"]'):
            if a.inner_text().strip() == '>':
                next_link = a
                break

        if not next_link:
            print("\n✅ No more pages.")
            break

        print(f"\n➡️  Next table page...")
        next_href = "http://www.iort.gov.tn" + next_link.get_attribute('href')
        page.goto(next_href, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(500)
        page_num += 1

    browser.close()
    print(f"""
{'='*50}
✅ Done!
  Downloaded : {total_downloaded}
  Skipped    : {total_skipped}
  Saved to   : ./{OUTPUT_DIR}/
{'='*50}
""")