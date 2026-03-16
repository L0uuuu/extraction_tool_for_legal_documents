from playwright.sync_api import sync_playwright
import os
import re
import time

OUTPUT_DIR = "pdfs/Journal_Officiel_Tribunal_Foncier"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_URL = "http://www.iort.gov.tn/WD120AWP/WD120Awp.exe/CONNECT/SITEIORT"
YEAR = "2026"

def parse_date(date_str):
    d, m, y = date_str.split('/')
    return f"{y}-{m}-{d}"

def get_rows(page):
    rows = []
    for tr in page.query_selector_all('tr[id^="A8_"]'):
        links = tr.query_selector_all('a')
        if len(links) >= 2:
            issue_num = links[0].inner_text().strip()
            date_str  = links[1].inner_text().strip()
            if re.match(r'\d{3}', issue_num) and '/' in date_str:
                rows.append((issue_num, date_str, links[1]))
    return rows

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()

    # Step 1: Homepage
    print("🌐 Loading homepage...")
    page.goto(START_URL, wait_until="networkidle", timeout=30000)

    # Step 2: Click M9
    print("🔗 Clicking M9...")
    page.click('a[name="M9"]')
    page.wait_for_load_state("networkidle")

    # Step 3: Click A5 - بحث
    print("🔍 Clicking search (A5)...")
    page.wait_for_selector('a[name="A5"]', timeout=15000)
    page.click('a[name="A5"]')
    page.wait_for_load_state("networkidle")

    # Step 4: Click A18 - سنة الرائد
    print("📋 Clicking A18 (سنة الرائد)...")
    page.wait_for_selector('a[name="A18"]', timeout=15000)
    page.click('a[name="A18"]')
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(500)

    # Step 5: Select 2026
    print(f"📅 Selecting year {YEAR}...")
    page.wait_for_selector('select#A7', timeout=15000)
    page.select_option('select#A7', label=YEAR)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(800)
    print("✅ Table loaded\n")

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
            filename = f"JORT_TribunalFoncier_{issue_num}_{date_iso}.pdf"
            filepath = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(filepath):
                print(f"  📋 {issue_num} ({date_str}) → ⏩ already exists")
                total_skipped += 1
                continue

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

        # Find ">" next page — specific to this section's table param
        next_link = None
        for a in page.query_selector_all('a[href*="TABLE_RequetteRechercheRequisitionJortAnnee"]'):
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