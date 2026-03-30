# Journal officiel des annonces du Tribunal Foncier
# Downloads all issues for a range of years using the year selector.
# Files are saved under:
#   pdfs/Journal_Officiel_Tribunal_Foncier/
#       2024/
#       2025/
#       2026/

from playwright.sync_api import sync_playwright
import os
import re
import time

# ============================================================
#  CONFIGURATION
# ============================================================
START_YEAR = 1991
END_YEAR   = 1995
# ============================================================

START_URL = "http://www.iort.gov.tn/WD120AWP/WD120Awp.exe/CONNECT/SITEIORT"
BASE_DIR  = "pdfs/Journal_Officiel_Tribunal_Foncier"

for y in range(START_YEAR, END_YEAR + 1):
    os.makedirs(os.path.join(BASE_DIR, str(y)), exist_ok=True)

def parse_date(date_str):
    d, m, y = date_str.split('/')
    return f"{y}-{m}-{d}", y

def navigate_to_table(page, year):
    print("🌐 Loading homepage...")
    page.goto(START_URL, wait_until="networkidle", timeout=30000)
    print("🔗 Clicking M9...")
    page.click('a[name="M9"]')
    page.wait_for_load_state("networkidle")
    print("🔍 Clicking search (A5)...")
    page.wait_for_selector('a[name="A5"]', timeout=15000)
    page.click('a[name="A5"]')
    page.wait_for_load_state("networkidle")
    print("📋 Clicking A18 (سنة الرائد)...")
    page.wait_for_selector('a[name="A18"]', timeout=15000)
    page.click('a[name="A18"]')
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(500)
    print(f"📅 Selecting year {year}...")
    page.wait_for_selector('select#A7', timeout=15000)
    page.select_option('select#A7', label=str(year))
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(800)
    print("✅ Table loaded\n")

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

    total_downloaded = 0
    total_skipped    = 0

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n{'='*50}")
        print(f"📅 Processing year {year}...")
        print(f"{'='*50}")

        try:
            navigate_to_table(page, year)
        except Exception as e:
            print(f"  ❌ Navigation error: {e}")
            continue

        year_downloaded = 0
        year_skipped    = 0
        page_num        = 1

        while True:
            print(f"📄 Table page {page_num}...")

            try:
                rows = get_rows(page)
            except Exception as e:
                print(f"   ❌ Error reading rows: {e}")
                break

            print(f"   Found {len(rows)} issues")

            if not rows:
                print("   No rows found, stopping.")
                break

            for issue_num, date_str, date_link in rows:
                date_iso, issue_year = parse_date(date_str)
                year_dir = os.path.join(BASE_DIR, issue_year)
                os.makedirs(year_dir, exist_ok=True)

                filename = f"JORT_TribunalFoncier_{issue_num}_{date_iso}.pdf"
                filepath = os.path.join(year_dir, filename)

                if os.path.exists(filepath):
                    print(f"  📋 {issue_num} ({date_str}) → ⏩ already exists")
                    year_skipped += 1
                    continue

                try:
                    with page.expect_download(timeout=60000) as dl_info:
                        date_link.click()
                    dl = dl_info.value
                    dl.save_as(filepath)
                    print(f"  📋 {issue_num} ({date_str}) → ✅ {filename}")
                    year_downloaded += 1
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  📋 {issue_num} ({date_str}) → ❌ {e}")

            # Find ">" next page button
            next_link = None
            try:
                for a in page.query_selector_all('a[href*="TABLE_RequetteRechercheRequisitionJortAnnee"]'):
                    if a.inner_text().strip() == '>':
                        next_link = a
                        break
            except:
                pass

            if not next_link:
                print(f"\n✅ No more pages for {year}.")
                break

            print(f"\n➡️  Next table page...")
            try:
                next_href = "http://www.iort.gov.tn" + next_link.get_attribute('href')
                page.goto(next_href, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(500)
                page_num += 1
            except Exception as e:
                print(f"  ❌ Pagination error: {e}")
                break

        total_downloaded += year_downloaded
        total_skipped    += year_skipped
        print(f"\n  📊 Year {year}: Downloaded {year_downloaded} | Skipped {year_skipped}")

    browser.close()
    print(f"""
{'='*50}
✅ All done!
  Downloaded : {total_downloaded}
  Skipped    : {total_skipped}
  Saved to   : ./{BASE_DIR}/<year>/
{'='*50}
""")