# Journal officiel des lois, décrets, décisions et avis
# Downloads all issues for a range of years using the year selector.
# Files are saved under:
#   pdfs/Journal_Officiel_Lois_Decrets_Decisions_Avis/
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
BASE_DIR  = "pdfs/Journal_Officiel_Lois_Decrets_Decisions_Avis"

for y in range(START_YEAR, END_YEAR + 1):
    os.makedirs(os.path.join(BASE_DIR, str(y)), exist_ok=True)

def parse_issue(text):
    match = re.search(r'(\d+)\s+بتاريخ\s+(\d{2}/\d{2}/\d{4})', text)
    if match:
        issue = match.group(1).zfill(3)
        d, m, y = match.group(2).split('/')
        return issue, f"{y}-{m}-{d}", y
    return None, None, None

def go_to_search(page):
    """Navigate from homepage to the search page."""
    page.goto(START_URL, wait_until="networkidle", timeout=30000)
    page.click('a[name="A8"]')
    page.wait_for_load_state("networkidle")
    page.wait_for_selector('select#A11', timeout=15000)
    page.wait_for_timeout(500)

def get_rows(page):
    """Extract all rows from current table page. Returns [] on error."""
    try:
        page.wait_for_selector('div[id^="A3_"]', timeout=10000)
        page.wait_for_timeout(500)
        rows = []
        for div in page.query_selector_all('div[id^="A3_"]'):
            try:
                div_id    = div.get_attribute('id')
                row_index = div_id.split('_')[1]
                info_el   = div.query_selector('a[name="A8"]')
                if info_el:
                    text = info_el.inner_text().strip()
                    issue_num, date_iso, issue_year = parse_issue(text)
                    if issue_num:
                        rows.append((issue_num, date_iso, issue_year, row_index))
            except:
                continue
        return rows
    except Exception as e:
        print(f"     ⚠️  get_rows error: {e}")
        return []

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

        # Step 1: Navigate to search page
        try:
            go_to_search(page)
        except Exception as e:
            print(f"  ❌ Navigation error: {e}")
            continue

        # Step 2: Select the year from dropdown
        try:
            page.select_option('select#A11', label=str(year))
            page.wait_for_load_state("networkidle", timeout=15000)
            page.wait_for_timeout(800)
        except Exception as e:
            print(f"  ❌ Year select error: {e}")
            continue

        # Step 3: Click the search button (mandatory)
        try:
            page.wait_for_selector('img[name="z_A40_IMG"]', timeout=10000)
            page.click('img[name="z_A40_IMG"]')
            page.wait_for_load_state("networkidle", timeout=15000)
            page.wait_for_timeout(1000)
            print(f"  ✅ Search submitted for {year}")
        except Exception as e:
            print(f"  ❌ Search button error: {e}")
            continue

        table_page      = 1
        year_downloaded = 0
        year_skipped    = 0

        while True:
            print(f"\n  📄 Table page {table_page}...")

            rows = get_rows(page)
            print(f"     Found {len(rows)} issues")

            if not rows:
                print("     No rows found, stopping.")
                break

            for issue_num, date_iso, issue_year, row_index in rows:
                folder_year = issue_year if issue_year else str(year)
                year_dir    = os.path.join(BASE_DIR, folder_year)
                os.makedirs(year_dir, exist_ok=True)

                filename = f"JORT_{issue_num}_{date_iso}.pdf"
                filepath = os.path.join(year_dir, filename)

                if os.path.exists(filepath):
                    print(f"     {issue_num} ({date_iso}) → ⏩ already exists")
                    year_skipped += 1
                    continue

                try:
                    page.evaluate(f"_PAGE_.A3.value = {row_index};")
                    page.wait_for_timeout(200)
                    div = page.query_selector(f'div#A3_{row_index}')
                    btn = div.query_selector('a[name="A15"]')
                    with page.expect_download(timeout=30000) as dl_info:
                        btn.click()
                    dl = dl_info.value
                    dl.save_as(filepath)
                    print(f"     {issue_num} ({date_iso}) → ✅ {filename}")
                    year_downloaded += 1
                    time.sleep(0.5)
                except Exception as e:
                    print(f"     {issue_num} ({date_iso}) → ❌ {e}")

            # Find ">" next page button
            next_link = None
            try:
                for a in page.query_selector_all('a[href*="ZR_RechercheArijJORTPlusieurs"]'):
                    if a.inner_text().strip() == '>':
                        next_link = a
                        break
            except:
                pass

            if not next_link:
                print(f"\n  ✅ No more pages for {year}.")
                break

            print(f"\n  ➡️  Next table page...")
            try:
                next_href = "http://www.iort.gov.tn" + next_link.get_attribute('href')
                page.goto(next_href, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(800)
                table_page += 1
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