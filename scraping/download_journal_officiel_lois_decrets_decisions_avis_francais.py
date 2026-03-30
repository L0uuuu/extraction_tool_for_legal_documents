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
START_YEAR = 2024
END_YEAR   = 2025
# ============================================================

START_URL = "http://www.iort.gov.tn/WD120AWP/WD120Awp.exe/CONNECT/SITEIORT"
BASE_DIR  = "pdfs/Journal_Officiel_Lois_Decrets_Decisions_Avis"

for y in range(START_YEAR, END_YEAR + 1):
    os.makedirs(os.path.join(BASE_DIR, str(y)), exist_ok=True)

def parse_issue(text):
    """
    Parse issue number and date from text like:
    "JORT n°: 156 du 31/12/2025"
    """
    match = re.search(r'JORT n°:\s*(\d+)\s+du\s+(\d{2}/\d{2}/\d{4})', text)
    if match:
        issue = match.group(1).zfill(3)
        d, m, y = match.group(2).split('/')
        return issue, f"{y}-{m}-{d}", y
    return None, None, None

def go_to_search(page):
    """Navigate from homepage to the search page."""
    print("  🌐 Loading homepage...")
    page.goto(START_URL, wait_until="networkidle", timeout=30000)
    
    # Step 1: Click the "Français" link to ensure correct language/interface
    try:
        page.wait_for_selector('a[name="M32"]', timeout=10000)
        page.click('a[name="M32"]')
        page.wait_for_load_state("networkidle", timeout=15000)
        print("  ✅ Clicked 'Français' link")
        page.wait_for_timeout(1000)
    except Exception as e:
        print(f"  ⚠️ Could not click Français link: {e}")
        # Continue anyway - maybe already in French or the link is not needed
    
    # Step 2: Click the "Journal officiel (lois, décrets, arrêtés et avis)" link (M7)
    try:
        page.wait_for_selector('a[name="M7"]', timeout=10000)
        page.click('a[name="M7"]')
        page.wait_for_load_state("networkidle", timeout=15000)
        print("  ✅ Clicked 'Journal officiel (lois, décrets, arrêtés et avis)' link")
        page.wait_for_timeout(1000)
    except Exception as e:
        print(f"  ❌ Could not click Journal officiel link: {e}")
        raise
    
    # Step 3: Click the "Recherche journal" link (A21)
    try:
        page.wait_for_selector('a[name="A21"]', timeout=10000)
        page.click('a[name="A21"]')
        page.wait_for_load_state("networkidle", timeout=15000)
        print("  ✅ Clicked 'Recherche journal' link")
        page.wait_for_timeout(1000)
    except Exception as e:
        print(f"  ❌ Could not click Recherche journal link: {e}")
        raise
    
    # Step 4: Wait for the search page to be fully loaded with the year selector
    try:
        page.wait_for_selector('select#A11', timeout=15000)
        print("  ✅ Search page loaded, year selector found")
        page.wait_for_timeout(500)
    except Exception as e:
        print(f"  ❌ Search page did not load properly: {e}")
        raise

def get_rows(page):
    """
    Extract all rows from current table page.
    Rows are contained in divs with id like "A3_1", "A3_2", etc.
    """
    try:
        page.wait_for_selector('div[id^="A3_"]', timeout=10000)
        page.wait_for_timeout(500)
        rows = []
        for div in page.query_selector_all('div[id^="A3_"]'):
            try:
                div_id    = div.get_attribute('id')
                row_index = div_id.split('_')[1]
                
                info_el = div.query_selector('a[name="A8"]')
                if info_el:
                    text = info_el.inner_text().strip()
                    issue_num, date_iso, issue_year = parse_issue(text)
                    if issue_num:
                        rows.append((issue_num, date_iso, issue_year, row_index))
            except Exception as e:
                print(f"     ⚠️  Error parsing row {div_id}: {e}")
                continue
        return rows
    except Exception as e:
        print(f"     ⚠️  get_rows error: {e}")
        return []

def get_next_page_url(page):
    """Find the '>' link for next page navigation."""
    try:
        pagination_td = page.query_selector('td#A6')
        if pagination_td:
            for a in pagination_td.query_selector_all('a'):
                if a.inner_text().strip() == '>':
                    href = a.get_attribute('href')
                    if href:
                        if href.startswith('/'):
                            return "http://www.iort.gov.tn" + href
                        return href
    except Exception as e:
        print(f"     ⚠️  Error finding next page: {e}")
    return None

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

        # Step 1: Navigate to search page (with all required clicks)
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
            print(f"  ✅ Selected year {year}")
        except Exception as e:
            print(f"  ❌ Year select error: {e}")
            continue

        # Step 3: Click the search button (the magnifying glass image)
        try:
            # The search button is an image with name="z_A40_IMG"
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
                    # Set the row index in the page's JavaScript variable
                    page.evaluate(f"_PAGE_.A3.value = {row_index};")
                    page.wait_for_timeout(200)
                    
                    div = page.query_selector(f'div#A3_{row_index}')
                    if not div:
                        print(f"     {issue_num} ({date_iso}) → ❌ Row div not found")
                        continue
                    
                    # Find the PDF download link (name="A15")
                    btn = div.query_selector('a[name="A15"]')
                    if not btn:
                        print(f"     {issue_num} ({date_iso}) → ❌ Download link not found")
                        continue
                    
                    with page.expect_download(timeout=60000) as dl_info:
                        btn.click()
                    dl = dl_info.value
                    dl.save_as(filepath)
                    print(f"     {issue_num} ({date_iso}) → ✅ {filename}")
                    year_downloaded += 1
                    time.sleep(0.5)
                except Exception as e:
                    print(f"     {issue_num} ({date_iso}) → ❌ {e}")

            next_url = get_next_page_url(page)
            if not next_url:
                print(f"\n  ✅ No more pages for {year}.")
                break

            print(f"\n  ➡️  Next table page...")
            try:
                page.goto(next_url, wait_until="networkidle", timeout=60000)
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
  Saved to   : ./{BASE_DIR}/<year>
{'='*50}
""")