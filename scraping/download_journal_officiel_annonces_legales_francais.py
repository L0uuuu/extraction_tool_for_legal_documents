# Journal officiel des annonces légales, sharia et judiciaires
# Downloads all issues for a range of years using the year selector.
# Files are saved under:
#   pdfs/Journal_Officiel_Annonces_Legales/
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
END_YEAR   = 2026
# ============================================================

START_URL = "http://www.iort.gov.tn/WD120AWP/WD120Awp.exe/CONNECT/SITEIORT"
BASE_DIR  = "pdfs/Journal_Officiel_Annonces_Legales"

for y in range(START_YEAR, END_YEAR + 1):
    os.makedirs(os.path.join(BASE_DIR, str(y)), exist_ok=True)

def parse_date(date_str):
    """Parse date string like '27/03/2026' to '2026-03-27' and return year"""
    d, m, y = date_str.split('/')
    return f"{y}-{m}-{d}", y

def navigate_to_table(page, year):
    """Navigate from homepage to the table of issues for the given year."""
    print("  🌐 Loading homepage...")
    page.goto(START_URL, wait_until="networkidle", timeout=30000)
    
    # Step 1: Click the "Français" link
    try:
        page.wait_for_selector('a[name="M32"]', timeout=10000)
        page.click('a[name="M32"]')
        page.wait_for_load_state("networkidle", timeout=15000)
        print("  ✅ Clicked 'Français' link")
        page.wait_for_timeout(1000)
    except Exception as e:
        print(f"  ⚠️ Could not click Français link: {e}")
    
    # Step 2: Click "Journal annonces légales" (M8)
    try:
        page.wait_for_selector('a[name="M8"]', timeout=10000)
        page.click('a[name="M8"]')
        page.wait_for_load_state("networkidle", timeout=15000)
        print("  ✅ Clicked 'Journal annonces légales' link")
        page.wait_for_timeout(1000)
    except Exception as e:
        print(f"  ❌ Could not click Journal annonces légales link: {e}")
        raise
    
    # Step 3: Click "Recherche" (A10)
    try:
        page.wait_for_selector('a[name="A10"]', timeout=10000)
        page.click('a[name="A10"]')
        page.wait_for_load_state("networkidle", timeout=15000)
        print("  ✅ Clicked 'Recherche' link")
        page.wait_for_timeout(1000)
    except Exception as e:
        print(f"  ❌ Could not click Recherche link: {e}")
        raise
    
    # Step 4: Hover over "Journal annonce" (A20)
    try:
        page.wait_for_selector('a[name="A20"]', timeout=10000)
        page.hover('a[name="A20"]')
        page.wait_for_timeout(500)
        print("  ✅ Hovered over 'Journal annonce'")
    except Exception as e:
        print(f"  ⚠️ Could not hover over A20: {e}")
    
    # Step 5: Click "Année journal" (A4)
    try:
        page.wait_for_selector('a[name="A4"]', timeout=10000)
        page.click('a[name="A4"]')
        page.wait_for_load_state("networkidle", timeout=15000)
        print("  ✅ Clicked 'Année journal' link")
        page.wait_for_timeout(500)
    except Exception as e:
        print(f"  ❌ Could not click Année journal link: {e}")
        raise
    
    # Step 6: Select the year from dropdown (A19)
    try:
        page.wait_for_selector('select#A19', timeout=15000)
        page.select_option('select#A19', label=str(year))
        page.wait_for_load_state("networkidle", timeout=15000)
        page.wait_for_timeout(1000)
        print(f"  ✅ Selected year {year}")
    except Exception as e:
        print(f"  ❌ Year select error: {e}")
        raise

def get_rows(page):
    """
    Extract all rows from current table page.
    Rows are in tr elements with id like "A5_1", "A5_2", etc.
    Each row contains two links: issue number and date.
    """
    rows = []
    try:
        # Wait for table rows to be present
        page.wait_for_selector('tr[id^="A5_"]', timeout=10000)
        
        for tr in page.query_selector_all('tr[id^="A5_"]'):
            try:
                links = tr.query_selector_all('a')
                if len(links) >= 2:
                    issue_num = links[0].inner_text().strip()
                    date_str = links[1].inner_text().strip()
                    # Validate format: issue should be 3 digits, date should have slashes
                    if re.match(r'\d{3}', issue_num) and '/' in date_str:
                        rows.append((issue_num, date_str, links[1]))
            except Exception as e:
                print(f"     ⚠️ Error parsing row: {e}")
                continue
    except Exception as e:
        print(f"     ⚠️ get_rows error: {e}")
    
    return rows

def get_next_page_url(page):
    """Find the '>' link for next page navigation."""
    try:
        pagination_td = page.query_selector('td#A9')
        if pagination_td:
            for a in pagination_td.query_selector_all('a'):
                if a.inner_text().strip() == '>':
                    href = a.get_attribute('href')
                    if href:
                        if href.startswith('/'):
                            return "http://www.iort.gov.tn" + href
                        return href
    except Exception as e:
        print(f"     ⚠️ Error finding next page: {e}")
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

        try:
            navigate_to_table(page, year)
        except Exception as e:
            print(f"  ❌ Navigation error: {e}")
            continue

        year_downloaded = 0
        year_skipped    = 0
        page_num        = 1

        while True:
            print(f"\n  📄 Table page {page_num}...")

            try:
                rows = get_rows(page)
            except Exception as e:
                print(f"     ❌ Error reading rows: {e}")
                break

            print(f"     Found {len(rows)} issues")

            if not rows:
                print("     No rows found, stopping.")
                break

            for issue_num, date_str, date_link in rows:
                date_iso, issue_year = parse_date(date_str)
                year_dir = os.path.join(BASE_DIR, issue_year)
                os.makedirs(year_dir, exist_ok=True)

                filename = f"JORT_Annonces_{issue_num}_{date_iso}.pdf"
                filepath = os.path.join(year_dir, filename)

                if os.path.exists(filepath):
                    print(f"     {issue_num} ({date_str}) → ⏩ already exists")
                    year_skipped += 1
                    continue

                try:
                    with page.expect_download(timeout=60000) as dl_info:
                        date_link.click()
                    dl = dl_info.value
                    dl.save_as(filepath)
                    print(f"     {issue_num} ({date_str}) → ✅ {filename}")
                    year_downloaded += 1
                    time.sleep(0.5)
                except Exception as e:
                    print(f"     {issue_num} ({date_str}) → ❌ {e}")

            # Find next page URL
            next_url = get_next_page_url(page)
            if not next_url:
                print(f"\n  ✅ No more pages for {year}.")
                break

            print(f"\n  ➡️  Next table page...")
            try:
                page.goto(next_url, wait_until="networkidle", timeout=60000)
                page.wait_for_timeout(800)
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