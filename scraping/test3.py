from playwright.sync_api import sync_playwright

START_URL = "http://www.iort.gov.tn/WD120AWP/WD120Awp.exe/CONNECT/SITEIORT"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()

    page.goto(START_URL, wait_until="networkidle", timeout=30000)
    page.click('a[name="A8"]')
    page.wait_for_load_state("networkidle")
    page.wait_for_selector('input#A4', timeout=15000)

    date_str = "02/01/2026"

    # Fill with JS
    page.evaluate(f"""
        var inp = document.getElementById('A4');
        inp.focus();
        inp.value = '{date_str}';
        inp.dispatchEvent(new Event('input', {{bubbles: true}}));
        inp.dispatchEvent(new Event('change', {{bubbles: true}}));
        inp.blur();
        inp.dispatchEvent(new Event('blur', {{bubbles: true}}));
    """)
    page.wait_for_timeout(500)

    val = page.input_value('input#A4')
    print(f"Input value: [{val}]")

    page.click('img[name="z_A40_IMG"]')
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1000)

    # Print full page text
    print("\n--- FULL PAGE TEXT ---")
    print(page.inner_text('body'))
    print("--- END ---")

    # Print all links with their names
    print("\n--- ALL NAMED LINKS ---")
    for el in page.query_selector_all('a[name]'):
        name = el.get_attribute('name')
        text = el.inner_text().strip()
        print(f"  a[name='{name}'] → '{text}'")

    input("\nPress ENTER to close...")
    browser.close()