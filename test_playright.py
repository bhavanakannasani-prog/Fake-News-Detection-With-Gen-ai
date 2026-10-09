from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    print("Opening browser...")

    browser = p.chromium.launch(headless=False)

    page = browser.new_page()

    page.goto("https://www.bbc.com/news")

    print("Page Title:", page.title())

    page.wait_for_timeout(10000)

    browser.close()