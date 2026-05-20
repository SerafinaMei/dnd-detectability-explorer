from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import os
import time

URL = os.environ["STREAMLIT_URL"]

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Opening {URL}")
        page.goto(URL, wait_until="networkidle", timeout=120_000)

        # If the app is asleep, Streamlit shows a wake-up button.
        # Text may vary slightly, so we search broadly.
        possible_buttons = [
            "Yes, get this app back up!",
            "Get this app back up",
            "Wake up",
        ]

        for text in possible_buttons:
            try:
                button = page.get_by_text(text, exact=False)
                if button.count() > 0:
                    print(f"Found wake-up button: {text}")
                    button.first.click()
                    time.sleep(30)
                    break
            except Exception:
                pass

        # Give Streamlit time to establish a real browser session.
        try:
            page.wait_for_load_state("networkidle", timeout=60_000)
        except PlaywrightTimeoutError:
            pass

        time.sleep(20)

        print("Visited app successfully.")
        browser.close()

if __name__ == "__main__":
    main()