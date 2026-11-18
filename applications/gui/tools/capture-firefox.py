"""Capture the live GUI with Firefox through Selenium.

This replaces the Edge/CDP screenshot helper so a capture never uses the
user's running browser or profile. Selenium starts Firefox with an isolated
temporary profile and takes a normal screenshot after the page has data.

Run from the repository root:

    python applications/gui/tools/capture-firefox.py \
        --url http://127.0.0.1:8765/ --out applications/gui/screenshots/live.png \
        --wait-ms 8000 --click-selector '.processor-tab[data-view="mask"]'

Requires the `selenium` Python package and a local Firefox installation.
Selenium Manager obtains a matching geckodriver.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8765/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-ms", type=int, default=6000, help="Extra wait after the page reports live data")
    parser.add_argument("--click-selector", default=None, help="Optional element to click before the screenshot")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    options = Options()
    options.add_argument("--headless")
    options.add_argument(f"--window-size={args.width},{args.height}")
    driver = webdriver.Firefox(options=options)
    try:
        driver.set_window_size(args.width, args.height)
        driver.get(args.url)

        def has_live_state(current) -> bool:
            text = current.execute_script("return document.body.innerText")
            return (
                "Real-Time" in text
                and "Frame: --" not in text
                and "Splats: --" not in text
            )

        WebDriverWait(driver, 60).until(has_live_state)
        time.sleep(max(0, args.wait_ms) / 1000.0)
        if args.click_selector:
            elements = driver.find_elements(By.CSS_SELECTOR, args.click_selector)
            if elements:
                elements[0].click()
                time.sleep(1.5)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        driver.save_screenshot(str(args.out))
        print(f"SCREENSHOT: {args.out}")
        print(json_summary(driver))
        return 0
    finally:
        driver.quit()


def json_summary(driver) -> str:
    text = driver.execute_script("return document.body.innerText")
    compact = " ".join(text.split())
    return f"PAGE_TEXT: {compact[:2000]}"


if __name__ == "__main__":
    raise SystemExit(main())
