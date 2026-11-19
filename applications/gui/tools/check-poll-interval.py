"""Measure visible update delay for two state-poll intervals.

Runs a headless Firefox page against a running app twice, with `?poll=300` and
`?poll=100`, and records how long the footer frame label takes to change. This
is a proxy for the visible-update delay: it measures the browser noticing and
displaying a new state, not the server's processing time.

Usage (from the repository root, with the app already running):

    python applications/gui/tools/check-poll-interval.py \
        --url http://127.0.0.1:8780/ --seconds 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait


def with_poll(url: str, poll_ms: int) -> str:
    parts = urlparse(url)
    query = dict(parse_qsl(parts.query))
    query["poll"] = str(poll_ms)
    return urlunparse(parts._replace(query=urlencode(query)))


def measure(driver, url: str, seconds: float) -> dict:
    driver.get(url)
    WebDriverWait(driver, 90).until(
        lambda d: not d.find_element(By.CSS_SELECTOR, ".splat-frame").text.endswith("--")
    )
    started = time.monotonic()
    last_frame = driver.find_element(By.CSS_SELECTOR, ".splat-frame").text
    changes = 0
    last_change_at = started
    gaps: list[float] = []
    while time.monotonic() - started < seconds:
        current = driver.find_element(By.CSS_SELECTOR, ".splat-frame").text
        now = time.monotonic()
        if current != last_frame:
            gaps.append(now - last_change_at)
            last_change_at = now
            last_frame = current
            changes += 1
        time.sleep(0.02)
    return {
        "poll_ms": int(dict(parse_qsl(urlparse(url).query)).get("poll", 0)),
        "changes": changes,
        "elapsed_s": round(time.monotonic() - started, 2),
        "gap_p50_ms": round(statistics.median(gaps) * 1000, 1) if gaps else None,
        "gap_p95_ms": round(sorted(gaps)[int(0.95 * (len(gaps) - 1))] * 1000, 1) if gaps else None,
        "gap_min_ms": round(min(gaps) * 1000, 1) if gaps else None,
        "gap_max_ms": round(max(gaps) * 1000, 1) if gaps else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8780/")
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--poll-a", type=int, default=300)
    parser.add_argument("--poll-b", type=int, default=100)
    args = parser.parse_args()

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1200,800")
    driver = webdriver.Firefox(options=options)
    results = []
    try:
        driver.set_window_size(1200, 800)
        for poll in (args.poll_a, args.poll_b):
            result = measure(driver, with_poll(args.url, poll), args.seconds)
            results.append(result)
            print(json.dumps(result))
    finally:
        driver.quit()

    if len(results) == 2 and results[0]["gap_p50_ms"] and results[1]["gap_p50_ms"]:
        ratio = results[0]["gap_p50_ms"] / results[1]["gap_p50_ms"]
        print(json.dumps({"gap_p50_ratio_300_over_100": round(ratio, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
