"""Browser check for the viewer display-latency label.

This exercises the real SplatOutputPanel render path, not only the pure latency
arithmetic. It checks:

1. the first splat state produces a non-placeholder estimate after its first
   completed render;
2. a later state with a different splat count updates the estimate;
3. a later state with the same splat count (the in-place descriptor path) also
   updates the estimate.

Run from the repository root against a running integrated app:

    python applications/gui/tools/check-latency-browser.py \
        --url http://127.0.0.1:8765/ --timeout 60

Requires Firefox and the `selenium` Python package. Exit code 0 means every
required case passed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait

PLACEHOLDER = "Display (est): --"


def read_state(driver) -> dict[str, str]:
    frame = driver.find_element(By.CSS_SELECTOR, ".splat-frame").text
    count = driver.find_element(By.CSS_SELECTOR, ".splat-count").text
    latency = driver.find_element(By.CSS_SELECTOR, ".splat-latency").text
    return {"frame": frame, "count": count, "latency": latency}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8765/")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    options = Options()
    options.add_argument("--headless")
    options.add_argument(f"--window-size={args.width},{args.height}")
    driver = webdriver.Firefox(options=options)
    failures: list[str] = []
    observations: list[dict[str, str]] = []
    try:
        driver.set_window_size(args.width, args.height)
        driver.get(args.url)
        WebDriverWait(driver, 60).until(
            lambda d: "Real-Time" in d.execute_script("return document.body.innerText")
        )
        # 1. First state: the label must leave the placeholder after a render.
        WebDriverWait(driver, args.timeout).until(
            lambda d: not read_state(d)["latency"].endswith("--")
        )
        first = read_state(driver)
        observations.append(first)
        if first["latency"].endswith("--"):
            failures.append("first state kept the placeholder latency label")
        if "est" not in first["latency"]:
            failures.append("first state label does not state that it is an estimate")

        # 2 and 3. Collect later states and classify by splat count.
        # The first state is still being replaced while the panel starts, so the
        # classification baseline is the next state after the first stable label.
        time.sleep(1.0)
        baseline = read_state(driver)
        observations.append(baseline)
        first_frame = baseline["frame"]
        first_count = baseline["count"]
        counts: set[str] = set()
        same_count_updated = False
        count_changed_updated = False
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            state = read_state(driver)
            if state["frame"] != first_frame and state["latency"] != baseline["latency"]:
                if state["count"] == first_count:
                    same_count_updated = True
                else:
                    count_changed_updated = True
                observations.append(state)
                counts.add(state["count"])
            if same_count_updated and count_changed_updated:
                break
            time.sleep(0.2)

        if not count_changed_updated:
            failures.append("no latency update observed after a splat-count change")
        if not same_count_updated:
            failures.append("no latency update observed after a same-count frame change")

        print(
            json.dumps(
                {
                    "first": first,
                    "same_count_update": same_count_updated,
                    "count_change_update": count_changed_updated,
                    "counts_seen": sorted(counts),
                    "observations": observations[-5:],
                    "failures": failures,
                },
                indent=2,
            )
        )
    finally:
        driver.quit()

    if failures:
        print("display latency browser check failed", file=sys.stderr)
        return 1
    print("display latency browser check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
