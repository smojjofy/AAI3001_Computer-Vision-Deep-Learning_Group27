"""Check a capture-run boundary against a running integrated application.

Use it with the app's diagnostic option:

    python scripts/run_integration_app.py ... --force-run-change-after-s 25
    python scripts/check_run_boundary.py --url http://127.0.0.1:8765 --timeout 30

The check waits for the new run generation, then requires:
- the first new-generation diagnostics sample has no splat, no active identity,
  no error, and zero refreshes for the new run;
- /api/splat and /api/mask return 204 at that moment;
- the first state after the boundary belongs to the new generation.

It exits 0 on success and 1 on a failed expectation. It uses urllib so the
polling rate is high enough to catch the short boundary window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def get_json(base: str, path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=2) as response:
        return response.status, json.loads(response.read())


def get_status(base: str, path: str) -> int:
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=2) as response:
        return response.status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for the boundary and repopulation")
    args = parser.parse_args()

    started = time.monotonic()
    empty: dict | None = None
    new_state: dict | None = None
    while time.monotonic() - started < args.timeout:
        try:
            status, data = get_json(args.url, "/api/diagnostics")
        except (OSError, ValueError):
            continue
        if status != 200 or data.get("run_generation", 0) < 1:
            continue
        splat = data.get("splat")
        if splat is None and empty is None:
            empty = {
                "generation": data.get("run_generation"),
                "active_identity": data.get("active_identity"),
                "refresh_count": data.get("refresh_count"),
                "last_error": data.get("last_error"),
                "splat_status": get_status(args.url, "/api/splat"),
                "mask_status": get_status(args.url, "/api/mask"),
            }
        if splat is not None and new_state is None:
            new_state = {
                "generation": data.get("run_generation"),
                "active_generation": (data.get("active_identity") or {}).get("generation"),
                "frame_id": splat.get("frame_id"),
                "refresh_count": data.get("refresh_count"),
            }
        if empty is not None and new_state is not None:
            break

    failures: list[str] = []
    if empty is None:
        failures.append("no empty boundary sample was observed")
    else:
        if empty["active_identity"] is not None:
            failures.append(f"boundary kept an active identity: {empty['active_identity']}")
        if empty["splat_status"] != 204:
            failures.append(f"/api/splat returned {empty['splat_status']} at the boundary, expected 204")
        if empty["mask_status"] != 204:
            failures.append(f"/api/mask returned {empty['mask_status']} at the boundary, expected 204")
        if empty["last_error"] is not None:
            failures.append(f"boundary kept a last_error: {empty['last_error']}")
        if empty["refresh_count"] != 0:
            failures.append(f"boundary refresh_count is {empty['refresh_count']}, expected 0 for the new run")
    if new_state is None:
        failures.append("no new-run state was published")
    elif new_state["active_generation"] != new_state["generation"]:
        failures.append(
            f"new state generation {new_state['active_generation']} does not match run generation {new_state['generation']}"
        )

    print(json.dumps({"empty": empty, "new_state": new_state, "failures": failures}, indent=2))
    if failures:
        print("boundary check failed", file=sys.stderr)
        return 1
    print("boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
