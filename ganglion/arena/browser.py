"""Launch an isolated Edge fixture. DOM access is confined to evaluator setup and truth."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import time
from uuid import uuid4


def find_fixture_window(title):
    from ganglion.core.window import api, describe
    u = api()
    matches = []
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.EnumWindows.argtypes = [callback, wintypes.LPARAM]

    @callback
    def visit(hwnd, _):
        text = ctypes.create_unicode_buffer(1024)
        u.GetWindowTextW(hwnd, text, len(text))
        if title in text.value and u.IsWindowVisible(hwnd):
            matches.append(describe(hwnd))
        return True

    u.EnumWindows(visit, 0)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one isolated fixture window; found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ready", "truth", "stop-file", "trial-file"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--scenario", choices=["reach", "manipulation"], default="reach")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright
    from ganglion.core.session import ensure_dpi_aware
    from .target import activate_own_window
    ensure_dpi_aware()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=False,
            args=["--window-position=180,30", "--window-size=760,620"])
        try:
            context = browser.new_context(no_viewport=True)
            page = context.new_page()
            html = "manipulation.html" if args.scenario == "manipulation" else "browser.html"
            page.goto(Path(__file__).with_name(html).as_uri(), wait_until="networkidle")
            title = "Ganglion fixture " + uuid4().hex
            page.evaluate("value => document.title = value", title)
            page.bring_to_front()
            page.wait_for_timeout(300)
            window = find_fixture_window(title)
            # Unique random title belongs to the isolated browser launched above.
            activate_own_window(window["hwnd"], fixture_pid=window["pid"])
            window["browser_version"] = browser.version
            Path(args.ready).write_text(json.dumps(window))
            deadline, previous = time.perf_counter() + args.seconds, None
            try:
                while time.perf_counter() < deadline and not Path(args.stop_file).exists():
                    if Path(args.trial_file).exists():
                        trial = json.loads(Path(args.trial_file).read_text())
                        if trial["id"] != previous:
                            page.evaluate("value => window.beginTrial(value)", trial)
                            previous = trial["id"]
                    page.wait_for_timeout(10)
            finally:
                Path(args.truth).write_text(json.dumps(page.evaluate("window.truth"), indent=2))
        finally:
            browser.close()


if __name__ == "__main__":
    main()
