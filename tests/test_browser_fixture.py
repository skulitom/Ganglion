"""Optional headless fixture check: set GANGLION_BROWSER_TESTS=1, install the browser extra + Edge."""
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("GANGLION_BROWSER_TESTS") != "1", reason="optional installed Edge fixture test")
def test_browser_truth_and_visible_completion(tmp_path):
    from playwright.sync_api import sync_playwright
    from ganglion.arena import browser as fixture
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": 760, "height": 500})
            page.goto(Path(fixture.__file__).with_name("browser.html").as_uri(), wait_until="networkidle")
            page.evaluate("window.beginTrial({id:'test', seed:1})")
            page.wait_for_timeout(50)
            stage = page.locator("#stage").bounding_box()
            page.mouse.click(stage["x"] + 5, stage["y"] + 5)
            choice = page.locator("#choice").bounding_box()
            page.mouse.click(choice["x"] + 18, choice["y"] + 18)
            assert page.locator("#result").inner_text() == "Selection confirmed"
            assert not page.locator("#choice").is_visible()
            truth = page.evaluate("window.truth")
            assert [e["kind"] for e in truth].count("hit") == 1
            assert [e["kind"] for e in truth].count("false_action") == 1
            assert all(e["trusted"] for e in truth if e["kind"] == "input_received")
            page.screenshot(path=str(tmp_path / "browser-completed.png"))
        finally:
            browser.close()


@pytest.mark.skipif(os.environ.get("GANGLION_BROWSER_TESTS") != "1", reason="optional installed Edge fixture test")
@pytest.mark.parametrize("mode, accepted", [("drop", True), ("reject", False)])
def test_browser_drop_truth_and_snap_back(mode, accepted):
    from playwright.sync_api import sync_playwright
    from ganglion.arena import browser as fixture
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": 760, "height": 500})
            page.goto(Path(fixture.__file__).with_name("manipulation.html").as_uri(), wait_until="networkidle")
            page.evaluate("value => window.beginTrial(value)", {"id": "test", "seed": 1, "mode": mode})
            stage = page.locator("#stage").bounding_box()
            page.mouse.move(stage["x"] + 100, stage["y"] + 155)
            page.mouse.down()
            page.mouse.move(stage["x"] + 520, stage["y"] + 180, steps=20)
            page.mouse.up()
            truth = page.evaluate("window.truth")
            assert sum(e["kind"] == ("drop_accepted" if accepted else "drop_rejected") for e in truth) == 1
            assert page.locator("#status").inner_text() == ("Condition met" if accepted else mode)
            assert all(e["trusted"] for e in truth if e["kind"] in ("pointer_down", "pointer_up"))
        finally:
            browser.close()
