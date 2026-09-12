"""Optional, disposable browser inspection for generated visual artifacts.

The inspector deliberately keeps browser state outside the run workspace.  It returns
small, JSON-safe evidence for the manager and only invokes vision through a callback
owned by the existing model layer.  No screenshot bytes or browser profile survive a
call unless the caller explicitly asks to retain evidence.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote


BOOKKEEPING = {
    "answer.md", "inspection.json", "notes.md", "plan.md", "result.json",
    "solution.py", "task.md", "tasks.json", "transcript.jsonl",
}


def _entrypoint(workspace: Path) -> Path | None:
    html = sorted(
        p for p in workspace.rglob("*.html")
        if p.is_file() and p.name not in BOOKKEEPING
    )
    if not html:
        return None
    return next((p for p in html if p.name.lower() == "index.html"), html[0])


def _safe_text(value: Any, limit: int = 1000) -> str:
    text = str(value or "").replace("\x00", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _semantic_evidence(page: Any) -> dict[str, Any]:
    return page.evaluate(
        """() => {
          const label = (el) => (el.getAttribute('aria-label') ||
            el.getAttribute('title') || el.innerText || el.value || '').trim();
          const interactive = [...document.querySelectorAll(
            'a,button,input,select,textarea,[role="button"],[tabindex]')]
            .slice(0, 80).map(el => ({
              tag: el.tagName.toLowerCase(), role: el.getAttribute('role') || '',
              label: label(el).slice(0, 160), disabled: !!el.disabled
            }));
          const landmarks = [...document.querySelectorAll(
            'main,nav,header,footer,aside,[role="main"],[role="navigation"]')]
            .slice(0, 30).map(el => ({
              tag: el.tagName.toLowerCase(), role: el.getAttribute('role') || '',
              label: label(el).slice(0, 160)
            }));
          const imagesWithoutAlt = [...document.images]
            .filter(img => !img.hasAttribute('alt'))
            .slice(0, 30).map(img => img.src.slice(0, 240));
          return {
            title: document.title,
            body_text: (document.body?.innerText || '').trim().slice(0, 2000),
            interactive, landmarks, images_without_alt: imagesWithoutAlt,
            canvas: [...document.querySelectorAll('canvas')].slice(0, 20).map(el => {
              const box = el.getBoundingClientRect();
              const style = getComputedStyle(el);
              return {width: el.width, height: el.height,
                visible: box.width > 0 && box.height > 0 && style.visibility !== 'hidden',
                css_width: Math.round(box.width), css_height: Math.round(box.height)};
            }),
            h1_count: document.querySelectorAll('h1').length,
            visible_text_length: (document.body?.innerText || '').trim().length,
          };
        }"""
    )


def _findings(evidence: dict[str, Any], console: list[dict[str, str]],
              page_errors: list[str], request_failures: list[str]) -> list[str]:
    findings: list[str] = []
    has_visible_canvas = any(x.get("visible") for x in evidence.get("canvas", []))
    if not evidence.get("visible_text_length") and not has_visible_canvas:
        findings.append("The page has no visible body text.")
    if (not evidence.get("interactive") and evidence.get("visible_text_length", 0) < 20
            and not has_visible_canvas):
        findings.append("No meaningful interactive or textual content was detected.")
    if evidence.get("images_without_alt"):
        findings.append(f"{len(evidence['images_without_alt'])} image(s) lack alt text.")
    unlabeled = [x for x in evidence.get("interactive", []) if not x.get("label")]
    if unlabeled:
        findings.append(f"{len(unlabeled)} interactive control(s) have no accessible label.")
    if page_errors:
        findings.append(f"The page raised {len(page_errors)} runtime error(s).")
    severe_console = [x for x in console if x.get("type") in {"error", "assert"}]
    if severe_console:
        findings.append(f"The console reported {len(severe_console)} error message(s).")
    if request_failures:
        findings.append(f"{len(request_failures)} request(s) failed to load.")
    return findings


def inspect_visual_artifact(
    workspace: str | os.PathLike[str],
    *,
    timeout_ms: int = 15000,
    vision: Callable[[bytes, dict[str, Any]], str] | None = None,
    retain_evidence: bool = False,
) -> dict[str, Any]:
    """Inspect the first generated HTML entrypoint with a disposable browser.

    ``vision`` receives screenshot bytes and the already-collected DOM evidence.  A
    missing Playwright installation is represented as a structured unavailable result;
    callers decide whether that is a warning or a required failure.
    """
    root = Path(workspace).resolve()
    entry = _entrypoint(root)
    base: dict[str, Any] = {
        "available": False,
        "ok": False,
        "entrypoint": entry.relative_to(root).as_posix() if entry else None,
        "findings": [],
        "console": [],
        "page_errors": [],
        "request_failures": [],
        "vision": {"available": False, "ok": False, "status": "not_configured"},
        "cleaned": True,
    }
    if entry is None:
        base["error"] = "No generated .html entrypoint was found in the workspace."
        base["findings"] = [base["error"]]
        return base
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        base["error"] = (
            "Playwright is unavailable. Install the Python package and its Chromium "
            "browser, or rerun with --inspect off."
        )
        base["findings"] = [base["error"]]
        return base

    console_messages: list[dict[str, str]] = []
    page_errors: list[str] = []
    request_failures: list[str] = []
    temp = tempfile.TemporaryDirectory(prefix="purelogic-visual-")
    browser = context = page = playwright = None
    screenshot = b""
    try:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.on("console", lambda msg: console_messages.append({
            "type": _safe_text(msg.type, 40), "text": _safe_text(msg.text),
        }))
        page.on("pageerror", lambda exc: page_errors.append(_safe_text(exc)))
        page.on("requestfailed", lambda req: request_failures.append(
            _safe_text(f"{req.method} {req.url}: {req.failure}", 500)))
        url = "file://" + quote(str(entry).replace("\\", "/"), safe="/:~!$&'()*+,;=@")
        page.goto(url, wait_until="load", timeout=timeout_ms)
        page.wait_for_timeout(100)
        evidence = _semantic_evidence(page)
        screenshot_path = Path(temp.name) / "page.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        screenshot = screenshot_path.read_bytes()
        findings = _findings(evidence, console_messages, page_errors, request_failures)
        base.update({
            "available": True,
            "ok": not any(x.get("type") == "error" for x in console_messages) and not page_errors,
            "url": url,
            "title": _safe_text(evidence.get("title"), 200),
            "dom": evidence,
            "console": console_messages[:30],
            "page_errors": page_errors[:20],
            "request_failures": request_failures[:20],
            "findings": findings,
            "screenshot": {"format": "png", "bytes": len(screenshot)},
        })
        if vision is not None:
            try:
                answer = vision(screenshot, {
                    "entrypoint": base["entrypoint"], "title": base.get("title"),
                    "dom": evidence, "findings": findings,
                })
                base["vision"] = {
                    "available": True, "ok": bool(answer.strip()), "status": "ok",
                    "findings": _safe_text(answer, 2400),
                }
            except Exception as exc:  # noqa: BLE001
                base["vision"] = {
                    "available": False, "ok": False, "status": "error",
                    "error": _safe_text(exc, 500),
                }
        if retain_evidence:
            evidence_dir = root / "visual-evidence"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            (evidence_dir / "latest.png").write_bytes(screenshot)
    except Exception as exc:  # noqa: BLE001
        base["error"] = _safe_text(exc, 700)
        base["findings"] = [f"Browser inspection failed: {base['error']}"]
    finally:
        for obj, method in ((page, "close"), (context, "close"), (browser, "close"),
                            (playwright, "stop")):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except Exception:  # noqa: BLE001
                    pass
        temp.cleanup()
        base["cleaned"] = not os.path.exists(temp.name)
    return base


def concise_summary(result: dict[str, Any]) -> str:
    """Return bounded manager-facing feedback without embedding screenshot data."""
    if not result.get("available"):
        return f"[VISUAL INSPECTION unavailable: {_safe_text(result.get('error'), 500)}]"
    parts = [
        "[VISUAL INSPECTION: browser check completed; "
        f"{'passed' if result.get('ok') else 'findings reported'}]",
    ]
    if result.get("findings"):
        parts.append("Browser findings: " + _safe_text(" ".join(result["findings"][:8]), 1200))
    vision = result.get("vision") or {}
    if vision.get("status") == "not_configured":
        parts.append("Vision review was not configured.")
    elif vision.get("findings"):
        parts.append("Vision findings: " + _safe_text(vision["findings"], 1600))
    elif vision.get("error"):
        parts.append("Vision review unavailable: " + _safe_text(vision["error"], 400))
    return " ".join(parts)


def write_result(path: str | os.PathLike[str], result: dict[str, Any]) -> None:
    """Write concise JSON evidence, intentionally excluding screenshot bytes."""
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
