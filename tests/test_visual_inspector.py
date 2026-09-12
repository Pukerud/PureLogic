import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tui.visual_inspector import _findings, concise_summary, inspect_visual_artifact
from tui.purelogic_tui import Cfg, _stream_one, _vision_review, parse_args, resolve_spec


class _FakePage:
    def __init__(self):
        self.handlers = {}

    def on(self, name, callback):
        self.handlers[name] = callback

    def goto(self, *_args, **_kwargs):
        return None

    def wait_for_timeout(self, _ms):
        return None

    def evaluate(self, _script):
        return {
            "title": "Mock page", "body_text": "Hello", "visible_text_length": 5,
            "interactive": [], "landmarks": [], "images_without_alt": [], "h1_count": 1,
        }

    def screenshot(self, path, **_kwargs):
        Path(path).write_bytes(b"mock-png")

    def close(self):
        return None


class _FakeContext:
    def __init__(self):
        self.page = _FakePage()

    def new_page(self):
        return self.page

    def close(self):
        return None


class _FakeBrowser:
    def __init__(self):
        self.context = _FakeContext()

    def new_context(self, **_kwargs):
        return self.context

    def close(self):
        return None


class _FakeChromium:
    def launch(self, **_kwargs):
        return _FakeBrowser()


class _FakePlaywright:
    chromium = _FakeChromium()

    def stop(self):
        return None


class _FakePlaywrightContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def start(self):
        return _FakePlaywright()


class VisualInspectorTests(unittest.TestCase):
    def test_stream_parser_accepts_final_message_content(self):
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def __iter__(self):
                yield b'data: {"choices":[{"message":{"content":"canvas review"},"finish_reason":"stop"}]}\n'
                yield b'data: [DONE]\n'

        class _Live:
            def note_delta(self, *_args):
                return None

        import tui.purelogic_tui as purelogic_tui
        with patch.object(purelogic_tui.urllib.request, "urlopen",
                          lambda *_args, **_kwargs: _Response()):
            content, _tools, _usage, finish, _reasoning = _stream_one(
                types.SimpleNamespace(base="http://local.test/v1"),
                {"messages": []}, _Live(), base="http://local.test/v1")
        self.assertEqual(content, "canvas review")
        self.assertEqual(finish, "stop")

    def test_canvas_only_page_is_not_reported_as_empty(self):
        findings = _findings(
            {"visible_text_length": 0, "interactive": [], "canvas": [{"visible": True}],
             "images_without_alt": []}, [], [], [])
        self.assertEqual(findings, [])

    def test_vision_review_disables_thinking_and_requires_visible_answer(self):
        cfg = types.SimpleNamespace(
            vision_base="http://local.test/v1", vision_model="vision-model",
            vision_cap=128, temp=0.7,
        )
        live = types.SimpleNamespace(
            begin_call=lambda *_args: None, inflight=0, calls=0,
            completion_tokens=0, prompt_tokens=0, last_finish_reason=None,
            truncated_calls=0,
        )
        with patch(
            "tui.purelogic_tui._stream_with_retry",
            return_value=("A visible review", [], {"completion_tokens": 3,
                          "prompt_tokens": 4}, "stop", ""),
        ) as request:
            answer = _vision_review(cfg, live, b"png", {"canvas": [{"visible": True}]})
        self.assertEqual(answer, "A visible review")
        self.assertEqual(
            request.call_args.kwargs["extra"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_vision_defaults_to_local_endpoint_and_model(self):
        with patch.dict(
            "os.environ",
            {"PURELOGIC_VISION_BASE": "", "PURELOGIC_VISION_MODEL": ""},
            clear=False,
        ):
            with tempfile.TemporaryDirectory() as directory:
                cfg = Cfg(parse_args([
                    "--base", "http://local.test/v1",
                    "--model", "local-vision-model",
                    "--workspace-dir", directory,
                ]))
                self.assertEqual(cfg.vision_base, "http://local.test/v1")
                self.assertEqual(cfg.vision_model, "local-vision-model")

    def test_smart_route_preserves_explicit_specs(self):
        self.assertEqual(resolve_spec("general", "Build an HTML dashboard"), "visual")
        self.assertEqual(resolve_spec("code", "Build an HTML dashboard"), "code")
        self.assertEqual(resolve_spec("general", "Solve 2 + 2", auto_visual=True), "general")

    def test_mock_browser_returns_evidence_and_cleans_temp_files(self):
        sync_api = types.ModuleType("playwright.sync_api")
        sync_api.sync_playwright = lambda: _FakePlaywrightContext()
        playwright = types.ModuleType("playwright")
        playwright.sync_api = sync_api
        old = {name: sys.modules.get(name) for name in ("playwright", "playwright.sync_api")}
        sys.modules["playwright"] = playwright
        sys.modules["playwright.sync_api"] = sync_api
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "index.html").write_text("<h1>Hello</h1>", encoding="utf-8")
                result = inspect_visual_artifact(root)
                self.assertTrue(result["available"])
                self.assertTrue(result["ok"])
                self.assertTrue(result["cleaned"])
                self.assertEqual(result["screenshot"]["bytes"], len(b"mock-png"))
        finally:
            for name, value in old.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_summary_is_bounded_and_json_safe(self):
        result = {
            "available": True, "ok": False,
            "findings": ["A" * 4000],
            "vision": {"status": "not_configured"},
        }
        summary = concise_summary(result)
        self.assertLess(len(summary), 2200)
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
