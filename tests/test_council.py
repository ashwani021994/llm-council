"""Safety and workflow checks. All provider requests are mocked."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import app
from backend import council, openrouter
from backend.config import MAX_QUESTION_CHARS, OPENROUTER_API_URL, Settings
import httpx
from gradio.routes import App
from starlette.testclient import TestClient

SETTINGS = Settings(api_key="test-secret-not-a-real-key", models=("test/alpha", "test/beta"),
                    chairman="test/alpha")
CATALOG = {m: {"id": m, "supported_parameters": ["max_tokens"]} for m in SETTINGS.models}
BALLOT = "Comments on both candidates.\nFINAL RANKING:\n1. Response B\n2. Response A"


class ConfigTests(unittest.TestCase):
    def test_reject_unsafe_or_unbounded_configuration(self):
        for kwargs in [dict(models=("test/a",)), dict(models=("test/a", "test/a")),
                       dict(chairman="<img src=bad>"), dict(max_output_tokens=100_000),
                       dict(local_password="short"), dict(port=80),
                       dict(api_key="same-secret-password", local_password="same-secret-password")]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Settings(**kwargs)

    def test_secrets_not_in_settings_repr(self):
        self.assertNotIn(SETTINGS.api_key, repr(SETTINGS))


class CouncilTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch("backend.council.get_settings", return_value=SETTINGS).start()
        patch("app.get_settings", return_value=SETTINGS).start()
        patch("backend.council.validate_model_configuration", new_callable=AsyncMock, return_value=CATALOG).start()

    async def test_invalid_input_makes_no_model_calls(self):
        with patch.object(council, "query_models_parallel", new_callable=AsyncMock) as query:
            for text in ["", "   ", "x" * (MAX_QUESTION_CHARS + 1), SETTINGS.api_key]:
                with self.subTest(text=text[:20]), self.assertRaises(council.CouncilError):
                    await council.run_full_council(text)
            query.assert_not_awaited()

    async def test_pipeline_and_label_mapping_survive_a_failed_model(self):
        settings = Settings(api_key=SETTINGS.api_key, models=("test/alpha", "test/beta", "test/gamma"), chairman="test/alpha")
        first = {"test/alpha": None, "test/beta": {"content": "Candidate answer"}, "test/gamma": {"content": "Other answer"}}
        second = {"test/beta": {"content": BALLOT}, "test/gamma": {"content": BALLOT}}
        with patch.object(council, "query_models_parallel", new_callable=AsyncMock,
                          side_effect=[first, second]) as batches, patch.object(
                              council, "query_model", new_callable=AsyncMock,
                              return_value={"content": "Final answer"}) as chairman, patch.object(
                                  council, "get_settings", return_value=settings):
            one, two, three, metadata = await council.run_full_council("Question")
        self.assertEqual(len(one), 2)
        self.assertEqual(len(two), 2)
        self.assertEqual(three["response"], "Final answer")
        self.assertEqual(metadata["label_to_model"], {"Response A": "test/beta", "Response B": "test/gamma"})
        self.assertEqual(metadata["aggregate_rankings"][0]["average_rank"], 1)
        self.assertIn("Response A (test/beta)", chairman.call_args.args[1][1]["content"])
        self.assertEqual(batches.await_count, 2)
        self.assertEqual(batches.await_args_list[1].args[0], ("test/beta", "test/gamma"))
        chairman.assert_awaited_once()

    async def test_all_failed_answers_stop_before_reviews_or_chairman(self):
        with patch.object(council, "query_models_parallel", new_callable=AsyncMock,
                          return_value={m: None for m in SETTINGS.models}) as query, patch.object(
                              council, "query_model", new_callable=AsyncMock) as chairman:
            with self.assertRaises(council.CouncilError):
                await council.run_full_council("Question")
            self.assertEqual(query.await_count, 1)
            chairman.assert_not_awaited()

    async def test_single_answer_stops_further_spending_but_remains_visible(self):
        failed = {"content": None, "error": "Token limit reached; generated tokens were billed.",
                  "usage": {"completion_tokens": 1024, "reasoning_tokens": 1024, "cost": 0.0000932},
                  "finish_reason": "length"}
        with patch.object(council, "query_models_parallel", new_callable=AsyncMock,
                          return_value={"test/alpha": {"content": "Available answer"}, "test/beta": failed}) as query, patch.object(
                              council, "query_model", new_callable=AsyncMock) as chairman:
            updates = [update async for update in app.answer_question("Question")]
        self.assertEqual(query.await_count, 1)
        chairman.assert_not_awaited()
        self.assertIn("Stopped before review", updates[-1][1])
        self.assertEqual(updates[-1][4], "Available answer")
        self.assertIn("Token limit reached", updates[-1][5])
        self.assertIn("cost=9.32e-05", updates[-1][-1])

    async def test_preflight_failure_makes_no_paid_requests(self):
        with patch.object(council, "validate_model_configuration", new_callable=AsyncMock,
                          side_effect=openrouter.ModelCheckError("This model does not support low reasoning")), patch.object(
                              council, "query_models_parallel", new_callable=AsyncMock) as query:
            updates = [update async for update in app.answer_question("Question")]
        self.assertIn("does not support low reasoning", updates[-1][1])
        query.assert_not_awaited()

    async def test_incomplete_duplicate_and_unknown_rankings_are_excluded(self):
        stage1 = [{"model": m, "response": "Answer"} for m in SETTINGS.models]
        for ballot in ["Response A is best", "FINAL RANKING:\n1. Response A",
                       "FINAL RANKING:\n1. Response A\n2. Response A",
                       "FINAL RANKING:\n1. Response A\n2. Response Z"]:
            with self.subTest(ballot=ballot), patch.object(council, "query_models_parallel",
                    new_callable=AsyncMock, return_value={"test/alpha": {"content": ballot}}):
                rankings, labels = await council.stage2_collect_rankings("Question", stage1)
                self.assertEqual(rankings[0]["parsed_ranking"], [])
                self.assertEqual(council.calculate_aggregate_rankings(rankings, labels), [])

    async def test_run_lock_released_on_close_and_failure(self):
        stream = council.iterate_council("Question")
        self.assertEqual((await anext(stream))["type"], "preflight_start")
        try:
            with self.assertRaisesRegex(council.CouncilError, "already in progress"):
                await anext(council.iterate_council("Second question"))
        finally:
            await stream.aclose()
        with patch.object(council, "stage1_collect_responses", new_callable=AsyncMock,
                          side_effect=RuntimeError("private provider details")):
            with self.assertRaises(RuntimeError):
                await council.run_full_council("Question")
        self.assertFalse(council._run_lock.locked())

    async def test_timeout_stops_run_and_releases_lock(self):
        async def slow(_, **kwargs):
            await asyncio.sleep(1)
        with patch.object(council, "RUN_TIMEOUT_SECONDS", 0.01), patch.object(
                council, "stage1_collect_responses", side_effect=slow):
            with self.assertRaisesRegex(council.CouncilError, "time limit"):
                await council.run_full_council("Question")
        self.assertFalse(council._run_lock.locked())

    async def test_ui_preserves_untrusted_output_as_text(self):
        untrusted = '<script>alert(1)</script> ![image](https://attacker.invalid/secret)'
        first = {m: {"content": untrusted} for m in SETTINGS.models}
        second = {m: {"content": BALLOT} for m in SETTINGS.models}
        with patch.object(council, "query_models_parallel", new_callable=AsyncMock,
                          side_effect=[first, second]), patch.object(
                              council, "query_model", new_callable=AsyncMock,
                              return_value={"content": untrusted}):
            updates = [update async for update in app.answer_question("Question")]
        self.assertEqual(updates[-1][0], "Complete.")
        self.assertEqual(updates[-1][1], untrusted)
        self.assertNotIn(SETTINGS.api_key, json.dumps(updates))

    async def test_ui_retains_answers_when_chairman_fails(self):
        first = {m: {"content": "Answer"} for m in SETTINGS.models}
        second = {m: {"content": BALLOT} for m in SETTINGS.models}
        with patch.object(council, "query_models_parallel", new_callable=AsyncMock,
                          side_effect=[first, second]), patch.object(
                              council, "query_model", new_callable=AsyncMock, return_value=None):
            updates = [update async for update in app.answer_question("Question")]
        self.assertIn("chairman could not respond", updates[-1][0])
        self.assertEqual(updates[-1][4], "Answer")


class OpenRouterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch("backend.openrouter.get_settings", return_value=SETTINGS).start()
        patch("backend.openrouter.validate_model_configuration", new_callable=AsyncMock, return_value=CATALOG).start()
        self.real_client = httpx.AsyncClient

    def mock_client(self, handler):
        transport = httpx.MockTransport(handler)
        return patch("backend.openrouter.httpx.AsyncClient",
                     side_effect=lambda **kwargs: self.real_client(transport=transport, **kwargs))

    async def test_fixed_destination_and_privacy_and_token_limits(self):
        def handler(request):
            self.assertEqual(str(request.url), OPENROUTER_API_URL)
            self.assertEqual(request.headers["authorization"], f"Bearer {SETTINGS.api_key}")
            payload = json.loads(request.content)
            self.assertNotIn(SETTINGS.api_key, request.content.decode())
            self.assertEqual(payload["max_tokens"], 1024)
            self.assertEqual(payload["provider"], {
                "data_collection": "deny", "zdr": True, "require_parameters": True})
            self.assertNotIn("tools", payload)
            self.assertFalse(payload["stream"])
            self.assertNotIn("reasoning", payload)
            return httpx.Response(200, json={"choices": [{"message": {"content": "Answer"}}]})
        with self.mock_client(handler) as factory:
            result = await openrouter.query_model("test/alpha", [{"role": "user", "content": "Question"}])
        self.assertEqual(result["content"], "Answer")
        self.assertFalse(factory.call_args.kwargs["trust_env"])
        self.assertFalse(factory.call_args.kwargs["follow_redirects"])

    async def test_redirect_not_followed_and_secrets_not_logged(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(307, headers={"location": "https://attacker.invalid/"},
                                  text=SETTINGS.api_key)
        with self.mock_client(handler), self.assertLogs("backend.openrouter") as logs:
            result = await openrouter.query_model("test/alpha", [])
        self.assertIn("HTTP 307", result["error"])
        self.assertEqual(len(calls), 1)
        self.assertNotIn(SETTINGS.api_key, str(logs.output))

    async def test_invalid_empty_and_oversized_responses_fail_closed(self):
        cases = [{}, {"choices": [{"message": {"content": None}}]},
                 {"choices": [{"message": {"content": "x" * 24_001}}]}]
        for data in cases:
            with self.subTest(data=str(data)[:30]), self.mock_client(lambda _: httpx.Response(200, json=data)):
                self.assertTrue((await openrouter.query_model("test/alpha", []))["error"])
        with patch.object(openrouter, "MAX_HTTP_RESPONSE_BYTES", 10), self.mock_client(
                lambda _: httpx.Response(200, content=b"x" * 11)):
            self.assertTrue((await openrouter.query_model("test/alpha", []))["error"])

    async def test_unconfigured_model_is_rejected_before_network_access(self):
        with patch("backend.openrouter.httpx.AsyncClient") as factory:
            with self.assertRaises(ValueError):
                await openrouter.query_model("attacker/expensive-model", [])
            factory.assert_not_called()

    async def test_low_reasoning_is_requested_for_optional_and_mandatory_models(self):
        for metadata in [
            {"mandatory": False},
            {"mandatory": True, "supported_efforts": ["high", "medium", "low"]},
            {"supported_efforts": ["high", "low", "none"]},
            {"supported_efforts": None},
            None,
        ]:
            info = {"supported_parameters": ["max_tokens", "reasoning"], "reasoning": metadata}
            def handler(request):
                self.assertEqual(json.loads(request.content)["reasoning"], {"effort": "low"})
                return httpx.Response(200, json={"choices": [{"message": {"content": "Answer"}}]})
            with self.subTest(metadata=metadata), self.mock_client(handler):
                result = await openrouter.query_model("test/alpha", [], model_info=info)
                self.assertEqual(result["content"], "Answer")

    async def test_models_without_low_effort_support_are_rejected_without_generation(self):
        for info in [
            {"supported_parameters": ["reasoning"], "reasoning": {"supported_efforts": ["high", "medium"]}},
            {"supported_parameters": ["reasoning"], "reasoning": {"supported_efforts": []}},
            {},
        ]:
            with self.subTest(info=info), patch("backend.openrouter.httpx.AsyncClient") as factory:
                with self.assertRaises(openrouter.ModelCheckError):
                    await openrouter.query_model("test/alpha", [], model_info=info)
                factory.assert_not_called()

    async def test_billed_reasoning_only_length_response_has_actionable_diagnostics(self):
        data = {"choices": [{"finish_reason": "length", "message": {
            "content": "", "reasoning": "PRIVATE MODEL REASONING"}}],
            "usage": {"prompt_tokens": 92, "completion_tokens": 1024, "cost": 0.0000932,
                      "completion_tokens_details": {"reasoning_tokens": 1024}, "secret": SETTINGS.api_key}}
        with self.mock_client(lambda _: httpx.Response(200, json=data)):
            result = await openrouter.query_model("test/alpha", [])
        self.assertIsNone(result["content"])
        self.assertIn("1,024-token", result["error"])
        self.assertIn("billed", result["error"])
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["usage"]["reasoning_tokens"], 1024)
        self.assertEqual(result["usage"]["cost"], 0.0000932)
        self.assertNotIn("PRIVATE MODEL REASONING", json.dumps(result))
        self.assertNotIn(SETTINGS.api_key, json.dumps(result))

    async def test_text_parts_and_truncated_visible_answers_are_preserved(self):
        data = {"choices": [{"finish_reason": "length", "message": {"content": [
            {"type": "text", "text": "First part"}, {"type": "output_text", "text": "Second part"},
            {"type": "reasoning", "text": "Not an answer"},
        ]}}]}
        with self.mock_client(lambda _: httpx.Response(200, json=data)):
            result = await openrouter.query_model("test/alpha", [])
        self.assertEqual(result["content"], "First part\nSecond part")
        self.assertIsNone(result["error"])
        self.assertTrue(result["warnings"])

    async def test_404_has_routing_and_privacy_explanation_without_echoing_body(self):
        with self.mock_client(lambda _: httpx.Response(404, json={"error": {"message": SETTINGS.api_key}})):
            result = await openrouter.query_model("test/alpha", [])
        self.assertIn("ZDR", result["error"])
        self.assertNotIn(SETTINGS.api_key, json.dumps(result))


class ModelCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def test_catalog_check_uses_only_public_get_without_key_or_prompt(self):
        real_client = httpx.AsyncClient
        calls = []
        def handler(request):
            calls.append(request)
            self.assertEqual(request.method, "GET")
            self.assertEqual(str(request.url), openrouter.OPENROUTER_MODELS_URL)
            self.assertNotIn("authorization", request.headers)
            self.assertEqual(request.content, b"")
            return httpx.Response(200, json={"data": list(CATALOG.values())})
        with patch("backend.openrouter.get_settings", return_value=SETTINGS), patch(
            "backend.openrouter.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(
                transport=httpx.MockTransport(handler), **kwargs)):
            catalog = await openrouter.validate_model_configuration()
        self.assertEqual(set(catalog), set(SETTINGS.models))
        self.assertEqual(len(calls), 1)

    async def test_unknown_models_and_unsupported_efforts_fail_preflight(self):
        real_client = httpx.AsyncClient
        for rows in [[], [{"id": model, "supported_parameters": ["reasoning"],
                           "reasoning": {"supported_efforts": ["high"]}} for model in SETTINGS.models]]:
            with self.subTest(rows=rows), patch("backend.openrouter.get_settings", return_value=SETTINGS), patch(
                "backend.openrouter.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(
                    transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": rows})), **kwargs)):
                with self.assertRaisesRegex(openrouter.ModelCheckError, "before any paid requests"):
                    await openrouter.validate_model_configuration()

    async def test_mandatory_reasoning_with_low_effort_passes_preflight(self):
        real_client = httpx.AsyncClient
        rows = [{"id": model, "supported_parameters": ["reasoning"],
                 "reasoning": {"mandatory": True, "supported_efforts": ["high", "low"]}}
                for model in SETTINGS.models]
        with patch("backend.openrouter.get_settings", return_value=SETTINGS), patch(
            "backend.openrouter.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": rows})), **kwargs)):
            catalog = await openrouter.validate_model_configuration()
        self.assertEqual(set(catalog), set(SETTINGS.models))


class LocalAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password = "test-only-local-password"
        with patch("app.get_settings", return_value=SETTINGS):
            cls.demo = app.create_app()
        cls.options = app.launch_options(SETTINGS, cls.password)
        cls.demo.auth = [("council", cls.password)]
        cls.demo.auth_message = "Local login"
        cls.demo.run_history = False
        cls.demo.config = cls.demo.get_config_file()
        cls.asgi_app = App.create_app(cls.demo, app_kwargs=cls.options["app_kwargs"], strict_cors=True)

    def setUp(self):
        self.client = TestClient(self.asgi_app, base_url=f"http://127.0.0.1:{SETTINGS.port}")
        self.addCleanup(self.client.close)

    def login(self):
        response = self.client.post("/login", data={"username": "council", "password": self.password})
        self.assertEqual(response.status_code, 200)

    def test_authentication_protects_configuration_and_queue(self):
        self.assertEqual(self.client.get("/config").status_code, 401)
        self.assertEqual(self.client.post("/gradio_api/queue/join", json={}).status_code, 401)
        self.assertEqual(self.client.post("/login", data={"username": "council", "password": "wrong"}).status_code, 400)
        self.login()
        response = self.client.get("/config")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(SETTINGS.api_key, response.text)
        self.assertNotIn(self.password, response.text)

    def test_foreign_origins_and_dns_rebinding_are_blocked(self):
        for origin in ["https://attacker.invalid", "null", "http://localhost:9999"]:
            with self.subTest(origin=origin):
                self.assertEqual(self.client.post("/login", headers={"Origin": origin},
                                                  data={"username": "council", "password": self.password}).status_code, 403)
        self.assertEqual(self.client.get("/", headers={"Host": "attacker.invalid"}).status_code, 400)

    def test_upload_file_and_proxy_routes_are_disabled_even_after_login(self):
        self.login()
        for path in ["/gradio_api/file=/etc/passwd", "/gradio_api/proxy=https://example.com"]:
            self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.post("/gradio_api/upload", content=b"test").status_code, 404)

    def test_oversized_body_is_blocked(self):
        self.assertEqual(self.client.post("/login", content=b"x" * 65_537).status_code, 413)

    def test_ui_and_launch_defaults_do_not_expose_secrets_or_render_model_html(self):
        self.assertEqual(self.options["server_name"], "127.0.0.1")
        for flag in ["share", "run_history", "ssr_mode", "mcp_server", "pwa"]:
            self.assertFalse(self.options[flag])
        self.assertEqual(self.options["blocked_paths"], ["/"])
        self.assertEqual(self.options["allowed_paths"], [])
        self.assertFalse(self.demo.analytics_enabled)
        config = self.demo.get_config_file()
        for component in config["components"]:
            if component["type"] == "markdown":
                self.assertTrue(component["props"]["sanitize_html"])
        self.assertTrue(all(fn.api_visibility == "private" for fn in self.demo.fns.values()))


if __name__ == "__main__":
    unittest.main()
