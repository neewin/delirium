"""
Tests for Module 3. Unit tests use mocks (no API key needed).
Run:  python -m unittest test_answer_generator -v
Optional live test (needs real .env): python test_answer_generator.py --live
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import httpx
from openai import APIConnectionError, AuthenticationError, RateLimitError

import answer_generator as ag

CHUNKS = [
    {"chunk_id": 12, "source": "ml.pdf", "page": 4,
     "text": "Supervised learning uses labelled data.", "score": 0.87},
    {"chunk_id": 15, "source": "ml.pdf", "page": 5,
     "text": "Classification is a supervised task.", "score": 0.82},
]
ENV = {"LLM_API_KEY": "k", "LLM_BASE_URL": "http://x", "LLM_MODEL": "m"}


def fake_client(content="Supervised learning uses labelled data."):
    """Build a mock client whose reply is `content`."""
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=content))
    ]
    return client


def _http_response(status):
    return httpx.Response(status, request=httpx.Request("POST", "http://x"))


class TestAnswerGenerator(unittest.TestCase):
    def test_empty_question(self):
        r = ag.generate_answer("   ", CHUNKS, client=fake_client())
        self.assertEqual(r["status"], ag.STATUS_EMPTY_QUESTION)

    def test_empty_context_makes_no_api_call(self):
        client = fake_client()
        r = ag.generate_answer("What is ML?", [], client=client)
        self.assertEqual(r["status"], ag.STATUS_NO_CONTEXT)
        client.chat.completions.create.assert_not_called()

    def test_chunks_with_empty_text(self):
        r = ag.generate_answer("Q?", [{"text": "  "}], client=fake_client())
        self.assertEqual(r["status"], ag.STATUS_NO_CONTEXT)

    def test_valid_question_and_context(self):
        r = ag.generate_answer("What is supervised learning?", CHUNKS, client=fake_client())
        self.assertEqual(r["status"], ag.STATUS_SUCCESS)
        self.assertIn("labelled", r["answer"])

    def test_multiple_passages_in_prompt_and_metadata_preserved(self):
        client = fake_client()
        r = ag.generate_answer("Q?", CHUNKS, client=client)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("[Passage 1]", prompt)
        self.assertIn("[Passage 2]", prompt)
        self.assertEqual([s["page"] for s in r["sources"]], [4, 5])
        self.assertEqual([s["chunk_id"] for s in r["sources"]], [12, 15])
        self.assertEqual(r["sources"][0]["source"], "ml.pdf")
        self.assertEqual(r["sources"][0]["score"], 0.87)

    def test_missing_metadata_does_not_crash(self):
        r = ag.generate_answer("Q?", [{"text": "Some text."}], client=fake_client())
        self.assertEqual(r["sources"][0]["source"], "unknown")

    def test_missing_credentials(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(ag, "_get_config",
                          return_value={"api_key": None, "base_url": None, "model": None}):
            r = ag.generate_answer("Q?", CHUNKS)
        self.assertEqual(r["status"], ag.STATUS_MISSING_CONFIG)

    def test_api_connection_failure(self):
        client = fake_client()
        client.chat.completions.create.side_effect = APIConnectionError(
            request=httpx.Request("POST", "http://x"))
        r = ag.generate_answer("Q?", CHUNKS, client=client)
        self.assertEqual(r["status"], ag.STATUS_API_ERROR)

    def test_auth_error(self):
        client = fake_client()
        client.chat.completions.create.side_effect = AuthenticationError(
            "bad key", response=_http_response(401), body=None)
        r = ag.generate_answer("Q?", CHUNKS, client=client)
        self.assertEqual(r["status"], ag.STATUS_AUTH_ERROR)

    def test_rate_limit(self):
        client = fake_client()
        client.chat.completions.create.side_effect = RateLimitError(
            "slow down", response=_http_response(429), body=None)
        r = ag.generate_answer("Q?", CHUNKS, client=client)
        self.assertEqual(r["status"], ag.STATUS_RATE_LIMIT)

    def test_empty_model_response(self):
        r = ag.generate_answer("Q?", CHUNKS, client=fake_client(content="  "))
        self.assertEqual(r["status"], ag.STATUS_INVALID_RESPONSE)

    def test_malformed_response(self):
        client = MagicMock()
        client.chat.completions.create.return_value.choices = []
        r = ag.generate_answer("Q?", CHUNKS, client=client)
        self.assertEqual(r["status"], ag.STATUS_INVALID_RESPONSE)

    def test_insufficient_marker(self):
        client = fake_client(content="INSUFFICIENT_EVIDENCE: The passages don't say.")
        r = ag.generate_answer("Q?", CHUNKS, client=client)
        self.assertEqual(r["status"], ag.STATUS_INSUFFICIENT)
        self.assertNotIn("INSUFFICIENT_EVIDENCE", r["answer"])


def run_live_test() -> None:
    """Optional integration test: needs real credentials in .env."""
    r = ag.generate_answer("What is supervised learning?", CHUNKS)
    print("Status:", r["status"])
    print("Answer:", r["answer"])
    print("Sources:", r["sources"])


if __name__ == "__main__":
    if "--live" in sys.argv:
        run_live_test()
    else:
        unittest.main(argv=[sys.argv[0], "-v"])
