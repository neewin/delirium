"""
Tests for Module 4 (mocked NLI + fake embedder: no model downloads).
Run from project root:  python -m unittest discover tests -v
Optional real-model test: python tests/test_hallucination_detector.py --live
"""
import os
import re
import sys
import unittest
import zlib
from unittest.mock import patch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # project root
sys.path.insert(0, HERE)
try:
    from hallucination_detector import HallucinationDetector, extract_claims, split_passage
except ModuleNotFoundError:                  # Module4/ folder layout
    from hallucination_detector import HallucinationDetector, extract_claims, split_passage


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words are similar."""
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False):
        vecs = np.zeros((len(texts), 256), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9%]+", t.lower()):
                vecs[i, zlib.crc32(w.encode()) % 256] += 1
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms


def S(e, c, n):
    return {"entailment": e, "contradiction": c, "neutral": n, "truncated": False}


def fake_nli(pairs):
    """Scripted NLI decided by words in the premise/hypothesis."""
    out = []
    for premise, hypothesis in pairs:
        if "contradicts" in premise:
            out.append(S(0.02, 0.95, 0.03))
        elif "labelled" in hypothesis and "labelled" in premise:
            out.append(S(0.95, 0.02, 0.03))
        else:
            out.append(S(0.05, 0.05, 0.90))
    return out


CHUNK = {"chunk_id": 12, "source": "ml.pdf", "page": 4,
         "text": "Supervised learning uses labelled training data.", "score": 0.87}
CLAIM = "Supervised learning uses labelled data."


def make():
    d = HallucinationDetector(embedder=FakeEmbedder())
    d._predict_batch = fake_nli
    return d


class TestDetector(unittest.TestCase):
    def test_empty_answer(self):
        r = make().verify_answer("Q?", "", [CHUNK])
        self.assertEqual(r["summary"]["total_claims"], 0)
        self.assertIsNone(r["summary"]["verification_percentage"])

    def test_empty_evidence(self):
        r = make().verify_answer("Q?", CLAIM, [])
        self.assertEqual(r["claims"][0]["status"], "not_supported")
        self.assertEqual(r["claims"][0]["reason"], "insufficient_evidence")

    def test_single_supported_claim(self):
        r = make().verify_answer("Q?", CLAIM, [CHUNK])
        self.assertEqual(r["claims"][0]["status"], "supported")
        self.assertEqual(r["summary"]["overall_status"], "verified")

    def test_multiple_claims_percentage_and_status(self):
        r = make().verify_answer("Q?", CLAIM + " It always achieves 100% accuracy.", [CHUNK])
        s = r["summary"]
        self.assertEqual((s["total_claims"], s["supported"], s["not_supported"]), (2, 1, 1))
        self.assertEqual(s["verification_percentage"], 50.0)
        self.assertEqual(s["overall_status"], "partially_verified")
        self.assertEqual(r["claims"][1]["reason"], "no_relevant_sentence")

    def test_contradicted(self):
        bad = dict(CHUNK, text="Supervised learning needs labelled data; this contradicts the claim.")
        r = make().verify_answer("Q?", "Supervised learning needs no data at all.", [bad])
        self.assertEqual(r["claims"][0]["status"], "contradicted")
        self.assertEqual(r["summary"]["overall_status"], "unverified")

    def test_unsupported(self):
        r = make().verify_answer("Q?", "The method was invented in Mars colonies.", [CHUNK])
        self.assertEqual(r["claims"][0]["status"], "not_supported")

    def test_conflicting_evidence(self):
        bad = dict(CHUNK, chunk_id=13, text="Supervised learning uses labelled data, but this contradicts it.")
        r = make().verify_answer("Q?", CLAIM, [CHUNK, bad])
        self.assertEqual(r["claims"][0]["status"], "conflicting_evidence")
        self.assertEqual(len(r["claims"][0]["evidence"]), 2)

    def test_offtopic_passage_is_ignored(self):
        offtopic = dict(CHUNK, chunk_id=14, text="Report structure contradicts everything entirely.")
        r = make().verify_answer("Q?", CLAIM, [CHUNK, offtopic])
        self.assertEqual(r["claims"][0]["status"], "supported")
        self.assertEqual(len(r["claims"][0]["evidence"]), 1)

    def test_missing_metadata(self):
        r = make().verify_answer("Q?", CLAIM, [{"text": "Supervised learning uses labelled data."}])
        self.assertEqual(r["claims"][0]["evidence"][0]["source"], "unknown")

    def test_invalid_input_types(self):
        self.assertEqual(make().verify_answer("Q?", None, [CHUNK])["summary"]["overall_status"],
                         "verification_failed")
        self.assertEqual(make().verify_answer("Q?", "Some answer text here.", "x")["summary"]["overall_status"],
                         "verification_failed")

    def test_nli_model_load_failure(self):
        d = HallucinationDetector(embedder=FakeEmbedder())
        with patch.object(HallucinationDetector, "load_model", side_effect=RuntimeError("no model")):
            r = d.verify_answer("Q?", CLAIM, [CHUNK])
        self.assertEqual(r["summary"]["overall_status"], "verification_failed")
        self.assertIn("no model", r["error"])
        self.assertEqual(r["claims"], [])

    def test_embedder_failure(self):
        d = make()
        d.embedder = type("Bad", (), {"encode": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("embed boom"))})()
        r = d.verify_answer("Q?", CLAIM, [CHUNK])
        self.assertEqual(r["summary"]["overall_status"], "verification_failed")

    def test_metadata_preserved(self):
        r = make().verify_answer("Q?", CLAIM, [CHUNK])
        ev = r["claims"][0]["evidence"][0]
        self.assertEqual((ev["source"], ev["page"], ev["chunk_id"]), ("ml.pdf", 4, 12))
        self.assertIn("labelled training data", ev["matched_sentence"])

    def test_bullet_items_get_lead_in(self):
        text = ("Types such as \u25e6a formal report of facts "
                "\u25e6a preliminary report to attorney \u25e6an examination plan for counsel")
        units = split_passage(text)
        self.assertEqual(len(units), 3)
        self.assertTrue(all(u["standalone"] for u in units))
        self.assertEqual(units[1]["text"], "Types such as a preliminary report to attorney")

    def test_list_item_used_with_lead_in_not_siblings(self):
        chunk = dict(CHUNK, text="Learning methods include \u25e6supervised learning uses labelled data "
                                 "\u25e6unsupervised learning has none")
        r = make().verify_answer("Q?", CLAIM, [chunk])
        ev = r["claims"][0]["evidence"]
        self.assertEqual(r["claims"][0]["status"], "supported")
        self.assertEqual(len(ev), 1)
        self.assertTrue(ev[0]["text"].startswith("Learning methods include"))
        self.assertNotIn("unsupervised", ev[0]["text"])

    def test_claim_extraction_skips_non_claims(self):
        text = "Sure! Here is the answer:\n- Supervised learning uses labelled data.\n- It is used for classification tasks.\nWhat else?"
        claims = [c["claim"] for c in extract_claims(text)]
        self.assertEqual(claims, ["Supervised learning uses labelled data.",
                                  "It is used for classification tasks."])


def run_live():
    d = HallucinationDetector()
    r = d.verify_answer("What is supervised learning?",
                        "Supervised learning uses labelled training data. It always achieves 100% accuracy.",
                        [CHUNK])
    for c in r["claims"]:
        print(c["status"], "|", c["claim"], "|", c.get("reason", ""))
    print(r["summary"], r["error"])


if __name__ == "__main__":
    if "--live" in sys.argv:
        run_live()
    else:
        unittest.main(argv=[sys.argv[0], "-v"])