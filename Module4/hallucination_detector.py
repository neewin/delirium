"""
Module 4: Hallucination Detection and Answer Verification (sentence-level evidence).

Splits an LLM answer into claims. For each claim it finds the most similar
SENTENCES in the retrieved passages (semantic gate), then runs Natural
Language Inference (NLI) on those sentences:
    premise    = evidence sentence (with neighbouring context)
    hypothesis = claim from the generated answer
Similarity is used only to pick candidate evidence, never to decide the
verdict. Results are automated estimates, not ground truth.
"""

import re
from typing import Dict, List, Optional

import numpy as np

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # same model as Module 2
ENTAILMENT_THRESHOLD = 0.80             # starting values; calibrate on labelled data
CONTRADICTION_THRESHOLD = 0.80
MIN_SIMILARITY = 0.35                   # a sentence must be at least this similar to a claim
TOP_K_SENTENCES = 3                     # max evidence sentences checked per claim
CONTEXT_WINDOW = 1                      # neighbouring sentences added to each premise
MAX_LENGTH = 512                        # NLI model input limit (tokens)
BATCH_SIZE = 8

# Statuses
SUPPORTED = "supported"
CONTRADICTED = "contradicted"
NOT_SUPPORTED = "not_supported"
CONFLICTING = "conflicting_evidence"

_BULLETS = "\u25E6\u2022\u25CF\u25AA\u2023\u25A0\u2043"   # bullet characters in PDFs
_SKIP_START = re.compile(
    r"^(hi|hello|hey|sure|certainly|of course|thanks|thank you|i hope|"
    r"let me know|here is|here's|based on|according to the)\b", re.I)
_SKIP_PHRASES = ("cannot be determined", "not enough information",
                 "do not contain", "does not contain", "insufficient")


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------
def extract_claims(answer: str) -> List[Dict]:
    """Split an answer into sentence-level factual claims.

    Returns dicts: {"claim": str, "compound": bool}. 'compound' flags
    sentences that may hold several assertions (not split further).
    """
    if not isinstance(answer, str) or not answer.strip():
        return []

    claims = []
    for line in answer.splitlines():
        line = re.sub(r"^\s*([-*\u2022]|\d+[.)])\s+", "", line)  # bullets
        line = line.replace("**", "").replace("__", "").strip()
        if not line:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", line):
            s = sentence.strip()
            if len(s.split()) < 4 or s.endswith("?") or s.endswith(":"):
                continue
            if _SKIP_START.match(s) or any(p in s.lower() for p in _SKIP_PHRASES):
                continue
            compound = ";" in s or ", and " in s.lower()
            claims.append({"claim": s, "compound": compound})
    return claims


def _sentences(part: str) -> List[str]:
    """Split text into stripped, non-empty sentences."""
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", part) if x.strip()]


def split_passage(text: str) -> List[Dict]:
    """Split a passage into evidence units: {"text": str, "standalone": bool}.

    Bullet lists (PDF bullet marks) are handled specially: each item is joined
    to the lead-in sentence before the list ("... required to create reports,
    such as" + "a formal report ..."), so every item is a complete statement
    and keeps its meaning. Such units are 'standalone': no neighbouring items
    are added to them later. Other sentences are ordinary units.
    """
    parts = re.split(f"[{_BULLETS}]", text)
    units: List[Dict] = []

    first = _sentences(parts[0])
    lead = ""
    if len(parts) > 1 and first:
        lead = first.pop()               # last fragment before the list = lead-in
    for sentence in first:
        if len(sentence.split()) >= 3:
            units.append({"text": sentence, "standalone": False})

    for part in parts[1:]:
        sents = _sentences(part)
        if not sents:
            continue
        units.append({"text": f"{lead} {sents[0]}".strip(), "standalone": True})
        for sentence in sents[1:]:       # text that follows the last bullet
            if len(sentence.split()) >= 3:
                units.append({"text": sentence, "standalone": False})
    return units


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class HallucinationDetector:
    """Verifies answer claims against evidence sentences using NLI."""

    def __init__(self, model_name: str = NLI_MODEL_NAME,
                 entailment_threshold: float = ENTAILMENT_THRESHOLD,
                 contradiction_threshold: float = CONTRADICTION_THRESHOLD,
                 min_similarity: float = MIN_SIMILARITY,
                 top_k: int = TOP_K_SENTENCES,
                 context_window: int = CONTEXT_WINDOW,
                 embedder=None):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.context_window = context_window
        self.embedder = embedder          # a SentenceTransformer (e.g. retriever.model)
        self.tokenizer = None
        self.model = None
        self.label_index: Dict[str, int] = {}

    # -- NLI model -------------------------------------------------------
    def load_model(self) -> None:
        """Load the NLI model once (CPU) and map its labels from its config."""
        if self.model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as exc:
            self.model = None
            raise RuntimeError(f"Could not load NLI model '{self.model_name}': {exc}")

        # Never assume label order: read it from the model config.
        id2label = {int(i): str(l).lower() for i, l in self.model.config.id2label.items()}
        for idx, label in id2label.items():
            for key in ("entail", "contradict", "neutral"):
                if key in label:
                    self.label_index[key] = idx
        if len(self.label_index) != 3:
            self.model = None
            raise RuntimeError(f"Unrecognised NLI labels {id2label}; expected entailment/contradiction/neutral.")

    def _predict_batch(self, pairs: List[tuple]) -> List[Dict]:
        """Run NLI on (premise, hypothesis) pairs. Returns scores per pair."""
        self.load_model()
        import torch
        results = []
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[start:start + BATCH_SIZE]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]
            full = self.tokenizer(premises, hypotheses, truncation=False)["input_ids"]
            enc = self.tokenizer(premises, hypotheses, truncation="only_first",
                                 max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            with torch.inference_mode():
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
            for row, ids in zip(probs, full):
                results.append({
                    "entailment": float(row[self.label_index["entail"]]),
                    "contradiction": float(row[self.label_index["contradict"]]),
                    "neutral": float(row[self.label_index["neutral"]]),
                    "truncated": len(ids) > MAX_LENGTH,
                })
        return results

    def compare_claim_with_evidence(self, claim: str, evidence_text: str) -> Dict:
        """NLI scores for one claim vs one text (premise=evidence, hypothesis=claim)."""
        return self._predict_batch([(evidence_text, claim)])[0]

    # -- embeddings --------------------------------------------------------------
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Unit-length embeddings, so a dot product equals cosine similarity."""
        if self.embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
            except Exception as exc:
                raise RuntimeError(f"Could not load embedding model: {exc}")
        vectors = self.embedder.encode(texts, convert_to_numpy=True,
                                       normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    # -- evidence ---------------------------------------------------------
    @staticmethod
    def prepare_evidence(retrieved_chunks: List[Dict]) -> List[Dict]:
        """Keep usable passages with their original metadata."""
        evidence = []
        for c in retrieved_chunks or []:
            if not isinstance(c, dict) or not str(c.get("text", "")).strip():
                continue
            evidence.append({
                "chunk_id": c.get("chunk_id", "unknown"),
                "source": c.get("source", "unknown"),
                "page": c.get("page", "unknown"),
                "text": str(c["text"]).strip(),
                "score": c.get("score"),   # retrieval score; NOT proof of truth
            })
        return evidence

    def build_units(self, evidence: List[Dict]) -> List[Dict]:
        """Break passages into evidence units. Each unit keeps its chunk metadata and a
        'window_text' (the NLI premise): a bullet item with its lead-in, or an
        ordinary sentence plus neighbouring ordinary sentences."""
        units = []
        w = self.context_window
        for e in evidence:
            items = split_passage(e["text"]) or [{"text": e["text"], "standalone": True}]
            for k, item in enumerate(items):
                if item["standalone"]:
                    window = item["text"]
                else:
                    window = " ".join(items[m]["text"]
                                      for m in range(max(0, k - w), min(len(items), k + w + 1))
                                      if not items[m]["standalone"])
                units.append({
                    "source": e["source"], "page": e["page"], "chunk_id": e["chunk_id"],
                    "sentence": item["text"], "window_text": window,
                })
        return units

    # -- classification ------------------------------------------------------
    def classify_claim(self, claim: str, units: List[Dict],
                       similarities: Optional[np.ndarray]) -> Dict:
        """Classify one claim using its most similar evidence sentences.

        1. Semantic gate: keep up to top_k sentences with similarity >= min_similarity.
        2. NLI on each kept sentence (premise) vs the claim (hypothesis).
        3. Rules (T_e / T_c = thresholds):
           entailed >= T_e AND contradicted >= T_c -> conflicting_evidence
           entailed >= T_e                         -> supported
           contradicted >= T_c                     -> contradicted
           otherwise                               -> not_supported
        """
        if not units or similarities is None:
            return {"claim": claim, "status": NOT_SUPPORTED,
                    "reason": "insufficient_evidence", "evidence": []}

        order = np.argsort(-similarities)[: self.top_k]
        picked = [(int(i), float(similarities[i])) for i in order
                  if similarities[i] >= self.min_similarity]
        if not picked:
            return {"claim": claim, "status": NOT_SUPPORTED,
                    "reason": "no_relevant_sentence", "evidence": []}

        scores = self._predict_batch([(units[i]["window_text"], claim) for i, _ in picked])
        scored = []
        for (i, sim), s in zip(picked, scores):
            u = units[i]
            scored.append({
                "source": u["source"], "page": u["page"], "chunk_id": u["chunk_id"],
                "text": u["window_text"], "matched_sentence": u["sentence"],
                "similarity": round(sim, 2),
                "entailment_score": s["entailment"],
                "contradiction_score": s["contradiction"],
                "neutral_score": s["neutral"],
                "truncated": s.get("truncated", False),
            })

        entail = [s for s in scored if s["entailment_score"] >= self.entailment_threshold]
        contra = [s for s in scored if s["contradiction_score"] >= self.contradiction_threshold]

        if entail and contra:
            status, used = CONFLICTING, entail + contra
        elif entail:
            status, used = SUPPORTED, sorted(entail, key=lambda s: -s["entailment_score"])
        elif contra:
            status, used = CONTRADICTED, sorted(contra, key=lambda s: -s["contradiction_score"])
        else:
            # Show every candidate that was checked; none met a threshold.
            status, used = NOT_SUPPORTED, sorted(scored, key=lambda s: -s["entailment_score"])
        return {"claim": claim, "status": status, "evidence": used}

    # -- summary ------------------------------------------------------------------
    @staticmethod
    def summarize(claims: List[Dict]) -> Dict:
        """Count statuses and compute the verification percentage."""
        total = len(claims)
        count = lambda s: sum(1 for c in claims if c["status"] == s)
        supported = count(SUPPORTED)
        if total == 0:
            overall, pct = "unverified", None
        else:
            pct = round(supported / total * 100, 1)
            if supported == total:
                overall = "verified"
            elif supported > 0:
                overall = "partially_verified"
            else:
                overall = "unverified"
        return {
            "total_claims": total, "supported": supported,
            "contradicted": count(CONTRADICTED), "not_supported": count(NOT_SUPPORTED),
            "conflicting_evidence": count(CONFLICTING),
            "verification_percentage": pct, "overall_status": overall,
        }

    # -- main --------------------------------------------------------------------
    def verify_answer(self, question: str, answer: str,
                      retrieved_chunks: List[Dict]) -> Dict:
        """Verify a Module 3 answer against Module 2 passages.

        The percentage = supported claims / evaluated claims x 100. It is NOT
        the probability that the answer is correct, and 'not_supported' does
        not mean false (the document may simply not mention it).
        """
        report = {"question": question, "answer": answer, "claims": [],
                  "summary": self.summarize([]), "error": None}

        if not isinstance(answer, str) or not isinstance(retrieved_chunks, (list, tuple)) \
                or not isinstance(question, str):
            return self._failed(report, "Invalid input: question/answer must be text and chunks a list.")

        extracted = extract_claims(answer)
        units = self.build_units(self.prepare_evidence(list(retrieved_chunks)))

        try:
            sims = None
            if extracted and units:
                unit_vecs = self._embed([u["sentence"] for u in units])
                claim_vecs = self._embed([c["claim"] for c in extracted])
                sims = claim_vecs @ unit_vecs.T      # cosine similarities
            for k, item in enumerate(extracted):
                result = self.classify_claim(item["claim"], units,
                                             None if sims is None else sims[k])
                result["compound"] = item["compound"]
                report["claims"].append(result)
        except Exception as exc:  # model load/inference failure: never fake a verdict
            return self._failed(report, f"Verification could not be completed: {exc}")

        report["summary"] = self.summarize(report["claims"])
        return report

    @staticmethod
    def _failed(report: Dict, message: str) -> Dict:
        report["claims"] = []
        report["summary"] = {"total_claims": 0, "supported": 0, "contradicted": 0,
                             "not_supported": 0, "conflicting_evidence": 0,
                             "verification_percentage": None,
                             "overall_status": "verification_failed"}
        report["error"] = message
        return report