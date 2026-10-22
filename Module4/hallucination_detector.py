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

import os
import re
import threading
from typing import Dict, List, Optional

import numpy as np

try:                                  # lets NLI_MODEL be set in the project's .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # same model as Module 2
ENTAILMENT_THRESHOLD = 0.80             # starting values; calibrate on labelled data
CONTRADICTION_THRESHOLD = 0.80
MIN_SIMILARITY = 0.35                   # a sentence must be at least this similar to a claim
TOP_K_SENTENCES = 3                     # max evidence sentences checked per claim
CONTEXT_WINDOW = 1                      # neighbouring sentences added to each premise
CONFLICT_MARGIN = 0.10                  # when passages disagree, the clearly more relevant side wins
MAX_LENGTH = 512                        # NLI model input limit (tokens)
BATCH_SIZE = 8

def configured_model_name() -> str:
    """NLI model name: NLI_MODEL from .env / environment, else the default."""
    return os.getenv("NLI_MODEL") or NLI_MODEL_NAME


# Statuses
SUPPORTED = "supported"
CONTRADICTED = "contradicted"
NOT_SUPPORTED = "not_supported"
CONFLICTING = "conflicting_evidence"

_BULLETS = "\u25E6\u2022\u25CF\u25AA\u2023\u25A0\u2043"   # bullet characters in PDFs
# A list entry such as "Android: Based on the Linux kernel..." stands on its own.
_LABEL_ITEM = re.compile(r"^[A-Z0-9][^:.!?]{0,60}:\s+\S")
# Words that let a single list entry rule something out ("... are not stored").
_NEGATION = re.compile(r"\b(not|no|never|cannot|can't|without|only|none|neither|nor)\b", re.I)
_SKIP_START = re.compile(
    r"^(hi|hello|hey|sure|certainly|of course|thanks|thank you|i hope|"
    r"let me know|here is|here's|based on|according to the)\b", re.I)
_SKIP_PHRASES = ("cannot be determined", "not enough information",
                 "do not contain", "does not contain", "insufficient")


# ---------------------------------------------------------------------------
# Acronym normalisation (used only for NLI and similarity, never for display)
# ---------------------------------------------------------------------------
_CONNECTORS = {"of", "the", "and", "for", "to", "in", "on", "a", "an"}
_LONG_THEN_ACRONYM = re.compile(r"\(([A-Z][A-Z0-9]{1,7})(s?)\)")
_ACRONYM_THEN_LONG = re.compile(r"\b([A-Z][A-Z0-9]{1,7})(s?)\s+\(([A-Za-z][^()]{3,80})\)")


def _initials(words: List[str]) -> str:
    """Upper-case initials of words, ignoring small connector words."""
    return "".join(w[0].upper() for w in words if w.lower() not in _CONNECTORS)


def normalize_acronyms(text: str) -> str:
    """Replace 'long form (ACR)' and 'ACR (long form)' with just 'ACR'.

    Example: 'subscriber identity module (SIM) cards' -> 'SIM cards'.
    The expansion is replaced only when the words' initials really spell the
    acronym, so ordinary parentheses are left alone. NLI models treat an
    expanded acronym as new information, which lowers entailment scores.
    """
    if not text:
        return text

    # Pattern 1: long form followed by (ACR). Work right-to-left so indices stay valid.
    for m in reversed(list(_LONG_THEN_ACRONYM.finditer(text))):
        acr, plural = m.group(1), m.group(2)
        words = [(w.group(), w.start()) for w in re.finditer(r"[A-Za-z][A-Za-z0-9-]*", text[:m.start()])]
        for k in range(len(acr), len(acr) + 4):          # allow a few connector words
            if k > len(words):
                break
            last = words[-k:]
            if last[0][0].lower() in _CONNECTORS or _initials([w for w, _ in last]) != acr:
                continue
            start = last[0][1]
            if re.search(r"[.!?;:]", text[start:m.start()]):
                continue                                  # do not cross a sentence boundary
            text = text[:start] + acr + plural + text[m.end():]
            break

    # Pattern 2: ACR followed by (long form).
    def _drop_long_form(m):
        acr, plural, inner = m.group(1), m.group(2), m.group(3)
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", inner)
        return acr + plural if _initials(words) == acr else m.group(0)

    return _ACRONYM_THEN_LONG.sub(_drop_long_form, text)


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
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(]|[a-z][A-Z])", line):
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
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(]|[a-z][A-Z])", part) if x.strip()]


def split_passage(text: str) -> List[Dict]:
    """Split a passage into evidence units: {"text": str, "standalone": bool}.

    Bullet lists (PDF bullet marks) are handled specially: each item is joined
    to the lead-in sentence before the list ("... required to create reports,
    such as" + "a formal report ..."), so every item is a complete statement
    and keeps its meaning. Such units are 'standalone': no neighbouring items
    are added to them later. 'Label: description' entries (e.g. "Android: Based on...")
    are standalone too, so a list of different items is never fed to the NLI model as
    one premise. Other sentences are ordinary units.
    """
    parts = re.split(f"[{_BULLETS}]", text)
    units: List[Dict] = []

    first = _sentences(parts[0])
    lead = ""
    if len(parts) > 1 and first:
        lead = first.pop()               # last fragment before the list = lead-in
    for sentence in first:
        if len(sentence.split()) >= 3:
            units.append({"text": sentence, "standalone": bool(_LABEL_ITEM.match(sentence)),
                          "list_item": False})

    for part in parts[1:]:
        sents = _sentences(part)
        if not sents:
            continue
        units.append({"text": f"{lead} {sents[0]}".strip(), "standalone": True,
                      "list_item": True})
        for sentence in sents[1:]:       # text that follows the last bullet
            if len(sentence.split()) >= 3:
                units.append({"text": sentence, "standalone": bool(_LABEL_ITEM.match(sentence)),
                          "list_item": False})
    return units


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class HallucinationDetector:
    """Verifies answer claims against evidence sentences using NLI."""

    def __init__(self, model_name: Optional[str] = None,
                 entailment_threshold: float = ENTAILMENT_THRESHOLD,
                 contradiction_threshold: float = CONTRADICTION_THRESHOLD,
                 min_similarity: float = MIN_SIMILARITY,
                 top_k: int = TOP_K_SENTENCES,
                 context_window: int = CONTEXT_WINDOW,
                 conflict_margin: float = CONFLICT_MARGIN,
                 embedder=None):
        # Priority: explicit argument, then NLI_MODEL in .env / environment, then the default.
        self.model_name = model_name or configured_model_name()
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.context_window = context_window
        self.conflict_margin = conflict_margin
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
            # Run in float32. Some checkpoints load in half precision (float16), which can
            # give unreliable, noisy scores on a CPU.
            self.model = self.model.float()
            self.model.eval()
        except Exception as exc:
            self.model = None
            hint = ""
            if os.environ.get("HF_HUB_OFFLINE") == "1":
                hint = (" (offline mode is on: run `python download_models.py` once, "
                        "or set ALLOW_MODEL_DOWNLOAD=1 to allow downloads)")
            raise RuntimeError(f"Could not load NLI model '{self.model_name}': {exc}{hint}")

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
        return self._predict_batch([(normalize_acronyms(evidence_text), normalize_acronyms(claim))])[0]

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
                    "list_item": item.get("list_item", False),
                })
        return units

    # -- classification ------------------------------------------------------
    def classify_claim(self, claim: str, units: List[Dict],
                       similarities: Optional[np.ndarray]) -> Dict:
        """Classify one claim using its most similar evidence sentences.

        1. Semantic gate: keep up to top_k sentences with similarity >= min_similarity.
        2. NLI on each kept sentence (premise) vs the claim (hypothesis).
        3. Rules (T_e / T_c = thresholds):
           entailed >= T_e AND contradicted >= T_c -> the more relevant (similar) side wins
                                                      if clearly ahead, else conflicting_evidence
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

        scores = self._predict_batch([(normalize_acronyms(units[i]["window_text"]),
                                       normalize_acronyms(claim)) for i, _ in picked])
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
                "list_item": u["list_item"],
            })

        entail = [s for s in scored if s["entailment_score"] >= self.entailment_threshold]
        # One bullet entry only names ONE item; it does not exclude other items, so a
        # bullet entry can confirm a claim but cannot contradict it (unless it contains a negation).
        contra = [s for s in scored
                  if s["contradiction_score"] >= self.contradiction_threshold
                  and not (s["list_item"] and not _NEGATION.search(s["text"]))]

        ignored = 0
        if entail and contra:
            # Passages disagree: the clearly more relevant side wins; if they are
            # about equally relevant, report conflicting evidence.
            gap = (max(s["similarity"] for s in entail)
                   - max(s["similarity"] for s in contra))
            if gap >= self.conflict_margin:
                status, ignored = SUPPORTED, len(contra)
                used = sorted(entail, key=lambda s: -s["entailment_score"])
            elif -gap >= self.conflict_margin:
                status, ignored = CONTRADICTED, len(entail)
                used = sorted(contra, key=lambda s: -s["contradiction_score"])
            else:
                status, used = CONFLICTING, entail + contra
        elif entail:
            status, used = SUPPORTED, sorted(entail, key=lambda s: -s["entailment_score"])
        elif contra:
            status, used = CONTRADICTED, sorted(contra, key=lambda s: -s["contradiction_score"])
        else:
            # Show every candidate that was checked; none met a threshold.
            status, used = NOT_SUPPORTED, sorted(scored, key=lambda s: -s["entailment_score"])
        return {"claim": claim, "status": status, "evidence": used,
                "ignored_conflicts": ignored}

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
                unit_vecs = self._embed([normalize_acronyms(u["sentence"]) for u in units])
                claim_vecs = self._embed([normalize_acronyms(c["claim"]) for c in extracted])
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


# ---------------------------------------------------------------------------
# One shared detector, loadable in the background (used by the Streamlit app)
# ---------------------------------------------------------------------------
_shared: Optional[HallucinationDetector] = None
_load_lock = threading.Lock()      # held while the model loads
_start_lock = threading.Lock()     # protects the 'started' flag only (never blocks the UI)
_started = False


def get_shared_detector() -> HallucinationDetector:
    """Return the one shared detector, loading its model first if needed.
    If a background load is running, this waits for it instead of loading twice."""
    global _shared
    with _load_lock:
        if _shared is None:
            detector = HallucinationDetector()
            detector.load_model()
            _shared = detector
    return _shared


def preload_in_background(also_run=None) -> None:
    """Start loading the NLI model in a background thread (once per process).

    also_run: optional function run first in the same thread (e.g. importing the
    heavy retrieval libraries). Errors are not raised here; they appear, with
    their message, the first time the detector is actually needed.
    """
    global _started
    with _start_lock:
        if _started:
            return
        _started = True

    def work():
        try:
            if also_run:
                also_run()
        except Exception:
            pass
        try:
            get_shared_detector()
        except Exception:
            pass

    threading.Thread(target=work, daemon=True).start()