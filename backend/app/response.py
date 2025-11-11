"""
Barebones RAG utilities for the debate tutor backend.

Goals:
- Keep it dependency-light and testable from a REPL or a quick script.
- Offer a TF‑IDF retriever (scikit‑learn if available; otherwise a tiny BOW+cosine fallback).
- Provide a simple, pluggable LLM call (OpenAI if env key present; otherwise a deterministic stub).
- Expose one main entrypoint: `answer(query: str, top_k: int = 4)`.

Usage (quick test):
    python -m app.response

You can also import `SimpleRAG` and feed your own docs.
"""
from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterable
from dotenv import load_dotenv
# Optional heavy deps
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    _HAS_SK = True
except Exception:
    _HAS_SK = False
load_dotenv()

# --- OpenAI client (modern SDK ≥ 1.0) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("Client object:", type(client))
    except Exception:
        client = None
        print("No Client!")


# ----------------------------
# Data structures & utilities
# ----------------------------

@dataclass
class Doc:
    id: str
    text: str
    meta: Optional[Dict] = None


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 150) -> List[str]:
    """Simple word-based chunking to keep snippets coherent.
    - chunk_size/overlap are in *words*.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
        i += max(1, chunk_size - overlap)
    return chunks


# ----------------------------
# Vectorization / Retrieval
# ----------------------------

class _FallbackVectorizer:
    """Tiny Bag-of-Words TF‑IDF-ish vectorizer with cosine.
    Only used if scikit-learn isn't available. Not production quality.
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self.docs: List[List[Tuple[int, float]]] = []
        self._fitted = False

    def _tokenize(self, s: str) -> List[str]:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return [t for t in s.split() if t]

    def fit_transform(self, texts: List[str]):
        # Build vocab
        df = {}
        tokenized = []
        for t in texts:
            toks = set(self._tokenize(t))
            tokenized.append(list(toks))
            for tok in toks:
                df[tok] = df.get(tok, 0) + 1
        self.vocab = {tok: i for i, tok in enumerate(sorted(df))}
        N = len(texts)
        # IDF
        self.idf = [math.log((N + 1) / (df.get(tok, 0) + 1)) + 1.0 for tok in sorted(df)]
        # TF‑IDF rows (sparse-ish as list of (idx, val))
        mat = []
        for t in texts:
            toks = self._tokenize(t)
            tf = {}
            for tok in toks:
                if tok in self.vocab:
                    tf[tok] = tf.get(tok, 0) + 1
            row = []
            norm = 0.0
            for tok, cnt in tf.items():
                j = self.vocab[tok]
                val = (cnt / len(toks)) * self.idf[j]
                norm += val * val
                row.append((j, val))
            norm = math.sqrt(norm) or 1.0
            row = [(j, v / norm) for (j, v) in row]
            mat.append(row)
        self.docs = mat
        self._fitted = True
        return mat

    def transform(self, texts: List[str]):
        assert self._fitted
        rows = []
        for t in texts:
            toks = self._tokenize(t)
            tf = {}
            for tok in toks:
                if tok in self.vocab:
                    tf[tok] = tf.get(tok, 0) + 1
            row = []
            norm = 0.0
            for tok, cnt in tf.items():
                j = self.vocab[tok]
                val = (cnt / len(toks)) * self.idf[j]
                norm += val * val
                row.append((j, val))
            norm = math.sqrt(norm) or 1.0
            row = [(j, v / norm) for (j, v) in row]
            rows.append(row)
        return rows

    @staticmethod
    def cosine(a: List[Tuple[int, float]], b: List[Tuple[int, float]]) -> float:
        i = j = 0
        dot = 0.0
        while i < len(a) and j < len(b):
            ia, va = a[i]
            ib, vb = b[j]
            if ia == ib:
                dot += va * vb
                i += 1
                j += 1
            elif ia < ib:
                i += 1
            else:
                j += 1
        # L2 norms are already normalized to 1.0 in fit/transform
        return float(dot)


class TFIDFRetriever:
    def __init__(self, docs: List[Doc]):
        self.docs = docs
        self._fit()

    def _fit(self):
        self.corpus = [d.text for d in self.docs]
        if _HAS_SK:
            self.vectorizer = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 2), stop_words="english")
            self.mat = self.vectorizer.fit_transform(self.corpus)
            self._fallback = None
        else:
            self.vectorizer = _FallbackVectorizer()
            self.mat = self.vectorizer.fit_transform(self.corpus)
            self._fallback = self.vectorizer

    def query(self, q: str, top_k: int = 4) -> List[Tuple[Doc, float]]:
        if _HAS_SK:
            qv = self.vectorizer.transform([q])
            sims = sk_cosine(qv, self.mat)[0]
            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
            return [(self.docs[i], float(s)) for (i, s) in ranked]
        else:
            qv = self._fallback.transform([q])[0]
            scores = [self._fallback.cosine(qv, row) for row in self.mat]
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
            return [(self.docs[i], float(s)) for (i, s) in ranked]


# ----------------------------
# Simple RAG Orchestrator
# ----------------------------

class SimpleRAG:
    def __init__(self, seed_docs: Optional[List[Doc]] = None):
        self.docs: List[Doc] = seed_docs or []
        self.retriever: Optional[TFIDFRetriever] = TFIDFRetriever(self.docs) if self.docs else None

    def add_documents(self, docs: Iterable[Doc]):
        for d in docs:
            self.docs.append(d)
        self.retriever = TFIDFRetriever(self.docs)

    def add_corpus_folder(self, folder: str, pattern: str = r".*\.txt$"):
        rx = re.compile(pattern)
        docs = []
        for root, _, files in os.walk(folder):
            for name in files:
                path = os.path.join(root, name)
                if rx.search(name):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        raw = f.read()
                    for idx, chunk in enumerate(chunk_text(raw)):
                        docs.append(Doc(id=f"{name}::chunk{idx}", text=_normalize_ws(chunk), meta={"path": path}))
        if docs:
            self.add_documents(docs)

    # -------------- LLM plumbing --------------
    def _call_llm(self, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
        if client is not None:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a concise debate coach. Cite only from provided context. If unsure, say so."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                return f"[LLM error: {e}]\n\nPROMPT\n-----\n{prompt}"
        # Deterministic stub: return a short extractive-looking answer
        return _stub_answer_from_prompt(prompt)

    # -------------- Public API --------------
    def answer(self, query: str, top_k: int = 4, model: str = "gpt-4o-mini", temperature: float = 0.2) -> Dict:
        if not self.retriever:
            return {"answer": "No index built. Add documents first.", "contexts": []}
        hits = self.retriever.query(query, top_k=top_k)
        context_blocks = []
        for doc, score in hits:
            meta_str = f"source={doc.meta.get('path','n/a')} | id={doc.id}" if doc.meta else f"id={doc.id}"
            context_blocks.append(f"[SCORE {score:.3f}] {meta_str}\n{doc.text}")
        joined = "\n\n".join(context_blocks)
        prompt = (
            "You are assisting a debate student. Use ONLY the context below to answer the question.\n"
            "If the context is insufficient, research.\n\n"
            f"CONTEXT\n=======\n{joined}\n\n"
            f"QUESTION\n========\n{query}\n\n"
            "Answer in 3-6 bullet points, concise, with evidence lines quoted when possible."
        )
        out = self._call_llm(prompt=prompt, model=model, temperature=temperature)
        return {"answer": out, "contexts": [c for c, _ in hits], "scores": [s for _, s in hits]}



def _stub_answer_from_prompt(prompt: str) -> str:
    """
    Deterministic fallback summarizer used when no LLM key is set.
    Supports two prompt styles:
    (A) Q&A: CONTEXT ... QUESTION ...
    (B) Debate: CONTEXT ... TASK ... with a 'Motion:' line
    """
    ctx_match = re.search(r"CONTEXT\n=+\n(.+?)(?:\n\n(?:QUESTION|TASK)\b|\Z)", prompt, flags=re.S)
    ctx = ctx_match.group(1).strip() if ctx_match else ""

    q_match = re.search(r"QUESTION\n=+\n(.+?)(?:\n\n|\Z)", prompt, flags=re.S)
    question = _normalize_ws(q_match.group(1)) if q_match else None

    if not question:
        motion_line = re.search(r"\bMotion:\s*(.+)", prompt)
        if motion_line:
            question = f"Motion — {motion_line.group(1).strip()}"

    if not question:
        tail_start = ctx_match.end() if ctx_match else 0
        tail = prompt[tail_start:tail_start + 200]
        question = _normalize_ws(tail) or "(no question)"

    if ctx:
        sentences = re.split(r"(?<=[.!?])\s+", ctx)
        summary = " ".join(sentences[:3])[:800]
    else:
        summary = "Not enough context."

    return (
        "- Provisional (no LLM): based on retrieved snippets, here’s the gist.\n"
        f"- Q: {question}\n"
        f"- Evidence: {summary}"
    )

DEBATE_RUBRIC = """You are an elite debate coach. Score the SPEECH on the rubric below.
Return strict JSON.

Rubric (0–3 each):
- Structure: Clear framing, burdens, 2–3 labeled contentions, crystallization.
- Weighing: Compares probability vs magnitude vs timeframe; comparative, not parallel.
- Warrants: Real mechanisms (causal chains), not assertion; at least 1 warrant per claim.
- Clash: Direct engagement with likely Opp arguments; named and answered.
- Evidence use: Quotes/paraphrases from CONTEXT if provided; otherwise plausible references.
- Efficiency: No fluff; short impact calculus; no meandering.

Output JSON schema:
{
  "scores": { "Structure": 0-3, "Weighing": 0-3, "Warrants": 0-3, "Clash": 0-3, "Evidence": 0-3, "Efficiency": 0-3 },
  "misses": ["bullet point describing gap #1", "..."],
  "action_items": ["specific rewrite instruction #1", "..."],
  "overall": 0-18
}
"""

SPEECH_PROMPT_TMPL = """You are an elite {format} debater. Side: {side}.
Motion: {motion}

{context_block}

Write a tight {format} speech (no greetings/thanks). Hard requirements:
0) Formatting notes: no need to ever repeat the motion verbatim. Sprinkle a human touch, for example, an Obama speech. Don't sound like a robot.
1) Setup: concise framing + burdens (about one minute of speech).
2) 2–3 arguments. Each: Tagline → Mechanism (warrants) → Impact (quantify/compare).
2a) At the end of each argument, do Weighing: do explicit probability vs magnitude vs timeframe and a clean "why we win".
2b) If relevant, do Pre-empts: name 2 likely Opp pushbacks and frontload answers.
5) Conclusion: 3–5 sentences that collapse the debate.

Style: zero fluff; short sentences; force comparisons; prefer numbers or concrete hooks.
If CONTEXT is provided, quote/paraphrase lines sparingly. Most importantly, use a similar logic flow. This version should be good to submit to a judge.

"""


def _format_context_blocks(hits):
    if not hits:
        return "CONTEXT: (none provided)\n"
    blocks = []
    for doc, score in hits:
        meta = f"source={doc.meta.get('path','n/a')} | id={doc.id}" if doc.meta else f"id={doc.id}"
        blocks.append(f"[SCORE {score:.3f}] {meta}\n{doc.text}")
    return "CONTEXT\n=======\n" + "\n\n".join(blocks)

def generate_debate_with_coach_loop(
    rag: SimpleRAG,
    motion: str,
    side: str = "Government",           # or "Opposition"
    format: str = "WSDC",               # label only; affects prompt text
    use_rag: bool = True,
    top_k: int = 6,
    min_score: float = 0.1,
    model: str = "gpt-4o-mini",
    temperature_gen: float = 0.2,
    temperature_rev: float = 0.2
) -> dict:
    """
    1) (Optional) Retrieve context via RAG.
    2) Generate a full speech with hard style/structure constraints.
    3) Critique against debate rubric (JSON).
    4) Revise speech to fix misses.
    Returns:
      {
        "initial_speech": str,
        "contexts": [Doc,...],
        "scores": [float,...]
      }
    """
    # Retrieve (or skip, allowing the model to argue from general knowledge)
    hits = rag.retriever.query(motion, top_k=top_k) if (use_rag and rag.retriever) else []
    if use_rag:
        hits = [(d, s) for (d, s) in hits if s >= min_score]
    context_block = _format_context_blocks(hits)

    # Generate
    gen_prompt = SPEECH_PROMPT_TMPL.format(
        format=format, side=side, motion=motion, context_block=context_block
    )
    initial = rag._call_llm(prompt=gen_prompt, model=model, temperature=temperature_gen)


    return {
        "initial_speech": initial,
        "contexts": [d for d, _ in hits],
        "scores": [s for _, s in hits],
    }

# ----------------------------
# CLI entrypoint
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debate RAG tester")
    parser.add_argument("--corpus-dir", default=os.environ.get("SPEECH_CORPUS_DIR", "./corpus"), help="Folder of .txt speeches")
    parser.add_argument("--use-rag", action="store_true", help="Ground the speech on retrieved context")
    parser.add_argument("--top-k", type=int, default=6, help="How many chunks to retrieve")
    parser.add_argument("--min-score", type=float, default=0.25, help="Minimum cosine score to keep a chunk")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--temperature-gen", type=float, default=0.2, help="Temperature for initial speech generation")
    parser.add_argument("--temperature-rev", type=float, default=0.2, help="Temperature for revision step")

    # Two modes: (A) full WSDC speech pipeline using motion+side, (B) raw query for rag.answer
    parser.add_argument("--motion", default=None, help="Debate motion for speech generation")
    parser.add_argument("--side", default="Government", choices=["Government", "Opposition"], help="Side to argue")
    parser.add_argument("--format", default="WSDC", help="Format label (affects prompt text only)")
    parser.add_argument("--query", default=None, help="Raw query to test rag.answer instead of speech pipeline")

    args = parser.parse_args()

    # Build RAG and index corpus
    rag = SimpleRAG()
    rag.add_corpus_folder(args.corpus_dir, pattern=r".*\.txt$")

    print(f"Indexed docs: {len(rag.docs)} from {args.corpus_dir}")

    if args.query:
        # Raw retrieval + answer mode (bulleted answer prompt)
        res = rag.answer(args.query, top_k=max(1, args.top_k), model=args.model, temperature=args.temperature_gen)
        print("\n=== ANSWER (rag.answer) ===\n", res["answer"])
        print("\n=== CONTEXT IDS & SCORES ===")
        for d, s in zip(res["contexts"], res["scores"]):
            print(f"{d.id}: {s:.3f}")
    else:
        # Full debate pipeline; require a motion
        motion = args.motion
        if not motion:
            # Provide a sensible default if not supplied
            motion = "This House would choose the job they are passionate about over a higher-paying, stressful career."
            print(f"[info] --motion not provided; using default motion: {motion}")

        out = generate_debate_with_coach_loop(
            rag,
            motion=motion,
            side=args.side,
            format=args.format,
            use_rag=args.use_rag,
            top_k=max(1, args.top_k),
            min_score=args.min_score,
            model=args.model,
            temperature_gen=args.temperature_gen,
            temperature_rev=args.temperature_rev,
        )

        print("\n=== INITIAL SPEECH ===\n", out["initial_speech"])
        print("\n=== CONTEXT IDS & SCORES ===")
        for d, s in zip(out["contexts"], out["scores"]):
            print(f"{d.id}: {s:.3f}")
"""
Space to draft run prompt: 
python3 response.py \
  --corpus-dir ./corpus \
  --motion "This House, as a Yale student, would choose NOT to sell out." \
  --side Government \
  --use-rag \
  --top-k 6 \
  --min-score 0.05
"""