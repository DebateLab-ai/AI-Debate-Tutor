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
    def _call_llm(self, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.5) -> str:
        if client is not None:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a world-class competitive debater. Your goal is to WIN through sharp, incisive argumentation—not through aggression. Be strategic, precise, and respectful. When you have context from the corpus, use it to build airtight cases. When context has gaps, fill them with compelling real-world examples that any educated voter would recognize—think NYT front page, not academic journals. Use concrete mechanisms and numbers. Make clear, fair comparisons that demonstrate why your case is stronger. Sound like a human champion debater who wins through superior logic and analysis, not an essay or a robot."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )


                # --- token usage from API ---
                prompt_tokens = resp.usage.prompt_tokens
                completion_tokens = resp.usage.completion_tokens
                total_tokens = resp.usage.total_tokens  # if you need it

                # --- pricing (check actual numbers on the pricing page) ---
                price_per_m_input = 0.15   # dollars per 1M input tokens
                price_per_m_output = 0.60  # dollars per 1M output tokens

                cost = (
                    (prompt_tokens / 1_000_000) * price_per_m_input
                    + (completion_tokens / 1_000_000) * price_per_m_output
                )

                # if you want to log it:
                print(
                    f"usage: prompt={prompt_tokens}, completion={completion_tokens}, "
                    f"total={total_tokens}, cost=${cost:.6f}"
                )



                return resp.choices[0].message.content.strip()
            except Exception as e:
                # Log internally but NEVER expose prompt or error details to user
                print(f"[ERROR] LLM call failed: {e}")
                return "[Error generating response. Please try again.]"
        # Deterministic stub: return a short extractive-looking answer
        return _stub_answer_from_prompt(prompt)

    # -------------- Public API --------------
    def answer(self, query: str, top_k: int = 4, model: str = "gpt-4o-mini", temperature: float = 0.3) -> Dict:
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

REBUTTAL_PROMPT_TMPL = """You are a {side} debater. Motion: {motion}

OTHER SIDE'S SPEECH:
{opponent_speech}

{context_block}

HARD RULE - OPPONENT IDENTIFICATION:
- NEVER say "my opponent", "the opponent", or "they/them" when referring to the other side
- ALWAYS identify them as "the proposition" (if arguing FOR the motion) or "the opposition" (if arguing AGAINST the motion)
- Examples: "the proposition claims...", "the opposition argues...", "the proposition's case..."

REQUIREMENTS:
1. MATCH DEPTH: If they made 3 developed points, you make 3-4.
2. ACCURACY: Only address what they ACTUALLY said. Never invent arguments.
3. GROUP BY THEME: Organize rebuttals by big-picture themes, not point-by-point.
4. FIRST PRINCIPLES: Build each response like a mathematical proof - state premise, derive conclusions step-by-step where each step FOLLOWS NECESSARILY from the previous.
5. WELL-KNOWN EXAMPLES ONLY: Use Amazon, COVID-19, major events laypeople know.
6. VARY LANGUAGE: Signpost clearly but naturally - don't robotically repeat "firstly... secondly...". Try "To start...", "Moving to...", "Beyond that...".
7. WEIGH CONSTANTLY: Compare throughout using "on our side vs their side", "even if we concede X, we still win because Y"

HARD RULE - NEW ARGUMENT COUNT:
- Count their arguments in their speech
- Make at least ONE new constructive argument, but NO MORE than their count
- Examples: They made 1 arg → you make 1. They made 2 → you make 1-2. They made 3 → you make 1-3.
- This ensures you focus on rebuttal while still extending the debate proportionally

CRITICAL - NEVER CITE SOURCES:
- DO NOT say "research by X", "studies show", "according to Y", "evidence suggests", "data indicates"
- Even if BACKGROUND KNOWLEDGE is provided, present the reasoning as YOUR OWN logical analysis
- Treat all information as common knowledge and first-principles reasoning
- Debate is about LOGIC, not citations
"""

SPEECH_PROMPT_TMPL = """You are delivering a {format} {side} speech. Motion: {motion}

CRITICAL: You are arguing on the {side} side. If you are Government, you SUPPORT the motion. If you are Opposition, you OPPOSE the motion. All your arguments must align with this position.

{context_block}

HARD RULE - OPPONENT IDENTIFICATION:
- NEVER say "my opponent", "the opponent", or "they/them" when referring to the other side
- ALWAYS identify them as "the proposition" (if arguing FOR the motion) or "the opposition" (if arguing AGAINST the motion)
- Examples: "the proposition will claim...", "the opposition might argue...", "the proposition's case..."

STRUCTURE:
1. ROADMAP (1-2 sentences): Flag your case structure. Vary language naturally. NO greetings.
2. OPENING (2-3 sentences): Hook with well-known example, pose the key question.
3. FRAMING & BURDENS (3-4 sentences): Define lens and establish burdens strategically.
4. CONTENTIONS (1-3 arguments):
   Each needs: PREMISE → MECHANISMS (2-3) → WEIGHING → optional PRE-EMPTION

   Mechanisms must:
   - Be INDEPENDENT causal pathways
   - Build from FIRST PRINCIPLES like a math proof: state premise, derive each step where steps FOLLOW NECESSARILY
   - Use 2-3 sentences each: causal chain + elaboration
   - Vary signposting ("To start...", "Beyond that..." not "First... Second...")

5. CONCLUSION (3-5 sentences): Collapse to key question and why you win.

CRITICAL RULES:
- WELL-KNOWN EXAMPLES ONLY: Amazon, iPhone, COVID-19, major events laypeople know
- FIRST PRINCIPLES: Each logical step must follow necessarily from the previous. No leaps.
- VARY LANGUAGE: Don't robotically repeat "firstly... secondly...". Sound human.
- Zero filler, sound like SPEAKING not essay

NEVER CITE SOURCES - THIS IS CRITICAL:
- DO NOT say "research by X", "studies show", "according to Y", "evidence suggests", "data indicates", "as reported by"
- Even if BACKGROUND KNOWLEDGE is provided, present it as YOUR OWN first-principles logical reasoning
- Treat all information as common knowledge you're deriving from first principles
- Debate is about LOGIC, not who cites more sources
"""


def _format_context_blocks(hits):
    if not hits:
        return "BACKGROUND KNOWLEDGE (DO NOT CITE): None\n"
    blocks = []
    for doc, score in hits:
        # DO NOT include source metadata - makes model think it should cite
        blocks.append(doc.text)
    return (
        "BACKGROUND KNOWLEDGE (DO NOT CITE AS 'RESEARCH' OR 'STUDIES'):\n"
        "Use the logical reasoning below, but present it as YOUR OWN first-principles analysis.\n"
        "NEVER say 'research shows' or 'studies indicate' - treat this as common knowledge.\n"
        "=======\n" + "\n\n".join(blocks)
    )

def generate_debate_with_coach_loop(
    rag: SimpleRAG,
    motion: str,
    side: str = "Government",           # or "Opposition"
    format: str = "WSDC",               # label only; affects prompt text
    use_rag: bool = True,
    top_k: int = 6,
    min_score: float = 0.1,
    model: str = "gpt-4o-mini",
    temperature_gen: float = 0.5,
    temperature_rev: float = 0.2,
    temp_low: float = 0.3,              # temperature when good context found
    temp_high: float = 0.8              # temperature when no/poor context
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

    # Dynamic temperature: use lower temp when we have good context, higher when we don't
    if len(hits) >= 1:
        # Good context found - stay on script
        adaptive_temp = temp_low
        temp_reason = "good corpus coverage (1 hits)"
    elif len(hits) == 0:
        # No context - be creative
        adaptive_temp = temp_high
        temp_reason = "no corpus hits"
    else:
        # Some context - interpolate between low and high
        ratio = len(hits) / 3.0
        adaptive_temp = temp_low + (temp_high - temp_low) * (1 - ratio)
        temp_reason = f"partial corpus coverage ({len(hits)} hit{'s' if len(hits) > 1 else ''})"

    # Print corpus usage stats
    print(f"\n[CORPUS STATS]")
    print(f"  Corpus hits above min_score ({min_score}): {len(hits)}")
    print(f"  Using corpus: {'Yes' if len(hits) > 0 else 'No'}")
    print(f"  Temperature: {adaptive_temp:.2f} ({temp_reason})")
    if len(hits) > 0:
        avg_score = sum(s for _, s in hits) / len(hits)
        print(f"  Average similarity score: {avg_score:.3f}")
    print()

    # Generate
    gen_prompt = SPEECH_PROMPT_TMPL.format(
        format=format, side=side, motion=motion, context_block=context_block
    )
    initial = rag._call_llm(prompt=gen_prompt, model=model, temperature=adaptive_temp)


    return {
        "initial_speech": initial,
        "contexts": [d for d, _ in hits],
        "scores": [s for _, s in hits],
    }

def generate_rebuttal_speech(
    rag: SimpleRAG,
    motion: str,
    opponent_speech: str,
    side: str = "Opposition",        # Usually you're responding, so Opposition if they're Gov
    format: str = "WSDC",
    use_rag: bool = True,
    top_k: int = 6,
    min_score: float = 0.1,
    model: str = "gpt-4o-mini",
    temp_low: float = 0.3,
    temp_high: float = 0.8
) -> dict:
    """
    Generate a rebuttal speech that:
    1) Tears down the opponent's arguments
    2) Forwards new constructive arguments
    3) Does comparative weighing

    Returns:
      {
        "rebuttal_speech": str,
        "contexts": [Doc,...],
        "scores": [float,...]
      }
    """
    # Retrieve context (or skip)
    hits = rag.retriever.query(motion, top_k=top_k) if (use_rag and rag.retriever) else []
    if use_rag:
        hits = [(d, s) for (d, s) in hits if s >= min_score]
    context_block = _format_context_blocks(hits)

    # Dynamic temperature
    if len(hits) >= 3:
        adaptive_temp = temp_low
        temp_reason = "good corpus coverage (≥3 hits)"
    elif len(hits) == 0:
        adaptive_temp = temp_high
        temp_reason = "no corpus hits"
    else:
        ratio = len(hits) / 3.0
        adaptive_temp = temp_low + (temp_high - temp_low) * (1 - ratio)
        temp_reason = f"partial corpus coverage ({len(hits)} hit{'s' if len(hits) > 1 else ''})"

    # Print corpus usage stats
    print(f"\n[CORPUS STATS]")
    print(f"  Corpus hits above min_score ({min_score}): {len(hits)}")
    print(f"  Using corpus: {'Yes' if len(hits) > 0 else 'No'}")
    print(f"  Temperature: {adaptive_temp:.2f} ({temp_reason})")
    if len(hits) > 0:
        avg_score = sum(s for _, s in hits) / len(hits)
        print(f"  Average similarity score: {avg_score:.3f}")
    print()

    # Generate rebuttal
    rebuttal_prompt = REBUTTAL_PROMPT_TMPL.format(
        format=format,
        side=side,
        motion=motion,
        opponent_speech=opponent_speech,
        context_block=context_block
    )
    rebuttal = rag._call_llm(prompt=rebuttal_prompt, model=model, temperature=adaptive_temp)

    return {
        "rebuttal_speech": rebuttal,
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
    parser.add_argument("--model", default="gpt-4.1-mini", help="LLM model name")
    parser.add_argument("--temperature-gen", type=float, default=0.5, help="Temperature for initial speech generation (deprecated, use temp-low/temp-high)")
    parser.add_argument("--temperature-rev", type=float, default=0.2, help="Temperature for revision step")
    parser.add_argument("--temp-low", type=float, default=0.3, help="Temperature when good context is found (≥3 hits above min-score)")
    parser.add_argument("--temp-high", type=float, default=0.8, help="Temperature when no/poor context is found")

    # Three modes: (A) full WSDC speech pipeline using motion+side, (B) raw query for rag.answer, (C) rebuttal mode
    parser.add_argument("--motion", default=None, help="Debate motion for speech generation")
    parser.add_argument("--side", default="Government", choices=["Government", "Opposition"], help="Side to argue")
    parser.add_argument("--format", default="WSDC", help="Format label (affects prompt text only)")
    parser.add_argument("--query", default=None, help="Raw query to test rag.answer instead of speech pipeline")
    parser.add_argument("--rebuttal-file", default=None, help="Path to file containing opponent's speech to rebut")

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
    elif args.rebuttal_file:
        # Rebuttal mode - tear down opponent's speech and rebuild
        motion = args.motion
        if not motion:
            print("[error] --motion is required for rebuttal mode")
            exit(1)

        # Read opponent's speech
        with open(args.rebuttal_file, "r", encoding="utf-8") as f:
            opponent_speech = f.read()

        print(f"[info] Generating rebuttal to opponent's speech from {args.rebuttal_file}")

        out = generate_rebuttal_speech(
            rag,
            motion=motion,
            opponent_speech=opponent_speech,
            side=args.side,
            format=args.format,
            use_rag=args.use_rag,
            top_k=max(1, args.top_k),
            min_score=args.min_score,
            model=args.model,
            temp_low=args.temp_low,
            temp_high=args.temp_high,
        )

        print("\n=== REBUTTAL SPEECH ===\n", out["rebuttal_speech"])
        print("\n=== CONTEXT IDS & SCORES ===")
        for d, s in zip(out["contexts"], out["scores"]):
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
            temp_low=args.temp_low,
            temp_high=args.temp_high,
        )

        print("\n=== INITIAL SPEECH ===\n", out["initial_speech"])
        print("\n=== CONTEXT IDS & SCORES ===")
        for d, s in zip(out["contexts"], out["scores"]):
            print(f"{d.id}: {s:.3f}")
"""
Space to draft run prompt: 
python3 response.py \
  --corpus-dir ./corpus \
  --motion "The world would be better off if bananas were straight rather than curved." \
  --side Government \
  --use-rag \
  --top-k 6 \
  --min-score 0.05

  python3 response.py \
    --corpus-dir ./corpus \
    --motion "This house believes that military aid is EVIL" \
    --side Government \
    --use-rag \
    --top-k 6 \
    --min-score 0.075 \
    --temp-low 0.4\
    --temp-high 0.8


      python3 response.py \
    --corpus-dir ./corpus \
    --motion "This house believes that the United States should stop intervening in countries to introduce democracy" \
    --side Government \
    --use-rag \
    --top-k 6 \
    --min-score 0.075 \
    --temp-low 0.4\
    --temp-high 0.8
    
"""
