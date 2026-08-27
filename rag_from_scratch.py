"""
================================================================================
RAG (Retrieval-Augmented Generation) FROM SCRATCH in Python
================================================================================

WHAT IS RAG?
------------
RAG = let an LLM answer questions using FACTS pulled from your own documents,
instead of relying only on what the model memorized during training.

THE 7 STAGES WE IMPLEMENT HERE (no LangChain, no LangGraph):
  1. LOAD      - read our sample knowledge base
  2. CHUNK     - split long docs into small pieces
  3. EMBED     - turn each chunk into a list of numbers (a "vector")
  4. STORE     - keep those vectors in memory (a tiny vector store)
  5. RETRIEVE  - for a question, find the most similar chunks
  6. AUGMENT   - put the found chunks into the prompt as "context"
  7. GENERATE  - ask an LLM to answer using ONLY that context

NOTE ON DEPENDENCIES:
  We implement embeddings with pure Python (TF-IDF + cosine similarity) so the
  code RUNS with ZERO external packages. For the LLM step we optionally call an
  OpenAI-compatible API (if OPENAI_API_KEY is set); otherwise we fall back to a
  local "template" generator so the demo always works offline.
================================================================================
"""

import math
import re
import os
import json

# =============================================================================
# STEP 1 — LOAD: Our sample knowledge base
# =============================================================================
# In a real system this might read PDFs, web pages, or a database. Here we just
# keep a few plain-text "documents" in a Python list. Each item is one document.

SAMPLE_DOCUMENTS = [
    {
        "id": "doc-planet-earth",
        "title": "Earth",
        "text": (
            "Earth is the third planet from the Sun and the only known world to "
            "support life. About 71 percent of its surface is covered by water. "
            "Earth has a protective atmosphere made mostly of nitrogen and oxygen. "
            "The atmosphere traps heat and shields living things from harmful solar "
            "radiation. Earth completes one orbit around the Sun in about 365 days."
        ),
    },
    {
        "id": "doc-planet-mars",
        "title": "Mars",
        "text": (
            "Mars is the fourth planet from the Sun and is often called the Red "
            "Planet because of its iron-oxide-rich soil. Mars has the largest "
            "volcano in the solar system, Olympus Mons. Scientists study Mars to "
            "understand whether life could ever have existed there. A Martian year "
            "lasts about 687 Earth days."
        ),
    },
    {
        "id": "doc-company-leave",
        "title": "Company Leave Policy",
        "text": (
            "Employees are entitled to 20 paid annual leave days per year. Sick "
            "leave of up to 10 days requires a medical certificate only after the "
            "third consecutive day. Requests for leave must be submitted through "
            "the HR portal at least one week in advance. Unused leave cannot be "
            "carried over to the next calendar year."
        ),
    },
    {
        "id": "doc-company-wfh",
        "title": "Company Work From Home Policy",
        "text": (
            "Employees may work from home up to two days per week with manager "
            "approval. Core working hours are 10am to 3pm in your local time zone. "
            "Remote workers must remain reachable on chat during core hours. A "
            "fully remote role requires director-level sign-off."
        ),
    },
    {
        "id": "doc-ai-rag",
        "title": "What is RAG",
        "text": (
            "Retrieval-Augmented Generation, or RAG, combines a search system with "
            "a language model. The system first retrieves relevant documents, then "
            "augments the prompt with that context, and finally the model generates "
            "an answer grounded in the retrieved facts. RAG reduces hallucination "
            "because the model is constrained to use provided evidence."
        ),
    },
]


def load_documents():
    """STEP 1: Return our knowledge base as a list of dicts.

    Why: We need a single, predictable place that holds ALL the text the RAG
    system is allowed to use as evidence.
    """
    return SAMPLE_DOCUMENTS


# =============================================================================
# STEP 2 — CHUNK: Split documents into smaller pieces
# =============================================================================
# Why chunk? Embeddings and LLM context windows work better on small, focused
# pieces. Searching one giant document returns all-or-nothing; small chunks let
# us fetch ONLY the sentence(s) that actually answer the question.

def chunk_text(text, chunk_size=3, overlap=1):
    """Split `text` into overlapping chunks of `chunk_size` sentences each.

    - chunk_size: how many sentences per chunk
    - overlap:    how many sentences to repeat at the start of the next chunk
                  (overlap keeps meaning continuous across chunk boundaries)

    Returns a list of chunk strings.
    """
    # 2a. Break the paragraph into sentences using simple punctuation rules.
    #     We treat '.', '!' or '?' followed by a space as a sentence boundary.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]  # drop any empty strings

    chunks = []
    step = max(1, chunk_size - overlap)  # how far to slide the window each step
    for start in range(0, len(sentences), step):
        # 2b. Take `chunk_size` sentences starting at `start`.
        piece = sentences[start:start + chunk_size]
        if not piece:
            continue
        chunks.append(" ".join(piece))
        # 2c. Stop once we have consumed all sentences.
        if start + chunk_size >= len(sentences):
            break
    return chunks


def build_chunks(documents):
    """Apply chunking to every document and tag each chunk with its source.

    Returns a list of dicts: {"doc_id", "title", "text", "chunk_index"}.
    """
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "text": piece,
                "chunk_index": i,
            })
    return all_chunks


# =============================================================================
# STEP 3 — EMBED: Turn text into a vector of numbers
# =============================================================================
# An "embedding" is a fixed-length list of numbers that captures the MEANING of
# a piece of text. Texts with similar meaning get similar vectors.
#
# We implement a classic, dependency-free embedding method: TF-IDF.
#   - TF  (term frequency): how often a word appears in THIS chunk
#   - IDF (inverse document frequency): how rare the word is across ALL chunks
#   - TF-IDF = TF * IDF  -> common words like "the" get low weight, important
#     distinctive words get high weight.
# Similarity between two vectors is measured with COSINE SIMILARITY (angle
# between them, ignoring length).

class TfidfEmbedder:
    """A tiny TF-IDF vectorizer + cosine similarity engine, written from scratch."""

    def __init__(self):
        self.vocab = {}        # word -> column index in our vector
        self.idf = {}          # word -> inverse document frequency weight
        self.fitted = False

    @staticmethod
    def _tokenize(text):
        """Turn text into a list of lowercase word tokens (letters only)."""
        # 3a. Find all sequences of letters/numbers, lowercase them.
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, corpus):
        """STEP 3 (learn): Build the vocabulary and IDF weights from `corpus`.

        `corpus` is a list of chunk texts. We must fit BEFORE we can embed,
        because IDF depends on how many chunks contain each word.
        """
        # 3b. Build vocabulary: assign each unique word a unique column index.
        for text in corpus:
            for word in self._tokenize(text):
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

        num_docs = len(corpus)
        # 3c. Count in how many chunks each word appears (document frequency).
        doc_freq = {word: 0 for word in self.vocab}
        for text in corpus:
            seen = set(self._tokenize(text))  # a word counts once per chunk
            for word in seen:
                doc_freq[word] += 1

        # 3d. IDF formula: log( (1 + N) / (1 + df) ) + 1  (smoothed so df=N -> ~0)
        #     Rare words (small df) get a BIG idf weight; common words get ~0.
        self.idf = {
            word: math.log((1 + num_docs) / (1 + df)) + 1
            for word, df in doc_freq.items()
        }
        self.fitted = True

    def embed(self, text):
        """STEP 3 (apply): Convert one text into a TF-IDF vector.

        Returns a dict {word_index: weight}. We use a sparse dict (only non-zero
        entries) because most words are absent from any single chunk.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() on the corpus before embed().")

        tf = {}
        tokens = self._tokenize(text)
        for word in tokens:
            if word in self.vocab:
                tf[word] = tf.get(word, 0) + 1  # raw count of this word

        vector = {}
        for word, count in tf.items():
            idx = self.vocab[word]
            # 3e. Weight = (count normalized by length) * idf
            vector[idx] = (count / len(tokens)) * self.idf[word]
        return vector

    @staticmethod
    def cosine(a, b):
        """Cosine similarity between two sparse vectors (dicts of idx->weight).

        Cosine = dot(a,b) / (|a| * |b|). Range is -1..1; for TF-IDF it is 0..1.
        """
        # 3f. Dot product: sum of products of shared dimensions.
        dot = 0.0
        for idx, val in a.items():
            if idx in b:
                dot += val * b[idx]

        # 3g. Magnitudes (lengths) of each vector.
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


# =============================================================================
# STEP 4 — STORE: Keep embeddings so we can search them later
# =============================================================================
# A "vector store" is just a database of (chunk_text, embedding) rows that we can
# scan quickly. For tiny demos an in-memory list is enough; production uses
# specialized engines (FAISS, Pinecone, pgvector). We build our own minimal one.

class VectorStore:
    """Minimal in-memory vector store: save chunks + their embeddings, then search."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.records = []  # each record: {"chunk": {...}, "vector": {...}}

    def add(self, chunks):
        """Embed every chunk and store it. This is our "indexing" step."""
        for chunk in chunks:
            vector = self.embedder.embed(chunk["text"])
            self.records.append({"chunk": chunk, "vector": vector})

    def search(self, query, top_k=3):
        """STEP 5 (retrieve): Embed the query, compare to every stored chunk,
        and return the `top_k` chunks with the highest cosine similarity."""
        query_vec = self.embedder.embed(query)
        scored = []
        for record in self.records:
            score = self.embedder.cosine(query_vec, record["vector"])
            scored.append((score, record["chunk"]))

        # Sort by score descending and keep the best `top_k`.
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(round(score, 4), chunk) for score, chunk in scored[:top_k]]


# =============================================================================
# STEP 6 + 7 — AUGMENT + GENERATE: Build a prompt and call the LLM
# =============================================================================
# We assemble a prompt that says: "Here are some facts. Answer the question
# using ONLY these facts." Then we send it to a model.

def build_prompt(question, retrieved_chunks):
    """STEP 6 (augment): Combine retrieved context + the user question.

    Formatting the context clearly (with source titles) helps the model quote
    the right evidence and lets us show where the answer came from.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[{i}] (Source: {chunk['title']}) {chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    # The classic RAG instruction: ground the answer in the provided context.
    prompt = (
        "You are a helpful assistant that answers ONLY using the context below.\n"
        "If the context does not contain the answer, say you don't know.\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    return prompt


def generate_answer(prompt, question, retrieved_chunks):
    """STEP 7 (generate): Get the final answer from an LLM, or a local fallback.

    Priority:
      1. If OPENAI_API_KEY is set, call an OpenAI-compatible chat model.
      2. Otherwise, use a simple local "generator" that composes an answer
         from the retrieved chunks so the demo still runs with no internet.
    """
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        # ---- REAL LLM CALL (kept optional so the demo runs offline) ----
        # We use the requests library only in this branch.
        try:
            import requests
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                },
                timeout=30,
            )
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # If the API call fails, fall through to the local generator.
            print(f"[warn] LLM call failed ({e}); using local generator.\n")

    # ---- LOCAL FALLBACK GENERATOR (no dependencies) ----
    # This is a stand-in that shows EXACTLY what an answer step receives.
    # It finds sentences in the retrieved chunks that contain question keywords.
    q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    best_sentences = []
    for chunk in retrieved_chunks:
        for sent in re.split(r"(?<=[.!?])\s+", chunk["text"]):
            sent_words = set(re.findall(r"[a-z0-9]+", sent.lower()))
            # Keep a sentence if it shares at least one meaningful word with the Q.
            if q_words & sent_words:
                best_sentences.append(sent.strip())
    if best_sentences:
        answer = " ".join(best_sentences[:3])
    else:
        answer = "I could not find that information in the provided documents."
    return answer


# =============================================================================
# ORCHESTRATION: Wire all 7 steps together into one pipeline
# =============================================================================
def run_rag(question, top_k=3):
    """Run the full pipeline and return (answer, retrieved_chunks)."""
    # STEP 1: load
    docs = load_documents()

    # STEP 2: chunk
    chunks = build_chunks(docs)

    # STEP 3 + 4: embed + store (fit embedder on all chunks, then index them)
    embedder = TfidfEmbedder()
    embedder.fit([c["text"] for c in chunks])
    store = VectorStore(embedder)
    store.add(chunks)

    # STEP 5: retrieve the most relevant chunks for this question
    retrieved = store.search(question, top_k=top_k)
    retrieved_chunks = [chunk for _, chunk in retrieved]

    # STEP 6: augment (build the prompt with context)
    prompt = build_prompt(question, retrieved_chunks)

    # STEP 7: generate the answer
    answer = generate_answer(prompt, question, retrieved_chunks)

    return answer, retrieved


# =============================================================================
# DEMO / INTERACTIVE LOOP
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SIMPLE RAG DEMO (from scratch, no LangChain)")
    print("=" * 70)
    print("Type a question, or 'exit' to quit.\n")

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not question:
            continue

        answer, retrieved = run_rag(question, top_k=3)

        print("\n--- Retrieved context (top matches) ---")
        for score, chunk in retrieved:
            print(f"  [{score}] {chunk['title']}: {chunk['text']}")
        print("\n--- Answer ---")
        print(f"  {answer}\n")
