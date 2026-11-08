"""Policy retrieval, TF-IDF grounding, answer generation, and evaluation."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm

POLICIES_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
_DOCUMENTS = None
_CHUNKS = None
_VECTORIZER = None
_TFIDF_MATRIX = None


def load_policy():
    """Load the five supplied policy documents."""
    global _DOCUMENTS
    documents = {}
    for file_path in sorted(POLICIES_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        documents[file_path.name] = text.split()
    _DOCUMENTS = documents
    print("Policy documents loaded:", len(documents))
    return documents


def create_chunks(documents=None, overlap=CHUNK_OVERLAP):
    """Split policy documents into the original 40-word overlapping chunks."""
    global _CHUNKS
    if documents is None:
        if _DOCUMENTS is None:
            load_policy()
        documents = _DOCUMENTS

    chunks = []
    step = CHUNK_SIZE - overlap
    for source, words in documents.items():
        start = 0
        while start < len(words):
            chunk_words = words[start:start + CHUNK_SIZE]
            chunks.append({"source": source, "text": " ".join(chunk_words)})
            if start + CHUNK_SIZE >= len(words):
                break
            start += step
    if overlap == CHUNK_OVERLAP:
        _CHUNKS = chunks
        chunk_counts = {}
        for chunk in chunks:
            category = chunk["source"].split(".")[0].lower()
            chunk_counts[category] = chunk_counts.get(category, 0) + 1

        print(f"\n{len(chunks)} chunks in total:")
        for category, count in chunk_counts.items():
            print(f"{category}: {count}")

    return chunks


def fit_tfidf(chunks=None):
    """Fit one TF-IDF vectorizer for the supplied policy chunks."""
    global _VECTORIZER, _TFIDF_MATRIX
    if chunks is None:
        if _CHUNKS is None:
            create_chunks()
        chunks = _CHUNKS
    chunk_texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    if chunks is _CHUNKS:
        _VECTORIZER = vectorizer
        _TFIDF_MATRIX = tfidf_matrix
    return vectorizer, tfidf_matrix


def retrieve(query=None, chunks=None,
             vectorizer=None, tfidf_matrix=None, k=3):
    """Retrieve the top policy chunks for a query using cosine similarity."""
    show_results = query is None
    if query is None:
        query = "refund policy late delivery"
    if chunks is None:
        if _CHUNKS is None:
            create_chunks()
        chunks = _CHUNKS
    if vectorizer is None or tfidf_matrix is None:
        if _VECTORIZER is None or _TFIDF_MATRIX is None:
            fit_tfidf(chunks)
        vectorizer = _VECTORIZER
        tfidf_matrix = _TFIDF_MATRIX

    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    ranked_indices = similarity_scores.argsort()[::-1][:k]
    results = []
    for index in ranked_indices:
        results.append({
            "source": chunks[index]["source"],
            "score": float(similarity_scores[index]),
            "text": chunks[index]["text"],
        })
    if show_results:
        print("\nTop 3 Retrieved Chunks:")
        for result in results:
            print("\nSource:", result["source"])
            print("Score:", result["score"])
            print("Text:", result["text"])
    return results


def answer(question, chunks=None, vectorizer=None, tfidf_matrix=None):
    results = retrieve(
        question, chunks=chunks, vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix, k=3
    )
    context = "\n".join(
        f"[{result['source']}]\n{result['text']}" for result in results
    )
    user_message = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    system_message = (
        "You are a POLICY ASSISTANT. "
        "Answer the question only using the supplied policy context. "
        "Do not use outside knowledge. "
        "If the answer is not available in the context, "
        "say exactly: I could not find the information in the policy documents."
    )
    response = mock_llm.chat(messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ])
    model_answer = response.choices[0].message.content
    sources = list(dict.fromkeys(result["source"] for result in results))
    return {"answer": model_answer, "sources": sources}


def answers():
    """Generate the original example answer grounded in retrieved policy text."""
    question = "What happens if an order is delivered late?"
    result = answer(question)
    print("\nPolicy Assistant Answer:")
    print(result["answer"])
    print("\nSources Used:")
    print(result["sources"])
    return result


def _evaluate(chunks, vectorizer, tfidf_matrix, show_failures=False):
    eval_df = pd.read_csv(EVAL_FILE)
    answerable_total = 0
    answerable_retrieval = 0
    answerable_grounding = 0
    unanswerable_total = 0
    unanswerable_fallback = 0

    for _, row in eval_df.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]
        results = retrieve(
            question, chunks=chunks, vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix, k=3
        )
        retrieved_sources = [result["source"] for result in results]
        result = answer(question, chunks, vectorizer, tfidf_matrix)
        model_answer = result["answer"]

        if pd.notna(expected_doc) and str(expected_doc).strip().upper() != "NONE":
            answerable_total += 1
            if expected_doc in retrieved_sources:
                answerable_retrieval += 1
            if pd.notna(expected_fact) and str(expected_fact) in model_answer:
                answerable_grounding += 1
            if show_failures and (
                expected_doc not in retrieved_sources
                or (
                    pd.notna(expected_fact)
                    and str(expected_fact) not in model_answer
                )
            ):
                print("\nFailed Question:", question)
                print("Expected Document:", expected_doc)
                print("Retrieved Sources:", retrieved_sources)
                print("Expected Fact:", expected_fact)
                print("Generated Answer:", model_answer)
        else:
            unanswerable_total += 1
            if model_answer.startswith("I could not find"):
                unanswerable_fallback += 1

    print("\nPolicy Evaluation Results")
    print("Expected document retrieved:", answerable_retrieval, "/", answerable_total)
    print("Expected fact in answer:", answerable_grounding, "/", answerable_total)
    print("Unanswerable fallback:", unanswerable_fallback, "/", unanswerable_total)
    return {
        "retrieval": (answerable_retrieval, answerable_total),
        "grounding": (answerable_grounding, answerable_total),
        "fallback": (unanswerable_fallback, unanswerable_total),
    }


def evaluate_policy():
    """Evaluate normal overlap and the original zero-overlap comparison."""
    if _DOCUMENTS is None:
        load_policy()
    if _CHUNKS is None:
        create_chunks()
    if _VECTORIZER is None or _TFIDF_MATRIX is None:
        fit_tfidf()

    print("\n========== NORMAL OVERLAP EVALUATION ==========")
    normal_results = _evaluate(_CHUNKS, _VECTORIZER, _TFIDF_MATRIX)

    zero_overlap_chunks = create_chunks(_DOCUMENTS, overlap=0)
    zero_vectorizer, zero_tfidf_matrix = fit_tfidf(zero_overlap_chunks)
    print("\nZero overlap chunks:", len(zero_overlap_chunks))
    print("\n========== ZERO OVERLAP EVALUATION ==========")
    zero_results = _evaluate(
        zero_overlap_chunks, zero_vectorizer, zero_tfidf_matrix,
        show_failures=True
    )
    return {"normal_overlap": normal_results, "zero_overlap": zero_results}


if __name__ == "__main__":
    load_policy()
    create_chunks()
    fit_tfidf()
    retrieve()
    answers()
    evaluate_policy()
