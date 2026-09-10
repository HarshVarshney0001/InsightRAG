import os
import re
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVector, models
from fastembed import SparseTextEmbedding, TextEmbedding

load_dotenv()

COLLECTION_NAME = "insightrag_docs"
SCORE_THRESHOLD = 0.5
PDF_PATH = "data/NIPS-2017-attention-is-all-you-need-Paper.pdf"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")  # Ollama ki jagah local dense model

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            text = re.sub(r'\n\s*\n+', '\n', text)
            text = re.sub(r' {2,}', ' ', text)
            pages.append((page_num, text))
    return pages


def chunk_text(pages, chunk_size=800, overlap_sentences=4):
    all_chunks = []
    for page_num, text in pages:
        sentences = re.split(r'(?<=[.!?])\s+|\n\s*\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            current_chunk.append(sentence)
            current_length += len(sentence)

            if current_length >= chunk_size:
                all_chunks.append({"text": " ".join(current_chunk), "page": page_num})
                current_chunk = current_chunk[-overlap_sentences:]
                current_length = sum(len(s) for s in current_chunk)

        if current_chunk:
            all_chunks.append({"text": " ".join(current_chunk), "page": page_num})

    return all_chunks


# --- Ollama HTTP calls ki jagah ab local fastembed dense model ---
def get_document_embedding(text):
    return list(dense_model.embed([text]))[0].tolist()


def get_query_embedding(text):
    # bge models query ke liye ye instruction-prefix use karne se retrieval better hota hai
    query_text = f"Represent this sentence for searching relevant passages: {text}"
    return list(dense_model.embed([query_text]))[0].tolist()


def store_in_qdrant(client, chunks):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    sample_embedding = get_document_embedding(chunks[0]["text"])
    vector_size = len(sample_embedding)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )

    texts = [c["text"] for c in chunks]

    print("Sparse embeddings ban rahe hain (keyword-based)...")
    sparse_embeddings = list(sparse_model.embed(texts))

    print("Dense embeddings ban rahe hain (meaning-based)...")
    points = []
    for idx, (chunk, sparse_emb) in enumerate(zip(chunks, sparse_embeddings)):
        dense_emb = get_document_embedding(chunk["text"])
        points.append(
            PointStruct(
                id=idx,
                vector={
                    "dense": dense_emb,
                    "sparse": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist()
                    )
                },
                payload={"text": chunk["text"], "page": chunk["page"]}
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def hybrid_search(client, dense_query, sparse_query, top_k=3):
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_query, using="dense", limit=10),
            models.Prefetch(
                query=SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist()
                ),
                using="sparse",
                limit=10
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points
    return results


def cosine_similarity(vec1, vec2):
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_actual_chunk_confidence(client, dense_query, top_result_id):
    point = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[top_result_id],
        with_vectors=True
    )[0]
    chunk_dense_vector = point.vector["dense"]
    return cosine_similarity(dense_query, chunk_dense_vector)


# --- Ollama /api/generate ki jagah ab Groq API ---
def get_answer(question, context):
    prompt = f"""You are a strict and accurate assistant that answers ONLY based on the given context.

Rules:
1. Only state facts that are CLEARLY written in the context — do not add or change anything on your own.
2. Answer in the SAME LANGUAGE as the question was asked...
3. Give a somewhat detailed answer (2-4 sentences)...
4. NEVER use phrases like "based on general knowledge" or "typically" or "usually" to fill gaps —
   if the context doesn't explicitly state something, say "I couldn't find this specific detail
   in the document" instead of guessing or using outside knowledge.
5. Give a direct, clean answer in your own words — do not copy raw fragments, figure captions,
   or broken text formatting from the context verbatim.
Context:
{context}

Question: {question}"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def setup_pipeline(pdf_path=PDF_PATH, force_rebuild=False):
    client = QdrantClient(url=QDRANT_URL)

    if force_rebuild or not client.collection_exists(COLLECTION_NAME):
        print("PDF padh raha hu...")
        pages = load_pdf(pdf_path)
        total_chars = sum(len(t) for _, t in pages)
        print(f"PDF loaded. Total pages: {len(pages)}, Total characters: {total_chars}")

        chunks = chunk_text(pages)
        print(f"Total chunks created: {len(chunks)}")

        print("Chunks ko Qdrant mein store kar raha hu (thoda time lagega)...")
        store_in_qdrant(client, chunks)
        print("Qdrant mein data store ho gaya! (hybrid: dense + sparse dono)")
    else:
        print("Qdrant mein data already maujood hai, dobara nahi bana raha.")

    return client


def answer_question(client, question, top_k=5):
    dense_query = get_query_embedding(question)
    sparse_query = list(sparse_model.embed([question]))[0]

    results = hybrid_search(client, dense_query, sparse_query, top_k=top_k)
    confidence = get_actual_chunk_confidence(client, dense_query, results[0].id)

    context_parts = [r.payload["text"] for r in results]
    pages_used = sorted(set(r.payload["page"] for r in results))

    if confidence < SCORE_THRESHOLD:
        return {
            "question": question,
            "answer": "Mujhe iska jawab is document mein nahi mila. (Relevant content nahi mila)",
            "contexts": context_parts,
            "pages": pages_used,
            "confidence": confidence,
            "answered": False
        }

    context = "\n\n---\n\n".join(context_parts)
    answer = get_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "contexts": context_parts,
        "pages": pages_used,
        "confidence": confidence,
        "answered": True
    }


def main():
    print("Qdrant se connect ho raha hu...")
    client = setup_pipeline()

    print("\nAb tum sawaal pooch sakte ho. Band karne ke liye 'exit' likho.\n")

    while True:
        question = input("Tumhara sawaal: ")

        if question.strip().lower() == "exit":
            print("Bye!")
            break

        result = answer_question(client, question)

        print(f"\n(Confidence on top chunk: {result['confidence']:.2f})")
        print("---- ANSWER ----")
        print(result["answer"])

        if result["answered"]:
            print(f"\n(Source: Page {', '.join(map(str, result['pages']))})")

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()