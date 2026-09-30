from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    text: str
    score: float = 0.0


class LocalRAG:
    def __init__(self, documents: List[str], chunk_size: int = 200, top_k: int = 3):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.chunks = self._chunk_documents(documents)
        if not self.chunks:
            raise ValueError("No document content provided for the knowledge base.")
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    @staticmethod
    def _chunk_documents(documents: List[str]) -> List[str]:
        chunks: List[str] = []
        for document in documents:
            words = document.split()
            for index in range(0, len(words), 200):
                chunk = " ".join(words[index:index + 200])
                if chunk.strip():
                    chunks.append(chunk.strip())
        return chunks

    def retrieve(self, query: str) -> List[Chunk]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1][: self.top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            results.append(Chunk(text=self.chunks[int(idx)], score=float(scores[int(idx)])))
        return results

    def answer(self, query: str) -> str:
        hits = self.retrieve(query)
        if not hits:
            return "I couldn't find relevant information in the local knowledge base for that question."

        context = "\n\n".join(f"{index + 1}. {hit.text}" for index, hit in enumerate(hits))
        answer = (
            "Based on the local knowledge base, here is the most relevant information:\n\n"
            f"{context}\n\n"
            f"In short: the retrieved context suggests that {query.strip()[:-1] if query.strip().endswith('?') else query.strip()} is best answered with the evidence above."
        )
        return answer


def load_documents(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {file_path}")
    return [text.strip() for text in file_path.read_text(encoding="utf-8").split("\n\n") if text.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple local RAG demonstration")
    parser.add_argument("--query", type=str, help="Question to ask the RAG system")
    parser.add_argument("--knowledge-file", type=str, default="data/knowledge.txt", help="Path to the knowledge base text file")
    args = parser.parse_args()

    file_path = Path(args.knowledge_file)
    documents = load_documents(file_path)
    rag = LocalRAG(documents)

    if args.query:
        print(rag.answer(args.query))
        return

    print("RAG assistant ready. Type a question or 'exit' to quit.")
    while True:
        user_query = input("Ask a question: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        print(rag.answer(user_query))


if __name__ == "__main__":
    main()
