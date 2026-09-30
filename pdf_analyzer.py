from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from io import BytesIO

import numpy as np
from pypdf import PdfReader


def _import_fitz():
    try:
        import fitz  # type: ignore

        return fitz
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "OCR and scanned-PDF rendering need the Microsoft Visual C++ 2015-2022 "
            "Redistributable (x64) on Windows. Regular selectable-text PDFs still work."
        ) from error


@lru_cache(maxsize=1)
def _get_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "OCR could not start. On Windows, install the Microsoft Visual C++ "
            "2015-2022 Redistributable (x64), restart this app, and try again."
        ) from error


def _recognize_page(page, page_number: int) -> str:
    fitz = _import_fitz()
    engine = _get_ocr_engine()
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    try:
        results, _ = engine(image)
    except Exception as error:
        raise RuntimeError(f"OCR failed on page {page_number}: {error}") from error

    if not results:
        return ""

    results.sort(
        key=lambda item: (
            min(point[1] for point in item[0]),
            min(point[0] for point in item[0]),
        )
    )
    return " ".join(item[1].strip() for item in results if item[1].strip())


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours",
}


def extract_pdf_pages(pdf_bytes: bytes) -> list[str]:
    """Extract text from each page, using OCR only when needed."""
    reader = PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        raise ValueError("This PDF is password-protected. Upload an unlocked copy.")

    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    if any(not page for page in pages):
        try:
            fitz = _import_fitz()
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                pages = [
                    text or _recognize_page(page, index + 1)
                    for index, (page, text) in enumerate(zip(document, pages))
                ]
        except RuntimeError:
            raise
        except Exception as error:
            raise ValueError(f"Could not render this PDF for OCR: {error}") from error

    if not any(pages):
        raise ValueError(
            "OCR could not find readable text in this PDF. The scan may be blank, "
            "too blurry, or written in a language not supported by the OCR model."
        )
    return pages


def _split_sentences(pages: list[str]) -> list[tuple[int, str]]:
    sentences: list[tuple[int, str]] = []
    for page_number, page_text in enumerate(pages, start=1):
        normalized = re.sub(r"\s+", " ", page_text).strip()
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])", normalized):
            sentence = part.strip()
            if len(sentence.split()) >= 5:
                sentences.append((page_number, sentence))
    return sentences


def summarize_pages(
    pages: list[str], sentence_count: int = 6
) -> list[tuple[int, str]]:
    """Select the most informative source sentences, retaining their page numbers."""
    sentences = _split_sentences(pages)
    if not sentences:
        return []

    tokens_by_sentence = [
        [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z'-]*", text)
         if word.lower() not in STOP_WORDS and len(word) > 2]
        for _, text in sentences
    ]
    document_frequency: Counter[str] = Counter()
    for tokens in tokens_by_sentence:
        document_frequency.update(set(tokens))

    total_sentences = len(sentences)
    scores: list[float] = []
    for tokens in tokens_by_sentence:
        if not tokens:
            scores.append(0.0)
            continue
        term_frequency = Counter(tokens)
        weighted_score = sum(
            count * (math.log((total_sentences + 1) /
                              (document_frequency[word] + 1)) + 1)
            for word, count in term_frequency.items()
        )
        scores.append(weighted_score / math.sqrt(len(tokens)))

    count = min(max(1, sentence_count), len(sentences))
    selected_indices = sorted(
        sorted(range(len(sentences)), key=lambda index: scores[index], reverse=True)[:count]
    )
    return [sentences[index] for index in selected_indices]


def explain_summary(summary: list[tuple[int, str]]) -> str:
    """Format the document's central claims as a concise, source-grounded overview."""
    if not summary:
        return "There is not enough readable text to explain this PDF."

    key_idea = summary[0][1]
    supporting_points = [sentence for _, sentence in summary[1:4]]
    explanation = f"The main idea is: {key_idea}"
    if supporting_points:
        explanation += "\n\nThe document also explains:\n"
        explanation += "\n".join(f"- {sentence}" for sentence in supporting_points)
    return explanation
