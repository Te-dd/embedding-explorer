from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def generate_embedding(text: str):
    if not text.strip():
        raise ValueError("Text cannot be empty")

    embedding = model.encode(text)

    return embedding


def compare_sentences(text1: str, text2: str):
    embedding1 = generate_embedding(text1)
    embedding2 = generate_embedding(text2)

    similarity = np.dot(
        embedding1,
        embedding2
    ) / (
        np.linalg.norm(embedding1)
        * np.linalg.norm(embedding2)
    )

    return float(similarity)