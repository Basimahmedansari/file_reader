
import os
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = os.getenv(
    "GEMINI_EMBED_MODEL", "gemini-embedding-001"
)


def make_chunks(text, size=1200, overlap=150):
    if not text or not text.strip():
        return []

    step = max(1, size - overlap)
    return [
        text[i:i + size].strip()
        for i in range(0, len(text), step)
        if text[i:i + size].strip()
    ]


def get_embeddings(texts):
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = []

    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]

        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch
        )

        vectors.extend(
            item.values for item in result.embeddings
        )

    return np.asarray(vectors, dtype=np.float32)
