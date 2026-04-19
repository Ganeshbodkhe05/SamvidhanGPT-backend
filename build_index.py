import pickle
import numpy as np

from src.loader import load_pdf
from src.chunker import chunk_text
from src.embeddings import embed
from src.vector_db import build_index, save_index

text = load_pdf("data/constitution.pdf")

chunks = chunk_text(text)

vectors = embed(chunks)
vectors = np.array(vectors)

index = build_index(vectors)

save_index(index)

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Index built")