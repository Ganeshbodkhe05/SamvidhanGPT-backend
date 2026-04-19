import faiss
import numpy as np

def build_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors.astype("float32"))
    return index

def save_index(index):
    faiss.write_index(index, "faiss_index.bin")

def load_index():
    return faiss.read_index("faiss_index.bin")

def search(index, query_vector, k=3):
    D, I = index.search(query_vector, k)
    return I