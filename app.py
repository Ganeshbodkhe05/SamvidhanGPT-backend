from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle

from src.vector_db import load_index
from src.rag_pipeline import generate_answer

app = FastAPI()

# CORS (IMPORTANT for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS + chunks
index = load_index()

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

@app.get("/")
def home():
    return {"message": "SamvidhanGPT API running"}

@app.post("/chat")
async def chat(data: dict):
    query = data.get("question")
    answer = generate_answer(query, index, chunks)
    return {"answer": answer}