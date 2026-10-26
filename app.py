from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
import ollama

app = FastAPI()
documents = []

@app.get("/")
def home():
    return {"message": "Atlas AI is running"}

@app.get("/ask")
def ask(question: str):
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": question}],
    )
    return {"answer": response["message"]["content"]}


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


@app.post("/upload")
async def upload(file: UploadFile = File("sample_pdf.pdf")):
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    text = " ".join(text.split())
    chunks = chunk_text(text)

    embeddings = ollama.embed(model="nomic-embed-text", input=chunks)["embeddings"]
    documents.clear()
    for chunk, emb in zip(chunks, embeddings):
        documents.append({"text": chunk, "embedding": emb})
    return {"filename": file.filename, "chunks": len(chunks), "embedded": len(documents), "dimensions": len(documents[0]["embedding"])}

def top_chunks(question, top_k=3):
    q_emb = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]
    scored = [(similarity(q_emb, d["embedding"]), d["text"]) for d in documents]
    scored.sort(reverse=True)
    return scored[:top_k]

@app.get("/search")
def search(question: str, top_k: int = 3):
    if not documents:
        return {"error": "No document uploaded yet. Use /upload first."}
    results = top_chunks(question, top_k)
    return {"question": question, "results": [{"score": round(s, 3), "text": t} for s, t in results]}

@app.get("/chat")
def chat(question: str):
    if not documents:
        return {"error": "No document uploaded yet. Use /upload first."}
    results = top_chunks(question, 3)
    context = "\n\n".join(t for s, t in results)
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}"""
    response = ollama.chat(model="qwen2.5:3b", messages=[{"role": "user", "content": prompt}])
    return {"question": question, "answer": response["message"]["content"], "sources": [round(s, 3) for s, t in results]}