from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
import chromadb
import ollama

app = FastAPI()

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("atlas", metadata={"hnsw:space": "cosine"})


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def ask_qwen(prompt):
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"]


def top_chunks(question, top_k=3):
    q_emb = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        (1 - dist, doc, meta)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def rag_prompt(question, results):
    context = "\n\n".join(t for s, t, m in results)
    return f"""Answer the question using only the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}"""


def source_list(results):
    return [{"source": m["source"], "page": m["page"], "score": round(s, 3)} for s, t, m in results]


def decide(question):
    if collection.count() == 0:
        return "DIRECT"
    topic = collection.get(limit=1)["documents"][0][:300]
    prompt = f"""You are a router. An uploaded document starts like this:
{topic}

Decide how to answer the user's question. Reply with exactly one word:
SEARCH - if the question is about the content or topic of the uploaded document
DIRECT - if it is a greeting or a general question unrelated to the document

Question: {question}
Reply:"""
    reply = ask_qwen(prompt).strip().upper()
    return "SEARCH" if "SEARCH" in reply else "DIRECT"


@app.get("/")
def home():
    return {"message": "Atlas AI is running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Please upload a PDF file."}
    reader = PdfReader(file.file)
    ids, texts, metas = [], [], []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = " ".join((page.extract_text() or "").split())
        for i, chunk in enumerate(chunk_text(page_text)):
            ids.append(f"{file.filename}-p{page_num}-c{i}")
            texts.append(chunk)
            metas.append({"source": file.filename, "page": page_num})
    if not texts:
        return {"error": "No text found in this PDF. It may be a scanned image."}
    embeddings = ollama.embed(model="nomic-embed-text", input=texts)["embeddings"]
    collection.delete(where={"source": file.filename})
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "chunks": len(texts),
        "total_in_database": collection.count(),
    }


@app.get("/search")
def search(question: str, top_k: int = 3):
    if collection.count() == 0:
        return {"error": "No document uploaded yet. Use /upload first."}
    results = top_chunks(question, top_k)
    return {
        "question": question,
        "results": [
            {"score": round(s, 3), "source": m["source"], "page": m["page"], "text": t}
            for s, t, m in results
        ],
    }


@app.get("/chat")
def chat(question: str):
    if collection.count() == 0:
        return {"error": "No document uploaded yet. Use /upload first."}
    results = top_chunks(question, 3)
    return {
        "question": question,
        "answer": ask_qwen(rag_prompt(question, results)),
        "sources": source_list(results),
    }


@app.get("/agent")
def agent(question: str):
    action = decide(question)
    if action == "SEARCH":
        results = top_chunks(question, 3)
        return {
            "question": question,
            "action": action,
            "answer": ask_qwen(rag_prompt(question, results)),
            "sources": source_list(results),
        }
    return {"question": question, "action": action, "answer": ask_qwen(question), "sources": []}

