from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
import ollama

app = FastAPI()

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

@app.post("/upload")
async def upload(file: UploadFile = File("sample_pdf.pdf")):
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    text = " ".join(text.split())
    chunks = chunk_text(text)
    return {"filename": file.filename, "pages": len(reader.pages), "characters": len(text), "chunks": len(chunks), "sample": chunks[:2]}