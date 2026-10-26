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


@app.post("/upload")
async def upload(file: UploadFile = File("sample_pdf.pdf")):
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return {"filename": file.filename, "pages": len(reader.pages), "characters": len(text), "preview": text[:300]}