from fastapi import FastAPI
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