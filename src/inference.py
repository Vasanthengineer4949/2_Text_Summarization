from transformers import pipeline
from fastapi import FastAPI
import uvicorn
import config

app = FastAPI(debug=True)
summarizer = pipeline("summarization", model=config.INFERENCE)

@app.get("/")
def home():
    return {"Project Name": "Arxiv Article Summarization"}

@app.get("/predict")
def predict(content:str):
    summarized_content = summarizer(content, truncation=True)
    return summarized_content

if __name__ == "__main__":
    uvicorn.run(app)


