from src.inference import NER
from fastapi import FastAPI
import uvicorn

app = FastAPI()

ner = NER()

@app.post("/NER_en")
def NER_en(text : str):
    result = ner.tagging_NER(text)
    print(result)
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

