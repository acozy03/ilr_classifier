from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import numpy as np
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# Load model and NLP tools
with open("tree_classifier.pkl", "rb") as f:
    model = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

nltk.download('punkt')

app = FastAPI()
# Add these lines
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
class TextRequest(BaseModel):
    text: str

def extract_features(text: str):
    tokens = nltk.word_tokenize(text)
    doc = nlp(text)
    
    wc = len(tokens)
    sents = list(doc.sents)
    avg_sent_len = np.mean([len(nltk.word_tokenize(sent.text)) for sent in sents]) if sents else 0
    readability = 50  # placeholder, or calculate using textstat
    avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0
    ttr = len(set(tokens)) / len(tokens) if tokens else 0
    pos_counts = {pos: 0 for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON']}
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
    pos_ratios = [pos_counts[pos] / (sum(pos_counts.values()) or 1) for pos in pos_counts]
    ner_count = len(doc.ents)
    parse_depth = np.mean([abs(token.head.i - token.i) for token in doc if token.head != token]) if doc else 0
    noun_chunks = len(list(doc.noun_chunks))
    embedding = embedder.encode([text])[0]

    # Replace below with actual z-score logic or keep as-is for demo
    features = [wc, avg_sent_len, readability, avg_word_len, ttr, *pos_ratios, ner_count, parse_depth, noun_chunks, *embedding]
    return np.array(features).reshape(1, -1)

@app.post("/predict")
def predict_ilr(request: TextRequest):
    features = extract_features(request.text)
    prediction = model.predict(features)
    return {"predicted_ilr": int(prediction[0])}
