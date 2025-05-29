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
    readability = 50  # Placeholder or use textstat.flesch_reading_ease(text)
    avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0
    ttr = len(set(tokens)) / len(tokens) if tokens else 0

    pos_counts = {pos: 0 for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON']}
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
    total_pos = sum(pos_counts.values()) or 1
    pos_ratios = [pos_counts[pos] / total_pos for pos in pos_counts]

    ner_count = len(doc.ents)
    parse_depth = np.mean([abs(token.head.i - token.i) for token in doc if token.head != token]) if doc else 0
    noun_chunks = len(list(doc.noun_chunks))

    embedding = embedder.encode([text])[0]

    features_for_model = [wc, avg_sent_len, readability, avg_word_len, ttr, *pos_ratios, ner_count, parse_depth, noun_chunks, *embedding]
    features_dict = {
        "Word Count": wc,
        "Avg Sentence Length": avg_sent_len,
        "Readability Score": readability,
        "Avg Word Length": avg_word_len,
        "Type-Token Ratio": ttr,
        "Noun Ratio": pos_ratios[0],
        "Verb Ratio": pos_ratios[1],
        "Adj Ratio": pos_ratios[2],
        "Adv Ratio": pos_ratios[3],
        "Pronoun Ratio": pos_ratios[4],
        "NER Count": ner_count,
        "Parse Depth": parse_depth,
        "Noun Chunks": noun_chunks,
    }

    return np.array(features_for_model).reshape(1, -1), features_dict

def normalize_features(features_dict):
    max_vals = {
        "Word Count": 500,  # adjust based on typical input size
        "Avg Sentence Length": 40,
        "Readability Score": 100,
        "Avg Word Length": 10,
        "Type-Token Ratio": 1,
        "Noun Ratio": 1,
        "Verb Ratio": 1,
        "Adj Ratio": 1,
        "Adv Ratio": 1,
        "Pronoun Ratio": 1,
        "NER Count": 30,
        "Parse Depth": 20,
        "Noun Chunks": 250,
    }

    return {
        k: features_dict[k] / max_vals[k] if max_vals[k] != 0 else 0
        for k in features_dict
    }


@app.post("/predict")
def predict_ilr(request: TextRequest):
    features_vector, features_dict = extract_features(request.text)
    prediction = model.predict(features_vector)

    normalized = normalize_features(features_dict)

    return {
        "predicted_ilr": int(prediction[0]),
        "features": normalized,
        "raw_features": features_dict
    }

