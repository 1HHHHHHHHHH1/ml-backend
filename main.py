"""
VentureBridge ML Backend
النموذج يُحمَّل من Hugging Face عند بدء السيرفر
"""
import pickle, os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="VentureBridge ML API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HF_REPO  = os.getenv("HF_REPO",  "YOUR_USERNAME/venturebridge-model")
HF_TOKEN = os.getenv("HF_TOKEN", "")

CATEGORIES = [
    "Food and Beverage","Fashion/Beauty","Fitness/Sports/Outdoors",
    "Children/Education","Health/Wellness","Technology","Home","Automotive",
    "Pet Products","Travel/Outdoor","Media/Entertainment","Other",
    "Lifestyle/Home","Electronics","Green/CleanTech","Finance/Legal"
]
INVESTORS = [
    "Mark Cuban","Barbara Corcoran","Lori Greiner","Robert Herjavec",
    "Daymond John","Kevin O'Leary","Investor_7","Investor_8","Investor_9",
    "Investor_10","Investor_11","Investor_12","Investor_13","Investor_14"
]
POSITIVE_KEYWORDS = ["revenue","patented","patent","retention","subscribers",
    "growth","margin","profit","profitable","exclusive","traction","recurring"]
NEGATIVE_KEYWORDS = ["pre-revenue","pre revenue","crowded","easily replicable",
    "single founder","saturated","no patent","no revenue","unproven","copycat","commodity"]
THRESHOLD = 0.47

model = None
tfidf = None

@app.on_event("startup")
def load_model():
    global model, tfidf
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading from {HF_REPO}...")
        kw = {"repo_id": HF_REPO, "filename": "investor_model.pkl"}
        if HF_TOKEN: kw["token"] = HF_TOKEN
        with open(hf_hub_download(**kw), "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded: {type(model)}")
        try:
            kw2 = {"repo_id": HF_REPO, "filename": "tfidf_vectorizer.pkl"}
            if HF_TOKEN: kw2["token"] = HF_TOKEN
            with open(hf_hub_download(**kw2), "rb") as f:
                tfidf = pickle.load(f)
            print("TF-IDF loaded")
        except Exception:
            print("tfidf_vectorizer.pkl not found — OK if model is a Pipeline")
    except Exception as e:
        print(f"ERROR loading model: {e}")

class PredictRequest(BaseModel):
    project_description: str
    project_category:    str
    funding_goal:        float
    project_title:       Optional[str] = ""
    investor_name:       str
    investor_bio:        Optional[str] = ""
    investor_industries: List[str] = []

class PredictResponse(BaseModel):
    decision: str; probability: float; match_percentage: float
    confidence_level: str; positive_signals: List[str]
    negative_signals: List[str]; explanation: str

class BulkRequest(BaseModel):
    project_description: str; project_category: str
    funding_goal: float; project_title: Optional[str] = ""
    investors: List[dict]

class BulkResult(BaseModel):
    investor_id: str; investor_name: str; decision: str
    probability: float; match_percentage: float
    confidence_level: str; positive_signals: List[str]; negative_signals: List[str]

@app.get("/")
def root():
    return {"status": "running", "model_loaded": model is not None}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None: raise HTTPException(503, "Model loading... retry in 10 seconds")
    pos, neg = _signals(req.project_description)
    prob = _proba(_features(req, pos, neg))
    dec  = "INVEST" if prob >= THRESHOLD else "SKIP"
    return PredictResponse(decision=dec, probability=round(prob,4),
        match_percentage=round(prob*100,1), confidence_level=_conf(prob),
        positive_signals=pos, negative_signals=neg,
        explanation=f"{'✅' if dec=='INVEST' else '❌'} {dec} — {prob*100:.1f}% with {req.investor_name}. "
            + (f"Signals: {', '.join(pos[:3])}." if pos else "No strong signals."))

@app.post("/predict/bulk", response_model=List[BulkResult])
def predict_bulk(req: BulkRequest):
    if model is None: raise HTTPException(503, "Model loading... retry in 10 seconds")
    pos, neg = _signals(req.project_description)
    out = []
    for inv in req.investors:
        try:
            r = PredictRequest(project_description=req.project_description,
                project_category=req.project_category, funding_goal=req.funding_goal,
                project_title=req.project_title or "", investor_name=inv.get("name",""),
                investor_bio=inv.get("bio",""), investor_industries=inv.get("industries",[]))
            prob = _proba(_features(r, pos, neg))
            out.append(BulkResult(investor_id=inv.get("id",""), investor_name=inv.get("name",""),
                decision="INVEST" if prob>=THRESHOLD else "SKIP",
                probability=round(prob,4), match_percentage=round(prob*100,1),
                confidence_level=_conf(prob), positive_signals=pos, negative_signals=neg))
        except Exception:
            out.append(BulkResult(investor_id=inv.get("id",""), investor_name=inv.get("name",""),
                decision="SKIP", probability=0.0, match_percentage=0.0,
                confidence_level="Low", positive_signals=[], negative_signals=[]))
    out.sort(key=lambda x: x.probability, reverse=True)
    return out

def _signals(text):
    t = text.lower()
    return [k for k in POSITIVE_KEYWORDS if k in t], [k for k in NEGATIVE_KEYWORDS if k in t]

def _features(req, pos, neg):
    f = []
    text = f"{req.project_title} {req.project_description}".strip()
    f.extend(tfidf.transform([text]).toarray()[0].tolist() if tfidf else [0.0]*500)
    for k in POSITIVE_KEYWORDS: f.append(1.0 if k in pos else 0.0)
    for k in NEGATIVE_KEYWORDS:  f.append(1.0 if k in neg else 0.0)
    cm = 1.0 if req.project_category in (req.investor_industries or []) else 0.0
    f += [0.40, cm, cm/max(len(req.investor_industries or [1]),1), float(len(pos)-3*len(neg))]
    for c in CATEGORIES: f.append(1.0 if req.project_category.lower() in c.lower() or c.lower() in req.project_category.lower() else 0.0)
    for i in INVESTORS:  f.append(1.0 if req.investor_name == i else 0.0)
    return np.array(f, dtype=np.float32).reshape(1,-1)

def _proba(features):
    if hasattr(model,"predict_proba"): return float(model.predict_proba(features)[0][1])
    if hasattr(model,"predict"):
        v = float(model.predict(features)[0])
        return v if 0<=v<=1 else float(v>0.5)
    return 0.0

def _conf(p):
    return "High" if p>=0.75 or p<=0.25 else ("Medium" if p>=0.60 or p<=0.40 else "Low")
