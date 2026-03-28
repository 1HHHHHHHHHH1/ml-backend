"""
VentureBridge ML Backend
Loads the model artifact from Hugging Face during startup.
"""

import os
import pickle
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="VentureBridge ML API", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_REPO = os.getenv("HF_REPO", "YOUR_USERNAME/venturebridge-model")
HF_TOKEN = os.getenv("HF_TOKEN", "")

CATEGORIES = [
    "Food and Beverage",
    "Fashion/Beauty",
    "Fitness/Sports/Outdoors",
    "Children/Education",
    "Health/Wellness",
    "Technology",
    "Home",
    "Automotive",
    "Pet Products",
    "Travel/Outdoor",
    "Media/Entertainment",
    "Other",
    "Lifestyle/Home",
    "Electronics",
    "Green/CleanTech",
    "Finance/Legal",
]
INVESTORS = [
    "Mark Cuban",
    "Barbara Corcoran",
    "Lori Greiner",
    "Robert Herjavec",
    "Daymond John",
    "Kevin O'Leary",
    "Investor_7",
    "Investor_8",
    "Investor_9",
    "Investor_10",
    "Investor_11",
    "Investor_12",
    "Investor_13",
    "Investor_14",
]
POSITIVE_KEYWORDS = [
    "revenue",
    "patented",
    "patent",
    "retention",
    "subscribers",
    "growth",
    "margin",
    "profit",
    "profitable",
    "exclusive",
    "traction",
    "recurring",
]
NEGATIVE_KEYWORDS = [
    "pre-revenue",
    "pre revenue",
    "crowded",
    "easily replicable",
    "single founder",
    "saturated",
    "no patent",
    "no revenue",
    "unproven",
    "copycat",
    "commodity",
]
THRESHOLD = 0.47

model = None
tfidf = None
ensemble_models = []


class PredictRequest(BaseModel):
    project_description: str
    project_category: str
    funding_goal: float
    project_title: Optional[str] = ""
    investor_name: str
    investor_bio: Optional[str] = ""
    investor_industries: List[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    decision: str
    probability: float
    match_percentage: float
    confidence_level: str
    positive_signals: List[str]
    negative_signals: List[str]
    explanation: str


class BulkRequest(BaseModel):
    project_description: str
    project_category: str
    funding_goal: float
    project_title: Optional[str] = ""
    investors: List[dict]


class BulkResult(BaseModel):
    investor_id: str
    investor_name: str
    decision: str
    probability: float
    match_percentage: float
    confidence_level: str
    positive_signals: List[str]
    negative_signals: List[str]


@app.on_event("startup")
def load_model():
    global model, tfidf, ensemble_models, THRESHOLD
    global POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS, CATEGORIES

    try:
        from huggingface_hub import hf_hub_download

        print(f"Downloading from {HF_REPO}...")
        kw = {"repo_id": HF_REPO, "filename": "investor_model.pkl"}
        if HF_TOKEN:
            kw["token"] = HF_TOKEN

        with open(hf_hub_download(**kw), "rb") as f:
            model = pickle.load(f)

        print(f"Model loaded: {type(model)}")

        if isinstance(model, dict):
            tfidf = model.get("tfidf", tfidf)
            THRESHOLD = float(model.get("threshold", THRESHOLD))
            POSITIVE_KEYWORDS = list(model.get("kw_pos", POSITIVE_KEYWORDS))
            NEGATIVE_KEYWORDS = list(model.get("kw_neg", NEGATIVE_KEYWORDS))
            CATEGORIES = list(model.get("categories", CATEGORIES))
            ensemble_models = [
                model[k]
                for k in ("rf", "et", "lr")
                if k in model and hasattr(model[k], "predict_proba")
            ]
            print(
                "Artifact keys:",
                sorted(model.keys()),
                "| ensemble models:",
                len(ensemble_models),
                "| threshold:",
                THRESHOLD,
            )

        if tfidf is None:
            try:
                kw2 = {"repo_id": HF_REPO, "filename": "tfidf_vectorizer.pkl"}
                if HF_TOKEN:
                    kw2["token"] = HF_TOKEN
                with open(hf_hub_download(**kw2), "rb") as f:
                    tfidf = pickle.load(f)
                print("TF-IDF loaded from standalone file")
            except Exception:
                print("Standalone tfidf_vectorizer.pkl not found")
    except Exception as e:
        print(f"ERROR loading model: {e}")


@app.get("/")
def root():
    return {"status": "running", "model_loaded": model is not None}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "ensemble_models": len(ensemble_models),
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model loading... retry in 10 seconds")

    pos, neg = _signals(req.project_description)
    prob = _proba(_features(req, pos, neg))
    dec = "INVEST" if prob >= THRESHOLD else "SKIP"

    explanation = f"{dec} - {prob * 100:.1f}% with {req.investor_name}. "
    explanation += (
        f"Signals: {', '.join(pos[:3])}." if pos else "No strong signals."
    )

    return PredictResponse(
        decision=dec,
        probability=round(prob, 4),
        match_percentage=round(prob * 100, 1),
        confidence_level=_conf(prob),
        positive_signals=pos,
        negative_signals=neg,
        explanation=explanation,
    )


@app.post("/predict/bulk", response_model=List[BulkResult])
def predict_bulk(req: BulkRequest):
    if model is None:
        raise HTTPException(503, "Model loading... retry in 10 seconds")

    pos, neg = _signals(req.project_description)
    out = []

    for inv in req.investors:
        try:
            single = PredictRequest(
                project_description=req.project_description,
                project_category=req.project_category,
                funding_goal=req.funding_goal,
                project_title=req.project_title or "",
                investor_name=inv.get("name", ""),
                investor_bio=inv.get("bio", ""),
                investor_industries=inv.get("industries", []),
            )
            prob = _proba(_features(single, pos, neg))
            out.append(
                BulkResult(
                    investor_id=inv.get("id", ""),
                    investor_name=inv.get("name", ""),
                    decision="INVEST" if prob >= THRESHOLD else "SKIP",
                    probability=round(prob, 4),
                    match_percentage=round(prob * 100, 1),
                    confidence_level=_conf(prob),
                    positive_signals=pos,
                    negative_signals=neg,
                )
            )
        except Exception as e:
            print(f"Bulk prediction failed for investor {inv.get('id', '')}: {e}")
            out.append(
                BulkResult(
                    investor_id=inv.get("id", ""),
                    investor_name=inv.get("name", ""),
                    decision="SKIP",
                    probability=0.0,
                    match_percentage=0.0,
                    confidence_level="Low",
                    positive_signals=[],
                    negative_signals=[],
                )
            )

    out.sort(key=lambda x: x.probability, reverse=True)
    return out


def _signals(text: str):
    lowered = text.lower()
    positives = [k for k in POSITIVE_KEYWORDS if k in lowered]
    negatives = [k for k in NEGATIVE_KEYWORDS if k in lowered]
    return positives, negatives


def _features(req: PredictRequest, pos: List[str], neg: List[str]):
    values = []
    text = f"{req.project_title} {req.project_description}".strip()
    tfidf_size = len(getattr(tfidf, "vocabulary_", {})) or 500

    if tfidf is not None:
        values.extend(tfidf.transform([text]).toarray()[0].tolist())
    else:
        values.extend([0.0] * tfidf_size)

    for key in POSITIVE_KEYWORDS:
        values.append(1.0 if key in pos else 0.0)
    for key in NEGATIVE_KEYWORDS:
        values.append(1.0 if key in neg else 0.0)

    category_match = 1.0 if req.project_category in (req.investor_industries or []) else 0.0
    values += [
        0.40,
        category_match,
        category_match / max(len(req.investor_industries or [1]), 1),
        float(len(pos) - 3 * len(neg)),
    ]

    category_lower = req.project_category.lower()
    for category in CATEGORIES:
        category_name = category.lower()
        values.append(
            1.0
            if category_lower in category_name or category_name in category_lower
            else 0.0
        )

    for investor_name in _active_investors():
        values.append(1.0 if req.investor_name == investor_name else 0.0)

    return _reshape_features(values)


def _proba(features):
    if ensemble_models:
        probs = [_predict_single(single_model, features) for single_model in ensemble_models]
        return float(sum(probs) / len(probs))

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0][1])

    if hasattr(model, "predict"):
        value = float(model.predict(features)[0])
        return value if 0 <= value <= 1 else float(value > 0.5)

    return 0.0


def _predict_single(single_model, features):
    if hasattr(single_model, "predict_proba"):
        return float(single_model.predict_proba(features)[0][1])
    if hasattr(single_model, "predict"):
        value = float(single_model.predict(features)[0])
        return value if 0 <= value <= 1 else float(value > 0.5)
    return 0.0


def _expected_feature_count():
    if ensemble_models:
        return getattr(ensemble_models[0], "n_features_in_", None)
    return getattr(model, "n_features_in_", None)


def _active_investors():
    expected = _expected_feature_count()
    if expected is None:
        return INVESTORS

    tfidf_size = len(getattr(tfidf, "vocabulary_", {})) or 500
    base_count = (
        tfidf_size
        + len(POSITIVE_KEYWORDS)
        + len(NEGATIVE_KEYWORDS)
        + 4
        + len(CATEGORIES)
    )
    investor_slots = max(0, expected - base_count)
    return INVESTORS[:investor_slots]


def _reshape_features(values):
    expected = _expected_feature_count()
    if expected is None:
        return np.array(values, dtype=np.float32).reshape(1, -1)

    if len(values) < expected:
        values = values + [0.0] * (expected - len(values))
    elif len(values) > expected:
        values = values[:expected]

    return np.array(values, dtype=np.float32).reshape(1, -1)


def _conf(probability: float):
    if probability >= 0.75 or probability <= 0.25:
        return "High"
    if probability >= 0.60 or probability <= 0.40:
        return "Medium"
    return "Low"
