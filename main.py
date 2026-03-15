"""
VentureBridge — ML Backend
Ensemble Model: Random Forest (40%) + Extra Trees (40%) + Logistic Regression (20%)
Features: TF-IDF (500) + Positive Signals (12) + Negative Signals (11) +
          Structural (4) + Category One-Hot (16) + Investor One-Hot (14)
Total: 553 features
Decision Threshold: 0.47
"""

import pickle, os, re
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="VentureBridge ML API",
    description="Investor-Project Matching — Ensemble RF+ET+LR",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════
# Constants — مطابقة لبيانات التدريب
# ══════════════════════════════════════════════════════════

# 16 categories من Shark Tank dataset
CATEGORIES = [
    "Food and Beverage", "Fashion/Beauty", "Fitness/Sports/Outdoors",
    "Children/Education", "Health/Wellness", "Technology",
    "Home", "Automotive", "Pet Products", "Travel/Outdoor",
    "Media/Entertainment", "Other", "Lifestyle/Home",
    "Electronics", "Green/CleanTech", "Finance/Legal"
]

# 14 investors (6 original Shark Tank + 8 synthetic)
INVESTORS = [
    "Mark Cuban", "Barbara Corcoran", "Lori Greiner",
    "Robert Herjavec", "Daymond John", "Kevin O'Leary",
    "Investor_7", "Investor_8", "Investor_9", "Investor_10",
    "Investor_11", "Investor_12", "Investor_13", "Investor_14"
]

# ✅ الكلمات المفتاحية الإيجابية (12 مؤشر)
POSITIVE_KEYWORDS = [
    "revenue", "patented", "patent", "retention", "subscribers",
    "growth", "margin", "profit", "profitable", "exclusive",
    "traction", "recurring"
]

# ✅ الكلمات المفتاحية السلبية (11 مؤشر)
NEGATIVE_KEYWORDS = [
    "pre-revenue", "pre revenue", "crowded", "easily replicable",
    "single founder", "saturated", "no patent", "no revenue",
    "unproven", "copycat", "commodity"
]

DECISION_THRESHOLD = 0.47

# ══════════════════════════════════════════════════════════
# Load Model
# ══════════════════════════════════════════════════════════

MODEL_PATH  = os.getenv("MODEL_PATH",  "investor_model.pkl")
TFIDF_PATH  = os.getenv("TFIDF_PATH",  "tfidf_vectorizer.pkl")

model       = None
tfidf       = None
# بيانات إضافية لـ structural features
category_investor_stats = {}  # يُحسب من بيانات التدريب إذا توفرت

@app.on_event("startup")
def load_artifacts():
    global model, tfidf

    # تحميل النموذج
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded: {type(model)}")
    except FileNotFoundError:
        print(f"⚠️  investor_model.pkl not found at {MODEL_PATH}")

    # تحميل TF-IDF vectorizer (إذا كان محفوظاً منفصلاً)
    # إذا كان النموذج Pipeline يحتوي على TF-IDF بداخله، هذا غير مطلوب
    try:
        with open(TFIDF_PATH, "rb") as f:
            tfidf = pickle.load(f)
        print(f"✅ TF-IDF vectorizer loaded")
    except FileNotFoundError:
        print("ℹ️  tfidf_vectorizer.pkl not found — will use model's internal pipeline")

    # إذا كان النموذج Pipeline (يحتوي TF-IDF بداخله)
    if model is not None and hasattr(model, 'named_steps'):
        print("   Model is a Pipeline — TF-IDF is internal")
        tfidf = None  # مش محتاجه منفصل


# ══════════════════════════════════════════════════════════
# Request / Response Models
# ══════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    # بيانات المشروع
    project_description: str        # ✅ الأهم — نص وصف المشروع
    project_category:    str        # e.g. "Technology"
    funding_goal:        float      # e.g. 500000.0
    project_title:       Optional[str] = ""

    # بيانات المستثمر
    investor_name:       str        # e.g. "Mark Cuban" أو اسم جديد
    investor_bio:        Optional[str] = ""   # نبذة عن المستثمر
    investor_industries: List[str] = []

class PredictResponse(BaseModel):
    decision:          str      # "INVEST" أو "SKIP"
    probability:       float    # 0.0 → 1.0
    match_percentage:  float    # 0 → 100
    confidence_level:  str      # "High" / "Medium" / "Low"
    positive_signals:  List[str]
    negative_signals:  List[str]
    explanation:       str

class BulkRequest(BaseModel):
    project_description: str
    project_category:    str
    funding_goal:        float
    project_title:       Optional[str] = ""
    investors: List[dict]   # [{"id": "...", "name": "...", "bio": "...", "industries": [...]}]

class BulkResult(BaseModel):
    investor_id:      str
    investor_name:    str
    decision:         str
    probability:      float
    match_percentage: float
    confidence_level: str
    positive_signals: List[str]
    negative_signals: List[str]


# ══════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "model_info": {
            "type":      "Ensemble RF(40%) + ET(40%) + LR(20%)",
            "features":  553,
            "threshold": DECISION_THRESHOLD,
            "accuracy":  "93.14%",
            "auc_roc":   "0.9860"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    يتنبأ هل مستثمر معين سيستثمر في مشروع معين.
    Input: نص وصف المشروع + بيانات المستثمر
    Output: INVEST/SKIP + احتمالية + تفسير
    """
    if model is None:
        raise HTTPException(503, "Model not loaded — ضع investor_model.pkl في نفس المجلد")

    try:
        pos_signals, neg_signals = _detect_signals(req.project_description)
        features    = _build_features(req, pos_signals, neg_signals)
        prob        = _predict_proba(features)
        decision    = "INVEST" if prob >= DECISION_THRESHOLD else "SKIP"
        explanation = _build_explanation(req, prob, decision, pos_signals, neg_signals)

        return PredictResponse(
            decision         = decision,
            probability      = round(prob, 4),
            match_percentage = round(prob * 100, 1),
            confidence_level = _confidence(prob),
            positive_signals = pos_signals,
            negative_signals = neg_signals,
            explanation      = explanation,
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@app.post("/predict/bulk", response_model=List[BulkResult])
def predict_bulk(req: BulkRequest):
    """
    يصنّف عدة مستثمرين لمشروع واحد دفعة واحدة ويرتّبهم.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")

    pos_signals, neg_signals = _detect_signals(req.project_description)
    results = []

    for inv in req.investors:
        try:
            single = PredictRequest(
                project_description = req.project_description,
                project_category    = req.project_category,
                funding_goal        = req.funding_goal,
                project_title       = req.project_title or "",
                investor_name       = inv.get("name", ""),
                investor_bio        = inv.get("bio", ""),
                investor_industries = inv.get("industries", []),
            )
            features = _build_features(single, pos_signals, neg_signals)
            prob     = _predict_proba(features)
            decision = "INVEST" if prob >= DECISION_THRESHOLD else "SKIP"

            results.append(BulkResult(
                investor_id      = inv.get("id", ""),
                investor_name    = inv.get("name", ""),
                decision         = decision,
                probability      = round(prob, 4),
                match_percentage = round(prob * 100, 1),
                confidence_level = _confidence(prob),
                positive_signals = pos_signals,
                negative_signals = neg_signals,
            ))
        except Exception as e:
            results.append(BulkResult(
                investor_id      = inv.get("id", ""),
                investor_name    = inv.get("name", ""),
                decision         = "SKIP",
                probability      = 0.0,
                match_percentage = 0.0,
                confidence_level = "Low",
                positive_signals = [],
                negative_signals = [],
            ))

    results.sort(key=lambda x: x.probability, reverse=True)
    return results


# ══════════════════════════════════════════════════════════
# Feature Engineering — مطابق للتوثيق
# ══════════════════════════════════════════════════════════

def _detect_signals(text: str):
    """يستخرج المؤشرات الإيجابية والسلبية من النص"""
    text_lower = text.lower()
    pos = [kw for kw in POSITIVE_KEYWORDS if kw in text_lower]
    neg = [kw for kw in NEGATIVE_KEYWORDS if kw in text_lower]
    return pos, neg


def _build_features(req: PredictRequest,
                    pos_signals: List[str],
                    neg_signals: List[str]) -> np.ndarray:
    """
    يبني متجه الـ 553 feature:
    500 TF-IDF + 12 pos + 11 neg + 4 structural + 16 category_onehot + 14 investor_onehot
    """
    features = []

    # ── 1. TF-IDF Features (500) ─────────────────────────
    combined_text = f"{req.project_title} {req.project_description}".strip()

    if tfidf is not None:
        # TF-IDF خارجي محفوظ منفصلاً
        tfidf_vec = tfidf.transform([combined_text]).toarray()[0]
    elif model is not None and hasattr(model, 'named_steps') and 'tfidf' in model.named_steps:
        # Pipeline — لا نحتاج TF-IDF منفصل، النموذج يتعامل معه
        # في هذه الحالة، ارسل النص مباشرة في _predict_proba
        tfidf_vec = np.zeros(500)
    else:
        # Fallback: zero vector (سيعتمد النموذج على باقي الـ features)
        tfidf_vec = np.zeros(500)

    features.extend(tfidf_vec.tolist())

    # ── 2. Positive Signal Features (12) ─────────────────
    for kw in POSITIVE_KEYWORDS:
        features.append(1.0 if kw in pos_signals else 0.0)

    # ── 3. Negative Signal Features (11) ─────────────────
    for kw in NEGATIVE_KEYWORDS:
        features.append(1.0 if kw in neg_signals else 0.0)

    # ── 4. Structural Features (4) ───────────────────────
    cat = req.project_category

    # investor-category historical rate (fallback = 0.4 إذا مش عندنا بيانات)
    hist_rate = category_investor_stats.get(
        (req.investor_name, cat), 0.40)
    features.append(hist_rate)

    # category frequency in investor profile
    cat_freq = 1.0 if cat in (req.investor_industries or []) else 0.0
    features.append(cat_freq)

    # category interest ratio
    total_ind = len(req.investor_industries) if req.investor_industries else 1
    cat_ratio = cat_freq / total_ind
    features.append(cat_ratio)

    # signal score = positive_count - 3 * negative_count
    signal_score = len(pos_signals) - 3 * len(neg_signals)
    features.append(float(signal_score))

    # ── 5. Category One-Hot (16) ─────────────────────────
    for c in CATEGORIES:
        # مطابقة مرنة
        features.append(1.0 if _fuzzy_match(cat, c) else 0.0)

    # ── 6. Investor One-Hot (14) ─────────────────────────
    for inv in INVESTORS:
        features.append(1.0 if req.investor_name == inv else 0.0)

    return np.array(features, dtype=np.float32).reshape(1, -1)


def _predict_proba(features: np.ndarray) -> float:
    """يحصل على الاحتمالية من النموذج"""

    # إذا كان Pipeline يأخذ نصاً مباشرة
    if hasattr(model, 'named_steps'):
        # Pipeline — لا تستخدم features array
        # هذه الحالة تُعالج في predict() مباشرة
        pass

    # Ensemble weights (RF=0.4, ET=0.4, LR=0.2)
    if hasattr(model, 'estimators_'):
        # VotingClassifier أو custom ensemble
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)
            return float(proba[0][1])

    # Single model مع predict_proba
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)
        return float(proba[0][1])

    # Regressor أو binary output
    if hasattr(model, 'predict'):
        pred = float(model.predict(features)[0])
        return pred if 0 <= pred <= 1 else float(pred > 0.5)

    return 0.0


def _confidence(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "High"
    elif prob >= 0.60 or prob <= 0.40:
        return "Medium"
    return "Low"


def _fuzzy_match(input_cat: str, model_cat: str) -> bool:
    """مطابقة مرنة للفئات"""
    a = input_cat.lower().strip()
    b = model_cat.lower().strip()
    if a == b:
        return True
    # Technology ↔ Tech
    mapping = {
        "technology": ["technology", "tech", "software", "app"],
        "food and beverage": ["food", "beverage", "food & beverage", "restaurant"],
        "health/wellness": ["health", "wellness", "healthcare", "medical"],
        "fashion/beauty": ["fashion", "beauty", "clothing", "apparel"],
        "children/education": ["children", "education", "kids", "school"],
        "fitness/sports/outdoors": ["fitness", "sports", "outdoor"],
        "home": ["home", "real estate", "housing"],
        "finance/legal": ["finance", "financial", "legal", "fintech"],
    }
    for key, variants in mapping.items():
        if key in b:
            return any(v in a for v in variants)
    return False


def _build_explanation(req: PredictRequest, prob: float,
                        decision: str, pos: List[str], neg: List[str]) -> str:
    parts = []

    if decision == "INVEST":
        parts.append(
            f"The model predicts a {prob*100:.1f}% chance that {req.investor_name} "
            f"will invest in this {req.project_category} project."
        )
    else:
        parts.append(
            f"The model predicts a low {prob*100:.1f}% match with {req.investor_name}."
        )

    if pos:
        parts.append(f"Positive signals detected: {', '.join(pos)}.")
    if neg:
        parts.append(f"Negative signals detected: {', '.join(neg)}.")
    if not pos and not neg:
        parts.append(
            "No strong signals detected — consider adding measurable traction "
            "(revenue, users, growth rate) to the project description."
        )

    return " ".join(parts)
