# 🤖 ربط investor_model.pkl بتطبيق VentureBridge

## ما يفعله النموذج
- **Ensemble**: RF (40%) + Extra Trees (40%) + Logistic Regression (20%)
- **Input**: نص وصف المشروع + بيانات المستثمر
- **Output**: INVEST / SKIP + احتمالية (0-1)
- **Threshold**: 0.47
- **Accuracy**: 93.14% | **AUC-ROC**: 0.986

---

## الجزء 1 — Backend

### الخطوة 1: هيكل المجلد
```
ml_backend/
├── main.py
├── requirements.txt
├── Procfile
├── investor_model.pkl      ← ضعه هنا
└── tfidf_vectorizer.pkl    ← ضعه هنا (إذا كان منفصلاً)
```

> **ملاحظة مهمة**: إذا كان النموذج `Pipeline` (يحتوي TF-IDF بداخله)،
> لا تحتاج `tfidf_vectorizer.pkl`. جرّب أولاً بدونه.

### الخطوة 2: تشغيل محلياً
```bash
cd ml_backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
اختبر: http://localhost:8000/docs

### الخطوة 3: اختبار يدوي
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "project_description": "We have 500K subscribers and 40% monthly growth. Patent pending. Revenue is $2M ARR with strong retention.",
    "project_title": "AI Health App",
    "project_category": "Health/Wellness",
    "funding_goal": 500000,
    "investor_name": "Mark Cuban",
    "investor_bio": "I invest in technology and health companies",
    "investor_industries": ["Technology", "Health/Wellness"]
  }'
```

**النتيجة المتوقعة:**
```json
{
  "decision": "INVEST",
  "probability": 0.82,
  "match_percentage": 82.0,
  "confidence_level": "High",
  "positive_signals": ["subscribers", "growth", "patent", "revenue", "retention"],
  "negative_signals": [],
  "explanation": "The model predicts 82.0% chance that Mark Cuban will invest..."
}
```

---

## الجزء 2 — النشر على Render (مجاناً)

1. ارفع المجلد على GitHub (بما فيه ملف `.pkl`)
2. اذهب إلى [render.com](https://render.com) → New Web Service
3. اربطه بالـ repo
4. **Build Command**: `pip install -r requirements.txt`
5. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. انتظر النشر (3-5 دقائق) → ستحصل على رابط مثل:
   ```
   https://venturebridge-ml-api.onrender.com
   ```

---

## الجزء 3 — Flutter

### 1. أضف package في pubspec.yaml
```yaml
http: ^1.2.0
```

### 2. انسخ الملفات
```
ml_service.dart      → lib/core/services/ml_service.dart
match_provider_ml.dart → lib/providers/match_provider.dart
```

### 3. غيّر الرابط في ml_service.dart
```dart
static const String remoteUrl = 'https://venturebridge-ml-api.onrender.com';
static const String baseUrl   = remoteUrl;
```

### 4. أضف للـ pubspec.yaml إذا لم يكن موجوداً
```yaml
http: ^1.2.0
```

---

## تشخيص المشاكل

### إذا كان النموذج Pipeline:
افتح Python وتحقق:
```python
import pickle
model = pickle.load(open('investor_model.pkl', 'rb'))
print(type(model))
print(dir(model))

# إذا كان Pipeline:
# <class 'sklearn.pipeline.Pipeline'>
# يمكنك تمرير النص مباشرة:
result = model.predict(["project description here"])
```

### إذا كان النموذج عبارة عن dict:
```python
# بعض النماذج تُحفظ كـ dict يحتوي على عدة objects
print(model.keys())
# مثل: {'rf': ..., 'et': ..., 'lr': ..., 'tfidf': ...}
```

في هذه الحالة عدّل `load_artifacts()` في `main.py`:
```python
# مثال إذا كان النموذج dict
model_dict = pickle.load(f)
rf    = model_dict['rf']
et    = model_dict['et']
lr    = model_dict['lr']
tfidf = model_dict['tfidf']
```
