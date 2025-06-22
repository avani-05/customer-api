import os
import re
import math
import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Tell NLTK to use /tmp for downloads ───────────────────────────────
nltk.data.path.insert(0, "/tmp/nltk_data")

# ── FastAPI setup ────────────────────────────────────────────────────
app = FastAPI(title="Customer Credibility API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Global placeholder for the analyzer ───────────────────────────────
sia: SentimentIntensityAnalyzer  # will be initialized on startup

# ── Download lexicon & init analyzer ──────────────────────────────────
@app.on_event("startup")
def setup_vader():
    # Download into /tmp/nltk_data (one time on boot)
    nltk.download("vader_lexicon", download_dir="/tmp/nltk_data", quiet=True)
    # Now that the data is there, instantiate the analyzer
    global sia
    sia = SentimentIntensityAnalyzer()

# ── Health check ──────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "ok"}

# ── Rule-based helpers ─────────────────────────────────────────────────
def tenure_score(m, t50=18, slope=5):
    return 100 / (1 + math.exp(-(m - t50) / slope))

def purchase_score(v, mid=6000):
    return 100 * (1 - math.exp(-v / mid))

def auth_score(text, verified, spam):
    # compound in [-1,1] → [0,100]
    sent = sia.polarity_scores(text)["compound"]
    raw  = (sent + 1) / 2 * 100
    boost = 25 if verified and not spam else 0
    pen   = -10 if spam else 0
    return max(0, min(100, raw + boost + pen))

# ── Request schema ────────────────────────────────────────────────────
class CustomerData(BaseModel):
    review_text: str
    star_rating: float
    verified_purchase: int
    customer_tenure_months: float
    purchase_value_rupees: float

# ── Prediction endpoint ───────────────────────────────────────────────
@app.post("/predict_customer")
def predict_customer(data: CustomerData):
    T = tenure_score(data.customer_tenure_months)
    P = purchase_score(data.purchase_value_rupees)
    spam_flag = bool(re.search(r"(buy now|limited time|!!!!!|free)", data.review_text, re.I))
    A = auth_score(data.review_text, data.verified_purchase == 1, spam_flag)

    cred = 0.25*T + 0.35*P + 0.40*A
    if data.verified_purchase == 1 and data.customer_tenure_months >= 24:
        cred = max(cred, 60)
    cred = round(cred, 2)

    grade = (
        "Excellent" if cred > 70 else
        "Good"      if cred > 50 else
        "Moderate"  if cred > 30 else
        "Suspicious"
    )

    return {
        "pred_tenure_score":   round(T, 2),
        "pred_purchase_score": round(P, 2),
        "pred_auth_score":     round(A, 2),
        "credibility_score":   cred,
        "grade":               grade
    }

# ── Entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
