# app.py – Customer Credibility API
from fastapi import FastAPI
from pydantic import BaseModel
import torch, re, math
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────
app = FastAPI(title="Customer Credibility API")   # ➊ must exist first
# ────────────────────────────────────────────────────────────────

# grade helper
def cred_grade(score: float) -> str:
    if score > 70: return "Excellent"
    if score > 50: return "Good"
    if score > 30: return "Moderate"
    return "Suspicious"

# exact MultiReg definition (same as training)
class MultiReg(torch.nn.Module):
    def __init__(self, num_numeric: int = 4, hidden_num: int = 16, fusion: int = 64):
        super().__init__()
        from transformers import AutoModel
        self.text = AutoModel.from_pretrained("bert-base-uncased")
        d = self.text.config.hidden_size
        self.num_net = torch.nn.Sequential(
            torch.nn.Linear(num_numeric, hidden_num), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_num, hidden_num), torch.nn.ReLU())
        self.fuse = torch.nn.Sequential(
            torch.nn.Linear(d + hidden_num, fusion), torch.nn.ReLU(), torch.nn.Dropout(0.3))
        self.head_T = torch.nn.Linear(fusion, 1)
        self.head_P = torch.nn.Linear(fusion, 1)
        self.head_A = torch.nn.Linear(fusion, 1)
    def forward(self, ids, mask, num):
        txt = self.text(ids, attention_mask=mask).last_hidden_state[:, 0]
        num = self.num_net(num)
        h   = self.fuse(torch.cat([txt, num], 1))
        return self.head_T(h), self.head_P(h), self.head_A(h)

# load tokenizer & model once
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model = MultiReg().to(DEVICE)
state = torch.load("customer_multireg.pt", map_location=DEVICE)
model.load_state_dict(state, strict=False)   # ignore unmatched keys
model.eval()

# payload schema
class CustomerData(BaseModel):
    review_text: str
    star_rating: float
    verified_purchase: int
    customer_tenure_months: float
    purchase_value_rupees: float

# === calibration helpers =======================================
def tenure_score_from_months(m, t50=18, slope=5):
    return 100 / (1 + math.exp(-(m - t50) / slope))          # logistic

def purchase_score_from_value(v, mid=6000):
    return 100 * (1 - math.exp(-v / mid))                    # S-curve

def auth_boost(raw_auth, verified, spam):
    bonus   = 25 if verified and not spam else 0
    penalty = -10 if spam else 0
    return max(0, min(100, raw_auth + bonus + penalty))
# ==============================================================

@app.post("/predict_customer")
def predict_customer(data: CustomerData):
    # 1) text → BERT tokens
    enc = tokenizer(data.review_text, truncation=True, padding="max_length",
                    max_length=128, return_tensors="pt")
    ids, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)

    # 2) numeric tensor
    num = torch.tensor([[data.star_rating, data.verified_purchase,
                         data.customer_tenure_months, data.purchase_value_rupees]],
                       dtype=torch.float32).to(DEVICE)

    # 3) raw network outputs
    with torch.no_grad():
        T_hat, P_hat, A_hat = model(ids, mask, num)
    T_raw, P_raw, A_raw = map(float, (T_hat, P_hat, A_hat))

    # 4) rule scores
    spam_flag = bool(re.search(r"(buy now|limited time|!!!!!|free)", data.review_text, re.I))
    T_rule = tenure_score_from_months(data.customer_tenure_months)
    P_rule = purchase_score_from_value(data.purchase_value_rupees)
    A_rule = auth_boost(A_raw, data.verified_purchase == 1, spam_flag)

    # 5) blend (50 % rule, 50 % model)
    BLEND = 0.5
    T = BLEND * T_rule + (1 - BLEND) * T_raw
    P = BLEND * P_rule + (1 - BLEND) * P_raw
    A = BLEND * A_rule + (1 - BLEND) * A_raw

    # 6) credibility & optional floor boost
    cred = 0.25 * T + 0.35 * P + 0.40 * A
    if data.verified_purchase == 1 and data.customer_tenure_months >= 24:
        cred = max(cred, 60)      # guarantee at least “Good” for loyal verified users
    cred = round(cred, 2)

    return {
        "pred_tenure_score":   round(T, 2),
        "pred_purchase_score": round(P, 2),
        "pred_auth_score":     round(A, 2),
        "credibility_score":   cred,
        "grade": cred_grade(cred)
    }
