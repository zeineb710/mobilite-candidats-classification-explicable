from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from explain import clean, make_explainer, origin, shap_explain, summary_sentence
from prep import convert_ordinal_text

ART = {}  # modèle et préparation, chargés une seule fois au démarrage

from fastapi.responses import RedirectResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    ART.update(joblib.load("artifacts/model.joblib"))
    ART["explainer"] = make_explainer(ART["name"], ART["model"], ART["background"])
    yield


app = FastAPI(title="API d'interprétation : mobilité des candidats",
              description="Prédit si un candidat cherche un nouvel emploi et explique pourquoi (SHAP).",
              lifespan=lifespan)


class Candidate(BaseModel):
    features: dict = Field(..., description="Variables brutes du candidat (voir GET /schema). "
                                            "Une variable absente ou null est traitée comme manquante.")
    threshold: float | None = Field(None, ge=0, le=1,
                                    description="Seuil de décision. Par défaut : celui choisi dans l'application.")


class Prediction(BaseModel):
    model: str
    probability: float
    threshold: float
    decision: str


class Explanation(Prediction):
    summary: str
    contributions: list[dict]
    by_variable: dict[str, float]


def to_matrix(features: dict) -> pd.DataFrame:
    """Applique à un candidat exactement la même préparation que pour l'entraînement."""
    unknown = set(features) - set(ART["raw_cols"])
    if unknown:
        raise HTTPException(422, f"Variables inconnues : {sorted(unknown)}")
    df = pd.DataFrame([features]).reindex(columns=ART["raw_cols"])
    if ART["convert_ordinal"]:
        df = convert_ordinal_text(df)
    for c in ART["num_cols"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ART["cat_cols"]:
        df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
    X = clean(ART["preprocessor"].transform(df))[ART["columns"]]
    if X.isna().any().any():
        raise HTTPException(422, "Valeurs manquantes non gérées par la préparation choisie.")
    return X


def score(req: Candidate):
    X = to_matrix(req.features)
    proba = float(ART["model"].predict_proba(X)[0, 1])
    thr = req.threshold if req.threshold is not None else ART["threshold"]
    return X, proba, thr, ("positive" if proba >= thr else "negative")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

@app.get("/health")
def health():
    return {"status": "ok", "model": ART["name"]}


@app.get("/schema")
def schema():
    """Variables attendues en entrée, avec les valeurs les plus fréquentes pour les catégories."""
    return {"model": ART["name"], "default_threshold": ART["threshold"],
            "numeric": ART["num_cols"], "categorical": ART["choices"]}


@app.post("/predict", response_model=Prediction)
def predict(req: Candidate):
    _, proba, thr, decision = score(req)
    return {"model": ART["name"], "probability": round(proba, 4), "threshold": thr, "decision": decision}


@app.post("/explain", response_model=Explanation)
def explain(req: Candidate, top: int = 10):
    X, proba, thr, decision = score(req)
    sv = shap_explain(ART["name"], ART["explainer"], X)[0]
    s = pd.Series(sv.values, index=sv.feature_names)
    best = s.reindex(s.abs().nlargest(top).index)
    by_var = s.groupby([origin(c, ART["raw_cols"]) for c in s.index]).sum()
    by_var = by_var.reindex(by_var.abs().sort_values(ascending=False).index).head(top)
    return {"model": ART["name"], "probability": round(proba, 4), "threshold": thr, "decision": decision,
            "summary": summary_sentence(sv, proba, thr),
            "contributions": [{"feature": f, "shap_value": round(float(v), 4),
                               "effect": "pousse vers la classe positive" if v > 0
                               else "pousse vers la classe négative"} for f, v in best.items()],
            "by_variable": {k: round(float(v), 4) for k, v in by_var.items()}}

