import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

NAMES = ["Régression logistique", "Arbre de décision", "k plus proches voisins",
         "Random Forest", "XGBoost"]

# Grilles de recherche (petites, pour que ça reste rapide)
GRIDS = {
    "Régression logistique": {"C": [0.01, 0.1, 1, 10]},
    "Arbre de décision": {"max_depth": [3, 5, 8, 12], "min_samples_leaf": [5, 20, 50]},
    "k plus proches voisins": {"n_neighbors": [5, 15, 31, 51]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [5, 10, 15]},
    "XGBoost": {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
}


def build(name, balance, pos_weight):
    """Crée un modèle avec ses réglages de base."""
    w = "balanced" if balance else None
    if name == "Régression logistique":
        return LogisticRegression(max_iter=1000, class_weight=w)
    if name == "Arbre de décision":
        return DecisionTreeClassifier(class_weight=w, random_state=42)
    if name == "k plus proches voisins":
        return KNeighborsClassifier()
    if name == "Random Forest":
        return RandomForestClassifier(class_weight=w, random_state=42)
    return XGBClassifier(eval_metric="logloss", random_state=42,
                         scale_pos_weight=pos_weight if balance else 1)


def train_one(name, params, X_tr, y_tr, X_te, y_te, balance=True, optimize=False):
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    model = build(name, balance, pos_weight)
    t0 = time.time()

    if optimize:  # recherche du meilleur réglage, validation croisée à 3 plis
        search = GridSearchCV(model, GRIDS[name], scoring="roc_auc", cv=3, n_jobs=-1)
        search.fit(X_tr, y_tr)
        model, used = search.best_estimator_, search.best_params_
    else:         # réglages choisis à la main
        model.set_params(**params).fit(X_tr, y_tr)
        used = params

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    row = {"Modèle": name,
           "AUC-ROC": roc_auc_score(y_te, proba),
           "Précision": precision_score(y_te, pred, zero_division=0),
           "Rappel": recall_score(y_te, pred),
           "F1": f1_score(y_te, pred),
           "Temps (s)": time.time() - t0,
           "Réglages": str(used)}
    return model, proba, row