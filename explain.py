import re
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance

TREES = ("Arbre de décision", "Random Forest", "XGBoost")


def clean(df):
    """Même nettoyage des noms de colonnes que dans prep.py (XGBoost refuse < > [ ])."""
    df = df.copy()
    df.columns = [re.sub(r"[\[\]<>]", "_", c) for c in df.columns]
    return df


def origin(col, raw_cols):
    """Retrouve la variable d'origine d'une colonne encodée (ex. gender_Male -> gender)."""
    matches = [r for r in raw_cols if col == r or col.startswith(r + "_")]
    return max(matches, key=len) if matches else col


# ---------- Importances globales ----------
def own_importance(name, model, cols):
    if name == "Régression logistique":
        return pd.Series(model.coef_[0], index=cols)
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=cols)
    return None  # k-NN n'a pas d'importance propre


def perm_importance(model, X, y, n=1000, repeats=3):
    """Baisse d'AUC-ROC quand on mélange une variable (même mesure pour tous les modèles)."""
    idx = X.sample(min(n, len(X)), random_state=0).index
    r = permutation_importance(model, X.loc[idx], y.loc[idx], scoring="roc_auc",
                               n_repeats=repeats, random_state=0, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns)


# ---------- SHAP ----------
def make_explainer(name, model, X_train):
    if name in TREES:
        return shap.TreeExplainer(model)
    if name == "Régression logistique":
        return shap.LinearExplainer(model, X_train)
    background = shap.sample(X_train, 50, random_state=0)  # k-NN : méthode générique, lente
    predict = lambda d: model.predict_proba(pd.DataFrame(d, columns=X_train.columns))[:, 1]
    return shap.KernelExplainer(predict, background)


def shap_explain(name, explainer, X):
    """Valeurs SHAP de la classe positive, dans un format unique pour tous les modèles."""
    if name == "k plus proches voisins":
        vals = explainer.shap_values(X, nsamples=300, silent=True)
    else:
        vals = explainer.shap_values(X)
    if isinstance(vals, list):          # anciennes versions de shap : une liste par classe
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:                  # nouvelles versions : (individus, variables, classes)
        vals = vals[:, :, 1]
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    return shap.Explanation(values=vals, base_values=np.full(len(X), base),
                            data=X.values, feature_names=list(X.columns))


# ---------- LIME ----------
def lime_explain(model, X_train, x_row, num_features=10):
    cols = list(X_train.columns)
    binary = [i for i, c in enumerate(cols) if X_train[c].nunique() <= 2]  # colonnes 0/1
    explainer = LimeTabularExplainer(X_train.values, feature_names=cols,
                                     class_names=["négatif", "positif"],
                                     categorical_features=binary, mode="classification",
                                     discretize_continuous=True, random_state=42)
    predict = lambda d: model.predict_proba(pd.DataFrame(d, columns=cols))
    return explainer.explain_instance(x_row.values[0], predict,
                                      num_features=num_features, num_samples=2000)


def compare(sv, exp, k=10):
    """Tableau SHAP vs LIME pour un individu + nombre de variables communes dans le top 5."""
    cols = sv.feature_names
    s = pd.Series(sv.values, index=cols)
    l = pd.Series({cols[i]: w for i, w in exp.as_map()[1]})
    keep = list(set(s.abs().nlargest(k).index) | set(l.index))
    df = pd.DataFrame({"SHAP": s, "LIME": l}).loc[keep]
    same = np.sign(df["SHAP"]) == np.sign(df["LIME"])
    df["Même sens"] = np.where(df["LIME"].isna(), "—", np.where(same, "oui", "non"))
    df = df.reindex(df["SHAP"].abs().sort_values(ascending=False).index)
    common = len(set(s.abs().nlargest(5).index) & set(l.abs().nlargest(5).index))
    return df, common


def summary_sentence(sv, proba, thr, k=2):
    s = pd.Series(sv.values, index=sv.feature_names)
    high = proba >= thr
    top = s[s > 0].nlargest(k) if high else s[s < 0].nsmallest(k)
    txt = (f"La probabilité de la classe positive est {'élevée' if high else 'faible'} "
           f"({proba:.0%}, seuil {thr:.2f})")
    if len(top):
        txt += ", principalement en raison de : " + " et ".join(f"« {c} »" for c in top.index)
    return txt + "."