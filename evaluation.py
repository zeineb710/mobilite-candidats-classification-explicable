import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def metrics_at(y_true, proba, thr):
    """Précision, rappel et F1 de la classe positive pour un seuil donné."""
    pred = (proba >= thr).astype(int)
    return {"Précision": precision_score(y_true, pred, zero_division=0),
            "Rappel": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred)}


def comparison_table(results, y_true, thr):
    """Un tableau : les métriques indépendantes du seuil + celles qui en dépendent."""
    rows = []
    for name, r in results.items():
        rows.append({"Modèle": name,
                     "AUC-ROC": roc_auc_score(y_true, r["proba"]),
                     "PR-AUC": average_precision_score(y_true, r["proba"]),
                     **metrics_at(y_true, r["proba"], thr)})
    return pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)