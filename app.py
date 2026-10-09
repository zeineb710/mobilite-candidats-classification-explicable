import os
import pandas as pd
import plotly.express as px
import streamlit as st
import json
import os
import joblib
import requests
import matplotlib.pyplot as plt
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from prep import prepare
from models import NAMES, train_one
from evaluation import comparison_table, metrics_at
from explain import (clean, compare, lime_explain, make_explainer, origin,
                     own_importance, perm_importance, shap_explain, summary_sentence)

from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

st.set_page_config(page_title="Mobilité des candidats", layout="wide")
st.title("Mobilité des candidats : classification explicable")

KAGGLE_SLUG = "arashnic/hr-analytics-job-change-of-data-scientists"
KAGGLE_FILE = "aug_train.csv"


@st.cache_data(show_spinner="Téléchargement du jeu de données...")
def load_kaggle() -> pd.DataFrame:
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_SLUG)
    return pd.read_csv(os.path.join(path, KAGGLE_FILE))


# ---------- 1. Chargement ----------
st.header("1. Données")
source = st.radio("Source", ["Téléchargement automatique", "Importer un CSV"], horizontal=True)

df = None
if source == "Téléchargement automatique":
    try:
        df = load_kaggle()
    except Exception as e:
        st.error(f"Téléchargement impossible ({e}). Importez le CSV à la place.")
else:
    file = st.file_uploader("Fichier CSV", type="csv")
    if file:
        df = pd.read_csv(file)

if df is None:
    st.stop()

# ---------- 2. Cible ----------
default_idx = list(df.columns).index("target") if "target" in df.columns else len(df.columns) - 1
target = st.selectbox("Variable cible", df.columns, index=default_idx)
values = sorted(df[target].dropna().unique().tolist(), key=str)
positive = st.selectbox("Classe d'intérêt (positive)", values, index=len(values) - 1)

df = df.dropna(subset=[target]).copy()
df["_y"] = (df[target] == positive).astype(int)
st.session_state["df"] = df
st.session_state["target"] = target
st.session_state["positive"] = positive

# ---------- 3. Aperçu ----------
st.subheader("Aperçu")
c1, c2, c3 = st.columns(3)
c1.metric("Lignes", f"{len(df):,}")
c2.metric("Colonnes", df.shape[1] - 1)
c3.metric("Classe positive", f"{df['_y'].mean():.1%}")
st.dataframe(df.drop(columns="_y").head(20), use_container_width=True)

with st.expander("Statistiques descriptives"):
    st.dataframe(df.drop(columns="_y").describe(include="all").T, use_container_width=True)

# ---------- 4. Valeurs manquantes ----------
st.subheader("Valeurs manquantes")
miss = (df.drop(columns="_y").isna().mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0].rename("% manquant").reset_index()
miss.columns = ["variable", "% manquant"]
if miss.empty:
    st.success("Aucune valeur manquante.")
else:
    fig = px.bar(miss, x="% manquant", y="variable", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ---------- 5. Proportion des classes ----------
st.subheader("Proportion des classes")
counts = df["_y"].map({1: f"Positive ({positive})", 0: "Négative"}).value_counts().reset_index()
counts.columns = ["classe", "effectif"]
st.plotly_chart(px.pie(counts, names="classe", values="effectif", hole=0.4), use_container_width=True)

# ---------- 6. Visualisations exploratoires ----------
st.header("Exploration")
features = [c for c in df.columns if c not in (target, "_y")]
num_cols = df[features].select_dtypes("number").columns.tolist()
tab1, tab2, tab3 = st.tabs(["Distributions", "Corrélations", "Taux de classe positive"])

with tab1:
    col = st.selectbox("Variable", features, key="dist")
    if col in num_cols:
        fig = px.histogram(df, x=col, color=df["_y"].astype(str), barmode="overlay",
                           opacity=0.6, labels={"color": "classe"})
    else:
        top = df[col].fillna("inconnu").value_counts().head(20).reset_index()
        top.columns = [col, "effectif"]
        fig = px.bar(top, x=col, y="effectif")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    corr = df[num_cols + ["_y"]].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                              color_continuous_scale="RdBu_r"), use_container_width=True)

with tab3:
    col = st.selectbox("Variable", features, key="rate")
    s = df[col]
    if col in num_cols and s.nunique() > 10:
        s = pd.qcut(s, 5, duplicates="drop").astype(str)
    s = s.fillna("inconnu").astype(str)
    rate = (df.assign(g=s).groupby("g")["_y"].agg(taux="mean", effectif="count")
            .sort_values("effectif", ascending=False).head(15).reset_index())
    fig = px.bar(rate, x="g", y="taux", hover_data=["effectif"], labels={"g": col})
    fig.add_hline(y=df["_y"].mean(), line_dash="dash", annotation_text="taux global")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Seules les 15 catégories les plus fréquentes sont affichées. Attention aux petits effectifs.")

# ---------- MODULE 2 : Préparation ----------
st.header("2. Préparation")
st.caption(f"Cible : **{target}** (classe positive : {positive}), choisie plus haut.")

all_feats = [c for c in df.columns if c not in (target, "_y")]
ids = [c for c in all_feats if df[c].nunique() == len(df)]  # colonnes-identifiants
if ids:
    st.info(f"Identifiants exclus par défaut : {', '.join(ids)}")
features = st.multiselect("Variables explicatives", all_feats,
                          default=[c for c in all_feats if c not in ids])

c1, c2, c3 = st.columns(3)
with c1:
    num_impute = st.selectbox("Numériques manquantes", ["median", "mean", "drop"],
                              format_func={"median": "Médiane", "mean": "Moyenne", "drop": "Supprimer les lignes"}.get)
    cat_impute = st.selectbox("Catégorielles manquantes", ["inconnu", "mode", "drop"],
                              format_func={"inconnu": "Catégorie « inconnu »", "mode": "Valeur la plus fréquente",
                                           "drop": "Supprimer les lignes"}.get)
with c2:
    encoding = st.selectbox("Encodage des catégories", ["onehot", "ordinal"],
                            format_func={"onehot": "One-hot", "ordinal": "Ordinal (codes entiers)"}.get)
    min_freq = st.slider("Regrouper les catégories rares (one-hot)", 0.0, 0.10, 0.01, 0.005,
                         help="Les catégories sous ce seuil sont regroupées en « infrequent ».")
    scaler = st.selectbox("Normalisation", ["standard", "minmax", "none"],
                          format_func={"standard": "Standardisation", "minmax": "Min-Max", "none": "Aucune"}.get)
with c3:
    test_size = st.slider("Taille du jeu de test", 0.1, 0.5, 0.2, 0.05)
    seed = st.number_input("Graine aléatoire", value=42, step=1)
    convert_ordinal = st.checkbox("Convertir « experience » et « last_new_job » en nombres", value=True)

if st.button("Appliquer la préparation", type="primary"):
    if not features:
        st.error("Choisissez au moins une variable explicative.")
    else:
        st.session_state["prep"] = prepare(
            df, features, num_impute=num_impute, cat_impute=cat_impute, scaler=scaler,
            encoding=encoding, min_freq=min_freq, test_size=test_size,
            seed=int(seed), convert_ordinal=convert_ordinal)

if "prep" in st.session_state:
    p = st.session_state["prep"]
    st.success("Préparation appliquée.")
    a, b, c, d = st.columns(4)
    a.metric("Train", f"{len(p['X_train']):,}")
    b.metric("Test", f"{len(p['X_test']):,}")
    c.metric("Variables après encodage", p["X_train"].shape[1])
    d.metric("Positifs (train / test)", f"{p['y_train'].mean():.1%} / {p['y_test'].mean():.1%}")
    st.write("Aperçu des données transformées (train)")
    st.dataframe(p["X_train"].head(10), use_container_width=True)
    st.caption(f"Valeurs manquantes restantes : {int(p['X_train'].isna().sum().sum())}")

    # ---------- MODULE 3 : Modélisation ----------
st.header("3. Modélisation")
if "prep" not in st.session_state:
    st.info("Appliquez d'abord la préparation (Module 2).")
    st.stop()
p = st.session_state["prep"]


def manual_params(name):
    k = name
    if name == "Régression logistique":
        return {"C": st.select_slider("C (plus petit = modèle plus simple)",
                                      [0.01, 0.1, 1, 10, 100], 1, key=k + "C")}
    if name == "Arbre de décision":
        return {"max_depth": st.slider("Profondeur max", 2, 20, 5, key=k + "d"),
                "min_samples_leaf": st.slider("Taille min d'une feuille", 1, 100, 20, key=k + "l")}
    if name == "k plus proches voisins":
        return {"n_neighbors": st.slider("Nombre de voisins", 1, 101, 15, 2, key=k + "n")}
    if name == "Random Forest":
        return {"n_estimators": st.slider("Nombre d'arbres", 50, 500, 200, 50, key=k + "e"),
                "max_depth": st.slider("Profondeur max", 2, 30, 10, key=k + "d")}
    return {"n_estimators": st.slider("Nombre d'arbres", 50, 500, 200, 50, key=k + "e"),
            "max_depth": st.slider("Profondeur max", 2, 10, 4, key=k + "d"),
            "learning_rate": st.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.2, 0.3], 0.1, key=k + "r")}


chosen = st.multiselect("Modèles à entraîner", NAMES, default=NAMES)
balance = st.checkbox("Compenser le déséquilibre des classes", value=True)
optimize = st.checkbox("Optimiser automatiquement les hyperparamètres (plus lent)", value=False)

params = {}
if not optimize:
    for name in chosen:
        with st.expander(f"Réglages : {name}"):
            params[name] = manual_params(name)

if st.button("Entraîner les modèles", type="primary"):
    if not chosen:
        st.error("Choisissez au moins un modèle.")
    else:
        results, rows = {}, []
        bar = st.progress(0.0)
        for i, name in enumerate(chosen):
            with st.spinner(f"Entraînement : {name}..."):
                model, proba, row = train_one(name, params.get(name, {}), p["X_train"], p["y_train"],
                                              p["X_test"], p["y_test"], balance, optimize)
            results[name] = {"model": model, "proba": proba}
            rows.append(row)
            bar.progress((i + 1) / len(chosen))
        st.session_state["results"] = results
        for k in ("shap_cache", "perm", "local"):
            st.session_state.pop(k, None)
        st.session_state["table"] = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)

if "table" in st.session_state:
    st.subheader("Comparaison des modèles (jeu de test, seuil 0,5)")
    st.dataframe(st.session_state["table"].round(3), width="stretch", hide_index=True)
    best = st.session_state["table"].iloc[0]
    st.success(f"Meilleur AUC-ROC : {best['Modèle']} ({best['AUC-ROC']:.3f})")

# ---------- MODULE 4 : Évaluation ----------
st.header("4. Évaluation")
if "results" not in st.session_state:
    st.info("Entraînez d'abord les modèles (Module 3).")
    st.stop()

results = st.session_state["results"]
y_te = p["y_test"]

# Seuil de décision (partagé avec le Module 5 via la clé "thr")
thr = st.slider("Seuil de décision : on prédit « positif » si la probabilité dépasse ce seuil",
                0.05, 0.95, 0.50, 0.01, key="thr")

# Tableau comparatif
st.subheader("Métriques")
st.dataframe(comparison_table(results, y_te, thr).round(3), width="stretch", hide_index=True)
st.caption("AUC-ROC et PR-AUC ne dépendent pas du seuil. Précision, rappel et F1 changent avec le curseur.")

# Courbes ROC superposées + précision-rappel
col_a, col_b = st.columns(2)

with col_a:
    fig = go.Figure()
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_te, r["proba"])
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=name)
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard",
                    line=dict(dash="dash", color="gray"))
    fig.update_layout(title="Courbes ROC", xaxis_title="Taux de faux positifs",
                      yaxis_title="Taux de vrais positifs (rappel)")
    st.plotly_chart(fig, width="stretch")

with col_b:
    fig = go.Figure()
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(y_te, r["proba"])
        fig.add_scatter(x=rec, y=prec, mode="lines", name=name)
    fig.add_hline(y=y_te.mean(), line_dash="dash", line_color="gray",
                  annotation_text="Hasard (taux de positifs)")
    fig.update_layout(title="Courbes précision-rappel", xaxis_title="Rappel", yaxis_title="Précision")
    st.plotly_chart(fig, width="stretch")

# Un modèle en détail : matrice de confusion + effet du seuil
st.subheader("Détail d'un modèle")
name = st.selectbox("Modèle", list(results.keys()))
proba = results[name]["proba"]
pred = (proba >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
m = metrics_at(y_te, proba, thr)

c1, c2, c3 = st.columns(3)
c1.metric("Précision", f"{m['Précision']:.1%}")
c2.metric("Rappel", f"{m['Rappel']:.1%}")
c3.metric("F1", f"{m['F1']:.2f}")

cm = pd.DataFrame([[tn, fp], [fn, tp]],
                  index=["Réel : négatif", "Réel : positif"],
                  columns=["Prédit : négatif", "Prédit : positif"])
st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues"), width="stretch")
st.write(f"À ce seuil, le modèle repère **{tp}** candidats sur **{tp + fn}** qui cherchent vraiment "
         f"un emploi, au prix de **{fp}** fausses alertes.")

# ---------- MODULE 5 : Explainable AI ----------
st.header("5. Explainable AI")
ename = st.selectbox("Modèle à expliquer", list(results.keys()), key="xai_model")
model = results[ename]["model"]
X_tr, X_te = p["X_train"], p["X_test"]
cols = list(X_tr.columns)
cache = st.session_state.setdefault("shap_cache", {})


def show_plot():
    st.pyplot(plt.gcf())
    plt.close("all")


# --- Calcul SHAP (une fois par modèle) ---
if ename not in cache:
    n = 50 if ename == "k plus proches voisins" else 500
    st.info(f"SHAP sera calculé sur {n} individus du jeu de test. Cela peut prendre un moment.")
    if st.button("Calculer SHAP", type="primary"):
        with st.spinner("Calcul SHAP en cours..."):
            sample = X_te.sample(min(n, len(X_te)), random_state=0)
            expl = make_explainer(ename, model, X_tr)
            cache[ename] = {"explainer": expl, "sv": shap_explain(ename, expl, sample)}
        st.rerun()
    st.stop()
sv = cache[ename]["sv"]
X_shap = pd.DataFrame(sv.data, columns=sv.feature_names)
order = pd.Series(np.abs(sv.values).mean(axis=0), index=sv.feature_names).sort_values(ascending=False)

# --- Explications globales ---
st.subheader("Explications globales")
t1, t2, t3, t4, t5 = st.tabs(["Importance du modèle", "Permutation", "SHAP summary",
                              "Dépendance SHAP", "Variables sensibles"])

with t1:
    imp = own_importance(ename, model, cols)
    if imp is None:
        st.info("Le k-NN n'a pas d'importance propre : voir l'onglet Permutation.")
    else:
        top = imp.reindex(imp.abs().nlargest(15).index)[::-1]
        label = "Coefficient" if ename == "Régression logistique" else "Importance"
        st.plotly_chart(px.bar(x=top.values, y=top.index, orientation="h",
                               labels={"x": label, "y": ""}), width="stretch")

with t2:
    if st.button("Calculer l'importance par permutation (tous les modèles)"):
        with st.spinner("Calcul en cours..."):
            st.session_state["perm"] = pd.DataFrame(
                {nm: perm_importance(r["model"], X_te, p["y_test"]) for nm, r in results.items()})
    if "perm" in st.session_state:
        perm = st.session_state["perm"]
        top = perm.mean(axis=1).nlargest(12).index
        long = perm.loc[top].reset_index().melt(id_vars="index", var_name="Modèle",
                                                value_name="Baisse d'AUC")
        fig = px.bar(long, x="Baisse d'AUC", y="index", color="Modèle", barmode="group", orientation="h")
        fig.update_yaxes(title="", autorange="reversed")
        st.plotly_chart(fig, width="stretch")

with t3:
    shap.summary_plot(sv.values, X_shap, max_display=15, show=False)
    show_plot()

with t4:
    feat = st.selectbox("Variable", order.index[:10].tolist())
    shap.dependence_plot(feat, sv.values, X_shap, show=False)
    show_plot()

with t5:
    raw_cols = list(p["X_train_raw"].columns)
    groups = order.groupby([origin(c, raw_cols) for c in order.index]).sum()
    share = (groups / groups.sum() * 100).sort_values(ascending=False)
    default = [c for c in ("gender", "city", "city_development_index") if c in share.index]
    sens = st.multiselect("Variables à considérer comme sensibles", share.index.tolist(), default=default)
    st.metric("Part de l'importance SHAP portée par ces variables", f"{share[sens].sum():.1f} %")
    st.bar_chart(share)

# --- Explication d'un individu ---
st.subheader("Explication d'un individu")
mode = st.radio("Individu", ["Du jeu de test", "Individu fictif"], horizontal=True)

if mode == "Du jeu de test":
    pos = int(st.number_input("Position dans le jeu de test", 0, len(X_te) - 1, 0))
    x_row = X_te.iloc[[pos]]
    st.dataframe(p["X_test_raw"].iloc[[pos]].astype(str).T, width="stretch")
    st.caption(f"Valeur réelle de la cible pour cet individu : {int(p['y_test'].iloc[pos])}")
else:
    raw = p["X_train_raw"]
    vals, grid = {}, st.columns(3)
    for i, c in enumerate(raw.columns):
        with grid[i % 3]:
            if c in p["num_cols"]:
                vals[c] = st.number_input(c, value=float(raw[c].median()), key=f"f_{c}")
            else:
                opts = sorted(raw[c].dropna().astype(str).unique()) + ["(manquant)"]
                choice = st.selectbox(c, opts, index=opts.index(str(raw[c].mode().iloc[0])), key=f"f_{c}")
                vals[c] = np.nan if choice == "(manquant)" else choice
    x_row = clean(p["preprocessor"].transform(pd.DataFrame([vals])))[cols]

proba = float(model.predict_proba(x_row)[0, 1])
a, b = st.columns(2)
a.metric("Probabilité prédite", f"{proba:.1%}")
b.metric(f"Décision au seuil {thr:.2f}", "Classe positive" if proba >= thr else "Classe négative")

if st.button("Expliquer cet individu (SHAP + LIME)", type="primary"):
    with st.spinner("Calcul des explications..."):
        sv1 = shap_explain(ename, cache[ename]["explainer"], x_row)[0]
        st.session_state["local"] = {"model": ename, "sv": sv1, "proba": proba,
                                     "exp": lime_explain(model, X_tr, x_row)}

loc = st.session_state.get("local")
if loc and loc["model"] == ename:
    st.caption("Recliquez sur le bouton si vous changez d'individu ou de modèle.")
    st.info(summary_sentence(loc["sv"], loc["proba"], thr))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**SHAP (waterfall)**")
        shap.plots.waterfall(loc["sv"], max_display=12, show=False)
        show_plot()
    with c2:
        st.markdown("**LIME**")
        st.pyplot(loc["exp"].as_pyplot_figure())
        plt.close("all")
    table, common = compare(loc["sv"], loc["exp"])
    st.write("**Comparaison SHAP / LIME**")
    st.dataframe(table.round(3), width="stretch")
    st.write(f"Parmi les 5 variables les plus importantes, SHAP et LIME en ont **{common}** en commun.")

# ---------- MODULE 6 : API d'interprétation ----------
st.header("6. API d'interprétation (FastAPI)")
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.markdown("**Étape 1 : exporter le modèle expliqué ci-dessus pour l'API**")
if st.button(f"Exporter « {ename} » vers artifacts/model.joblib"):
    os.makedirs("artifacts", exist_ok=True)
    raw = p["X_train_raw"]
    joblib.dump({"name": ename, "model": model, "preprocessor": p["preprocessor"],
                 "raw_cols": list(raw.columns), "num_cols": p["num_cols"], "cat_cols": p["cat_cols"],
                 "columns": cols, "convert_ordinal": p["convert_ordinal"], "threshold": float(thr),
                 "background": X_tr.sample(min(200, len(X_tr)), random_state=0),
                 "choices": {c: raw[c].dropna().astype(str).value_counts().head(30).index.tolist()
                             for c in p["cat_cols"]}},
                "artifacts/model.joblib", compress=3)
    size = os.path.getsize("artifacts/model.joblib") / 1e6
    st.success(f"Modèle exporté ({size:.1f} Mo). Relancez ensuite l'API.")

st.markdown("**Étape 2 : appeler l'API**")
st.caption(f"Adresse de l'API : {API_URL}")
pos6 = int(st.number_input("Candidat du jeu de test", 0, len(X_te) - 1, 0, key="pos6"))
row = p["X_test_raw"].iloc[pos6]
default_json = json.dumps({k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                           for k, v in row.items()}, ensure_ascii=False, indent=2)
body = st.text_area("Variables envoyées à l'API (modifiables)", default_json, height=280)

if st.button("Appeler POST /explain", type="primary"):
    try:
        r = requests.post(f"{API_URL}/explain", json={"features": json.loads(body), "threshold": float(thr)},
                          timeout=60)
        r.raise_for_status()
        out = r.json()
        st.info(out["summary"])
        a, b = st.columns(2)
        a.metric("Probabilité (API)", f"{out['probability']:.1%}")
        b.metric("Décision (API)", out["decision"])
        st.write("Contributions SHAP par variable d'origine")
        st.bar_chart(pd.Series(out["by_variable"]))
        st.dataframe(pd.DataFrame(out["contributions"]), width="stretch", hide_index=True)
    except json.JSONDecodeError:
        st.error("Le JSON saisi est invalide.")
    except requests.RequestException as e:
        st.error(f"Appel à l'API impossible : {e}")