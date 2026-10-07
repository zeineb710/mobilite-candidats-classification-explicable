import os
import pandas as pd
import plotly.express as px
import streamlit as st
from prep import prepare

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