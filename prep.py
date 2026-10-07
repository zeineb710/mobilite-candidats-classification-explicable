import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler


def convert_ordinal_text(df: pd.DataFrame) -> pd.DataFrame:
    """Spécifique au cas 9 : 'experience' et 'last_new_job' contiennent du texte ('<1', '>20', 'never')."""
    out = df.copy()
    if "experience" in out and out["experience"].dtype == object:
        out["experience"] = pd.to_numeric(out["experience"].replace({"<1": "0", ">20": "21"}), errors="coerce")
    if "last_new_job" in out and out["last_new_job"].dtype == object:
        out["last_new_job"] = pd.to_numeric(out["last_new_job"].replace({"never": "0", ">4": "5"}), errors="coerce")
    return out


def prepare(df, features, y_col="_y", *, num_impute="median", cat_impute="inconnu",
            scaler="standard", encoding="onehot", min_freq=0.01,
            test_size=0.2, seed=42, convert_ordinal=False):
    data = df[features + [y_col]].copy()
    if convert_ordinal:
        data = convert_ordinal_text(data)

    num_cols = data[features].select_dtypes("number").columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]
    data[cat_cols] = data[cat_cols].astype("object")

    # Suppression de lignes (si choisie), avant la séparation
    if num_impute == "drop":
        data = data.dropna(subset=num_cols)
    if cat_impute == "drop":
        data = data.dropna(subset=cat_cols)

    # Pipeline numérique
    num_steps = []
    if num_impute in ("median", "mean"):
        num_steps.append(("imp", SimpleImputer(strategy=num_impute)))
    if scaler != "none":
        num_steps.append(("sc", StandardScaler() if scaler == "standard" else MinMaxScaler()))
    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

    # Pipeline catégoriel
    cat_steps = []
    if cat_impute == "inconnu":
        cat_steps.append(("imp", SimpleImputer(strategy="constant", fill_value="inconnu")))
    elif cat_impute == "mode":
        cat_steps.append(("imp", SimpleImputer(strategy="most_frequent")))
    if encoding == "onehot":
        cat_steps.append(("enc", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                               min_frequency=min_freq, sparse_output=False)))
    else:
        cat_steps.append(("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
    cat_pipe = Pipeline(cat_steps)

    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
                            verbose_feature_names_out=False).set_output(transform="pandas")

    # Séparation PUIS ajustement sur le train uniquement (évite la fuite de données)
    X, y = data[features], data[y_col]
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    X_tr = pre.fit_transform(X_tr_raw)
    X_te = pre.transform(X_te_raw)

    return dict(X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
                X_train_raw=X_tr_raw, X_test_raw=X_te_raw,
                preprocessor=pre, num_cols=num_cols, cat_cols=cat_cols)