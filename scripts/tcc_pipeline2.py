# ========================================
# Pipeline Final - Prudential Life Insurance Assessment
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb

# 1. Carregar dados
data = pd.read_csv(r"C:\Users\Administrador\Desktop\train.csv")

# 2. Separar features e target
X = data.drop("Response", axis=1)
y = data["Response"]

# Ajuste target para XGBoost/LightGBM (0-indexed)
y_xgb = y - 1

# 3. Colunas numéricas e categóricas
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

print("Numéricas:", len(num_cols), "| Categóricas:", len(cat_cols))

# 4. Imputação
X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])

# 5. Encoding e Scaling
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
scaler = StandardScaler()

X_cat = encoder.fit_transform(X[cat_cols])
X_num = scaler.fit_transform(X[num_cols])

# Concatenar
X_all = np.hstack([X_num, X_cat])
feature_names = list(num_cols) + list(encoder.get_feature_names_out(cat_cols))
X_all_df = pd.DataFrame(X_all, columns=feature_names)  # mantém nomes para LightGBM/XGBoost

# 6. Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_all_df, y, test_size=0.2, random_state=42, stratify=y
)
y_train_xgb = y_train - 1
y_test_xgb = y_test - 1

# 7. Balanceamento SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train_res_xgb, y_train_res_xgb = smote.fit_resample(X_train, y_train_xgb)

print("Antes SMOTE:", y_train.value_counts().to_dict())
print("Depois SMOTE:", pd.Series(y_train_res).value_counts().to_dict())

# =========================
# XGBoost com RandomizedSearchCV
# =========================
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    num_class=8,
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

param_dist_xgb = {
    "n_estimators": [300, 500, 700],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rand_search_xgb = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist_xgb,
    n_iter=10,
    scoring="accuracy",
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=42
)
rand_search_xgb.fit(X_train_res_xgb, y_train_res_xgb)

best_xgb = rand_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test) + 1  # reverter para original
print("\n=== XGBoost ===")
print("Melhores parâmetros:", rand_search_xgb.best_params_)
print("Acurácia:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred_xgb))

# =========================
# LightGBM com RandomizedSearchCV
# =========================
lgb_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=8,
    random_state=42,
    n_jobs=-1
)

param_dist_lgb = {
    "n_estimators": [300, 500, 700],
    "max_depth": [4, 6, 8, -1],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

rand_search_lgb = RandomizedSearchCV(
    lgb_model,
    param_distributions=param_dist_lgb,
    n_iter=10,
    scoring="accuracy",
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=42
)
rand_search_lgb.fit(X_train_res, y_train_res)

best_lgb = rand_search_lgb.best_estimator_
y_pred_lgb = best_lgb.predict(X_test)
print("\n=== LightGBM ===")
print("Melhores parâmetros:", rand_search_lgb.best_params_)
print("Acurácia:", accuracy_score(y_test, y_pred_lgb))
print(classification_report(y_test, y_pred_lgb))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred_lgb))

# =========================
# Importância das features
# =========================
def plot_feature_importance(model, features, top_n=15, title="Feature Importance"):
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": features, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)
    plt.figure(figsize=(10,6))
    sns.barplot(x="importance", y="feature", data=df)
    plt.title(title)
    plt.xlabel("Importância")
    plt.ylabel("Feature")
    plt.show()

plot_feature_importance(best_xgb, feature_names, top_n=15, title="Top 15 Features - XGBoost")
plot_feature_importance(best_lgb, feature_names, top_n=15, title="Top 15 Features - LightGBM")
