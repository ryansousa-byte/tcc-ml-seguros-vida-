# ================================
# Pipeline Completo - Prudential Life Insurance Assessment
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from imblearn.over_sampling import SMOTE

# 1. Carregar os dados
data = pd.read_csv(r"C:\Users\Administrador\Desktop\train.csv")

# 2. Separar features e target
X = data.drop("Response", axis=1)
y = data["Response"]

# 3. Ajustar target para XGBoost (começando em 0)
y_xgb = y - 1  # classes 1-8 → 0-7

# 4. Identificar colunas numéricas e categóricas
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

print("Numéricas:", len(num_cols), "| Categóricas:", len(cat_cols))

# 5. Imputação de valores ausentes
imputer_num = SimpleImputer(strategy="median")
X[num_cols] = imputer_num.fit_transform(X[num_cols])

imputer_cat = SimpleImputer(strategy="most_frequent")
X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])

# 6. Encoding e Scaling
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
scaler = StandardScaler()

X_cat = encoder.fit_transform(X[cat_cols])
X_num = scaler.fit_transform(X[num_cols])

# Concatenar numérico + categórico
X_all = np.hstack([X_num, X_cat])

# 7. Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

# Para XGBoost
_, _, y_train_xgb, y_test_xgb = train_test_split(
    X_all, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
)

# 8. Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train_res_xgb, y_train_res_xgb = smote.fit_resample(X_train, y_train_xgb)

print("Antes SMOTE:", y_train.value_counts().to_dict())
print("Depois SMOTE:", y_train_res.value_counts().to_dict())

# 9. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
)
rf_model.fit(X_train_res, y_train_res)
y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest ===")
print("Acurácia:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred_rf))

# 10. XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    num_class=8
)
xgb_model.fit(X_train_res_xgb, y_train_res_xgb)
y_pred_xgb = xgb_model.predict(X_test)

# Reverter classes para o original
y_pred_xgb = y_pred_xgb + 1

print("\n=== XGBoost ===")
print("Acurácia:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred_xgb))


# ==============================================
# Otimização de Modelos com GridSearchCV
# ==============================================

from sklearn.model_selection import GridSearchCV
import time

print("\n🚀 Iniciando otimização de hiperparâmetros...\n")

# ----------------------------
# Random Forest - Grid Search
# ----------------------------
param_grid_rf = {
    'n_estimators': [200, 300, 500],
    'max_depth': [8, 10, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

start = time.time()
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid_rf,
    cv=3,
    scoring='accuracy',
    verbose=2
)

grid_rf.fit(X_train_res, y_train_res)
end = time.time()

print("\n=== RESULTADOS RANDOM FOREST ===")
print(f"⏱️ Tempo total: {(end - start)/60:.2f} min")
print("Melhores parâmetros:", grid_rf.best_params_)
print("Melhor acurácia média (CV):", grid_rf.best_score_)

# Treinar novamente com os melhores parâmetros
best_rf = grid_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

print("\nAcurácia (teste final):", accuracy_score(y_test, y_pred_best_rf))
print(classification_report(y_test, y_pred_best_rf))


# ----------------------------
# XGBoost - Grid Search
# ----------------------------
param_grid_xgb = {
    'n_estimators': [300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

start = time.time()
grid_xgb = GridSearchCV(
    xgb.XGBClassifier(
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        num_class=8
    ),
    param_grid_xgb,
    cv=3,
    scoring='accuracy',
    verbose=2
)

grid_xgb.fit(X_train_res_xgb, y_train_res_xgb)
end = time.time()

print("\n=== RESULTADOS XGBOOST ===")
print(f"⏱️ Tempo total: {(end - start)/60:.2f} min")
print("Melhores parâmetros:", grid_xgb.best_params_)
print("Melhor acurácia média (CV):", grid_xgb.best_score_)

# Treinar novamente com os melhores parâmetros
best_xgb = grid_xgb.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test)
y_pred_best_xgb = y_pred_best_xgb + 1  # voltar classes para 1-8

print("\nAcurácia (teste final):", accuracy_score(y_test, y_pred_best_xgb))
print(classification_report(y_test, y_pred_best_xgb))
print(confusion_matrix(y_test, y_pred_best_xgb))

print("\n✅ Otimização concluída com sucesso!")




import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Recuperar nomes das features
# ==========================
num_features = num_cols.tolist()
cat_features = encoder.get_feature_names_out(cat_cols)
all_features = np.concatenate([num_features, cat_features])

# ==========================
# Importância - Random Forest
# ==========================
importances_rf = rf_model.feature_importances_
features_rf = pd.DataFrame({
    'feature': all_features,
    'importance': importances_rf
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=features_rf.head(15))
plt.title("Importância das Variáveis - Random Forest (Top 15)")
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.show()

# ==========================
# Importância - XGBoost
# ==========================
importances_xgb = xgb_model.feature_importances_
features_xgb = pd.DataFrame({
    'feature': all_features,
    'importance': importances_xgb
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=features_xgb.head(15))
plt.title("Importância das Variáveis - XGBoost (Top 15)")
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.show()
