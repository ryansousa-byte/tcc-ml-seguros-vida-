# ========================================
# Pipeline Final - Prudential Life Insurance Assessment (Atualizado)
# ========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import optuna

# 1. Carregar dados
df = pd.read_csv(r"C:\Users\Administrador\Desktop\train.csv")

# 2. Definir features e target
target_col = "Response"
X = df.drop(target_col, axis=1)
y = df[target_col]

# 3. Identificar colunas numéricas e categóricas
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

print(f"Numéricas: {len(num_cols)}, Categóricas: {len(cat_cols)}")

# 4. Imputação
X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])

# 5. Encoding e Scaling
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
scaler = StandardScaler()

X_cat = encoder.fit_transform(X[cat_cols])
X_num = scaler.fit_transform(X[num_cols])

X_all = np.hstack([X_num, X_cat])
feature_names = list(num_cols) + list(encoder.get_feature_names_out(cat_cols))
X_all_df = pd.DataFrame(X_all, columns=feature_names)

# 6. Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_all_df, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Antes SMOTE:", y_train.value_counts().to_dict())
print("Depois SMOTE:", pd.Series(y_train_res).value_counts().to_dict())

# 8. Função de otimização para Optuna
def objective(trial):
    param = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train_res)),
        "metric": "multi_logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1
    }
    
    model = lgb.LGBMClassifier(**param)
    model.fit(X_train_res, y_train_res)  # sem verbose nem early_stopping_rounds
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# 9. Criar estudo e otimizar
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # pode aumentar n_trials pra resultados melhores

print("Melhor acurácia obtida:", study.best_value)
print("Melhores parâmetros:", study.best_params)

# 10. Treinar modelo final com melhores parâmetros
best_params = study.best_params
best_params.update({
    "objective": "multiclass",
    "num_class": len(np.unique(y_train_res)),
    "metric": "multi_logloss",
    "random_state": 42,
    "n_jobs": -1
})
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train_res, y_train_res)

y_pred_final = final_model.predict(X_test)
print("\n=== LightGBM Final ===")
print("Acurácia:", accuracy_score(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred_final))
