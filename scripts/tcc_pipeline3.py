# ========================================
# Pipeline Final - Prudential Life Insurance Assessment (Atualizado)
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import lightgbm as lgb
import optuna
from lightgbm import early_stopping, log_evaluation

# 1. Carregar dados
data = pd.read_csv(r"C:\Users\Administrador\Desktop\train.csv")

# 2. Separar features e target
X = data.drop("Response", axis=1)
y = data["Response"]

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
X_all_df = pd.DataFrame(X_all, columns=feature_names)

# 6. Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_all_df, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Balanceamento SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

print("Antes SMOTE:", y_train.value_counts().to_dict())
print("Depois SMOTE:", pd.Series(y_res).value_counts().to_dict())

# ========================================
# Otimização LightGBM com Optuna
# ========================================

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1
    }

    model = lgb.LGBMClassifier(**params)

    # Split treino/validação dentro do treino SMOTE
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    model.fit(
        X_train_opt, y_train_opt,
        eval_set=[(X_val_opt, y_val_opt)],
        callbacks=[early_stopping(50), log_evaluation(0)]
    )

    preds = model.predict(X_val_opt)
    from sklearn.metrics import f1_score
    return f1_score(y_val_opt, preds, average="macro")

# Criar estudo Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("=== Otimização finalizada ===")
print("Melhores parâmetros:", study.best_params)

# Treinar modelo final com todos os dados de treino SMOTE
best_model = lgb.LGBMClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_res, y_res)

# ========================================
# Avaliação no conjunto de teste
# ========================================
y_pred = best_model.predict(X_test)

print("\n=== LightGBM Final ===")
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

# ========================================
# Importância das features
# ========================================
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

plot_feature_importance(best_model, feature_names, top_n=15, title="Top 15 Features - LightGBM")
