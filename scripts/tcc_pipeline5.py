import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1️⃣ Carregar a base
df = pd.read_csv(r"C:\Users\Administrador\Desktop\train.csv")  # substitua pelo nome correto do arquivo

# 2️⃣ Separar alvo e features
target_col = "Response"  # coluna alvo
X = df.drop(columns=[target_col])
y = df[target_col]

# 3️⃣ Criar novas features se quiser
# Exemplo: idade por altura
if "Ins_Age" in X.columns and "Ht" in X.columns:
    X["Age_per_Ht"] = X["Ins_Age"] / X["Ht"]

# 4️⃣ Separar colunas numéricas e categóricas
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# 5️⃣ Tratar valores infinitos e missing apenas nas numéricas
X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# 6️⃣ One-Hot Encoding das categóricas
if len(cat_cols) > 0:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 7️⃣ Normalizar numéricas
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# 8️⃣ Balancear classes com SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"Antes SMOTE: {y.value_counts().to_dict()}")
print(f"Depois SMOTE: {y_res.value_counts().to_dict()}")

# 9️⃣ Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# 10️⃣ Treinar modelo (Random Forest como exemplo)
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 11️⃣ Avaliar
y_pred = model.predict(X_test)
print("=== Random Forest Final ===")
print(f"Acurácia: {model.score(X_test, y_test)}")
print(classification_report(y_test, y_pred))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
