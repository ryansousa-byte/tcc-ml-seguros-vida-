# Uso de Aprendizado de Máquina para Previsão de Riscos em Seguros de Vida

Este repositório contém as imagens dos códigos utilizados no Trabalho de Conclusão de Curso (TCC) **"Uso de Aprendizado de Máquina para Previsão de Riscos em Seguros de Vida: Uma Análise das Aplicações e Impactos no Mercado Securitário Brasileiro"**.  

O objetivo do projeto é aplicar técnicas de **Machine Learning** para previsão de risco em seguros de vida, avaliando diferentes modelos, estratégias de pré-processamento, otimização de hiperparâmetros e balanceamento de classes.

---

## 🗂 Estrutura do Repositório

/tcc-ml-seguros-vida
│
├── scripts/ # códigos utilizados nos testes
├── README.md # Documentação principal do projeto

yaml
Copiar código


---

> Obs.: Cada script implementa um pipeline de aprendizado de máquina utilizado nos testes do TCC. Não há imagens neste repositório, apenas código Python funcional.

---

## 🖥 Códigos Utilizados nos Testes (Seção 4 do TCC)

### 📌 Teste 1 — Random Forest e XGBoost com GridSearchCV
[Ver código Python](scripts/tcc_pipeline1.py)

**Descrição:**  
Este pipeline implementa e otimiza dois modelos de ensemble tradicionais: Random Forest (RF) e XGBoost (XGB).  

- **Pré-processamento:** imputação (mediana/moda), One-Hot Encoding e StandardScaler.  
- **Ajuste do Target:** o target para o XGBoost é ajustado de 1–8 para 0–7, conforme exigido para classificação multiclasse.  
- **Balanceamento:** SMOTE no conjunto de treino para lidar com desbalanceamento de classes.  
- **Otimização:** GridSearchCV para busca exaustiva de hiperparâmetros.  
- **Análise:** avaliação final e plot da importância das features para RF e XGBoost.  

---

### 📌 Teste 2 — XGBoost e LightGBM com RandomizedSearchCV
[Ver código Python](scripts/tcc_pipeline2.py)

**Descrição:**  
Pipeline focado em dois modelos baseados em Gradient Boosting: XGBoost e LightGBM (LGBM). A otimização usa **RandomizedSearchCV**, mais eficiente que GridSearch.  

- **Pré-processamento e Balanceamento:** mesmo que Pipeline 1.  
- **Otimização:** RandomizedSearchCV com StratifiedKFold.  
- **Avaliação:** acurácia, classification report e matriz de confusão.  
- **Análise:** plot da importância das features dos melhores modelos.  

---

### 📌 Teste 3 — Otimização Avançada LightGBM com Optuna e Early Stopping
[Ver código Python](scripts/tcc_pipeline3.py)

**Descrição:**  
Refina a abordagem LightGBM com **otimização bayesiana usando Optuna** e Early Stopping.  

- **Pré-processamento e Balanceamento:** mesmas etapas do Pipeline 2.  
- **Otimização:** função objective do Optuna com validação interna e Early Stopping para evitar overfitting.  
- **Métrica de Otimização:** maximizar o F1-Score (macro) no conjunto de validação, adequado para classes desbalanceadas.  
- **Análise:** função para plotar importância das features do modelo final LightGBM.  

---

### 📌 Teste 4 — LightGBM com Optuna focado em Acurácia
[Ver código Python](scripts/tcc_pipeline4.py)

**Descrição:**  
Semelhante ao Pipeline 3, mas a métrica principal é a **Acurácia**.  

- **Otimização:** Optuna treina o modelo no conjunto de treino balanceado (SMOTE) e avalia acurácia no conjunto de teste.  
- **Parâmetros Otimizados:** num_leaves, max_depth, feature_fraction, bagging_fraction, lambda_l1/lambda_l2.  

---

### 📌 Teste 5 — Pipeline Simplificado com Feature Engineering e Random Forest
[Ver código Python](scripts/tcc_pipeline5.py)

**Descrição:**  
Pipeline mais conciso, mostrando o fluxo completo com Random Forest e introduzindo **Feature Engineering**.  

- **Feature Engineering:** criação da feature `Age_per_Ht` (Idade / Altura).  
- **Tratamento de Missing e Infinitos:** lida com valores `np.inf` e `-np.inf` antes da imputação.  
- **Target/Balanceamento:** SMOTE aplicado antes do split treino/teste.  
- **Modelo:** Random Forest básico, treino e avaliação final.  

---

## 📚 Finalidade Acadêmica

Este repositório serve para documentação e visualização dos códigos utilizados nos testes do TCC, garantindo **transparência e reprodutibilidade** das análises realizadas sobre previsão de risco em seguros de vida.

---

## ✉ Contato

**Autores:** Ryan Paulo, Gabriel Lima  
**Ano:** 2025  
**Curso:** Sistemas de Informação
