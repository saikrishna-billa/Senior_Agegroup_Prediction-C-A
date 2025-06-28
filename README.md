# Senior Age Group Prediction | XGBoost + SHAP + Feature Engineering

Predicting whether a person belongs to the **Senior Age Group** using medical and survey features from a structured dataset.  
This end-to-end machine learning pipeline is built for a hackathon and implements advanced preprocessing, interpretability, and evaluation strategies.

---

##  Problem Statement

Classify individuals as either `Adult` or `Senior` based on biometric, demographic, and lifestyle-related attributes.  
This classification aids in targeted healthcare and policy-making.

---

##  Dataset Overview

- `Train_Data.csv` — labeled training dataset with features and `age_group` as target.
- `Test_Data.csv` — unlabeled dataset for prediction.
- `Sample_Submission.csv` — format reference for submission.

---

##  Features Used

- **Numerical:** BMXBMI (BMI), LBXGLU (Glucose), LBXIN (Insulin), LBXGLT (Glutamate)
- **Categorical:** RIAGENDR (Gender), PAQ605 (Activity), DIQ010 (Diabetes)
- **Engineered Features:**
  - `GLU_INS = LBXGLU * LBXIN`
  - `BMI_PAQ = BMXBMI * PAQ605`
  - `GLT_DIV_GLUC = LBXGLT / (LBXGLU + 1)`
  - `BMI_Category` derived from raw BMI

---

##  Techniques Used

- **Missing Value Imputation:** `SimpleImputer` with median (numerical) and mode (categorical)
- **Scaling:** `StandardScaler` for numerical columns
- **Encoding:** `OneHotEncoder` for categorical features
- **Class Imbalance Handling:** `scale_pos_weight` in XGBoost
- **Cross-Validation:** Stratified 5-fold
- **Hyperparameter Tuning:** `RandomizedSearchCV`
- **Model:** `XGBoostClassifier`
- **Interpretability:** `SHAP` summary plots for feature impact

---

##  Performance

- **Metric Used:** F1 Score (macro)
- **Cross-Validation Avg F1 Score:** `~0.91+`
- **Training Accuracy:** `~0.94+`

