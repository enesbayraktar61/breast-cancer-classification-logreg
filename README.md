# Breast Cancer Classification (Binary Classification)

This project focuses on predicting whether a breast tumor is **malignant** or **benign**
using machine learning classification techniques.
A Logistic Regression model is trained, evaluated, and deployed as an interactive web application.

---

## Project Overview

- **Problem Type:** Binary Classification  
- **Target Variable:** `target`  
  - `0` → Malignant  
  - `1` → Benign  
- **Dataset:** Breast Cancer Wisconsin Dataset (`sklearn.datasets`)  
- **Final Model:** Logistic Regression (with feature scaling)  
- **Deployment:** Streamlit app deployed on Hugging Face Spaces  

---

## Dataset Description

The dataset consists of 569 samples with 30 numerical features describing
characteristics of cell nuclei extracted from breast mass images.

- No missing values  
- Well-structured and suitable for classical classification models  

---

## Project Structure

breast_cancer_classification/
├── app.py
├── requirements.txt
├── models/
│ ├── breast_cancer_logreg_pipeline.joblib
│ └── training_columns.json
├── notebooks/
│ └── breast_cancer_classification.ipynb
└── README.md


---

## Methodology

### Exploratory Data Analysis (EDA)
- Verified dataset quality and structure  
- Inspected class distribution  
- Analyzed feature correlations  

### Preprocessing
- Feature/target split  
- Train/test split with stratification  
- Feature scaling using `StandardScaler`  

### Modeling
- **Baseline & Final Model:** Logistic Regression  
- **Comparison Model:** Random Forest Classifier  
- Logistic Regression achieved superior performance, indicating strong linear separability in the data.

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  

---

## Deployment

- The final model was saved as a single sklearn **Pipeline** (scaler + classifier).
- A Streamlit app allows interactive predictions and probability estimation.
- The app is deployed on Hugging Face Spaces for public access.

---

Conclusion

This project demonstrates a complete end-to-end binary classification workflow,
including data exploration, preprocessing, model training, evaluation, and deployment.

The Logistic Regression model proved to be highly effective for this dataset,
outperforming more complex models while maintaining excellent interpretability.
This highlights the importance of selecting models that align well with the underlying data structure.

---

Future Improvements

Hyperparameter tuning

Comparison with SVM and Gradient Boosting

Model interpretability techniques (e.g. SHAP or coefficient analysis)

---

## How to Run Locally

```bash
git clone https://github.com/enesbayraktar61/breast-cancer-classification-logreg.git
cd breast-cancer-classification-logreg
pip install -r requirements.txt
streamlit run app.py

---
