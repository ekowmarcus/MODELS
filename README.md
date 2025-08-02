"""
# Loan Default Prediction App (Group 5)

This is a **Streamlit-based machine learning app** that predicts the likelihood of loan default.  
It includes full workflow steps: data upload, preprocessing, feature selection, model training, evaluation, and interactive prediction.

---

##  How to Run

Install required libraries (if needed):

    pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib

Then run the app:

    streamlit run Group_5.py

---

##  Features

- Upload any loan dataset with a `Status` column (0 = non-default, 1 = default)
- Visualize numeric distributions and outliers
- Automatically preprocess missing values, categorical encodings, scaling
- Use **RandomForestClassifier** for fast, accurate feature selection
- Train and evaluate Random Forest model (metrics: accuracy, precision, recall, F1)
- Make interactive predictions and view default risk level

---

##  File Structure

- Group_5.py — ✅ Entire Streamlit app
- saved_data/ — Stores intermediate files, models, and outputs

---

##  Risk Thresholds

- **High Risk**: Probability > 30%
- **Medium Risk**: 10% – 30%
- **Low Risk**: < 10%

---

##  Contributors (Group 5)

| Name                     | ID        | Role                           |
|--------------------------|-----------|--------------------------------|
| Kingsley Sarfo           | 22252461  | Preprocessing, Coordination    |
| Francisca Manu Sarpong   | 22255796  | Documentation, Deployment      |
| George Owell             | 22256146  | Model Evaluation               |
| Barima Owiredu Addo      | 22254055  | UI, Prediction Interface       |
| Akrobettoe Marcus        | 11410687  | Feature Selection & Training   |

---

##  Deployment (Demo)

- [App 1](https://group5-vvhhfpcyg6qkpbswhhtckw.streamlit.app/)
- [App 2](https://kftalde5ypwd5a3qqejuvo.streamlit.app)

---

## 📌 License

For educational use only. Not suitable for production without validation and compliance.

"""

