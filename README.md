
# Loan Default Prediction System

A full-stack Streamlit application developed by **Group 5** to predict the likelihood of loan default using machine learning techniques. This app allows users to upload loan data, visualize and clean it, train predictive models, and assess applicant risk in real time.

---

## Features

- 📊 **Data Import & EDA**: Upload CSV data and explore it through visualizations and statistics.
- 🧹 **Data Preprocessing**: Handles missing values, scales numerical data, and encodes categoricals.
- 📉 **Feature Selection**: Includes correlation filtering and sequential feature selection.
- 🌲 **Model Training**: Random Forest Classifier with tunable hyperparameters.
- ✅ **Evaluation**: View confusion matrix, key metrics, and feature importance.
- 🔮 **Interactive Prediction**: Input applicant data and receive instant risk classification.
- 📘 **Conclusion**: Interpretation, business impact, and future enhancement suggestions.

---

## Machine Learning Stack

- **scikit-learn**: Modeling, preprocessing, and validation
- **RandomForestClassifier**: Core predictive model
- **SimpleImputer**, **StandardScaler**, **OneHotEncoder**: Preprocessing pipeline
- **SequentialFeatureSelector**: Feature selection
- **cross_val_score**: 5-fold validation

---

## UI/UX Libraries

- **Streamlit**: Web app interface
- **matplotlib** & **seaborn**: Plotting and analytics visuals
- **PIL**: Logo/image support

---

## Project Structure

```
Group_5.py             # Main application script
Loan_Default.csv       # Source dataset (not included)
saved_data/            # Folder created by the app for caching
├── 1_raw_data.csv
├── 2_column_types.pkl
├── 3_preprocessor.pkl
├── 4_processed_data.csv
├── 5_best_subset_features.pkl
├── 6_cv_results.pkl
├── 7_trained_model.pkl
├── 8_predictions.csv
├── 9_evaluation_results.pkl
```

---

## How to Run the App

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit**:
   ```bash
   streamlit run Group_5.py
   ```

3. **Use the sidebar** to navigate between app modules.

---

## Team Members (Group 5)

| Name                     | Student ID | Role                                         | Deployment |
|--------------------------|------------|----------------------------------------------|------------|
| Kingsley Sarfo           | 22252461   | Coordination, App Design, Preprocessing       | [Launch](https://loan-predictor-hbbz24vwfzaue2qx4hwcat.streamlit.app) |
| Francisca Manu Sarpong   | 22255796   | Documentation & Deployment                    | [Launch](https://kftalde5ypwd5a3qqejuvo.streamlit.app) |
| George Owell             | 22256146   | Model Evaluation & Cross-validation           | [Launch](https://loandefaultpredictionapp-utmbic9znd7uzqqhs9zgo6.streamlit.app/) |
| Barima Owiredu Addo      | 22254055   | UI & Prediction Testing                       | [Launch](https://loandefaultapp-ky4yy9kmt6ehsq8jqdcgs2.streamlit.app/) |
| Akrobettoe Marcus        | 11410687   | Feature Selection & Model Training            | [Launch](https://models-loan-default-prediction.streamlit.app/) |

---

## Business Value

This predictive system supports:

- Early identification of high-risk loan applicants
- Reduction in default-related losses
- Data-informed lending decisions

---

## Limitations & Future Work

- Current model limited to structured CSV input
- Doesn't incorporate external data like credit bureau scores
- Future versions could include:
  - XGBoost or deep learning models
  - Scheduled retraining with new data
  - Deployment as a microservice or API

---

## License

This project is for academic demonstration purposes only. For production use, ensure regulatory compliance and robust testing.
