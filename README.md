
# Loan Default Prediction Web App

This Streamlit web application is an end-to-end interactive machine learning tool for **Loan Default Prediction**. It is designed for educational, research, and practical demonstration of regression analysis, feature engineering, and interactive ML deployment.

---

## 🚀 Features

- **Data Import and Exploration**: Upload and explore your dataset with instant statistics and visualizations.
- **Data Preprocessing**: Clean, impute, encode, and standardize your data with reproducible ML pipelines.
- **Feature Selection**: Use sequential forward selection and Ridge Regression for optimal feature selection.
- **Model Training**: Train a Ridge Regression model with cross-validation.
- **Model Evaluation**: View key performance metrics (RMSE, R²), visualizations, and feature importances.
- **Interactive Prediction**: Enter new values and predict loan default in real-time.
- **Result Interpretation & Insights**: Understand model output, business implications, and next steps.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- PIL (Python Imaging Library)

---

## 📂 Project Structure

```
.
├── saved_data/
│   ├── 4_processed_data.csv      # Generated or committed processed data
│   └── ... (other artifacts)
├── Loan_Default.csv              # Raw data (if using local runs)
├── Group 5.py                    # Main Streamlit App
├── requirements.txt
├── README.md
└── ...
```

---

## 📥 Installation

1. **Clone the repository**  
   ```bash
   git clone <your_repo_url>
   cd <your_repo_directory>
   ```

2. **Create and activate a virtual environment (recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```

3. **Install required dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the App Locally

```bash
streamlit run "Group 5.py"
```

Then visit [http://localhost:8501](http://localhost:8501) in your web browser.

---

## 📦 Deployment on Streamlit Cloud

- **Ensure `saved_data/4_processed_data.csv` is committed to your repository** if you want the app to access preprocessed data on every run.
- Do **NOT** rely on files created at runtime unless you allow uploads or regeneration.
- For best results, keep all required files in your GitHub repo.

---

## 🚨 Common Issues & Troubleshooting

- **FileNotFoundError for `4_processed_data.csv`:**
    - Confirm `saved_data/4_processed_data.csv` is present in your GitHub repo.
    - Path should be exactly `saved_data/4_processed_data.csv` (case-sensitive).
    - Use a relative path in your code:  
      ```python
      file_path = "saved_data/4_processed_data.csv"
      processed_df = pd.read_csv(file_path)
      ```
    - If generating at runtime, ensure that the app has the raw data and full preprocessing pipeline is run every time.

---

## 📚 Data Source

- [Loan Default Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)

---

## 👥 Team Members

| Name               | Student ID | Role                | Deployment Link                                         |
|--------------------|------------|---------------------|---------------------------------------------------------|
| Kingsley Sarfo     | 22252461   | Project Coordinator | [Deployment](https://loan-predictor-hbbz24vwfzaue2qx4hwcat.streamlit.app)        |
| Francisca Sarpong  | 22255796   | Data Preprocessing  | [Deployment](https://kftalde5ypwd5a3qqejuvo.streamlit.app)                      |
| George Owell       | 22256146   | Model Evaluation    | [Deployment](https://loandefaultpredictionapp-utmbic9znd7uzqqhs9zgo6.streamlit.app) |
| Barima Addo        | 22254055   | UI Testing          | [Deployment](https://loandefaultapp-ky4yy9kmt6ehsq8jqdcgs2.streamlit.app)         |
| Marcus Akrobettoe  | 11410687   | Feature Selection   | [Deployment](https://models-loan-default-prediction.streamlit.app)               |

---

## 📝 License

This project is for academic and educational use.

---
