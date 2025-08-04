# Loan Default Prediction Web App

This Streamlit application is an interactive Machine Learning (ML) platform for predicting loan default amounts. It is designed for both educational and practical purposes, showcasing the full lifecycle of an ML regression project.

## Features

- **Data Import and Exploration**
- **Data Preprocessing** (cleaning, imputation, encoding, standardization)
- **Feature Selection** (Best subset selection with Ridge Regression)
- **Model Training** (Ridge Regression)
- **Model Evaluation** (RMSE, R², Cross-validation)
- **Interactive Prediction**
- **Insights and Interpretation**

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

```plaintext
.
├── saved_data/
│   ├── 1_raw_data.csv
│   ├── 2_column_types.pkl
│   ├── 3_preprocessor.pkl
│   ├── 4_processed_data.csv
│   ├── 5_best_subset_features.pkl
│   ├── 6_cv_results.pkl
│   ├── 7_trained_model.pkl
│   └── 8_predictions.csv
├── Loan_Default.csv
├── Group 5.py (Main Streamlit App)
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <your_repo_url>
cd <your_repo_directory>
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App Locally

Run the following command to launch the Streamlit app:

```bash
streamlit run Another1.py
```

Then, navigate to:

```
http://localhost:8501
```

## Common Issues & Troubleshooting

**FileNotFoundError (e.g., `4_processed_data.csv`):**
- Ensure the preprocessing step (`Data Preprocessing`) is completed successfully.
- Check that the file exists in the `saved_data` directory.
- Verify your file paths are correctly referenced in the script.

**Streamlit Deployment Issues:**
- Ensure all data files are pushed to your GitHub repository.
- Avoid using absolute file paths; use relative paths from your project root.

## Data Source

- [Loan Default Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)

## Project Contributors

| Name              | Student ID | Role                | Deployment Link                                         |
|-------------------|------------|---------------------|---------------------------------------------------------|
| Kingsley Sarfo    | 22252461   | Project Coordinator | [Deployment](https://loan-predictor-hbbz24vwfzaue2qx4hwcat.streamlit.app) |
| Francisca Sarpong | 22255796   | Data Preprocessing  | [Deployment](https://kftalde5ypwd5a3qqejuvo.streamlit.app)                 |
| George Owell      | 22256146   | Model Evaluation    | [Deployment](https://loandefaultpredictionapp-utmbic9znd7uzqqhs9zgo6.streamlit.app) |
| Barima Addo       | 22254055   | UI Testing          | [Deployment](https://loandefaultapp-ky4yy9kmt6ehsq8jqdcgs2.streamlit.app)  |
| Marcus Akrobettoe | 11410687   | Feature Selection   | [Deployment](https://models-loan-default-prediction.streamlit.app)         |

---

## License

This project is licensed under the MIT License.

