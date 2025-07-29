# 📊 Loan Default Prediction Web App

This is a **Streamlit-powered interactive web app** that helps predict loan default risk using machine learning (Ridge Regression). It allows users to explore data, preprocess, select features, train a model, and get predictions with real-time inputs.

---

## 🚀 Features

- ✅ Upload and preview loan data  
- 🧼 Data cleaning & preprocessing  
- 🧠 Feature selection using best subset (forward stepwise)  
- 📈 Ridge Regression model training  
- 📊 Model evaluation (R², RMSE, cross-validation)  
- 🔍 Interactive prediction form for new applicants  
- 📋 Final insights with coefficient interpretation  

---

## 📁 Project Structure

```
Loan_Default_Predictor/
│
├── loan_Default.csv             # Raw loan dataset (user-provided)
├── README.md                    # Project documentation (this file)
└── requirements.txt             # Required Python libraries
```

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)  
- **ML Model**: Ridge Regression (Scikit-learn)  
- **Data Handling**: pandas, NumPy  
- **Visualization**: seaborn, matplotlib  

---

## 🔧 Setup Instructions

1. **Clone this repo** or download the ZIP:
   ```bash
   git clone https://github.com/yourusername/Loan_Default_Predictor.git
   ```

2. **Navigate to the project folder**:
   ```bash
   cd Loan_Default_Predictor
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   streamlit run 29.07.25.py
   ```

---

## 📂 Sample Data File: `loan_Default.csv`

Your dataset should include:
- `loan_amount`, `income`, `Credit_Score`, `LTV`
- `interest_rate_spread`, `upfront_charges`, `dtir1`, etc.
- Categorical columns like: `loan_type`, `property_type`, `credit_type`
- **Target column**: `status` (1 = default, 0 = non-default)

---

## 🤖 Machine Learning Model

We use **Ridge Regression**, ideal for:
- Handling multicollinearity
- Reducing overfitting with L2 regularization
- Handling both numeric and encoded categorical variables

---

## 📊 Performance

- **Cross-Validation R²**: ~0.41  
- **Test Set R²**: ~0.41  
- **RMSE**: ~0.33  
> These metrics suggest the model moderately explains the variance in default risk.

---

## 🔍 Prediction Workflow

- Input your loan features via form
- Model predicts probability of default
- Output shows predicted risk + classification (Default or Not)

---

## 👥 Group 5 – Developers

**Interactive Loan Default Prediction Web App**  
- Kingsley Sarfo – 22252461  
- Francisca Manu Sarpong – 22255796  
- George Owell – 22256146  
- Barima Owiredu Addo – 22254055  
- Akrobettoe Marcus – 11410687  

---

## 📄 License

This project is released under the MIT License. Free to use for learning or academic purposes.

---

## 📦 requirements.txt (Paste into file)

```
streamlit==1.35.0
pandas
numpy
scikit-learn
matplotlib
seaborn
Pillow
```
