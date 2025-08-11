
# =========================
# Loan Default App (Polished UI) – Full Version
# =========================

# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
from PIL import Image

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector

# --- CONFIG ---
DATA_DIR = "saved_data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💳",
    layout="wide",
)

# Global CSS (clean theme)
st.markdown('''
<style>
:root{ --primary:#0f4c81; --bg:#f8fafc; --card:#fff; --text:#0f172a; --muted:#64748b; }
.main{ background:var(--bg); }
section[data-testid="stSidebar"] { background:#0f4c8110; }
.stButton>button{
  background:var(--primary); color:#fff; border:0; padding:.55rem 1rem; border-radius:12px;
  font-weight:600; box-shadow:0 6px 16px rgba(15,76,129,.18);
}
.stButton>button:hover{ filter:brightness(1.08); }
.card{ background:var(--card); border-radius:16px; padding:16px; border:1px solid #e5e7eb;
       box-shadow:0 4px 18px rgba(2,6,23,.08); }
.card h4{ margin:0 0 .2rem 0; color:var(--muted); font-weight:600; }
.card .big{ font-size:28px; font-weight:800; color:var(--text); }
.pill{ display:inline-block; padding:.2rem .6rem; border-radius:999px; background:#0f4c8115;
      color:var(--primary); font-size:12px; font-weight:700; }
.dataframe tbody tr:hover{ background:#0f4c810d; }
</style>
''', unsafe_allow_html=True)

def card_metric(title, value, sub=""):
    st.markdown(f'''
    <div class="card">
      <h4>{title}</h4>
      <div class="big">{value}</div>
      <div class="pill">{sub}</div>
    </div>
    ''', unsafe_allow_html=True)

# --- HELPERS ---
def pretty_label(name: str) -> str:
    """Human-friendly column labels for display."""
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    s = s.replace('_', ' ').strip()
    keep_upper = {'ID', 'DTI', 'LTV', 'APR'}
    words = [w.upper() if w.upper() in keep_upper else w.capitalize() for w in s.split()]
    return ' '.join(words)

def save_artifact(obj, filename):
    with open(f"{DATA_DIR}/{filename}", 'wb') as f:
        pickle.dump(obj, f)

def load_artifact(filename):
    with open(f"{DATA_DIR}/{filename}", 'rb') as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_data():
    """
    Priority:
    1) st.session_state['df'] (if set by uploader)
    2) saved_data/1_raw_data.csv (cached)
    3) Loan_Default.csv (if present locally)
    """
    if 'df' in st.session_state:
        df = st.session_state['df'].copy()
    elif os.path.exists(f"{DATA_DIR}/1_raw_data.csv"):
        df = pd.read_csv(f"{DATA_DIR}/1_raw_data.csv")
    elif os.path.exists("Loan_Default.csv"):
        df = pd.read_csv("Loan_Default.csv")
    else:
        raise FileNotFoundError("No dataset found. Upload a CSV on 'Data Import & Overview'.")

    df = df.drop(columns=['ID','dtir1','submission_of_application','year'], errors='ignore')
    df.to_csv(f"{DATA_DIR}/1_raw_data.csv", index=False)
    return df

def create_preprocessor():
    """Fit and persist preprocessing pipeline; output processed CSV."""
    # Use uploaded df if available
    df = st.session_state.get('df')
    if df is None:
        df = load_data()

    if 'loan_amount' not in df.columns:
        raise ValueError("Target column 'loan_amount' not found in your file. "
                         "Rename your target to 'loan_amount' or add a target selector.")

    X = df.drop('loan_amount', axis=1)
    numerical_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    save_artifact({'numerical': numerical_cols, 'categorical': categorical_cols},"2_column_types.pkl")

    num_tf = Pipeline([('imputer', SimpleImputer(strategy='median')),
                       ('scaler', StandardScaler())])
    cat_tf = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                       ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('num', num_tf, numerical_cols),
                             ('cat', cat_tf, categorical_cols)])
    pre.fit(X)
    save_artifact(pre, "3_preprocessor.pkl")

    Xp = pre.transform(X)
    num_features = pre.named_transformers_['num'].get_feature_names_out()
    cat_features = pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    all_features = np.concatenate([num_features, cat_features])
    processed_df = pd.DataFrame(Xp, columns=all_features)
    processed_df['loan_amount'] = df['loan_amount'].values
    processed_df.to_csv(f"{DATA_DIR}/4_processed_data.csv", index=False)
    return pre

# --- PAGES ---
def Home_Page():
    cols = st.columns([1,2], gap="large")
    with cols[0]:
        try:
            st.image(Image.open("LDP.jpeg"), width=220)
        except Exception:
            st.markdown("### 💳 Loan Default Prediction")
    with cols[1]:
        st.markdown("## Loan Default Prediction")
        st.write("Explore the dataset, build a model, and make predictions with a clean, guided UI.")
        c1, c2, c3 = st.columns(3)
        with c1: card_metric("Course", "Applied ML", "MSBA")
        with c2: card_metric("Tech Stack", "Python + Streamlit", "scikit-learn • pandas")
        with c3: card_metric("Model", "Ridge Regression", "Regularized")
    st.divider()
    st.caption("Kaggle loan default dataset • For learning/demo purposes.")

def Data_Import_and_Overview_page():
    st.title("1. Data Import & Overview")
    st.subheader("Upload Your Loan Data (CSV)")
    uploaded_file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.drop(columns=['ID','dtir1','submission_of_application','year'], errors='ignore')
        st.session_state['df'] = df
        df.to_csv(f"{DATA_DIR}/1_raw_data.csv", index=False)  # cache for later steps
        st.success("✅ File uploaded and cached for the workflow.")
    elif 'df' in st.session_state:
        df = st.session_state['df']
    else:
        st.info("Upload a dataset to continue.")
        return

    # KPI row
    k1, k2, k3 = st.columns(3)
    with k1: card_metric("Total Records", f"{df.shape[0]:,}")
    with k2: card_metric("Total Features", f"{df.shape[1]:,}")
    with k3:
        avg = df['loan_amount'].mean() if 'loan_amount' in df.columns else 0
        card_metric("Average Loan Amount", f"${avg:,.0f}")

    t1, t2, t3 = st.tabs(["🔎 Preview", "📊 Distributions", "🔗 Correlations"])

    with t1:
        st.dataframe(df.head(50), use_container_width=True, height=320)
        st.markdown("#### Summaries")
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Numerical Features**")
            num_summary = df.describe().T
            num_summary.index = [pretty_label(c) for c in num_summary.index]
            st.dataframe(num_summary.style.format("{:.2f}"), use_container_width=True, height=300)
        with cB:
            cat_cols = df.select_dtypes(include=['object']).columns
            st.markdown("**Categorical Features**")
            if len(cat_cols) > 0:
                cat_summary = pd.DataFrame({
                    'Unique Values': df[cat_cols].nunique(),
                    'Most Common': df[cat_cols].mode().iloc[0],
                    'Missing Values': df[cat_cols].isnull().sum()
                })
                cat_summary.index = [pretty_label(c) for c in cat_summary.index]
                st.dataframe(cat_summary, use_container_width=True, height=300)
            else:
                st.info("No categorical columns detected.")
        st.markdown("**Missing Values**")
        missing = df.isnull().sum().to_frame('Missing Values')
        missing['Percentage'] = (missing['Missing Values']/len(df))*100
        missing.index = [pretty_label(c) for c in missing.index]
        st.dataframe(missing.style.format({'Percentage':'{:.2f}%'}), use_container_width=True)

    with t2:
        st.markdown("#### Target: Loan Amount")
        if 'loan_amount' in df.columns:
            df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
            fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            sns.histplot(df['loan_amount'], ax=ax1, bins='auto')
            try: sns.kdeplot(df['loan_amount'], ax=ax1)
            except Exception: pass
            ax1.set_title('Loan Amount Distribution'); ax1.set_xlabel('Loan Amount')
            sns.boxplot(x=df['loan_amount'], ax=ax2)
            ax2.set_title('Loan Amount Spread'); ax2.set_xlabel('Loan Amount')
            st.pyplot(fig1)
        else:
            st.warning("Column 'loan_amount' not found—upload a file with this target to see target plots.")

        num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        if len(num_cols) > 0:
            st.markdown("#### Numerical Features")
            chosen = st.multiselect("Pick features to visualize", num_cols,
                                    default=[c for c in ['income','Credit_Score','property_value'] if c in num_cols])
            if chosen:
                fig2, axes2 = plt.subplots(nrows=len(chosen), ncols=2,
                                           figsize=(14, 5*len(chosen)), squeeze=False)
                for i, col in enumerate(chosen):
                    sns.histplot(df[col], ax=axes2[i,0], bins='auto')
                    try: sns.kdeplot(df[col], ax=axes2[i,0])
                    except Exception: pass
                    axes2[i,0].set_title(f'{col} Distribution')
                    sns.boxplot(x=df[col], ax=axes2[i,1])
                    axes2[i,1].set_title(f'{col} Boxplot')
                st.pyplot(fig2)

    with t3:
        num_cols = df.select_dtypes(include=['int64','float64']).columns
        if len(num_cols)>1 and 'loan_amount' in num_cols:
            corr = df[num_cols].corr()
            top = corr['loan_amount'].sort_values(key=abs, ascending=False)
            fig, ax = plt.subplots(figsize=(10,7))
            sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Correlation Matrix (Numeric)"); st.pyplot(fig)

            st.markdown("**Top Correlations with Loan Amount**")
            top_disp = top.copy(); top_disp.index = [pretty_label(c) for c in top.index]
            st.dataframe(top_disp.to_frame('Correlation').iloc[1:11], use_container_width=True)
        else:
            st.info("Need at least two numeric columns including 'loan_amount' for correlation view.")

def Data_Preprocessing_page():
    st.title("2. Data Preprocessing")
    if st.button("Run Data Preprocessing"):
        try:
            create_preprocessor()
            processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
            st.subheader("Processed Data Sample")
            st.dataframe(processed_df.head(), use_container_width=True)
            st.success("Preprocessing completed and saved!")
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")

def Feature_Selection_page():
    st.title("3. Feature Selection")
    try:
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
        X = processed_df.drop('loan_amount', axis=1)
        y = processed_df['loan_amount']
    except Exception:
        st.warning("Please complete preprocessing first")
        return

    st.subheader("Best Subset Selection (Ridge)")
    max_features = st.slider("Maximum features to evaluate", 1, 15, 5)
    scoring_metric = st.selectbox("Selection metric", ['neg_root_mean_squared_error', 'r2'])

    if st.button("Run Best Subset Selection"):
        with st.spinner("Searching for optimal feature combinations..."):
            estimator = Ridge()
            sfs = SequentialFeatureSelector(estimator,
                                            n_features_to_select=max_features,
                                            direction='forward',
                                            scoring=scoring_metric,
                                            cv=5)
            sfs.fit(X, y)
            selected_features = X.columns[sfs.get_support()].tolist()
            save_artifact({'selected_features': selected_features,
                           'selection_metric': scoring_metric,
                           'support_mask': sfs.get_support()}, "5_best_subset_features.pkl")
            st.success(f"Selected {len(selected_features)} optimal features:")
            st.write(selected_features)
            st.subheader("CV on Selected Features")
            cv_results = cross_validate(estimator, X[selected_features], y, cv=5,
                                        scoring=['neg_root_mean_squared_error','r2'])
            metrics_df = pd.DataFrame({
                'Mean': [abs(cv_results['test_neg_root_mean_squared_error'].mean()), cv_results['test_r2'].mean()],
                'Std': [cv_results['test_neg_root_mean_squared_error'].std(), cv_results['test_r2'].std()]
            }, index=['RMSE','R2 Score'])
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

def Model_Selection_And_Training_page():
    st.title("4. Model Selection & Training")
    try:
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
        X = processed_df.drop('loan_amount', axis=1)
        y = processed_df['loan_amount']
    except Exception:
        st.warning("Please complete preprocessing first")
        return

    st.subheader("Ridge Parameters")
    alpha = st.slider("Ridge regularization strength (alpha)", 0.01, 10.0, 1.0)
    model = Ridge(alpha=alpha)

    if st.button("Run Cross-Validation"):
        rmse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
        r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        st.write(f"Mean RMSE: {abs(rmse_scores.mean()):.2f} (±{rmse_scores.std():.2f})")
        st.write(f"Mean R² : {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
        save_artifact({'rmse_scores': rmse_scores, 'r2_scores': r2_scores, 'params': model.get_params()}, "6_cv_results.pkl")

    if st.button("Train Final Model"):
        model.fit(X, y)
        save_artifact(model, "7_trained_model.pkl")
        y_pred = model.predict(X)
        processed_df['Predicted_Amount'] = y_pred
        processed_df.to_csv(f"{DATA_DIR}/8_predictions.csv", index=False)
        st.success("Model trained and saved!")

def Model_Evaluation_page():
    st.title("5. Model Evaluation")
    try:
        model = load_artifact("7_trained_model.pkl")
        predictions_df = pd.read_csv(f"{DATA_DIR}/8_predictions.csv")
    except Exception:
        st.warning("Please train the model first")
        return

    y_true = predictions_df['loan_amount']
    y_pred = predictions_df['Predicted_Amount']

    c1, c2 = st.columns(2)
    with c1:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        st.metric("RMSE", f"{rmse:.2f}")
    with c2:
        r2 = r2_score(y_true, y_pred)
        st.metric("R²", f"{r2:.4f}")

    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.3)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Actual Loan Amount"); ax.set_ylabel("Predicted Loan Amount")
    st.pyplot(fig)

def Interactive_Prediction_page():
    st.title("6. Interactive Prediction")
    try:
        model = load_artifact("7_trained_model.pkl")
        preprocessor = load_artifact("3_preprocessor.pkl")
        original_features = load_artifact("2_column_types.pkl")
    except Exception:
        st.warning("Please complete model training first")
        return

    st.subheader("Enter Applicant Information")
    input_data = {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numerical Features**")
        for f in original_features['numerical']:
            input_data[feature] = st.number_input(
            label=pretty_label(feature), value=0.0, step=0.01, format="%.2f",
            key=f"num_{feature}" 

    if st.button("Predict Default Amount", type="primary"):
        df_template = pd.DataFrame([input_data]).replace('', np.nan)
        try:
            X_processed = preprocessor.transform(df_template)
            predicted_amount = float(model.predict(X_processed)[0])
            st.success(f"Predicted Loan Default Amount: ${predicted_amount:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

def Results_Interpretation_And_Conclusion_page():
    st.title("7. Results & Conclusion")
    try:
        model = load_artifact("7_trained_model.pkl")
        predictions_df = pd.read_csv(f"{DATA_DIR}/8_predictions.csv")
    except Exception:
        st.warning("Please train the model first")
        return

    y_true = predictions_df['loan_amount']; y_pred = predictions_df['Predicted_Amount']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    st.markdown(f"""## Model Performance Insights

    - The final Ridge Regression model achieved an RMSE of *{rmse_value}*, meaning that on average, predictions deviate from actual defaults by this amount.
    - The R² score of *{r2_value}* indicates that the model explains approximately *{float(r2)*100:.1f}%* of the variation in loan default amounts.

    ## Feature Insights

    - The most important features for prediction were: {top_features_str}

    ## Practical Impact

    - This model can help banks identify high-risk loans, personalize loan limits, and automate risk assessment workflows.
    - Outlier predictions (where model error is high) may reveal cases needing manual review.

    ## Limitations

    - The model’s accuracy depends on the quality and representativeness of the training data.
    - It may not fully account for macroeconomic shifts, fraud, or abrupt life events impacting borrowers.

    ## Future Work

    - Explore ensemble models (e.g XGBoost) for potentially higher accuracy.
    - Enhance interpretability using tools like SHAP.
    - Update the model periodically to capture changing economic conditions and borrower behaviors.
    """)

# --- NAVIGATION ---
pages = {
    "Home Page": Home_Page,
    "Data Import and Overview": Data_Import_and_Overview_page,
    "Data Preprocessing": Data_Preprocessing_page,
    "Feature Selection": Feature_Selection_page,
    "Model Selection and Training": Model_Selection_And_Training_page,
    "Model Evaluation": Model_Evaluation_page,
    "Interactive Prediction": Interactive_Prediction_page,
    "Result Interpretation and Conclusion": Results_Interpretation_And_Conclusion_page,
}

selection = st.sidebar.selectbox("📚 Navigate", list(pages.keys()))
pages[selection]()

