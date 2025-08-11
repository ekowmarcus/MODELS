
# --- IMPORT SECTION ---
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

# --- CONFIGURATION ---
DATA_DIR = "saved_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---- UI CONFIG ----
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💳",
    layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

# Global CSS for a cleaner aesthetic
st.markdown("""
<style>
:root{
  --primary:#0f4c81;
  --bg:#f8fafc;
  --card:#ffffff;
  --text:#0f172a;
  --muted:#64748b;
}
.main { background: var(--bg); }
section[data-testid="stSidebar"] { background: #0f4c8110; }
h1, h2, h3 { color: var(--text); }
.stButton>button {
  background: var(--primary); color: #fff; border: 0;
  padding: .55rem 1rem; border-radius: 12px; font-weight: 600;
  box-shadow: 0 6px 16px rgba(15,76,129,.18);
}
.stButton>button:hover{ filter: brightness(1.1); }
[data-testid="stMetricValue"] { color: var(--text); }
.dataframe tbody tr:hover { background:#0f4c810d; }
.card{
  background: var(--card); border-radius:16px; padding:16px;
  box-shadow: 0 4px 18px rgba(2,6,23,.08); border:1px solid #e5e7eb;
}
.card h4{ margin:0 0 .25rem 0; color: var(--muted); font-weight:600; }
.card .big{ font-size: 28px; font-weight: 800; color: var(--text); }
.pill{
  display:inline-block; padding:.2rem .6rem; border-radius:999px;
  background:#0f4c8115; color: var(--primary); font-size:12px; font-weight:700;
}
hr { border:0; border-top:1px solid #e5e7eb; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

def metric_card(title:str, value:str, subtext:str=""):
    st.markdown(f"""
    <div class="card">
      <h4>{title}</h4>
      <div class="big">{value}</div>
      <div class="pill">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def pretty_label(name: str) -> str:
    """Make column names human-friendly for display only."""
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    s = s.replace('_', ' ').strip()
    keep_upper = {'ID','DTI','LTV','APR'}
    words = [w.upper() if w.upper() in keep_upper else w.capitalize() for w in s.split()]
    return ' '.join(words)

def save_artifact(obj, filename):
    """Serialize and save Python objects to disk for later use.

        Args:
            obj: Any Python object to be serialized
            filename: Name of the file to save to (within DATA_DIR)
        """
    with open(f"{DATA_DIR}/{filename}", 'wb') as f:
        pickle.dump(obj, f)

def load_artifact(filename):
    """Load previously saved Python objects from disk.

        Args:
            filename: Name of the file to load (within DATA_DIR)
        Returns:
            The deserialized Python object
        """
    with open(f"{DATA_DIR}/{filename}", 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("Loan_Default.csv")
    df = df.drop(columns=['ID', 'dtir1', 'submission_of_application', 'year'], errors='ignore')
    df.to_csv(f"{DATA_DIR}/1_raw_data.csv", index=False)
    return df

def create_preprocessor():
    """Create and fit a preprocessing pipeline for the loan data."""
    df = load_data()
    X = df.drop('loan_amount', axis=1)

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    save_artifact({'numerical': numerical_cols, 'categorical': categorical_cols},
                  "2_column_types.pkl")

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    preprocessor.fit(X)
    save_artifact(preprocessor, "3_preprocessor.pkl")

    X_processed = preprocessor.transform(X)
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    all_features = np.concatenate([num_features, cat_features])
    processed_df = pd.DataFrame(X_processed, columns=all_features)
    processed_df['loan_amount'] = df['loan_amount'].values
    processed_df.to_csv(f"{DATA_DIR}/4_processed_data.csv", index=False)
    return preprocessor

# --- STREAMLIT PAGES ---
def Home_Page():
    """Landing page with visual cards and team info."""
    try:
        logo = Image.open("LDP.jpeg")
        st.image(logo, width=220)
    except Exception:
        st.markdown("### 💳 Loan Default Prediction")

    st.markdown(
        "Build, evaluate, and **predict loan default amounts** with a guided workflow. "
        "Use the left sidebar to move through each step."
    )
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Course", "Applied ML", "MSBA")
    with c2: metric_card("Tech Stack", "Python + Streamlit", "scikit-learn • pandas")
    with c3: metric_card("Model", "Ridge Regression", "Regularized")

    st.markdown("### Team")
    team_members = [
        ["1","Kingsley Sarfo", "22252461", "Project Coordinator", "https://loan-predictor-hbbz24vwfzaue2qx4hwcat.streamlit.app"],
        ["2","Francisca Sarpong", "22255796", "Data Preprocessing", "https://kftalde5ypwd5a3qqejuvo.streamlit.app"],
        ["3","George Owell", "22256146", "Model Evaluation", "https://loandefaultpredictionapp-utmbic9znd7uzqqhs9zgo6.streamlit.app"],
        ["4","Barima Addo", "22254055", "UI Testing", "https://loandefaultapp-ky4yy9kmt6ehsq8jqdcgs2.streamlit.app"],
        ["5","Marcus Akrobettoe", "11410687", "Feature Selection", "https://models-loan-default-prediction.streamlit.app"]
    ]
    df = pd.DataFrame(team_members,
                      columns=["SN", "Name", "Student ID", "Role", "Deployment Link"])
    st.dataframe(df,
                 hide_index=True,
                 use_container_width=True,
                 column_config={"Deployment Link": st.column_config.LinkColumn("Deployment")})
    st.caption("Kaggle dataset • This app is for learning/demonstration purposes.")

def Data_Import_and_Overview_page():
    """Page for data upload, exploration and visualization."""
    st.title("1. Data Import & Overview")

    st.subheader("Upload Your Loan Data (CSV)")
    uploaded_file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.drop(columns=['ID', 'dtir1', 'submission_of_application', 'year'], errors='ignore')
        st.session_state['df'] = df
        st.success("✅ File uploaded and loaded. This data will be used across the app.")
    elif 'df' in st.session_state:
        df = st.session_state['df']
    else:
        st.info("Upload a dataset to continue.")
        return

    # KPI Cards
    k1, k2, k3 = st.columns(3)
    with k1: metric_card("Total Records", f"{df.shape[0]:,}")
    with k2: metric_card("Total Features", f"{df.shape[1]:,}")
    with k3:
        avg = df['loan_amount'].mean() if 'loan_amount' in df.columns else 0
        metric_card("Average Loan Amount", f"${avg:,.0f}")

    # Tabs
    t1, t2, t3 = st.tabs(["🔎 Preview", "📊 Distributions", "🔗 Correlations"])

    with t1:
        st.dataframe(df.head(50), use_container_width=True, height=320)

        st.markdown("#### Summaries")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Numerical Features**")
            num_summary = df.describe().T
            num_summary.index = [pretty_label(c) for c in num_summary.index]
            st.dataframe(num_summary.style.format("{:.2f}"), use_container_width=True, height=300)
        with colB:
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
            # ensure numeric
            df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
            fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            sns.histplot(df['loan_amount'], ax=ax1, bins='auto')
            try: sns.kdeplot(df['loan_amount'], ax=ax1)
            except Exception: pass
            ax1.set_title('Loan Amount Distribution'); ax1.set_xlabel('Loan Amount')
            sns.boxplot(x=df['loan_amount'], ax=ax2)
            ax2.set_title('Loan Amount Spread'); ax2.set_xlabel('Loan Amount')
            st.pyplot(fig1)

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
            top_disp = top.copy()
            top_disp.index = [pretty_label(c) for c in top_disp.index]
            st.dataframe(top_disp.to_frame('Correlation').iloc[1:11], use_container_width=True)

def Data_Preprocessing_page():
    """Manages the data preprocessing workflow."""
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
    """Page for feature selection using sequential forward selection."""
    st.title("3. Feature Selection")
    try:
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
        X = processed_df.drop('loan_amount', axis=1)
        y = processed_df['loan_amount']
    except:
        st.warning("Please complete preprocessing first")
        return

    st.subheader("Best Subset Selection (Ridge Regression)")
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
            save_artifact({
                'selected_features': selected_features,
                'selection_metric': scoring_metric,
                'support_mask': sfs.get_support()
            }, "5_best_subset_features.pkl")
            st.success(f"Selected {len(selected_features)} optimal features:")
            st.write(selected_features)
            st.subheader("Model Performance with Selected Features")
            cv_results = cross_validate(estimator,
                                       X[selected_features], y,
                                       cv=5,
                                       scoring=['neg_root_mean_squared_error', 'r2'])
            metrics_df = pd.DataFrame({
                'Mean': [abs(cv_results['test_neg_root_mean_squared_error'].mean()), cv_results['test_r2'].mean()],
                'Std': [cv_results['test_neg_root_mean_squared_error'].std(), cv_results['test_r2'].std()]
            }, index=['RMSE', 'R2 Score'])
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
            estimator.fit(X[selected_features], y)
            if hasattr(estimator, 'coef_'):
                importance = pd.Series(np.abs(estimator.coef_),
                                       index=selected_features).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                importance.plot(kind='barh', ax=ax)
                ax.set_title("Feature Importance (Absolute Coefficients)")
                st.pyplot(fig)

def Model_Selection_And_Training_page():
    """Page for Ridge Regression model selection, cross-validation, and final training."""
    st.title("4. Model Selection and Training")
    try:
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
        X = processed_df.drop('loan_amount', axis=1)
        y = processed_df['loan_amount']
    except:
        st.warning("Please complete preprocessing first")
        return

    st.subheader("Ridge Regression Parameters")
    alpha = st.slider("Ridge regularization strength (alpha)", 0.01, 10.0, 1.0)
    model = Ridge(alpha=alpha)

    if st.button("Run Cross-Validation"):
        rmse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
        r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        st.write(f"Mean RMSE: {abs(rmse_scores.mean()):.2f} (±{rmse_scores.std():.2f})")
        st.write(f"Mean R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
        save_artifact({
            'rmse_scores': rmse_scores,
            'r2_scores': r2_scores,
            'params': model.get_params()
        }, "6_cv_results.pkl")

    if st.button("Train Final Model"):
        model.fit(X, y)
        save_artifact(model, "7_trained_model.pkl")
        y_pred = model.predict(X)
        processed_df['Predicted_Amount'] = y_pred
        processed_df.to_csv(f"{DATA_DIR}/8_predictions.csv", index=False)
        st.session_state['model'] = model
        st.session_state['features'] = X.columns.tolist()
        st.success("Model trained and saved!")

def Model_Evaluation_page():
    """Displays evaluation metrics and visualizations for the trained model."""
    st.title("5. Model Evaluation")
    try:
        model = load_artifact("7_trained_model.pkl")
        predictions_df = pd.read_csv(f"{DATA_DIR}/8_predictions.csv")
    except:
        st.warning("Please train the model first")
        return

    y_true = predictions_df['loan_amount']
    y_pred = predictions_df['Predicted_Amount']

    st.subheader("Model Performance")
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

    st.subheader("Feature Importance")
    if hasattr(model, 'coef_'):
        importance = pd.Series(np.abs(model.coef_), index=predictions_df.drop(['loan_amount','Predicted_Amount'], axis=1).columns)
        top_features = importance.nlargest(5)
        fig, ax = plt.subplots()
        top_features.plot(kind='barh', ax=ax)
        ax.set_title("Top Feature Importance (|coef|)")
        st.pyplot(fig)

def Interactive_Prediction_page():
    """Interactive page for making real-time predictions."""
    st.title("6. Interactive Prediction")
    try:
        model = load_artifact("7_trained_model.pkl")
        preprocessor = load_artifact("3_preprocessor.pkl")
        original_features = load_artifact("2_column_types.pkl")
    except:
        st.warning("Please complete model training first")
        return

    st.subheader("Enter Applicant Information")
    input_data = {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numerical Features**")
        for feature in original_features['numerical']:
            input_data[feature] = st.number_input(label=f"{feature}", value=0.0, step=0.01, format="%.2f")
    with col2:
        st.markdown("**Categorical Features**")
        for feature in original_features['categorical']:
            input_data[feature] = st.text_input(label=f"{feature}", value="", help=f"Enter {feature} value")

    if st.button("Predict Default Amount", type="primary"):
        df_template = pd.DataFrame([input_data])
        try:
            X_processed = preprocessor.transform(df_template)
            predicted_amount = model.predict(X_processed)[0]
            st.success(f"Predicted Loan Default Amount: ${predicted_amount:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check your input values and try again")

def Results_Interpretation_And_Conclusion_page():
    st.title("7. Results & Conclusion")
    try:
        model = load_artifact("7_trained_model.pkl")
        predictions_df = pd.read_csv(f"{DATA_DIR}/8_predictions.csv")
    except:
        st.warning("Please train the model first")
        return

    y_true = predictions_df['loan_amount']
    y_pred = predictions_df['Predicted_Amount']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if hasattr(model, 'coef_'):
        importance = pd.Series(np.abs(model.coef_), index=predictions_df.drop(['loan_amount','Predicted_Amount'], axis=1).columns)
        top_features = importance.nlargest(5)
        feats = ", ".join([f"{k} ({v:.2f})" for k,v in top_features.items()])
    else:
        feats = "N/A"

    st.markdown(f"""
    ### Model Performance
    - **RMSE:** {rmse:.2f}
    - **R²:** {r2:.4f}

    ### Feature Insights
    - Most influential features: {feats}

    ### Business Impact
    - Use predictions to prioritize high-risk applicants and right-size loan limits.
    - Automate risk flags for manual review.

    ### Caveats & Next Steps
    - Skew/outliers can affect linear models; consider log-transform/winsorizing.
    - Try Regularization search (RidgeCV), or Elastic Net, and add robust validation.
    """)

# --- ROUTING ---
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
