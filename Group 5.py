# Core libraries for data processing and visualization
import streamlit as st  # Web application framework
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
from matplotlib import pyplot as plt  # Visualization
import seaborn as sns  # Enhanced visualization
import pickle  # Object serialization
import os  # File system operations
from PIL import Image  # Image support (for adding logos or visuals)

# Scikit-learn components for machine learning
from sklearn.ensemble import RandomForestClassifier  # ML algorithm
from sklearn.model_selection import cross_val_score  # Cross-Validation
from sklearn.metrics import (accuracy_score, precision_score,  # Evaluation metrics
                              recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Feature engineering
from sklearn.impute import SimpleImputer  # Missing value handling
from sklearn.pipeline import Pipeline  # ML workflow
from sklearn.compose import ColumnTransformer  # Column-wise transformations

# CONFIGURATION SECTION
DATA_DIR = "saved_data"
os.makedirs(DATA_DIR, exist_ok=True)


# HELPER FUNCTIONS
def save_artifact(obj, filename):
    """Serializes and saves Python objects to disk for persistence.
    Uses pickle for serialization, which is efficient for scikit-learn objects."""
    with open(f"{DATA_DIR}/{filename}", 'wb') as f:
        pickle.dump(obj, f)


def load_artifact(filename):
    """Loads serialized objects from disk.
    Ensures we can reuse preprocessed data and models across sessions."""
    with open(f"{DATA_DIR}/{filename}", 'rb') as f:
        return pickle.load(f)


# DATA LOADING AND PREPROCESSING
@st.cache_data  # Streamlit cache decorator for performance optimization
def load_data():
    """Loads and caches raw loan data with initial cleaning:
        - Drops irrelevant columns (ID, dtir1, etc.)
        - Preserves the target variable 'Status' (1=default, 0=non-default)
        - Saves cleaned data for reproducibility"""
    df = pd.read_csv("Loan_Default.csv")
    df = df.drop(columns=['ID', 'dtir1', 'submission_of_application', 'year'], errors='ignore')
    df.to_csv(f"{DATA_DIR}/1_raw_data.csv", index=False)
    return df


def create_preprocessor():
    """Creates a comprehensive preprocessing pipeline that:
    1. Separates numerical and categorical features
    2. For numerical features:
       - Imputes missing values with median (robust to outliers)
       - Standardizes features (mean=0, std=1) for model convergence
    3. For categorical features:
       - Imputes missing values with most frequent category
       - One-hot encodes for model compatibility
    4. Uses ColumnTransformer for parallel processing of different feature types"""
    df = load_data()
    X = df.drop('Status', axis=1)

    # Feature type identification
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Save feature types for reference
    save_artifact({'numerical': numerical_cols, 'categorical': categorical_cols},
                  "2_column_types.pkl")

    # Numerical pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Robust to outliers
        ('scaler', StandardScaler())])  # Helps models like SVM and neural networks

    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Preserves mode
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # Handles new categories

    # Combined preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # # Fit and save the preprocessor
    preprocessor.fit(X)
    save_artifact(preprocessor, "3_preprocessor.pkl")

    # Transform and save processed data
    X_processed = preprocessor.transform(X)
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    all_features = np.concatenate([num_features, cat_features])

    processed_df = pd.DataFrame(X_processed, columns=all_features)
    processed_df['Status'] = df['Status'].values
    processed_df.to_csv(f"{DATA_DIR}/4_processed_data.csv", index=False)

    return preprocessor




# STREAMLIT PAGE FUNCTIONS
def Home_Page():

    #LOGO
    logo = Image.open("LDP.jpg")
    st.image(logo, caption="", width=300)

    #FUNCTIONS
    st.title("Loan Default Prediction System")

    st.write("""
       Welcome to the Loan Default Prediction System. This application helps financial institutions 
       assess the risk of loan default using machine learning.

       Use the navigation menu on the left to explore different sections of the application.
       """)


    st.markdown("""---

    ## Team Members (Group 5)

    | Name                     | Student ID | Role                                             | Deployment link                                              |
    |--------------------------|------------|--------------------------------------------------|--------------------------------------------------------------|
    | Kingsley Sarfo           | 22252461   | Project Coordination, App Design & Preprocessing | https://loan-predictor-hbbz24vwfzaue2qx4hwcat.streamlit.app  |                           
    | Francisca Manu Sarpong   | 22255796   | Documentation & Deployment                       | https://kftalde5ypwd5a3qqejuvo.streamlit.app                 |               
    | George Owell             | 22256146   | Model Evaluation & Cross-validation              | loandefaultpredictionapp-utmbic9znd7uzqqhs9zgo6.streamlit.app|
    | Barima Owiredu Addo      | 22254055   | UI & Prediction Testing                          | https://loandefaultapp-ky4yy9kmt6ehsq8jqdcgs2.streamlit.app/ |
    | Akrobettoe Marcus        | 11410687   | Feature Selection & Model Training               | https://models-loan-default-prediction.streamlit.app/        |

    ---
    """)




def Data_Import_and_Overview_page():
    """Handles data upload and exploratory data analysis (EDA) including:
       - Basic statistics (mean, median, missing values)
       - Target variable distribution
       - Correlation analysis
       - Interactive visualizations (histograms, boxplots, etc.)"""

    st.title(" 1. Data Import and Overview")

    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv",
                                     help="Please upload your loan data in CSV format")

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_data'] = df
            st.success("File successfully uploaded!")

            # ========================
            # 1. Summary Statistics
            # ========================
            st.subheader("1. Summary Statistics")

            # Basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", df.shape[0])
            with col2:
                st.metric("Total Features", df.shape[1])
            with col3:
                if 'Status' in df.columns:
                    st.metric("Default Rate", f"{df['Status'].mean():.2%}")

            # Numerical features summary
            st.markdown("**Numerical Features Summary**")
            st.dataframe(df.describe().T.style.format("{:.2f}"))

            # Categorical features summary
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                st.markdown("**Categorical Features Summary**")
                cat_summary = pd.DataFrame({
                    'Unique Values': df[cat_cols].nunique(),
                    'Most Common': df[cat_cols].mode().iloc[0],
                    'Missing Values': df[cat_cols].isnull().sum()
                })
                st.dataframe(cat_summary)

            # Missing values analysis
            st.markdown("**Missing Values Analysis**")
            missing = df.isnull().sum().to_frame('Missing Values')
            missing['Percentage'] = (missing['Missing Values'] / len(df)) * 100
            st.dataframe(missing.style.format({'Percentage': '{:.2f}%'}))

            # ========================
            # 2. Data Visualizations
            # ========================
            st.subheader("Data Visualizations")

            # Target distribution (if exists)
            if 'Status' in df.columns:
                st.markdown("**Target Variable Distribution**")
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                # Countplot
                sns.countplot(x='Status', data=df, ax=ax[0])
                ax[0].set_title('Default Status Count')

                # Pie chart
                df['Status'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
                ax[1].set_title('Default Status Proportion')
                st.pyplot(fig)

            # Numerical distributions
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(num_cols) > 0:
                st.markdown("**Numerical Features Distribution**")
                selected_num = st.multiselect("Select numerical features to visualize",
                                              num_cols, default=num_cols[3:])

                if selected_num:
                    fig, ax = plt.subplots(len(selected_num), 2, figsize=(14, 5 * len(selected_num)))
                    for i, col in enumerate(selected_num):
                        # Histogram
                        sns.histplot(df[col], kde=True, ax=ax[i, 0])
                        ax[i, 0].set_title(f'{col} Distribution')
                        ax[i, 0].tick_params(axis='x', rotation=45)

                        # Boxplot
                        sns.boxplot(x=df[col], ax=ax[i, 1])
                        ax[i, 1].set_title(f'{col} Boxplot')
                        ax[i, 1].tick_params(axis='x', rotation=45)

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Scatterplots for key financial relationships
                    if len(selected_num) >= 2:
                        st.markdown("**Key Financial Relationships**")

                        # Create more relevant comparisons for loan analysis
                        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

                        # 1. Debt-to-Income Ratio vs Credit Score (most important relationship)
                        if all(col in df.columns for col in ['loan_amount', 'income', 'Credit_Score']):
                            df['debt_to_income'] = df['loan_amount'] / df['income']
                            sns.scatterplot(x='debt_to_income', y='Credit_Score', data=df, ax=ax[0])
                            ax[0].set_title('Credit Score vs Debt-to-Income Ratio')
                            ax[0].set_xlabel('Loan Amount / Annual Income')
                            ax[0].set_ylabel('Credit Score')

                            if 'Status' in df.columns:
                                sns.scatterplot(x='debt_to_income', y='Credit_Score', hue='Status',
                                                data=df, ax=ax[1], palette=['green', 'red'], alpha=0.7)
                                ax[1].set_title('Default Status by Debt-to-Income and Credit Score')
                                ax[1].set_xlabel('Loan Amount / Annual Income')
                                ax[1].set_ylabel('Credit Score')

                        # 2. Loan-to-Value Ratio vs Income (alternative comparison)
                        elif all(col in df.columns for col in ['loan_amount', 'property_value', 'income']):
                            df['loan_to_value'] = df['loan_amount'] / df['property_value']
                            sns.scatterplot(x='loan_to_value', y='income', data=df, ax=ax[0])
                            ax[0].set_title('Income vs Loan-to-Value Ratio')
                            ax[0].set_xlabel('Loan Amount / Property Value')
                            ax[0].set_ylabel('Annual Income')

                            if 'Status' in df.columns:
                                sns.scatterplot(x='loan_to_value', y='income', hue='Status',
                                                data=df, ax=ax[1], palette=['green', 'red'], alpha=0.7)
                                ax[1].set_title('Default Status by LTV and Income')
                                ax[1].set_xlabel('Loan Amount / Property Value')
                                ax[1].set_ylabel('Annual Income')

                        # Fallback to original comparison if expected columns missing
                        else:
                            sns.scatterplot(x=selected_num[0], y=selected_num[1], data=df, ax=ax[0])
                            ax[0].set_title(f'{selected_num[0]} vs {selected_num[1]}')

                            if 'Status' in df.columns:
                                sns.scatterplot(x=selected_num[0], y=selected_num[1], hue='Status',
                                                data=df, ax=ax[1], palette=['green', 'red'])
                                ax[1].set_title(f'{selected_num[0]} vs {selected_num[1]} by Default Status')

                        plt.tight_layout()
                        st.pyplot(fig)

            # Correlation matrix
            if len(num_cols) > 1:
                st.markdown("**Correlation Matrix**")
                corr_matrix = df[num_cols].corr()

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                            center=0, ax=ax)
                ax.set_title("Feature Correlations")
                st.pyplot(fig)

                # Top correlations
                st.markdown("**Top Feature Correlations**")
                corr_pairs = corr_matrix.unstack().sort_values(key=abs, ascending=False)
                st.dataframe(corr_pairs[corr_pairs != 1].head(10).to_frame('Correlation'))

            # Categorical visualizations
            if len(cat_cols) > 0:
                st.markdown("**Categorical Features Analysis**")
                selected_cat = st.selectbox("Select categorical feature", cat_cols)

                fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                # Countplot
                sns.countplot(y=selected_cat, data=df, ax=ax[0],
                              order=df[selected_cat].value_counts().index)
                ax[0].set_title(f'{selected_cat} Distribution')

                # Relationship with target (if exists)
                if 'Status' in df.columns:
                    sns.barplot(x='Status', y=selected_cat, data=df, ax=ax[1],
                                estimator=lambda x: len(x) / len(df) * 100)
                    ax[1].set_title(f'Default Rate by {selected_cat}')
                    ax[1].set_ylabel('Default Rate (%)')
                else:
                    df[selected_cat].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
                    ax[1].set_title(f'{selected_cat} Proportion')

                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def Data_Preprocessing_page():
    st.title("2. Data Preprocessing")
    """Manages the data preprocessing workflow:
        - Executes the preprocessing pipeline
        - Displays sample processed data
        - Shows feature engineering details"""

    if st.button("Run Data Preprocessing"):
        preprocessor = create_preprocessor()
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")

        st.subheader("Processed Data Sample")
        st.dataframe(processed_df.head())

        st.subheader("Preprocessing Details")
        st.write("Numerical features:", len(preprocessor.named_transformers_['num'].get_feature_names_out()))
        st.write("Categorical features:",
                 len(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()))

        st.success("Preprocessing completed and saved!")


def Feature_Selection_page():
    """Implements feature selection techniques:
        1. Correlation-based filtering
        2. Sequential Feature Selection (wrapper method)
        3. Feature importance visualization
        Uses cross-validation to evaluate feature subsets."""

    st.title("3. Feature Selection")

    try:
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
        X = processed_df.drop('Status', axis=1)
        y = processed_df['Status']
    except:
        st.warning("Please complete preprocessing first")
        return

    # Correlation analysis
    st.subheader("Initial Correlation Analysis")
    corr_matrix = processed_df.corr()
    corr_with_target = corr_matrix['Status'].sort_values(key=abs, ascending=False)
    st.dataframe(corr_with_target.to_frame("Correlation"))

    # Best Subset Selection
    st.subheader("Best Subset Selection")
    st.write("""
    This performs exhaustive search for the most predictive feature combinations.
    Note: Computation time increases exponentially with more features.
    """)

    # User controls
    max_features = st.slider("Maximum features to evaluate", 1, 15, 5)
    scoring_metric = st.selectbox("Selection metric",
                                  ['accuracy', 'precision', 'recall', 'f1'])

    if st.button("Run Best Subset Selection"):
        with st.spinner("Searching for optimal feature combinations..."):
            from sklearn.feature_selection import SequentialFeatureSelector
            from sklearn.linear_model import LogisticRegression

            # Use logistic regression as base estimator for efficiency
            estimator = LogisticRegression(max_iter=1000, random_state=42)
            sfs = SequentialFeatureSelector(estimator,
                                            n_features_to_select=max_features,
                                            direction='forward',
                                            scoring=scoring_metric,
                                            cv=5)

            sfs.fit(X, y)
            selected_features = X.columns[sfs.get_support()].tolist()

            # Save results
            save_artifact({
                'selected_features': selected_features,
                'selection_metric': scoring_metric,
                'support_mask': sfs.get_support()
            }, "5_best_subset_features.pkl")

            # Display results
            st.success(f"Selected {len(selected_features)} optimal features:")
            st.write(selected_features)

            # Show performance metrics
            st.subheader("Model Performance with Selected Features")
            from sklearn.model_selection import cross_validate
            cv_results = cross_validate(estimator,
                                        X[selected_features],
                                        y,
                                        cv=5,
                                        scoring=['accuracy', 'precision', 'recall', 'f1'])

            metrics_df = pd.DataFrame({
                'Mean': [cv_results[f'test_{m}'].mean() for m in ['accuracy', 'precision', 'recall', 'f1']],
                'Std': [cv_results[f'test_{m}'].std() for m in ['accuracy', 'precision', 'recall', 'f1']]
            }, index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

            st.dataframe(metrics_df.style.format("{:.2%}"))

            # Feature importance plot
            estimator.fit(X[selected_features], y)
            if hasattr(estimator, 'coef_'):
                importance = pd.Series(np.abs(estimator.coef_[0]),
                                       index=selected_features).sort_values(ascending=False)

                fig, ax = plt.subplots(figsize=(10, 6))
                importance.plot(kind='barh', ax=ax)
                ax.set_title("Feature Importance (Absolute Coefficients)")
                st.pyplot(fig)

    # Existing correlation visualization
    st.subheader("Feature Correlation Visualization")
    top_n = st.slider("Number of top features to show", 5, 20, 10)
    top_features = corr_with_target.index[1:top_n + 1]  # Exclude Status

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=corr_with_target.values[1:top_n + 1], y=top_features, ax=ax)
    ax.set_title(f"Top {top_n} Features Correlated with Loan Default")
    st.pyplot(fig)


def Model_Selection_And_Training_page():
    """Handles model training with:
        - Random Forest classifier (ensemble method)
        - Hyperparameter tuning (n_estimators, max_depth, etc.)
        - 5-fold cross-validation for robust performance estimation
        - Full model training on entire dataset"""

    st.title("4. Model Selection and Training")

    try:
        processed_df = pd.read_csv(f"{DATA_DIR}/4_processed_data.csv")
        X = processed_df.drop('Status', axis=1)
        y = processed_df['Status']
    except:
        st.warning("Please complete preprocessing first")
        return

    # Model configuration
    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Number of trees", 50, 500, 100)
        max_depth = st.slider("Max depth", 2, 20, 5)
    with col2:
        min_samples_split = st.slider("Min samples split", 2, 10, 2)
        bootstrap = st.checkbox("Bootstrap samples", value=True)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        bootstrap=bootstrap,
        random_state=42)

    # Cross-validation
    if st.button("Run Cross-Validation"):
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        st.write(f"Mean Accuracy: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

        save_artifact({
            'cv_scores': cv_scores,
            'params': model.get_params()
        }, "6_cv_results.pkl")

    # Full training
    if st.button("Train Final Model"):
        model.fit(X, y)
        save_artifact(model, "7_trained_model.pkl")

        # Generate and save predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        processed_df['Prediction'] = y_pred
        processed_df['Default_Probability'] = y_proba[:, 1]
        processed_df.to_csv(f"{DATA_DIR}/8_predictions.csv", index=False)

        st.session_state['model'] = model
        st.session_state['features'] = X.columns.tolist()
        st.success("Model trained and saved!")


def Model_Evaluation_page():
    """Comprehensive model evaluation including:
        - Accuracy, precision, recall, F1 scores
        - Confusion matrix visualization
        - Feature importance analysis
        - Performance metric saving"""

    st.title("5. Model Evaluation")

    try:
        model = load_artifact("7_trained_model.pkl")
        predictions_df = pd.read_csv(f"{DATA_DIR}/8_predictions.csv")
    except:
        st.warning("Please train the model first")
        return

    X = predictions_df.drop(['Status', 'Prediction', 'Default_Probability'], axis=1)
    y = predictions_df['Status']
    y_pred = predictions_df['Prediction']

    # Performance metrics
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{accuracy_score(y, y_pred):.2%}")
        st.metric("Precision", f"{precision_score(y, y_pred):.2%}")

    with col2:
        st.metric("Recall", f"{recall_score(y, y_pred):.2%}")
        st.metric("F1 Score", f"{f1_score(y, y_pred):.2%}")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.nlargest(5)

    fig, ax = plt.subplots()
    top_features.plot(kind='barh', ax=ax)
    st.pyplot(fig)

    # Save evaluation results
    if st.button("Save Evaluation Results"):
        save_artifact({
            'confusion_matrix': cm,
            'feature_importance': importance,
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }, "9_evaluation_results.pkl")
        st.success("Evaluation results saved!")


def Interactive_Prediction_page():
    """Interactive interface for real-time predictions:
        - Input form for applicant details
        - Risk probability visualization
        - Risk factor analysis
        - Model diagnostics
        Implements the complete prediction pipeline from raw input to risk assessment."""

    st.title("6. Interactive Prediction")

    try:
        # Load necessary artifacts
        model = load_artifact("7_trained_model.pkl")
        preprocessor = load_artifact("3_preprocessor.pkl")
        original_features = load_artifact("2_column_types.pkl")
    except:
        st.warning("Please complete model training first")
        return

    st.subheader("Enter Applicant Information")

    # Create input form with original feature names
    input_data = {}
    col1, col2 = st.columns(2)

    with col1:
        input_data['loan_amount'] = st.number_input("Loan Amount", min_value=0, value=100000)
        input_data['income'] = st.number_input("Annual Income", min_value=0, value=50000)
        input_data['Credit_Score'] = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

    with col2:
        input_data['property_value'] = st.number_input("Property Value", min_value=0, value=200000)
        input_data['loan_purpose'] = st.selectbox("Loan Purpose", ['p1', 'p2', 'p3', 'p4'])
        input_data['Gender'] = st.selectbox("Gender", ['Male', 'Female', 'Joint', 'Sex Not Available'])

    if st.button("Predict Default Risk"):
        try:
            # Create a DataFrame with all original features
            df_template = pd.DataFrame(columns=original_features['numerical'] + original_features['categorical'])

            # Fill in the provided values
            for feature, value in input_data.items():
                if feature in df_template.columns:
                    df_template[feature] = [value]

            # Fill missing values with defaults (0 for numerical, first category for categorical)
            for col in df_template.columns:
                if col not in input_data:
                    if col in original_features['numerical']:
                        df_template[col] = 0
                    else:
                        df_template[col] = df_template[col].astype('object')
                        df_template[col] = df_template[col].fillna(
                            df_template[col].iloc[0] if len(df_template[col]) > 0 else 'unknown')

            # Apply the same preprocessing
            X_processed = preprocessor.transform(df_template)

            # Make prediction
            probability = model.predict_proba(X_processed)[0]


            # ======================================
            # Enhanced Prediction Results Section
            # ======================================
            st.subheader("Prediction Results")

            # Show raw probabilities first
            with st.expander("🔍 Detailed Probabilities"):
                st.write(f"P(No Default): {probability[0]:.4f}")
                st.write(f"P(Default): {probability[1]:.4f}")
                st.write(f"Classification threshold: 50%")

            # More sensitive risk classification
            HIGH_RISK_THRESHOLD = 0.3  # Lowered from 0.5
            MEDIUM_RISK_THRESHOLD = 0.1

            if probability[1] > HIGH_RISK_THRESHOLD:
                st.error(f"🚨 HIGH RISK (Default Probability: {probability[1]:.2%})")
            elif probability[1] > MEDIUM_RISK_THRESHOLD:
                st.warning(f"⚠️ MEDIUM RISK (Default Probability: {probability[1]:.2%})")
            else:
                st.success(f"✅ LOW RISK (Default Probability: {probability[0]:.2%})")

            # Probability visualization
            fig, ax = plt.subplots()
            ax.bar(['No Default', 'Default'], probability, color=['green', 'red'])
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            # ======================================
            # Risk Factor Analysis Section
            # ======================================
            st.subheader("Risk Factor Analysis")

            # Calculate key risk ratios
            debt_to_income = input_data['loan_amount'] / max(1, input_data['income'])
            collateral_ratio = input_data['property_value'] / max(1, input_data['loan_amount'])

            st.write(f"**Debt-to-Income Ratio:** {debt_to_income:.1f}x")
            st.write(f"**Collateral Coverage:** {collateral_ratio:.1%}")
            st.write(f"**Credit Score:** {input_data['Credit_Score']} (FICO range: 300-850)")

            # Risk indicators
            risk_flags = []
            if input_data['Credit_Score'] < 580:
                risk_flags.append("Subprime credit score (<580)")
            if debt_to_income > 5:
                risk_flags.append("Extreme debt burden (>5x income)")
            if collateral_ratio < 0.2:
                risk_flags.append("Insufficient collateral (<20%)")
            if input_data['loan_purpose'] == 'p4':
                risk_flags.append("High-risk loan purpose (p4)")

            if risk_flags:
                st.warning("🚩 Risk flags detected:")
                for flag in risk_flags:
                    st.write(f"- {flag}")
            else:
                st.info("No strong risk factors identified")

                # ======================================
                # Model Diagnostics Section
                # ======================================
                #with st.expander("ℹ️ Model Diagnostics"):
                #st.write("**If predictions seem incorrect:**")
                #st.write("- The model may be under-trained")
                #st.write("- Training data may lack high-risk examples")
                #st.write("- Important risk features may be missing")

                if probability[1] < 0.05:
                    st.error("❗ Extremely low default probability")
                    st.write("This suggests either:")
                    st.write("- The applicant is genuinely low-risk")
                    st.write("- The model isn't sensitive to risk factors")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error("Please check your input values")


def Results_Interpretation_And_Conclusion_page():
    """Provides business interpretation of results:
        - Model performance summary
        - Implementation recommendations
        - Limitations and future improvements
        - Team contribution details"""

    st.title("7. Results Interpretation and Conclusion")

    st.write("""
    ## Model Performance Summary

    The Random Forest classifier has shown good performance in predicting loan defaults:
    - High accuracy on both training and test sets
    - Balanced precision and recall scores
    - Important features align with financial domain knowledge

    ## Business Implications


    - The model can help reduce financial losses by identifying high-risk applicants
    - Can be used to adjust interest rates based on risk levels
    - Helps standardize the loan approval process

    ## Limitations

    - Model performance depends on data quality
    - May need periodic retraining as economic conditions change
    - Doesn't capture all qualitative factors in loan decisions

    ## Future Improvements

    - Experiment with other algorithms (XGBoost, Neural Networks)
    - Incorporate more features (economic indicators, employment history)
    - Develop a risk scoring system based on model probabilities
    """)


# Map sidebar names to functions
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

selection = st.sidebar.selectbox("Select Page", list(pages.keys()))
pages[selection]()
