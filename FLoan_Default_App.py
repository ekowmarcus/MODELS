import base64

import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import re
from PIL import Image
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image
from streamlit import logo

# Load uploaded data once and store in session
#def load_uploaded_data():
 #   uploaded_file = st.sidebar.file_uploader(" Upload your Loan Default CSV file", type="csv")
  #  if uploaded_file is not None:
   #     df = pd.read_csv(uploaded_file)
    #    st.session_state["df"] = df
     #   return df
    #return None

# Load once at app start (optional)
#if "df" not in st.session_state:
 #   load_uploaded_data()


# Setting up Home page configuration
# Set page configuration
st.set_page_config(
    page_title="Loan Default Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and display logo image
def get_base64_image(path):
    with open(path, "rb") as img_file:
        data = base64.b64encode(img_file.read()).decode()
    return data

img_data = get_base64_image("Loan_Default_Image.png")
st.markdown(
    f"""
    <div style="text-align: right;">
        <img src="data:image/png;base64,{img_data}" width="300"/>
    </div>
    """,
    unsafe_allow_html=True
)

# Function to hold pages
def Home_Page():
    st.title("Loan Default Prediction Web App")
    st.markdown("""
    ---

    ### Project Overview

    This interactive web application was developed as part of an applied regression and machine learning course project. It simulates a real-world scenario where a data science team is tasked with building a system to *predict the probability of loan default* based on demographic and financial features.

    Users can explore the dataset, follow the full machine learning workflow, and interact with the final model to generate real-time predictions.

    ---

    ### What This App Covers:
    - *Data Import and Exploration*
    - *Cleaning, Encoding, and Preprocessing*
    - *Feature Selection using Best Subset Selection*
    - *Model Training with Ridge Regression*
    - *Model Evaluation (RMSE, R², Cross-Validation)*
    - *Interactive Prediction Interface*
    - *Final Results Interpretation and Conclusion*

    ---

    ### How to Use This App
    Use the sidebar to navigate through the project steps:

    1. Data Import and Overview – Explore the dataset. 
    
    2. Data Preprocessing – Clean, impute, encode, and standardize
    3. Feature Selection – Best subset based on Ridge
    4. Model Training – Fit Ridge regression to selected features
    5. Model Evaluation – RMSE, R², and k-Fold CV
    6. Prediction – Enter values and predict default probability
    7. Conclusion – Insights and limitations

    ---

    *Developed by:* [Group 5 ]  
    *Tool:* Python + Streamlit + Scikit-learn

    """)

    # Members
    st.markdown("---")
    st.markdown("### Project Team")
    team_members = [
        ("Kingsley Sarfo", "22252461", "Project Coordinator, App Design"),
        ("Francisca Manu Sarpong", "22255796", "Preprocessing,Documentation,Deployment"),
        ("George Owell", "22256146", "Evaluation, Cross-validation"),
        ("Barima Owiredu Addo", "22254055", "Interactive Prediction UI, Testing"),
        ("Akrobettoe Marcus", "11410687", "Feature Selection, Model Training")
    ]

    # Create table-like layout
    col1, col2, col3 = st.columns([4, 1.5, 5])

    with col1:
        st.markdown("*Name of Student*")
        for name, _, _ in team_members:
            st.markdown(name)

    with col2:
        st.markdown("*Student ID*")
        for _, student_id, _ in team_members:
            st.markdown(student_id)

    with col3:
        st.markdown("*Role in Project*")
        for _, _, role in team_members:
            st.markdown(role)

    # Project Overview Section
    st.markdown("""

    ### Instructions:

    1. Use the sidebar menu on the left to navigate between the pages.
    2. Start from *"1. Data Upload and Overview"*.
    3. Follow each step in sequence for best results.

    ---
    ###  Dataset Information:
    - Source: [Kaggle - Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
    - Target variable: Status (indicating default)
    - Due to GitHub’s size limitations, the dataset was hosted on Google Drive.Here is the secure link to access the dataset: (https://drive.google.com/file/d/1NGurIkGeLmFVIjJVu-oUuIvbUUScpiLL/view?usp=sharing)

    """)

    ### Defining the metadata
    st.markdown("""
    ### *Data Attributes* """)
    data_dict = [
        {"Column": "ID", "Data Type": "int", "Model Role": "Ignore", "Description": "Unique record ID."},
        {"Column": "year", "Data Type": "int", "Model Role": "Ignore", "Description": "Year of loan application."},
        {"Column": "loan_limit", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Loan amount limit type."},
        {"Column": "Gender", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Gender of primary applicant."},
        {"Column": "approv_in_adv", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Was loan approved in advance?"},
        {"Column": "loan_type", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Loan product type."},
        {"Column": "loan_purpose", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Purpose of the loan."},
        {"Column": "Credit_Worthiness", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Applicant credit profile."},
        {"Column": "open_credit", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Whether applicant has open credit lines."},
        {"Column": "business_or_commercial", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Is the loan for business or commercial use?"},
        {"Column": "loan_amount", "Data Type": "float", "Model Role": "Numerical",
         "Description": "Total amount requested."},
        {"Column": "rate_of_interest", "Data Type": "float", "Model Role": "Numerical",
         "Description": "Interest rate on the loan."},
        {"Column": "Interest_rate_spread", "Data Type": "float", "Model Role": "Numerical",
         "Description": "Difference in interest rate and benchmark."},
        {"Column": "Upfront_charges", "Data Type": "float", "Model Role": "Numerical",
         "Description": "Initial fees paid upfront."},
        {"Column": "term", "Data Type": "int", "Model Role": "Numerical",
         "Description": "Loan repayment period (months)."},
        {"Column": "Neg_ammortization", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Is there negative amortization?"},
        {"Column": "interest_only", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Interest-only loan?"},
        {"Column": "lump_sum_payment", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Any lump-sum payment options?"},
        {"Column": "property_value", "Data Type": "float", "Model Role": "Numerical",
         "Description": "Market value of the property."},
        {"Column": "construction_type", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Construction classification of property."},
        {"Column": "occupancy_type", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Primary, Secondary, or Investment home."},
        {"Column": "Secured_by", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Collateral Type (e.g., landed property, Motor vehicles, Cash)."},
        {"Column": "total_units", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Total dwelling units."},
        {"Column": "income", "Data Type": "float", "Model Role": "Numerical",
         "Description": "Applicant's monthly income."},
        {"Column": "credit_type", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Main credit reporting agency."},
        {"Column": "Credit_Score", "Data Type": "float", "Model Role": "Numerical",
         "Description": "Numerical credit score."},
        {"Column": "co-applicant_credit_type", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Co-applicant's credit agency."},
        {"Column": "age", "Data Type": "object", "Model Role": "Ordinal",
         "Description": "Applicant age range (e.g., 25-34)."},
        {"Column": "submission_of_application", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Was the application submitted online or in person?"},
        {"Column": "LTV", "Data Type": "float", "Model Role": "Numerical", "Description": "Loan to value ratio."},
        {"Column": "Region", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Region where application was submitted."},
        {"Column": "Security_Type", "Data Type": "object", "Model Role": "Categorical",
         "Description": "Form of security for loan (e.g., direct, indirect)."},
        {"Column": "Status", "Data Type": "int", "Model Role": "Target",
         "Description": "Loan status (1 = defaulted, 0 = paid)."},
        {"Column": "dtir1", "Data Type": "float", "Model Role": "Numerical", "Description": "Debt-to-Income Ratio."}
    ]

    metadata_df = pd.DataFrame(data_dict)

    st.dataframe(metadata_df, use_container_width=True, height=600)

    st.info("This table helps to understand what each column means and how it's used in the prediction model.")

    # Footer
    st.markdown("---")
    st.markdown("#### Navigate through the sidebar to explore each stage of the machine learning pipeline.")

def Data_Import_and_Overview_page():
    st.title("Data Import & Overview")
    st.markdown("This section shows the raw data, summary statistics, and visualizations.")

    # Functionality to Upload CSV file
    uploaded_file = st.file_uploader("Upload your loan default dataset (.csv)", type="csv")

    if uploaded_file is None:
        st.warning("⚠ Please upload a dataset to continue.")
        return  # ✅ safely exits the function here

        # This only runs if a file is uploaded
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset successfully loaded!")

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10))

    #  Shape & Types
    st.subheader("Dataset Shape & Column Types")
    st.markdown(f"- **Rows:** {df.shape[0]}  \n- **Columns:** {df.shape[1]}")
    st.write(df.dtypes)

    #  Missing values & duplicates
    st.subheader("Missing Values & Duplicates")
    missing = df.isnull().sum()
    dup_count = df.duplicated().sum()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Missing Values**")
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        st.write(missing)
    else:
        st.success("No missing values")
    with col2:
        st.markdown("**Duplicate Rows**")
        if dup_count:
            st.warning(f"{dup_count} duplicate rows found")
        else:
            st.success("No duplicates found")
    # Numeric vs. Categorical
    st.subheader("Numeric vs. Categorical Columns")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    st.markdown(f"- **Numeric columns ({len(num_cols)}):** {num_cols}")
    st.markdown(f"- **Categorical columns ({len(cat_cols)}):** {cat_cols}")

    #  Summary Statistics using df.describe()
    st.subheader("Summary Statistics")
    st.write(df.describe())

    #  Count plots for categorical (pick up to 3 most frequent)
    st.subheader("Top Categories for Categorical Features")
    for col in cat_cols[:3]:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=col, order=df[col].value_counts().index[:10], ax=ax)
        ax.set_title(col)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    #  Correlation heatmap
    st.subheader("Correlation Matrix")
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Scatter plots: pick two pairs
    st.subheader("Sample Scatter plots")
    pairs = [("income", "loan_amount"), ("age", "loan_amount")]
    for x, y in pairs:
        if x in df.columns and y in df.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x, y=y, hue=cat_cols[0] if cat_cols else None, ax=ax)
            ax.set_title(f"{y} vs {x}")
            st.pyplot(fig)

    # Target distribution
    st.subheader("Target Distribution (`Status`)")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Status', order=df['Status'].value_counts().index, ax=ax)
    ax.set_title("Default vs Non-default")
    st.pyplot(fig)

    # Box plots for outliers (key numerics)
    st.subheader("Box plots of Numeric Features")
    for col in ['loan_amount','income','Credit_Score','LTV']:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

    #  Feature vs target (box plots)
    st.subheader("Numeric Features by Target Class")
    for col in ['income','loan_amount','Credit_Score','LTV']:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Status', y=col, ax=ax)
        ax.set_title(f"{col} by Status")
        st.pyplot(fig)

    #  Missing-value heatmap
    st.subheader("Missing-Value Heatmap")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
    ax.set_title("Where Are Missing Values?")
    st.pyplot(fig)

    # Skewness & kurtosis
    st.subheader("Skewness & Kurtosis of Numerics")
    skew_kurt = pd.DataFrame({
        'skewness': df[num_cols].skew(),
        'kurtosis': df[num_cols].kurt()
    })
    st.dataframe(skew_kurt)

    # Pair plot of key features
    st.subheader("Pair plot of Selected Features")
    sel = ['loan_amount','income','Credit_Score','LTV','Status']
    sns.pairplot(df[sel], hue='Status', corner=True, plot_kws={'alpha':0.5})
    st.pyplot(plt.gcf())  # get current figure

    # Distributions: histograms for numeric
    st.subheader("Distributions of Numeric Features")
    num_cols = ['loan_amount', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges','property_value', 'income', 'Credit_Score', 'LTV', 'dtir1']
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

def missing_value_summary(df):
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage (%)': missing_percent
    })
    # Show only columns that have missing values
    missing_df = missing_df[missing_df['Missing Values'] > 0]
    return missing_df.sort_values(by='Percentage (%)', ascending=False)


def Data_Preprocessing_page():
    st.title("Data Preprocessing")
    st.markdown("""
    This section focuses on preparing the dataset for machine learning by: transforming all categorical variables into numerical formats that can be fed into the machine learning model.
    Different encoding strategies are used based on the nature of each variable.

    Let's start by examining the missing values in the dataset.
    """)
    # Upload CSV
    uploaded_file = st.file_uploader("Upload your loan default dataset (.csv)", type="csv")
    if uploaded_file is None:
        st.warning("⚠ Please upload a dataset to continue.")
        return

    df = pd.read_csv(uploaded_file)


    # Define your 21 categorical columns
    cat_cols = [
        'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
        'Credit_Worthiness', 'open_credit', 'business_or_commercial',
        'Neg_ammortization', 'interest_only', 'lump_sum_payment',
        'construction_type', 'occupancy_type', 'Secured_by', 'total_units',
        'credit_type', 'co-applicant_credit_type', 'age',
        'submission_of_application', 'Region', 'Security_Type'
    ]

    # Compute and display unique counts
    st.subheader("Unique Value Counts for Categorical Features")
    unique_counts = {col: df[col].nunique() for col in cat_cols}
    uniq_df = (
        pd.DataFrame.from_dict(unique_counts, orient='index', columns=['n_unique'])
        .sort_values('n_unique')
    )
    st.dataframe(uniq_df, use_container_width=True)

    st.title("Data Cleaning: Gender, Age, and Region")

    # Making a Copy of the original Dataset
    df_cleaned = df.copy()

    # Standardizing column names
    df_cleaned.columns = df_cleaned.columns.str.strip().str.lower().str.replace(" ", "_")

    # Show before and after
    st.subheader("Original Column Names")
    st.write(list(df.columns))

    st.subheader("Standardized Column Names")
    st.write(list(df_cleaned.columns))

    # Show original values before cleaning
    st.subheader("Original Categorical Values")
    st.write("Gender:", df['Gender'].unique())
    st.write("Age:", df['age'].unique())
    st.write("Region:", df['Region'].unique())

    # -------------------------
    # CLEANING STARTS HERE
    # -------------------------

    # 1. Standardize text
    df_cleaned['gender'] = df_cleaned['gender'].str.lower().str.strip()
    df_cleaned['age'] = df_cleaned['age'].str.lower().str.strip()
    df_cleaned['region'] = df_cleaned['region'].str.lower().str.strip()

    # 2. Gender cleanup
    df_cleaned['gender'] = df_cleaned['gender'].replace({
        'sex not available': 'unknown',
        'joint': 'unknown'
    })

    # 3. Age group labels cleanup
    df_cleaned['age'] = df_cleaned['age'].replace({
        '<25': 'under_25',
        '>74': '75+'
    })

    # Cleaning  Region
    df_cleaned['region'] = df_cleaned['region'].str.lower().str.strip()

    # -------------------------
    # DISPLAY CLEANED OUTPUT
    # -------------------------

    st.subheader("Cleaned Categorical Values")
    st.write("gender:", df_cleaned['gender'].unique())
    st.write("Age:", df_cleaned['age'].unique())
    st.write("Region:", df_cleaned['region'].unique())

    # Optional: Save cleaned copy to CSV for use in next steps
    # df_cleaned.to_csv("cleaned_data.csv", index=False)




    # Handling Missing Values
    st.subheader("Missing Values After Cleaning")

    # Check for any remaining missing values
    missing_after = df_cleaned.isnull().sum()
    missing_after = missing_after[missing_after > 0]

    st.write("Columns with missing values:")
    st.dataframe(missing_after.to_frame(name='Missing Count'))

    st.subheader("Count of 'unknown' in Gender")
    st.write(df_cleaned['gender'].value_counts())

    # flagging the Unknown for feature importance analysis later
    df_cleaned['gender_unknown_flag'] = (df_cleaned['gender'] == 'unknown').astype(int)
    st.info("Flagging the Unknown helps the model explicitly recognize that gender info was missing.")


    st.subheader("Missing Values Before Imputation")

    # Calculate missing counts before
    missing_before = df_cleaned.isnull().sum()
    missing_before = missing_before[missing_before > 0].sort_values(ascending=False)

    # Display table
    st.dataframe(missing_before.to_frame(name="Missing Count"))

    # Visualize missing before
    fig1, ax1 = plt.subplots()
    missing_before.plot(kind='bar', ax=ax1, color='orange')
    ax1.set_title("Missing Values Before Imputation")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Columns")
    st.pyplot(fig1)

    st.subheader("Handling Missing Values")
    # 🔹 High Missing (10–30%) — Numeric → Median
    high_missing_numeric = [
        'upfront_charges', 'interest_rate_spread', 'rate_of_interest',
        'dtir1', 'property_value', 'ltv', 'income'
    ]
    df_cleaned[high_missing_numeric] = df_cleaned[high_missing_numeric].fillna(
        df_cleaned[high_missing_numeric].median()
    )

    # 🔹 Low Missing (<1%) — Mixed
    # Numeric → Median
    df_cleaned['term'] = df_cleaned['term'].fillna(df_cleaned['term'].median())

    # Categorical → Mode
    low_missing_categorical = [
        'loan_limit', 'approv_in_adv', 'submission_of_application',
        'age', 'loan_purpose', 'neg_ammortization'
    ]
    for col in low_missing_categorical:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

    st.markdown("""
    ### Why These Imputation Methods Were Used

    We applied different imputation techniques based on the **type of variable** and the **percentage of missing values**:

    ---

    #### High Missing (10–30%) → **Median Imputation** (Numeric Columns)

    **Columns:** `upfront_charges`, `interest_rate_spread`, `rate_of_interest`, `dtir1`, `property_value`, `LTV`, `income`

    - **Why median?** These columns contain financial values which often include **outliers** (e.g., extremely high loan amounts or incomes).
    - The **median** is resistant to outliers and better represents the central tendency of skewed data than the mean.

    ---

    #### Low Missing (< 1%) → **Mode or Median Imputation**

    **Numeric Column:** `term` → **Median**
    - This is a continuous feature with few missing values. Median safely fills gaps without affecting distribution.

    **Categorical Columns:** `loan_limit`, `approv_in_adv`, `submission_of_application`, `age`, `loan_purpose`, `neg_ammortization` → **Mode**
    - For categorical features, the **most frequent category (mode)** was used.
    - This avoids introducing noise and maintains the dominant class pattern in the data.

    ---

    #### 'Unknown' Category (like Gender)
    - Instead of imputing or dropping, we treated `"unknown"` as a valid category (especially since it covers over 50%).
    - This helps preserve data volume while allowing the model to learn patterns even with missing demographic info.

    ---
    """)

    st.subheader("Missing Values After Imputation")

    # Calculate missing after
    missing_after = df_cleaned.isnull().sum()
    missing_after = missing_after[missing_after > 0].sort_values(ascending=False)

    if not missing_after.empty:
        st.warning("Some columns still have missing values:")
        st.dataframe(missing_after.to_frame("Missing Count"))

        # Visualize remaining missing
        fig2, ax2 = plt.subplots()
        missing_after.plot(kind='bar', ax=ax2, color='red')
        ax2.set_title("Missing Values After Imputation")
        ax2.set_ylabel("Count")
        ax2.set_xlabel("Columns")
        st.pyplot(fig2)
    else:
        st.success(" All missing values handled successfully!")




    # Encoding Categorical Variables
    st.subheader("Encoding Categorical Variables")

    # Viewing the unique values in the Categorical Columns
    st.subheader("Unique Values in Categorical Columns")

    # Define categorical columns
    categorical_cols = [
        'loan_limit', 'gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
        'credit_worthiness', 'open_credit', 'business_or_commercial', 'neg_ammortization',
        'interest_only', 'lump_sum_payment', 'construction_type', 'occupancy_type',
        'secured_by', 'total_units', 'credit_type', 'co-applicant_credit_type', 'age',
        'submission_of_application', 'region', 'security_type'
    ]

    # Display unique values for each column
    for col in categorical_cols:
        st.write(f"**{col}**: {df_cleaned[col].unique().tolist()}")

    st.subheader("Final Encoding: One-Hot Encoding Applied")

    # List of cleaned categorical columns
    categorical_cols = [
        'loan_limit', 'gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
        'credit_worthiness', 'open_credit', 'business_or_commercial', 'neg_ammortization',
        'interest_only', 'lump_sum_payment', 'construction_type', 'occupancy_type',
        'secured_by', 'total_units', 'credit_type', 'co-applicant_credit_type', 'age',
        'submission_of_application', 'region', 'security_type'
    ]

    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

    # Display shapes and example columns
    st.write(f"Original Shape: {df_cleaned.shape}")
    st.write(f"Encoded Shape: {df_encoded.shape}")
    st.write("Sample Encoded Columns:", df_encoded.columns[:10].tolist())

    st.success(" One-hot encoding applied to all categorical features.")

    # Saving the Cleaned and Encoded Data in CSV for later use

    df_encoded.to_csv("encoded_cleaned_data.csv", index=False)
    st.success(" Encoded data saved as 'encoded_cleaned_data.csv'.")

    st.markdown("""
    ### 💾 Why Save the Encoded Data?

    - The **encoded dataset** contains all cleaned and transformed features, ready for modeling.
    - By saving it to a CSV, we can **reuse it in other Streamlit pages** (e.g., feature selection, training, prediction) without repeating the cleaning steps.
    - This keeps the app modular and **improves performance** by avoiding repeated transformations.

    """)


def Feature_Selection_page():

    st.title("Feature Selection: Best Subset Selection")

    # Loading the encoded dataset
    @st.cache_data
    def load_data():
        return pd.read_csv("encoded_cleaned_data.csv")

    new_df = load_data()

    # Separate predictors (X) and target (y)
    X = new_df.drop(columns=['status'])  # 'status' is the target variable (loan default)
    y = new_df['status']

    # Show how many features are available before selection
    st.write("Number of features before selection:", X.shape[1])

    # Initialize Ridge regression (required for the subset selector)
    # Ridge is used because it's your chosen model, and it handles multicollinearity well
    model = Ridge()

    # Perform best subset selection using forward stepwise method
    # Apply Sequential Forward Selection to choose best 15 features
    # This starts with zero features and adds one at a time, choosing the best at each step
    sfs = SequentialFeatureSelector(
        estimator=model,
        n_features_to_select=15,  # you can change this number based on model performance
        direction='forward'
    )
    sfs.fit(X, y)  # Fit the selector to the data

    # Extracting the names of the selected features
    selected_features = X.columns[sfs.get_support()]

    # Display selected features in the Streamlit app
    st.write("Best subset of selected features (n=15):")
    st.write(selected_features.tolist())


    # Save new subset to CSV for modeling
    new_df_selected = new_df[selected_features.tolist() + ["status"]]
    new_df_selected.to_csv("selected_features_data.csv", index=False)
    st.success("Saved to 'selected_features_data.csv'")


def Model_Selection_And_Training_page():

    st.title("Model Training – Ridge Regression")

    # Load dataset with selected features from previous step
    @st.cache_data
    def load_data():
        return pd.read_csv("selected_features_data.csv")

    new_df_selected = load_data()

    # Split data into features (X) and target (y)
    X = new_df_selected.drop(columns=["status"])  # Features
    y = new_df_selected["status"]  # Target: loan default status

    # Split into training and testing sets (80% train, 20% test)
    # Why? This helps evaluate the model on unseen data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Ridge regression model
    # Why Ridge? It adds L2 regularization to reduce overfitting and handle multicollinearity
    ridge_model = Ridge(alpha=1.0)

    # Train the model on the training data
    ridge_model.fit(X_train, y_train)

    # Show success message
    st.success("Ridge Regression model trained successfully on selected features.")

    # Display model coefficients (optional for interpretation)
    st.subheader("Model Coefficients")
    coeffs = pd.Series(ridge_model.coef_, index=X.columns)
    st.write(coeffs.sort_values(ascending=False))

    st.markdown("""
    ### Model Coefficient Interpretation – Ridge Regression

    The table below shows the **coefficients** assigned to each selected feature by the Ridge regression model. Here's how to interpret them:

    ---

    #### **Positive Coefficients (Increase Default Risk):**

    - **`credit_type_EQUI` (+0.80):**  
      Applicants with credit reports from EQUI are associated with **higher default probability**.

    - **`submission_of_application_to_inst` (+0.12):**  
      Submitting the loan application through an institution slightly increases the chance of default.

    - **`loan_type_type2` (+0.09):**  
      Type 2 loans may carry slightly more risk compared to the baseline.

    ---

    #### **Negative Coefficients (Reduce Default Risk):**

    - **`lump_sum_payment_not_lpsm` (-0.35):**  
      Applicants who **did not choose lump sum payment** have a notably higher risk of default.

    - **`neg_ammortization_not_neg` (-0.15):**  
      Loans without negative amortization are linked to **lower default risk**.

    - **`interest_rate_spread` (-0.10):**  
      Slightly surprising — higher rate spread correlates with **lower default** in this model, potentially due to interaction effects.

    ---

    #### **Features with Minimal Effect:**

    - Features like `upfront_charges` and `dtir1` have **coefficients near zero**, meaning their contribution to the model is minimal.

    ---

    ### Notes:

    - The **magnitude** shows the **impact strength**.
    - The **sign** (positive/negative) shows the **direction of the effect**.
    - Coefficients are **regularized** (shrunk) due to Ridge’s L2 penalty — reducing overfitting.

    """)

def Model_Evaluation_page():

    st.title("Model Evaluation")

    # Load selected feature dataset
    @st.cache_data
    def load_data():
        return pd.read_csv("selected_features_data.csv")

    new_df_selected = load_data()

    # Split into features (X) and target (y)
    X = new_df_selected.drop(columns=["status"])
    y = new_df_selected["status"]

    # Train-test split for visual comparison
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Ridge model
    model = Ridge(alpha=1.0)

    # k-Fold Cross-Validation (k=5)
    # R² scores
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    st.write("**5-Fold Cross-Validation R² Scores**:", r2_scores)
    st.write("Mean R² Score:", np.round(np.mean(r2_scores), 4))

    # Fit and Predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # RMSE and R²
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.metric("RMSE", f"{rmse:.4f}")
    st.metric("R² Score", f"{r2:.4f}")

    # Visual comparison: Predicted vs. Actual
    st.subheader("Predicted vs. Actual Plot")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # ideal line
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual Values")
    st.pyplot(fig)

    st.markdown("""
    ### Model Evaluation Summary

    ---

    #### **5-Fold Cross-Validation (R² Scores):**
    - We used **k-Fold Cross-Validation (k=5)** to assess model reliability across different subsets of data.
    - This prevents overfitting to a single train/test split and gives a more **robust estimate** of model performance.

    ---

    #### **Root Mean Square Error (RMSE):**
    - RMSE tells us the **average size of prediction error**.
    - Lower RMSE indicates better model accuracy.
    - It’s sensitive to outliers and penalizes large errors more heavily.

    ---

    ####  **R-Squared (R²):**
    - R² indicates the proportion of variance in the target variable explained by the model.
    - Values range from 0 to 1 — higher is better.
    - R² closer to 1 means the model predicts the output well.

    ---

    ####  **Predicted vs. Actual Plot:**
    - Each point shows a real vs predicted value.
    - The closer the points are to the red dashed line (perfect prediction), the better the model.
    - Deviation from the line shows where the model under/overestimates defaults.

    ---
    """)

def Interactive_Prediction_page():
    st.title("Loan Default Prediction")

    # Load selected features and data
    @st.cache_data
    def load_data():
        return pd.read_csv("selected_features_data.csv")

    new_df_selected = load_data()

    # Get feature list (exclude target)
    features = new_df_selected.drop(columns=["status"]).columns.tolist()

    # Collect user input
    st.subheader("Enter Feature Values")
    user_input = {}
    # Smart UI: dropdowns for dummies, sliders/numbers for continuous
    for feature in features:
        if feature.endswith("_ncf") or feature.endswith("_type2") or feature.endswith("_type3") \
                or feature.endswith("_p2") or feature.endswith("_p3") \
                or feature.endswith("_not_neg") or feature.endswith("_not_lpsm") \
                or feature.endswith("_pr") or feature.endswith("_EQUI") \
                or feature.endswith("_EXP") or feature.endswith("_to_inst"):
            # These are one-hot encoded binary features → dropdown
            user_input[feature] = st.selectbox(f"{feature}", [0, 1])
        else:
            user_input[feature] = st.number_input(f"{feature}", value=0.0)

    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Train Ridge model on full dataset
    X = new_df_selected[features]
    y = new_df_selected["status"]
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Predict probability of default
    prediction = model.predict(input_df)[0]

    # Show result
    st.subheader("🧾 Prediction Result")
    st.write(f"**Predicted Default Probability:** `{prediction:.4f}`")

    # Optional: classify based on threshold
    threshold = 0.5
    pred_class = "Will Default" if prediction >= threshold else "Will Not Default"
    st.markdown(f"### Classification: **{pred_class}** (Threshold = {threshold})")

    st.subheader(" Prediction Result")
    st.write(f"**Predicted Default Probability:** `{prediction:.4f}`")

    # Confidence bar (visual)
    st.progress(min(max(prediction, 0.0), 1.0))  # bar range: 0–1


def Results_Interpretation_And_Conclusion_page():

    st.title("Results Interpretation and Conclusion")

    # Section 1 – Regression Interpretation
    st.markdown("""
    ### Regression Output Interpretation

    The Ridge regression model selected 15 key features after feature selection. The coefficients indicate how each feature influences the probability of loan default:

    - **Positive coefficients** (e.g., `credit_type_EQUI`, `loan_type_type2`) increase the likelihood of default.
    - **Negative coefficients** (e.g., `lump_sum_payment_not_lpsm`, `interest_rate_spread`) reduce the likelihood.

    These weights help us understand **which borrower characteristics are riskier** or safer.

    ---
    """)

    # Section 2 – Model Performance
    st.markdown("""
    ### Model Performance

    - **Cross-Validation R² Score**: ~0.41  
    - **Test Set R²**: ~0.41  
    - **RMSE**: ~0.33

    These values indicate the model explains about **41% of the variance** in loan default, with moderate prediction error. This is **acceptable for financial behavior prediction**, but not highly precise.

    ---
    """)

    # Section 3 – Implications and Limitations
    st.markdown("""
    ### Limitation: Ridge Regression on a Binary Target

    The target variable used in this project, status, is binary:
    - 0 → Non-default
    - 1 → Default

    We applied *Ridge Regression, which is a **linear regression model*, not a classification model.

    ---

    #### Implications:

    - *Continuous Output*:  
      Ridge predicts values between 0 and 1 (e.g., 0.32, 0.78), which we interpret as the *probability of default*. A threshold (e.g., 0.5) is then used to classify applicants.

    - *Performance Tradeoff*:  
      Since Ridge is not optimized for binary classification, the *R² and RMSE* may not reflect classification quality as well as precision, recall, or AUC would in a classifier.

    - *Interpretation Caution*:  
      Coefficients indicate *linear influence on the predicted probability*, but not on log-odds (as in logistic regression). Interpretation is still meaningful but less precise for binary targets.

    ---

    #### Justification for Using Ridge:

    - Ridge regression was required by the project instructions.
    - It enables us to demonstrate a *full supervised ML pipeline* using regularization and subset feature selection.
    - The output helps estimate *risk of default*, which is often more useful than a strict Yes/No in real-world loan evaluation.

    """)

    st.markdown("""
    ### Data Upload Approach – Why We Used st.file_uploader() and the pd.read_csv():

    - During local development in PyCharm, we loaded the dataset using pd.read_csv("Loan_Default.csv") 
    - This allowed us to preprocess, clean, and prepare the data efficiently while building the machine learning pipeline.
    
    - However, for online deployment on Streamlit Community Cloud, large CSV files cannot be bundled or accessed directly from the app's file system. To overcome this limitation, we used:

    - st.file_uploader("Upload your dataset", type="csv") which was also a requirement in the project instructions

    - This allows users (including you, our wonderful Profesor;lol) to upload the dataset manually at runtime when using the app online.
    
    ---
    """)

    # Section 4 – Visual Summary (optional table)
    coeff_data = {
        'Feature': ['credit_type_EQUI', 'loan_type_type2', 'submission_of_application_to_inst',
                    'lump_sum_payment_not_lpsm'],
        'Coefficient': [0.80, 0.09, 0.12, -0.35],
        'Effect': ['↑ Default Risk', '↑ Default Risk', '↑ Default Risk', '↓ Default Risk']
    }
    st.write("### Key Feature Effects")
    st.dataframe(pd.DataFrame(coeff_data))

def Project_Report_page():
    st.title("📄 Final Project Report")

    st.markdown("""
###  Group 5 – University of Ghana

---

## 1. Introduction

This interactive web application simulates a real-world scenario where a data science team is tasked with building a system to **predict loan default probability** using demographic and financial features. The goal is to apply the **full machine learning pipeline** from preprocessing to deployment using Ridge Regression.

---

## 2. Dataset Overview

- Source: Kaggle Loan Default Dataset  
- Target variable: `Status` (0 = Paid, 1 = Default)  
- ~148,671 records, 34 features (categorical & numerical)  
-  Some columns had missing values and inconsistent formats.

---

## 3. Tools and Libraries

- Python, Streamlit
- pandas, NumPy
- scikit-learn (Ridge, feature selection, CV)
- seaborn, matplotlib
- GitHub, Streamlit Community Cloud (Deployment)

---

## 4. Machine Learning Pipeline

###  Data Preprocessing

- Handled missing values:
  - Numeric → Median
  - Categorical → Mode
- Cleaned and standardized text (e.g., gender, age, region)
- Flagged unknown categories (e.g., `gender_unknown_flag`)

###  Encoding

- Applied **One-Hot Encoding** on 21 categorical variables  
- Saved encoded dataset as `encoded_cleaned_data.csv`

###  Feature Selection

- Used **Sequential Feature Selector** with Ridge Regression  
- Selected top **15 most predictive features**

###  Model Training

- Used **Ridge Regression (L2)** to reduce multicollinearity
- Split: 80% train / 20% test
- `alpha=1.0` regularization parameter

###  Model Evaluation

- **RMSE** ≈ `0.33`  
- **R² Score** ≈ `0.41`  
- Used **5-fold Cross-Validation**  
- Visualized *Predicted vs Actual*

---

## 5. Interactive Prediction

- User inputs 15 selected feature values via Streamlit UI  
- Ridge model predicts **default probability**  
- Classification based on threshold (0.5):  
  - ≥ 0.5 → “Will Default”  
  - < 0.5 → “Will Not Default”

---

## 6. Model Insights

| Feature                              | Coefficient | Effect              |
|--------------------------------------|-------------|---------------------|
| `credit_type_EQUI`                   | +0.80       | ↑ Default Risk      |
| `submission_of_application_to_inst` | +0.12       | ↑ Default Risk      |
| `loan_type_type2`                    | +0.09       | ↑ Default Risk      |
| `lump_sum_payment_not_lpsm`         | -0.35       | ↓ Default Risk      |
| `interest_rate_spread`              | -0.10       | ↓ Default Risk      |

---

## 7. Limitation

- `Status` is binary (0/1), but Ridge is a regression model.
- Ridge estimates probabilities, not class labels.
- Would benefit from **Logistic Regression or Random Forest** in future.

---

## 8. Deployment

- App hosted on Streamlit Community Cloud  
- GitHub Repository: [Group 5 Loan Default App](https://kftalde5ypwd5a3qqejuvo.streamlit.app)  
- File upload handled using `st.file_uploader()`  
  *(CSV not included in GitHub due to size limits)*

---

## 9. Team Members

| Name                    | Student ID | Role                              |
|-------------------------|------------|-----------------------------------|
| Kingsley Sarfo          | 22252461   | App Design, Coordinator           |
| Francisca Manu Sarpong  | 22255796   | Preprocessing, Deployment         |
| George Owell            | 22256146   | Evaluation, Cross-validation      |
| Barima Owiredu Addo     | 22254055   | UI & Interactive Prediction       |
| Akrobettoe Marcus       | 11410687   | Feature Selection, Model Training |

---

## 10. Conclusion

This project demonstrates:
- Real-world application of machine learning to financial risk.
- Importance of careful data cleaning, encoding, and feature selection.
- Interactive ML apps improve interpretability for decision makers.

---
""")


# Map sidebar names to functions
pages = {
    "Home Page": Home_Page,
    "Data Import and Overview": Data_Import_and_Overview_page,
    "Data Preprocessing": Data_Preprocessing_page,
    "Feature Selection" : Feature_Selection_page,
    "Model Selection and Training" : Model_Selection_And_Training_page,
    "Model Evaluation": Model_Evaluation_page,
    "Interactive Prediction" : Interactive_Prediction_page,
    "Result Interpretation and Conclusion" : Results_Interpretation_And_Conclusion_page,
    "Project Report" : Project_Report_page,
}

selection = st.sidebar.selectbox("Select Page", list(pages.keys()))
pages[selection]()