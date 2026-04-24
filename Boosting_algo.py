import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
#Streamlit config
st.set_page_config(page_title="Customer Churn Prediction ",layout="wide")
st.title("Customer Churn using Boosting")
st.write("Enter Customer Details below to predict churn.")
# Load and train model
@st.cache_resource
def load_and_train_model():
    file_path=r"D:\\Riddhi\\Customer-Churn.csv"
    df=pd.read_csv(file_path)
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    df["Churn"]=df["Churn"].map({"No":0,"Yes":1})
    X=df.drop(columns=["Churn","customerID"])
    y=df["Churn"]
    numeric_features=X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_features=X.select_dtypes(include=["object"]).columns.tolist()
    # Train-test split
    X_train,X_test,y_train,y_test=train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    # Preprocessing pipeline
    numeric_transformer=Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
        ]
    )
    categorical_transformer=Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("scaler",OneHotEncoder(handle_unknown="ignore",sparse_output=False)),
        ]
    )
    preprocessor=ColumnTransformer(
      transformers=[
          ("num",numeric_transformer,numeric_features),
          ("cat", categorical_transformer,categorical_features),
      ]
  )
  # Process train and test
    X_train_processed=preprocessor.fit_transform(X_train)
    X_test_processed=preprocessor.transform(X_test)
    # feature Selection
    selector_model=RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )  
    selector_model.fit(X_train_processed,y_train)
    selector=SelectFromModel(selector_model,threshold="median",prefit=True)
    X_train_selected=selector.transform(X_train_processed)
    X_test_selected=selector.transform(X_test_processed)
    #Choose best model
    ada_model=AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1,random_state=42),
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,  
    )
    gb_model=GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    models={
        "AdaBoost":ada_model,
        "GradientBoosting":gb_model
    }
    best_model=None
    best_auc=-1
    results={}
    for name,model in models.items():
        model.fit(X_train_selected,y_train)
        y_pred=model.predict(X_test_selected)
        y_prob=model.predict_proba(X_test_selected)[:,1]
        acc=accuracy_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_prob)
        results[name]={
            "model":model,
            "accuracy":acc,
            "auc":auc
        }
        if auc>best_auc:
            best_auc=auc
            best_model=model
            best_model_name=name
    return{
            "preprocessor":preprocessor,
            "selector":selector,
            "best_model":best_model,
            "best_model_name":best_model_name,
            "results":results,
            "numeric_features":numeric_features,
            "categorical_features":categorical_features
            }
    
artifacts = load_and_train_model()

preprocessor = artifacts["preprocessor"]
selector = artifacts["selector"]
model = artifacts["best_model"]
best_model_name = artifacts["best_model_name"]
results = artifacts["results"]

# -------------------------------
# 2. Show model results
# -------------------------------
st.subheader("Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.write("AdaBoost")
    st.write(f"Accuracy: {results['AdaBoost']['accuracy']:.4f}")
    st.write(f"ROC-AUC: {results['AdaBoost']['auc']:.4f}")

with col2:
    st.write("GradientBoosting")
    st.write(f"Accuracy: {results['GradientBoosting']['accuracy']:.4f}")
    st.write(f"ROC-AUC: {results['GradientBoosting']['auc']:.4f}")

st.success(f"Best model selected automatically: {best_model_name}")

# -------------------------------
# 3. User Input Form
# -------------------------------
st.subheader("Enter Customer Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)

        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.1)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=850.0, step=0.1)

    submitted = st.form_submit_button("Predict Churn")

# -------------------------------
# 4. Prediction
# -------------------------------
if submitted:
    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
    })

    # Apply same preprocessing + feature selection
    input_processed = preprocessor.transform(input_df)
    input_selected = selector.transform(input_processed)

    prediction = model.predict(input_selected)[0]
    probability = model.predict_proba(input_selected)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {probability:.4f}")

    if prediction == 1:
        st.error("Prediction: Customer is likely to Churn")
    else:
        st.success("Prediction: Customer is likely to Stay")

