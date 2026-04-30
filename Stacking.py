import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    recall_score,
    precision_score,
    make_scorer,
)
#Streamlit config
st.set_page_config(page_title="Customer Churn Prediction ",layout="wide")
st.title("Customer Churn using Stacking")
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
        stratify=y,
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
    
    # Stacking Model with FN reduction focus
    base_models=[
        (
            "dt",
            DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced",
            )
        ),
        (
            "rf",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),
        ( 
         "knn",
         KNeighborsClassifier()
         ),
    ]
    meta_model=LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    stack_model=StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        passthrough=False,
        stack_method="predict_proba",
    )
    recall_scorer=make_scorer(recall_score)
    param_grid={
          "dt__max_depth": [3, 5, 7],
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [5, 10, None],
        "knn__n_neighbors": [3, 5, 7],
        "final_estimator__C":[0.1 ,1.0, 10.0],
    }
    grid=GridSearchCV(
        estimator=stack_model,
        param_grid=param_grid,
        scoring=recall_scorer,
        cv=3,
        n_jobs=-1,
        verbose=0,
        
    )
    grid.fit(X_train_selected,y_train)
    best_model=grid.best_estimator_
    #probabilities on test set
    y_prob=best_model.predict_proba(X_test_selected)[:,1]
    # threshold tuning to reduce FN
    threshold_candidates=np.arange(0.20,0.61,0.05)
    best_threshold=0.50
    best_recall=-1
    best_precision=-1,
    best_cm=None
    best_y_pred=None
    
    for threshold in threshold_candidates:
        y_pred_temp=(y_prob>=threshold).astype(int)
        recall_temp=recall_score(y_test,y_pred_temp)
        precision_temp=precision_score(y_test,y_pred_temp,zero_division=0)
        cm_temp=confusion_matrix(y_test,y_pred_temp)
        # priority:high recall
        if(recall_temp>best_recall)   or (
            recall_temp== best_recall and precision_temp>best_precision):
            best_recall=recall_temp
            best_precision=precision_temp
            best_threshold=threshold
            best_cm=cm_temp
            best_y_pred=y_pred_temp
            
            # Final Evaluation
            acc=accuracy_score(y_test,best_y_pred)
            auc=roc_auc_score(y_test,y_prob)
            report=classification_report(y_test,best_y_pred,digits=4)
             
            tn,fp,fn,tp=best_cm.ravel()
            results={
                     "Stacking":{
                         "model":best_model,
                         "accuracy":acc,
                         "auc":auc,
                         "recall":best_recall,
                         "precision":best_precision,
                         "best_threshold":best_threshold,
                         "confusion_matrix":best_cm,
                         "classification_report":report,
                         "tn":tn,
                         "fp":fp,
                         "fn":fn,
                         "tp":tp,
                         "best_params":grid.best_params_
                     }
                 }
                 
            return {
                     "preprocessor":preprocessor,
                     "selector":selector,
                     "best_model":best_model,
                     "best_model_name":"Stacking",
                     "results":results,
                     "numeric_features":numeric_features,
                     "categorical_features":categorical_features,
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
    st.write("Stacking Model")
    st.write(f"Accuracy: {results['Stacking']['accuracy']:.4f}")
    st.write(f"ROC-AUC: {results['Stacking']['auc']:.4f}")
    st.write(f"Recall: {results['Stacking']['recall']:.4f}")
    st.write(f"Precision: {results['Stacking']['precision']:.4f}")
    st.write(f"Best Threshold: {results['Stacking']['best_threshold']:.2f}")

with col2:
    st.write("Confusion Matrix")
    st.write(results["Stacking"]["confusion_matrix"])
    st.write(f"False Negatives:{ results["Stacking"]["fn"]}")
    st.write(f"False Positives:{ results["Stacking"]["fp"]}")  
    st.write(f"True Positives:{ results["Stacking"]["tp"]}")
    st.write(f"True Negatives:{ results["Stacking"]["tn"]}")      

st.success(f"Best model selected automatically: {best_model_name}")
with st.expander("classification Report"):
    st.text(results["Stacking"]["classification_report"])
with st.expander("Best Parameters from GridSearchCV"):
    st.write(results["Stacking"]["best_params"])

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

