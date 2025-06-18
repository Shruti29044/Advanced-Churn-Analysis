
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import shap
import xgboost as xgb

st.title("Customer Churn Analyzer — Professional Advanced Edition")

uploaded_file = st.file_uploader("Upload your customer dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    if 'Churn' not in df.columns:
        st.error("Dataset must contain 'Churn' column (1=churn, 0=no churn).")
    else:
        df.fillna(df.mean(numeric_only=True), inplace=True)
        features = df.drop(columns=['Churn']).select_dtypes(include=[np.number]).columns.tolist()
        st.write("Using numeric features:", features)

        X = df[features]
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"])
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)
        else:
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        st.subheader("Model Evaluation")
        st.text(classification_report(y_test, y_pred))
        auc = roc_auc_score(y_test, y_proba)
        st.write(f"AUC Score: {auc:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        st.pyplot(plt)

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        st.pyplot(plt)

        # SHAP Explainability
        st.subheader("Feature Importance (SHAP)")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(plt)

        # Retention Simulation
        st.subheader("Retention Campaign Simulation")
        uplift = st.slider("Retention Effect (reduce churn probability by %):", 0, 50, 10)
        adjusted_probs = y_proba * (1 - uplift / 100)
        adjusted_churn = (adjusted_probs >= 0.5).astype(int)
        reduced_churn_rate = np.mean(adjusted_churn)
        st.write(f"Projected Churn Rate after campaign: {reduced_churn_rate*100:.2f}%")

        revenue_per_customer = st.number_input("Average revenue per customer ($):", min_value=1.0, value=500.0)
        baseline_loss = np.sum(y_test) * revenue_per_customer
        adjusted_loss = np.sum(adjusted_churn) * revenue_per_customer
        st.write(f"Current Revenue Loss: ${baseline_loss:,.2f}")
        st.write(f"Projected Revenue Loss after retention campaign: ${adjusted_loss:,.2f}")

        if st.button("Export Results"):
            output = X_test.copy()
            output['ActualChurn'] = y_test.values
            output['PredictedChurn'] = y_pred
            output['AdjustedChurn'] = adjusted_churn
            output.to_csv("adjusted_churn_results.csv", index=False)
            st.success("Exported adjusted_churn_results.csv")
