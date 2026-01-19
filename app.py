import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")

@st.cache_resource
def load_model(model_name):
    """Load the selected model"""
    model_files = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "Decision Tree": "models/decision_tree.pkl",
        "kNN": "models/knn.pkl",
        "Naive Bayes": "models/naive_bayes.pkl",
        "Random Forest": "models/random_forest.pkl",
        "XGBoost": "models/xgboost.pkl"
    }
    with open(model_files[model_name], 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    """Load the feature scaler"""
    with open('models/scaler.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_names():
    """Load feature names"""
    with open('models/feature_names.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    """Load the full dataset"""
    return pd.read_csv('data/dataset.csv')

@st.cache_data
def load_results():
    """Load model comparison results"""
    return pd.read_csv('models/model_comparison_results.csv')

def preprocess_data(df):
    """Preprocess the data to match training format"""
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df_encoded

tab1, tab2, tab3 = st.tabs(["📊 Dataset Description", "📈 Training Analysis", "🎯 Try It Out"])

with tab1:
    try:
        df = load_data()

        st.header("📋 Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Customers", f"{len(df):,}", border=True)

        with col2:
            st.metric("Features", df.shape[1] - 1, border=True)

        with col3:
            churn_pct = (df['Churn'] == 'Yes').sum() / len(df) * 100
            st.metric("Churn Rate", f"{churn_pct:.1f}%", border=True)

        with col4:
            st.metric("Data Quality", "100%" if df.isnull().sum().sum() == 0 else "Has Missing", border=True)

        st.divider()

        st.header("🎯 Problem Statement")
        st.markdown("""
        **Objective:** Predict which customers are likely to churn (leave the service) based on their
        demographic information, account details, and service usage patterns.

        **Business Impact:**
        - Retaining existing customers is 5-25x cheaper than acquiring new ones
        - Early churn prediction enables proactive retention strategies
        - Personalized interventions can reduce customer loss
        """)

        st.divider()

        st.header("🏷️ Feature Categories")

        subtab1, subtab2, subtab3, subtab4 = st.tabs(["Demographic", "Account Info", "Services", "Target"])

        with subtab1:
            st.subheader("Demographic Information")
            st.markdown("""
            - **gender**: Customer gender (Male/Female)
            - **SeniorCitizen**: Whether customer is a senior citizen (0/1)
            - **Partner**: Whether customer has a partner (Yes/No)
            - **Dependents**: Whether customer has dependents (Yes/No)
            """)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                df['gender'].value_counts().plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
                ax.set_title('Gender Distribution')
                ax.set_xlabel('Gender')
                ax.set_ylabel('Count')
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                df['SeniorCitizen'].value_counts().plot(kind='bar', ax=ax, color=['#2ca02c', '#d62728'])
                ax.set_title('Senior Citizen Distribution')
                ax.set_xlabel('Senior Citizen (0=No, 1=Yes)')
                ax.set_ylabel('Count')
                st.pyplot(fig)

        with subtab2:
            st.subheader("Account Information")
            st.markdown("""
            - **tenure**: Number of months with the company
            - **Contract**: Contract type (Month-to-month, One year, Two year)
            - **PaperlessBilling**: Paperless billing enabled (Yes/No)
            - **PaymentMethod**: Payment method used
            - **MonthlyCharges**: Current monthly charges
            - **TotalCharges**: Total charges accumulated
            """)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df['tenure'], bins=30, color='skyblue', edgecolor='black')
                ax.set_title('Tenure Distribution')
                ax.set_xlabel('Months')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                df['Contract'].value_counts().plot(kind='bar', ax=ax, color=['#9467bd', '#8c564b', '#e377c2'])
                ax.set_title('Contract Type Distribution')
                ax.set_xlabel('Contract Type')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)

        with subtab3:
            st.subheader("Services")
            st.markdown("""
            - **PhoneService**: Phone service (Yes/No)
            - **MultipleLines**: Multiple phone lines (Yes/No/No phone service)
            - **InternetService**: Internet service type (DSL/Fiber optic/No)
            - **OnlineSecurity**: Online security add-on (Yes/No/No internet)
            - **OnlineBackup**: Online backup add-on (Yes/No/No internet)
            - **DeviceProtection**: Device protection add-on (Yes/No/No internet)
            - **TechSupport**: Technical support add-on (Yes/No/No internet)
            - **StreamingTV**: Streaming TV service (Yes/No/No internet)
            - **StreamingMovies**: Streaming movies service (Yes/No/No internet)
            """)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                df['InternetService'].value_counts().plot(kind='bar', ax=ax, color=['#17becf', '#bcbd22', '#7f7f7f'])
                ax.set_title('Internet Service Distribution')
                ax.set_xlabel('Service Type')
                ax.set_ylabel('Count')
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                services = ['PhoneService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport']
                service_counts = [df[s].value_counts().get('Yes', 0) for s in services]
                ax.bar(services, service_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                ax.set_title('Service Adoption')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

        with subtab4:
            st.subheader("Target Variable")
            st.markdown("""
            - **Churn**: Whether the customer left (Yes/No)

            **Class Distribution:**
            """)

            churn_counts = df['Churn'].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                st.metric("No Churn", f"{churn_counts.get('No', 0):,} ({churn_counts.get('No', 0)/len(df)*100:.1f}%)", border=True)
                st.metric("Churn", f"{churn_counts.get('Yes', 0):,} ({churn_counts.get('Yes', 0)/len(df)*100:.1f}%)", border=True)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#2ca02c', '#d62728']
                churn_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('Churn Distribution')
                ax.set_ylabel('')
                st.pyplot(fig)

        st.divider()

        st.header("📄 Data Sample")
        st.dataframe(df.head(10), use_container_width=True)

        st.divider()

        st.header("📊 Statistical Summary")

        summary_type = st.segmented_control("Select Summary Type:", ["Numerical Features", "Categorical Features"], default="Numerical Features")

        if summary_type == "Numerical Features":
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            st.dataframe(df[numerical_cols].describe(), use_container_width=True)
        else:
            categorical_cols = df.select_dtypes(include=['object']).columns
            st.dataframe(df[categorical_cols].describe(), use_container_width=True)

        st.divider()

        st.header("📥 Download Dataset")

    
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset as CSV",
            data=csv,
            file_name="telco_customer_churn.csv",
            mime="text/csv"
        )

        test_df = pd.read_csv('data/test.csv')
        st.download_button(
            label="Download Test Dataset as CSV",
            data=test_df.to_csv(index=False),
            file_name="test.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.error("❌ Dataset file not found! Please ensure 'data/dataset.csv' exists.")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")

with tab2:
    try:
        results_df = load_results()

        st.header("📊 Performance Overview")

        col1, col2, col3, col4 = st.columns(4)

        best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        best_accuracy = results_df['Accuracy'].max()

        with col1:
            st.metric("Best Model", best_model, border=True)

        with col2:
            st.metric("Best Accuracy", f"{best_accuracy:.2%}", border=True)

        with col3:
            st.metric("Best AUC", f"{results_df['AUC'].max():.4f}", border=True)

        with col4:
            st.metric("Models Trained", len(results_df), border=True)

        st.divider()

        st.header("📋 Complete Results Table")

        st.dataframe(
            results_df.style.highlight_max(
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                color='lightgreen'
            ),
            use_container_width=True
        )

        st.divider()

        st.header("📊 Visual Comparison")

        subtab1, subtab2 = st.tabs(["Bar Charts", "Heatmap"])

        with subtab1:
            st.subheader("Metric Comparison - Bar Charts")

            metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

            for i in range(0, 6, 2):
                col1, col2 = st.columns(2)

                with col1:
                    metric = metrics[i]
                    fig, ax = plt.subplots(figsize=(8, 5))
                    results_df.plot(
                        x='Model',
                        y=metric,
                        kind='bar',
                        ax=ax,
                        legend=False,
                        color='steelblue'
                    )
                    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
                    ax.set_xlabel('')
                    ax.set_ylabel(metric)
                    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
                    ax.grid(axis='y', alpha=0.3)

                    best_idx = results_df[metric].idxmax()
                    ax.patches[best_idx].set_color('green')
                    ax.patches[best_idx].set_alpha(0.7)

                    st.pyplot(fig)

                if i + 1 < len(metrics):
                    with col2:
                        metric = metrics[i + 1]
                        fig, ax = plt.subplots(figsize=(8, 5))
                        results_df.plot(
                            x='Model',
                            y=metric,
                            kind='bar',
                            ax=ax,
                            legend=False,
                            color='steelblue'
                        )
                        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
                        ax.set_xlabel('')
                        ax.set_ylabel(metric)
                        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
                        ax.grid(axis='y', alpha=0.3)

                        best_idx = results_df[metric].idxmax()
                        ax.patches[best_idx].set_color('green')
                        ax.patches[best_idx].set_alpha(0.7)

                        st.pyplot(fig)

        with subtab2:
            st.subheader("Heatmap - Model vs Metric Performance")

            fig, ax = plt.subplots(figsize=(10, 6))

            heatmap_data = results_df.set_index('Model')[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]

            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.4f',
                cmap='RdYlGn',
                center=0.7,
                ax=ax,
                cbar_kws={'label': 'Score'}
            )

            ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
            st.pyplot(fig)

        st.divider()

        st.header("📊 Model Rankings")

        ranking_metric = st.selectbox("Rank models by:", metrics)

        ranked_df = results_df.sort_values(by=ranking_metric, ascending=False).reset_index(drop=True)
        ranked_df.index = ranked_df.index + 1

        st.dataframe(
            ranked_df[['Model', ranking_metric]],
            use_container_width=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="large"),
                ranking_metric: st.column_config.ProgressColumn(
                    ranking_metric,
                    help=f"Score for {ranking_metric}",
                    format="%.4f",
                    min_value=0,
                    max_value=1,
                ),
            }
        )

        st.divider()

        st.header("⚙️ Training Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Dataset Split:**
            - Training: 70% (4,930 samples)
            - Testing: 30% (2,113 samples)
            - Stratification: Applied

            **Preprocessing:**
            - Feature Scaling: StandardScaler
            - Categorical Encoding: One-hot encoding
            - Missing Values: Handled
            """)

        with col2:
            st.markdown("""
            **Model Configuration:**
            - Random State: 42 (reproducibility)
            - Cross-Validation: Not applied
            - Hyperparameter Tuning: Default parameters
            - Ensemble Methods: 100 estimators

            **Evaluation:**
            - 6 metrics per model
            - Total evaluations: 36
            """)

    except FileNotFoundError:
        st.error("❌ Results file not found! Please ensure models have been trained.")
    except Exception as e:
        st.error(f"❌ Error loading results: {e}")

with tab3:
    st.header("⚙️ Configuration")
    col1, col2 = st.columns(2)

    model_choice = col1.selectbox(
        "Select Model:",
        [
            "Logistic Regression",
            "Decision Tree",
            "kNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ],
        help="Choose which model to use for predictions"
    )

    input_method = col2.segmented_control(
        "How would you like to provide data?",
        ["📁 Upload CSV File", "✏️ Manual Input"],
        default="📁 Upload CSV File"
    )

    st.divider()

    if input_method == "📁 Upload CSV File":
        st.header("📁 Upload Test Dataset")

        st.info("""
        Upload a CSV file containing customer data in the same format as the training data.
        The file should include all required features (without customerID and Churn columns).
        """)

        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                st.success(f"✅ File uploaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")

                with st.expander("📋 View Data Preview"):
                    st.dataframe(df.head(10), use_container_width=True)

                st.divider()

                with st.spinner("Preprocessing data..."):
                    has_labels = 'Churn' in df.columns
                    if has_labels:
                        y_true = df['Churn'].map({'Yes': 1, 'No': 0})

                    df_processed = preprocess_data(df.copy())

                    if 'Churn' in df_processed.columns:
                        X = df_processed.drop('Churn', axis=1)
                    else:
                        X = df_processed

                    feature_names = load_feature_names()
                    for col in feature_names:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[feature_names]

                model = load_model(model_choice)

                with st.spinner(f"Making predictions with {model_choice}..."):
                    if model_choice in ["Logistic Regression", "kNN"]:
                        scaler = load_scaler()
                        X_scaled = scaler.transform(X)
                        predictions = model.predict(X_scaled)
                        predictions_proba = model.predict_proba(X_scaled)
                    else:
                        predictions = model.predict(X)
                        predictions_proba = model.predict_proba(X)

                st.header("🎯 Prediction Results")

                col1, col2 = st.columns(2)

                churn_count = np.sum(predictions == 1)
                no_churn_count = np.sum(predictions == 0)

                with col1:
                    st.metric(
                        "Predicted Churners",
                        churn_count,
                        delta=f"{churn_count/len(predictions)*100:.1f}%",
                        delta_color="inverse",
                        border=True
                    )

                with col2:
                    st.metric(
                        "Predicted Non-Churners",
                        no_churn_count,
                        delta=f"{no_churn_count/len(predictions)*100:.1f}%",
                        border=True
                    )

                st.divider()

                st.subheader("📊 Detailed Predictions")

                results_df = pd.DataFrame({
                    'Customer Index': range(len(predictions)),
                    'Prediction': ['Will Churn' if p == 1 else 'Will Not Churn' for p in predictions],
                    'Churn Probability': [f"{p[1]:.2%}" for p in predictions_proba],
                    'Confidence': [f"{max(p):.2%}" for p in predictions_proba]
                })

                st.dataframe(results_df, use_container_width=True)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{model_choice.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )

                if has_labels:
                    st.divider()
                    st.header("📈 Model Evaluation")

                    st.info("True labels detected in uploaded data. Showing performance metrics.")

                    accuracy = accuracy_score(y_true, predictions)
                    precision = precision_score(y_true, predictions, zero_division=0)
                    recall = recall_score(y_true, predictions, zero_division=0)
                    f1 = f1_score(y_true, predictions, zero_division=0)

                    try:
                        auc = roc_auc_score(y_true, predictions_proba[:, 1])
                    except:
                        auc = None

                    try:
                        mcc = matthews_corrcoef(y_true, predictions)
                    except:
                        mcc = None

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2%}", border=True)
                        st.metric("Precision", f"{precision:.4f}", border=True)

                    with col2:
                        st.metric("Recall", f"{recall:.4f}", border=True)
                        st.metric("F1 Score", f"{f1:.4f}", border=True)

                    with col3:
                        if auc is not None:
                            st.metric("AUC", f"{auc:.4f}", border=True)
                        if mcc is not None:
                            st.metric("MCC", f"{mcc:.4f}", border=True)

                    st.divider()

                    st.subheader("📊 Confusion Matrix")

                    cm = confusion_matrix(y_true, predictions)

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['No Churn', 'Churn'],
                            yticklabels=['No Churn', 'Churn'],
                            ax=ax
                        )
                        ax.set_ylabel('True Label')
                        ax.set_xlabel('Predicted Label')
                        ax.set_title(f'Confusion Matrix - {model_choice}')
                        st.pyplot(fig)

                    with col2:
                        st.markdown("### Classification Report")
                        report = classification_report(
                            y_true, predictions,
                            target_names=['No Churn', 'Churn'],
                            output_dict=True
                        )
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

        else:
            st.info("👆 Upload a CSV file to get started")

    else:
        st.header("✏️ Manual Input")

        st.info("Enter customer information below to predict churn probability for a single customer.")

        with st.form("manual_input_form"):
            st.subheader("Demographics")

            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

            with col2:
                partner = st.selectbox("Has Partner", ["No", "Yes"])
                dependents = st.selectbox("Has Dependents", ["No", "Yes"])

            st.divider()
            st.subheader("Account Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

            with col2:
                monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 50.0, 0.5)
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

            with col3:
                total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, float(monthly_charges * tenure), 1.0)
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ])

            st.divider()
            st.subheader("Services")

            col1, col2, col3 = st.columns(3)

            with col1:
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])

            with col2:
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

            with col3:
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

            st.divider()

            submit_button = st.form_submit_button("🎯 Predict Churn", use_container_width=True)

        if submit_button:
            input_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }

            df = pd.DataFrame([input_data])

            df_processed = preprocess_data(df.copy())

            feature_names = load_feature_names()
            for col in feature_names:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            X = df_processed[feature_names]

            model = load_model(model_choice)

            if model_choice in ["Logistic Regression", "kNN"]:
                scaler = load_scaler()
                X_scaled = scaler.transform(X)
                prediction = model.predict(X_scaled)[0]
                prediction_proba = model.predict_proba(X_scaled)[0]
            else:
                prediction = model.predict(X)[0]
                prediction_proba = model.predict_proba(X)[0]

            st.divider()
            st.header("🎯 Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                if prediction == 1:
                    st.error("### ⚠️ WILL CHURN")
                else:
                    st.success("### ✅ WILL NOT CHURN")

            with col2:
                st.metric("Churn Probability", f"{prediction_proba[1]:.2%}", border=True)

            with col3:
                st.metric("Confidence", f"{max(prediction_proba):.2%}", border=True)

            st.divider()
            st.subheader("📊 Probability Distribution")

            fig, ax = plt.subplots(figsize=(10, 4))
            categories = ['Will Not Churn', 'Will Churn']
            probabilities = [prediction_proba[0], prediction_proba[1]]
            colors = ['green', 'red']

            bars = ax.barh(categories, probabilities, color=colors, alpha=0.7)
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)

            for bar, prob in zip(bars, probabilities):
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{prob:.2%}', ha='left', va='center', fontweight='bold')

            st.pyplot(fig)