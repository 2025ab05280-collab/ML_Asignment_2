#!/usr/bin/env python3

import pickle
from typing import Dict, List, Tuple, Any
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class Config:
    """Configuration class for training pipeline"""

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = Path(__file__).parent / 'models'

    # Dataset
    DATASET_FILE = 'dataset.csv'
    TARGET_COLUMN = 'Churn'

    # Preprocessing
    BINARY_COLUMNS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    CATEGORICAL_COLUMNS = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]

    # Training
    TEST_SIZE = 0.3
    RANDOM_STATE = 42

    # Models requiring scaling
    MODELS_REQUIRING_SCALING = ['Logistic Regression', 'kNN']

    # Metrics to calculate
    METRICS = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

class DataLoader:
    """Handles data loading and preprocessing"""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV file"""
        data_path = self.config.DATA_DIR / self.config.DATASET_FILE

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        df = pd.read_csv(data_path)
        print(f"   ✓ Dataset loaded: {df.shape[0]} instances, {df.shape[1]} columns")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset"""
        print("\n[2/6] Preprocessing data...")

        # Remove customerID
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            print("   ✓ Dropped customerID column")

        # Convert TotalCharges to numeric
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(0, inplace=True)
            print("   ✓ Converted TotalCharges to numeric")

        # Convert binary columns
        for col in self.config.BINARY_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
        print(f"   ✓ Converted {len(self.config.BINARY_COLUMNS)} binary columns to numeric")

        # Convert target variable
        if self.config.TARGET_COLUMN in df.columns:
            df[self.config.TARGET_COLUMN] = df[self.config.TARGET_COLUMN].map({'Yes': 1, 'No': 0})
            print(f"   ✓ Converted target variable ({self.config.TARGET_COLUMN}) to numeric")

        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(
            df,
            columns=self.config.CATEGORICAL_COLUMNS,
            drop_first=True
        )
        print(f"   ✓ One-hot encoded {len(self.config.CATEGORICAL_COLUMNS)} categorical variables")
        print(f"   ✓ Final shape after encoding: {df_encoded.shape}")

        return df_encoded

    def prepare_train_test_split(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """Split data into train and test sets with scaling"""
        print("\n[3/6] Preparing data for training...")

        # Separate features and target
        X = df.drop(self.config.TARGET_COLUMN, axis=1)
        y = df[self.config.TARGET_COLUMN]

        print(f"   ✓ Feature matrix X: {X.shape}")
        print(f"   ✓ Target vector y: {y.shape}")
        print(f"   ✓ Class distribution: {dict(y.value_counts())}")

        class_balance = y.value_counts()
        print(f"   ✓ Class balance: No={class_balance[0]} ({class_balance[0]/len(y)*100:.1f}%), "
              f"Yes={class_balance[1]} ({class_balance[1]/len(y)*100:.1f}%)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )

        print(f"   ✓ Training set: {X_train.shape}")
        print(f"   ✓ Test set: {X_test.shape}")

        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save artifacts
        self._save_artifacts(X.columns)

        return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

    def _save_artifacts(self, feature_names: pd.Index):
        """Save scaler and feature names"""
        # Save scaler
        scaler_path = self.config.MODEL_DIR / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print("   ✓ Features scaled and scaler saved")

        # Save feature names
        features_path = self.config.MODEL_DIR / 'feature_names.pkl'
        with open(features_path, 'wb') as f:
            pickle.dump(list(feature_names), f)
        print("   ✓ Feature names saved")

class ModelManager:
    """Manages model initialization and configuration"""

    @staticmethod
    def get_models() -> Dict[str, Any]:
        """Initialize and return all models"""
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=Config.RANDOM_STATE
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=Config.RANDOM_STATE,
                max_depth=10
            ),
            'kNN': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=Config.RANDOM_STATE,
                eval_metric='logloss',
                n_jobs=-1
            )
        }
        return models

class ModelEvaluator:
    """Evaluates model performance"""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_pred_proba),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        return {k: round(v, 4) for k, v in metrics.items()}

    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Get detailed classification report"""
        return classification_report(
            y_true, y_pred,
            target_names=['No Churn', 'Churn'],
            output_dict=True
        )


class ModelTrainer:
    """Handles model training and evaluation pipeline"""

    def __init__(self, config: Config):
        self.config = config
        self.evaluator = ModelEvaluator()
        self.results = []

    def train_and_evaluate(
        self,
        models: Dict[str, Any],
        X_train: np.ndarray,
        X_test: np.ndarray,
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> List[Dict[str, Any]]:
        """Train and evaluate all models"""
        print("\n[5/6] Training and evaluating models...")
        print("-" * 70)

        for model_name, model in models.items():
            print(f"\n{model_name}:")
            print("   Training...")

            # Select appropriate data (scaled or unscaled)
            X_train_use, X_test_use = self._select_data(
                model_name, X_train, X_test, X_train_scaled, X_test_scaled
            )

            # Train model
            model.fit(X_train_use, y_train)

            # Make predictions
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]

            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)

            # Print metrics
            self._print_metrics(metrics)

            # Print confusion matrix
            cm = self.evaluator.get_confusion_matrix(y_test, y_pred)
            print(f"   ✓ Confusion Matrix:\n      {cm[0]}\n      {cm[1]}")

            # Save model
            self._save_model(model, model_name)

            # Store results
            result = {'Model': model_name, **metrics}
            self.results.append(result)

        print("\n" + "-" * 70)
        return self.results

    def _select_data(
        self,
        model_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select scaled or unscaled data based on model type"""
        if model_name in self.config.MODELS_REQUIRING_SCALING:
            return X_train_scaled, X_test_scaled
        return X_train, X_test

    def _print_metrics(self, metrics: Dict[str, float]):
        """Print evaluation metrics"""
        for metric_name, value in metrics.items():
            print(f"   ✓ {metric_name:12s} {value:.4f}")

    def _save_model(self, model: Any, model_name: str):
        """Save trained model to disk"""
        model_filename = f"{model_name.replace(' ', '_').lower()}.pkl"
        model_path = self.config.MODEL_DIR / model_filename

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✓ Model saved as {model_filename}")


class ResultsAnalyzer:
    """Analyzes and displays training results"""

    def __init__(self, config: Config):
        self.config = config

    def create_results_table(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create results comparison table"""
        print("\n[6/6] Creating results summary...")
        results_df = pd.DataFrame(results)

        print("\n" + "=" * 70)
        print("RESULTS COMPARISON TABLE")
        print("=" * 70)
        print(results_df.to_string(index=False))

        return results_df

    def save_results(self, results_df: pd.DataFrame):
        """Save results to CSV"""
        results_path = self.config.MODEL_DIR / 'model_comparison_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to {results_path.name}")

    def print_best_models(self, results_df: pd.DataFrame):
        """Print best model for each metric"""
        print("\n" + "=" * 70)
        print("BEST MODELS BY METRIC")
        print("=" * 70)

        for metric in self.config.METRICS:
            best_idx = results_df[metric].idxmax()
            best_model = results_df.loc[best_idx, 'Model']
            best_value = results_df.loc[best_idx, metric]
            print(f"{metric:12s}: {best_model:25s} ({best_value:.4f})")

class TrainingPipeline:
    """Main training pipeline orchestrator"""

    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader(self.config)
        self.model_manager = ModelManager()
        self.trainer = ModelTrainer(self.config)
        self.analyzer = ResultsAnalyzer(self.config)

    def run(self):
        """Execute the complete training pipeline"""

        # Step 1: Load data
        print("\n[1/6] Loading dataset...")
        df = self.data_loader.load_data()

        # Step 2-3: Preprocess and prepare data
        df_processed = self.data_loader.preprocess_data(df)
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = \
            self.data_loader.prepare_train_test_split(df_processed)

        # Step 4: Initialize models
        print("\n[4/6] Initializing models...")
        models = self.model_manager.get_models()
        print(f"   ✓ {len(models)} models initialized")

        # Step 5: Train and evaluate
        results = self.trainer.train_and_evaluate(
            models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
        )

        # Step 6: Analyze and save results
        results_df = self.analyzer.create_results_table(results)
        self.analyzer.save_results(results_df)
        self.analyzer.print_best_models(results_df)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
