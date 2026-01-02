import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)


class ModelTraining:

    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Dataset path
        self.data_path = os.path.join(
            self.project_root,
            "artifacts",
            "raw",
            "data.csv"
        )

        # Model save directory
        self.model_dir = os.path.join(self.project_root, "artifacts", "models")
        os.makedirs(self.model_dir, exist_ok=True)

        logger.info("ModelTraining initialized")
        logger.info(f"Dataset path: {self.data_path}")

    # ---------------------------------------------------------
    # LOAD + PREPARE DATA
    # ---------------------------------------------------------
    def load_and_prepare_data(self):
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Dataset not found at {self.data_path}")

            df = pd.read_csv(self.data_path)
            logger.info("Dataset loaded successfully")

            # -------- CREATE REALISTIC TARGET --------
            def survival_logic(row):
                score = 0
                if row["Cancer_Stage"] == "Localized":
                    score += 5
                elif row["Cancer_Stage"] == "Regional":
                    score += 3
                elif row["Cancer_Stage"] == "Metastatic":
                    score -= 5

                if row["Tumor_Size_mm"] < 30:
                    score += 2
                if row["Tumor_Size_mm"] > 60:
                    score -= 2

                if row["Age"] < 50:
                    score += 1
                if row["Age"] > 80:
                    score -= 2

                score += np.random.randint(-2, 3)
                return "Yes" if score > 0 else "No"

            df["Real_Survival"] = df.apply(survival_logic, axis=1)
            logger.info("Target variable created")

            # -------- FEATURES & TARGET --------
            X = df.drop(
                columns=[
                    "Patient_ID",
                    "Survival_Prediction",
                    "Survival_5_years",
                    "Mortality",
                    "Real_Survival"
                ],
                errors="ignore"
            )

            y = df["Real_Survival"]

            # Encode target
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y)
            logger.info(f"Target classes: {list(y_encoder.classes_)}")

            # Encode categorical features
            for col in X.select_dtypes(include=["object"]).columns:
                X[col] = LabelEncoder().fit_transform(X[col])

            # Convert numeric features to float32 to reduce model size
            X = X.astype(np.float32)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            logger.info("Train-test split completed")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error("Error during data preparation", exc_info=True)
            raise CustomException(e)

    # ---------------------------------------------------------
    # TRAIN MODEL + METRICS + SAVE
    # ---------------------------------------------------------
    def train_model(self):
        try:
            logger.info("Model training started")

            X_train, X_test, y_train, y_test = self.load_and_prepare_data()

            # ------------------------
            # Random Forest Model
            # ------------------------
            model = RandomForestClassifier(
                n_estimators=150,  # reduced from 200
                max_depth=10,      # reduced from 12
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            logger.info("Model training completed")

            # ------------------------
            # Predictions + Metrics
            # ------------------------
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_proba)

            logger.info("===== MODEL METRICS =====")
            logger.info(f"Accuracy  : {acc:.4f}")
            logger.info(f"Precision : {prec:.4f}")
            logger.info(f"Recall    : {rec:.4f}")
            logger.info(f"F1 Score  : {f1:.4f}")
            logger.info(f"ROC-AUC   : {roc:.4f}")

            # ------------------------
            # Save model with compression
            # ------------------------
            model_path = os.path.join(self.model_dir, "random_forest_model.pkl")
            joblib.dump(model, model_path, compress=3)  # compress=1-9, 3 is a good tradeoff
            logger.info(f"Model saved successfully at: {model_path}")
            logger.info("Training pipeline completed successfully")

        except Exception as e:
            logger.error("Error during model training", exc_info=True)
            raise CustomException(e)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.train_model()
