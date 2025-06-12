import pandas as pd
import numpy as np
import logging
import yaml
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import optuna
import joblib
import json
import torch
import sys
import os
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
)

# It's good practice to ensure the project's root is in the Python path.
# This makes imports from other modules (like data, features) more reliable.
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_loader import DataLoader
    from features.feature_pipeline import FeaturePipeline
    from models.ensemble import SupplyChainRiskEnsemble
    from training.evaluator import ModelEvaluator
    from models.lstm_model import LSTMConfig
except ImportError as e:
    print(f"Error: Failed to import project modules. Ensure paths are correct and dependencies are installed. Details: {e}")
    sys.exit(1)

# Setup a logger for consistent, informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles the end-to-end model training pipeline, including data preparation,
    hyperparameter optimization, final model training, and evaluation.
    """
    def __init__(self, config_path: str):
        """
        Initializes the trainer with configuration and sets up necessary components.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        try:
            logger.info(f"Loading configuration from: {config_path}")
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self._validate_config()
        except FileNotFoundError:
            logger.error(f"Configuration file not found at: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing essential key in configuration: {e}")
            raise

        self.data_loader = DataLoader(self.config['data'])
        self.feature_pipeline = FeaturePipeline(self.config)
        self.evaluator = ModelEvaluator(self.config['evaluation'])
        
        # MLflow setup with error handling
        try:
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            mlflow.set_experiment(self.config['mlflow']['experiment_name'])
            logger.info(f"MLflow tracking URI set to '{self.config['mlflow']['tracking_uri']}'")
            logger.info(f"MLflow experiment set to '{self.config['mlflow']['experiment_name']}'")
        except Exception as e:
            logger.error(f"Failed to configure MLflow: {e}")
            raise

        # Create artifact and log directories safely
        try:
            self.model_dir = Path("models/artifacts")
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir = Path("logs")
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create necessary directories: {e}")
            raise
    
    def _validate_config(self):
        """Checks for the presence of essential keys in the config file."""
        required_keys = ['data', 'features', 'training', 'evaluation', 'mlflow']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Configuration missing required section: '{key}'")
        if 'tracking_uri' not in self.config['mlflow'] or 'experiment_name' not in self.config['mlflow']:
            raise KeyError("MLflow configuration missing 'tracking_uri' or 'experiment_name'")


    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Loads raw data and prepares initial features and labels."""
        try:
            logger.info("Loading training data...")
            raw_data = self.data_loader.load_training_data()
            
            if raw_data is None or raw_data.empty:
                raise ValueError("No training data loaded. The source might be empty or inaccessible.")
            
            logger.info("Preparing initial features and labels from raw data.")
            X, y, metadata = self.prepare_features_and_labels(raw_data)
            
            logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} initial features.")
            logger.info(f"Target distribution: \n{y.value_counts(normalize=True)}")
            
            return X, y, metadata
        except Exception as e:
            logger.error(f"Failed during data loading and preparation: {e}")
            raise

    def prepare_features_and_labels(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Extracts features, labels, and metadata from the raw dataframe."""
        feature_columns = self.data_loader.get_feature_columns(self.config['features'])
        available_features = [col for col in feature_columns if col in raw_data.columns]
        
        if not available_features:
            logger.warning("No configured features found in data. Falling back to all numeric columns.")
            available_features = raw_data.select_dtypes(include=np.number).columns.tolist()
            available_features = [col for col in available_features if not col.endswith('_id') and col != 'target']
            if not available_features:
                raise ValueError("No suitable feature columns could be identified in the dataset.")
        
        X = raw_data[available_features].copy()
        y = self.data_loader.create_labels(raw_data)
        
        metadata = {
            'feature_columns': available_features,
            'n_samples': len(X),
            'n_features': len(available_features)
        }
        return X, y, metadata
    
    def train_with_hyperparameter_optimization(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Performs hyperparameter optimization using Optuna.

        Returns:
            Dict: The best hyperparameters found.
        """
        logger.info("Starting hyperparameter optimization with Optuna...")

        def objective(trial: optuna.trial.Trial) -> float:
            try:
                # --- WARNING ---
                # This hyperparameter search defines an ensemble with weights, but the training
                # loop below only trains and evaluates a single model at a time. More importantly,
                # it does not prepare or use the LSTM features. For a true ensemble search,
                # the logic should construct and train the full ensemble model here.
                # The current implementation will find good parameters for the tabular models
                # but they might not be optimal for the final LSTM-inclusive ensemble.
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }

                tscv = TimeSeriesSplit(n_splits=self.config['training']['cv_folds'])
                scores = []
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    # Using a nested try-except to ensure one failed fold doesn't stop the whole trial
                    try:
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        # Apply feature pipeline
                        X_train_processed = self.feature_pipeline.fit_transform(X_train, y_train)
                        X_val_processed = self.feature_pipeline.transform(X_val)
                        
                        # --- RECOMMENDATION ---
                        # For a classification task, use a classifier like XGBClassifier.
                        # The ensemble class should be adapted accordingly.
                        # from xgboost import XGBClassifier
                        # model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
                        # model.fit(X_train_processed, y_train)
                        # preds_proba = model.predict_proba(X_val_processed)[:, 1]

                        # This part needs to be adapted based on your actual ensemble model's API
                        model = SupplyChainRiskEnsemble(xgb_params=params) # Simplified for tuning
                        model.fit(X_train_processed, y_train) # Assumes a simplified fit
                        predictions = model.predict(X_val_processed) # Assumes a simplified predict
                        
                        opt_metric = self.config['training'].get('optimization_metric', 'auc')
                        score = roc_auc_score(y_val, predictions) if opt_metric == 'auc' else f1_score(y_val, (predictions > 0.5).astype(int))
                        scores.append(score)

                    except Exception as e:
                        logger.warning(f"Fold {fold+1} in Optuna trial {trial.number} failed: {e}")
                        scores.append(0.0) # Penalize failure
                
                return np.mean(scores)

            except Exception as e:
                logger.error(f"Optuna trial {trial.number} failed catastrophically: {e}")
                return 0.0 # Return a score that indicates failure

        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config['training']['n_trials'], timeout=self.config['training'].get('timeout_seconds'))
            
            logger.info(f"Hyperparameter optimization finished.")
            logger.info(f"Best score ({self.config['training']['optimization_metric']}): {study.best_value:.4f}")
            logger.info(f"Best hyperparameters: {study.best_params}")
            
            return study.best_params
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            raise

    def train_final_model(self, X: pd.DataFrame, y: pd.Series, best_params: Dict) -> Tuple[Any, Dict, Dict]:
        """Trains, evaluates, and saves the final model with the best hyperparameters."""
        logger.info("Training final model with best parameters...")
        
        with mlflow.start_run(run_name="final_model_training") as run:
            try:
                mlflow.log_params(best_params)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                logger.info("Fitting feature pipeline on training data...")
                X_train_processed = self.feature_pipeline.fit_transform(X_train.copy(), y_train)
                logger.info("Transforming test data with fitted pipeline...")
                X_test_processed = self.feature_pipeline.transform(X_test.copy())
                
                logger.info("Preparing LSTM features...")
                X_train_lstm = self._prepare_lstm_features(X_train_processed)
                X_test_lstm = self._prepare_lstm_features(X_test_processed)
                
                model_config = self._convert_params_to_config(best_params)
                model_config['lstm_config'].input_size = X_train_lstm.shape[2] # Dynamically set input size

                # --- RECOMMENDATION ---
                # Ensure the SupplyChainRiskEnsemble internally uses RandomForestClassifier, not Regressor.
                model = SupplyChainRiskEnsemble(
                    lstm_config=model_config['lstm_config'],
                    rf_params=model_config['rf_params'],
                    xgb_params=model_config['xgb_params']
                )
                
                logger.info("Training the final ensemble model...")
                model.train(
                    X_lstm=torch.FloatTensor(X_train_lstm),
                    X_tabular=X_train_processed.values,
                    y=y_train.values
                )
                
                logger.info("Evaluating model on the test set...")
                test_predictions = model.predict(
                    X_lstm=torch.FloatTensor(X_test_lstm),
                    X_tabular=X_test_processed.values
                )
                test_metrics = self._calculate_metrics(y_test, test_predictions)
                logger.info(f"Test Set Metrics: {test_metrics}")
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

                logger.info("Saving model artifacts...")
                run_id = run.info.run_id
                model_path = self.model_dir / f"run_{run_id}"
                model_path.mkdir(exist_ok=True, parents=True)

                model.save(str(model_path / "ensemble_model"))
                self.feature_pipeline.save_pipeline(str(model_path / "feature_pipeline.joblib"))
                
                metadata = {
                    'mlflow_run_id': run_id,
                    'training_date': datetime.now().isoformat(),
                    'n_train_samples': len(X_train),
                    'n_test_samples': len(X_test),
                    'feature_names_in': self.feature_pipeline.feature_names,
                    'feature_names_out': X_train_processed.columns.tolist(),
                    'hyperparameters': best_params,
                    'test_metrics': test_metrics
                }
                with open(model_path / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                mlflow.log_artifact(str(model_path), artifact_path="final_model")
                logger.info(f"Final model and artifacts saved and logged to MLflow run {run_id}.")

                return model, test_predictions, metadata

            except Exception as e:
                logger.error(f"Final model training failed: {e}\n{traceback.format_exc()}")
                mlflow.end_run(status="FAILED")
                raise

    def run_full_training_pipeline(self) -> Dict:
        """Executes the complete training pipeline from data loading to evaluation."""
        logger.info("Starting full training pipeline.")
        try:
            # Step 1: Load and prepare initial data
            X, y, metadata = self.load_and_prepare_data()

            # Step 2: Hyper-parameter search
            best_params = self.train_with_hyperparameter_optimization(X, y)

            # Step 3: Train the final model on split data (train/test)
            # This step also saves the model and pipeline fitted on the training set.
            model, test_predictions, model_metadata = self.train_final_model(
                X, y, best_params
            )

            # The feature pipeline is now fitted. We use it to transform the *entire* dataset
            # for a final, comprehensive evaluation.
            logger.info("Applying final feature pipeline to the entire dataset for evaluation.")
            X_processed = self.feature_pipeline.transform(X)

            # Step 4: Prepare the full dataset into the format required by the model's predict method
            X_lstm_full = self._prepare_lstm_features(X_processed)
            X_tabular_full = X_processed.values

            # Step 5: Evaluate the final model on the full, processed dataset
            # NOTE: This requires updating self.evaluator.evaluate_model to accept
            # X_lstm and X_tabular arguments instead of a single DataFrame.
            logger.info("Evaluating final model on the full processed dataset.")
            evaluation_results = self.evaluator.evaluate_model(
                model=model,
                X_lstm=torch.FloatTensor(X_lstm_full),
                X_tabular=X_tabular_full,
                y=y,
                metadata=metadata
            )
            logger.info(f"Full Dataset Evaluation Metrics: {evaluation_results.get('metrics')}")

            logger.info("Training pipeline completed successfully!")

            return {
                "model": model,
                "test_set_predictions": test_predictions,
                "metadata": model_metadata,
                "full_dataset_evaluation": evaluation_results,
                "best_params": best_params,
            }

        except Exception as e:
            logger.error(f"The training pipeline failed catastrophically.")
            logger.error(f"ERROR: {e}\n{traceback.format_exc()}")
            raise
            
    def _prepare_lstm_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepares time-series features for the LSTM model."""
        # This is a placeholder. In a real scenario, this would involve
        # reshaping data based on time steps, e.g., using a sliding window.
        # The number of features must be consistent.
        if X.empty:
            return np.array([])
        n_samples = len(X)
        # Assuming all columns in X_processed are used for the time series.
        n_features = X.shape[1]
        seq_length = 1 # Mocking a sequence length of 1
        
        # Reshape to (n_samples, seq_length, n_features)
        return X.values.reshape(n_samples, seq_length, n_features)

    def _convert_params_to_config(self, params: Dict) -> Dict:
        """Converts Optuna parameters to a structured model configuration."""
        # This function should be aligned with the parameters searched in the 'objective' function.
        lstm_config = LSTMConfig(
            input_size=10,  # This will be updated dynamically later
            hidden_size=params.get('hidden_size', 128),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.3),
            bidirectional=True,
            batch_first=True
        )
        
        # Use a single source of truth for hyperparameters
        xgb_params = {
            'n_estimators': params.get('n_estimators', 500),
            'max_depth': params.get('max_depth', 5),
            'learning_rate': params.get('learning_rate', 0.05),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'random_state': 42
        }

        # Placeholder for RF, as it wasn't in the provided Optuna search
        rf_params = {
            'n_estimators': params.get('rf_n_estimators', 100),
            'max_depth': params.get('rf_max_depth', 10),
            'random_state': 42
        }
        
        return {'lstm_config': lstm_config, 'rf_params': rf_params, 'xgb_params': xgb_params}
    
    def _calculate_metrics(self, y_true: pd.Series, predictions: np.ndarray) -> Dict:
        """Calculates a dictionary of evaluation metrics."""
        y_pred_binary = (predictions > 0.5).astype(int)
        metrics = {}
        
        try:
            metrics['auc'] = roc_auc_score(y_true, predictions)
        except ValueError as e:
            metrics['auc'] = 0.0
            logger.warning(f"Could not calculate AUC: {e}")
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred_binary, zero_division=0)
        
        return metrics


if __name__ == "__main__":
    logger.info("Starting model training script execution.")
    # Assuming the config file is in a 'config' directory relative to the script's location
    config_path = "config/config.yaml"
    
    if not os.path.exists(config_path):
        logger.error(f"CRITICAL: Main config file not found at '{config_path}'. Exiting.")
        sys.exit(1)

    try:
        trainer = ModelTrainer(config_path)
        results = trainer.run_full_training_pipeline()
        logger.info("Main training script finished successfully.")
        logger.info(f"Final test metrics: {results['metadata']['test_metrics']}")
    except Exception as e:
        logger.error("An unhandled exception occurred during the training pipeline execution.")
        logger.error(f"Error details: {e}\n{traceback.format_exc()}")
        sys.exit(1)