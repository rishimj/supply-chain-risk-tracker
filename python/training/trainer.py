import pandas as pd
import numpy as np
import logging
import yaml
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import optuna
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import torch

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from features.feature_pipeline import FeaturePipeline
from models.ensemble import SupplyChainRiskEnsemble
from training.evaluator import ModelEvaluator
from models.lstm_model import LSTMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = DataLoader(self.config['data'])
        self.feature_pipeline = FeaturePipeline(self.config)
        self.evaluator = ModelEvaluator(self.config['evaluation'])
        
        # MLflow setup
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        # Create directories
        self.model_dir = Path("models/artifacts")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Load and prepare training data"""
        logger.info("Loading training data...")
        
        # Load raw data
        raw_data = self.data_loader.load_training_data()
        
        if raw_data.empty:
            raise ValueError("No training data loaded")
        
        # Create features and labels
        X, y, metadata = self.prepare_features_and_labels(raw_data)
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, metadata
    
    def prepare_features_and_labels(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Prepare features and labels from raw data"""
        
        # Extract feature columns based on configuration
        feature_columns = self.data_loader.get_feature_columns(self.config['features'])
        
        # Select available feature columns
        available_features = [col for col in feature_columns if col in raw_data.columns]
        
        if not available_features:
            logger.warning("No configured features found in data. Using all numeric columns.")
            available_features = raw_data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove ID columns
            available_features = [col for col in available_features if not col.endswith('_id')]
        
        X = raw_data[available_features].copy()
        
        # Create labels
        y = self.data_loader.create_labels(raw_data)
        
        # Create metadata
        metadata = {
            'company_ids': raw_data.get('id', range(len(raw_data))).tolist(),
            'symbols': raw_data.get('symbol', ['UNK'] * len(raw_data)).tolist(),
            'sectors': raw_data.get('sector', ['Unknown'] * len(raw_data)).tolist(),
            'feature_columns': available_features,
            'n_samples': len(X),
            'n_features': len(available_features)
        }
        
        return X, y, metadata
    
    def train_with_hyperparameter_optimization(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train model with hyperparameter optimization using Optuna"""
        logger.info("Starting hyperparameter optimization...")
        
        def objective(trial):
            # Sample hyperparameters
            params = {
                'ensemble_weights': {
                    'xgboost': trial.suggest_float('xgb_weight', 0.2, 0.6),
                    'lstm': trial.suggest_float('lstm_weight', 0.2, 0.5),
                    'gnn': trial.suggest_float('gnn_weight', 0.1, 0.4)
                },
                'xgboost': {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                },
                'lstm': {
                    'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
                    'num_layers': trial.suggest_int('num_layers', 1, 4),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('lstm_lr', 0.0001, 0.01, log=True)
                },
                'gnn': {
                    'hidden_dim': trial.suggest_categorical('gnn_hidden', [32, 64, 128]),
                    'num_layers': trial.suggest_int('gnn_layers', 2, 5),
                    'dropout': trial.suggest_float('gnn_dropout', 0.1, 0.4)
                }
            }
            
            # Normalize ensemble weights
            total_weight = sum(params['ensemble_weights'].values())
            params['ensemble_weights'] = {k: v/total_weight for k, v in params['ensemble_weights'].items()}
            
            # Cross-validation with time series split
            tscv = TimeSeriesSplit(n_splits=self.config['training']['cv_folds'])
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                try:
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Apply feature pipeline
                    X_train_processed = self.feature_pipeline.fit_transform(X_train, y_train)
                    X_val_processed = self.feature_pipeline.transform(X_val)
                    
                    # Train model
                    model = SupplyChainRiskEnsemble(params)
                    model.fit(X_train_processed, y_train, X_val_processed, y_val)
                    
                    # Evaluate
                    predictions = model.predict(X_val_processed)
                    
                    # Calculate score based on optimization metric
                    opt_metric = self.config['training']['optimization_metric']
                    if opt_metric == 'auc':
                        score = roc_auc_score(y_val, predictions['guidance_miss_probability'])
                    elif opt_metric == 'f1':
                        binary_preds = (predictions['guidance_miss_probability'] > 0.5).astype(int)
                        score = f1_score(y_val, binary_preds)
                    else:
                        score = accuracy_score(y_val, (predictions['guidance_miss_probability'] > 0.5).astype(int))
                    
                    scores.append(score)
                    
                except Exception as e:
                    logger.warning(f"Fold {fold} failed: {e}")
                    scores.append(0.0)
            
            return np.mean(scores) if scores else 0.0
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['training']['n_trials'])
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series, best_params: Dict) -> Tuple[Any, Dict, Dict]:
        """Train final model with best hyperparameters"""
        logger.info("Training final model with best parameters...")
        
        with mlflow.start_run(run_name="final_model"):
            # Log hyperparameters
            for key, value in best_params.items():
                mlflow.log_param(key, value)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.25, random_state=42
            )
            
            logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Apply feature pipeline
            X_train_processed = self.feature_pipeline.fit_transform(X_train, y_train)
            X_val_processed = self.feature_pipeline.transform(X_val)
            X_test_processed = self.feature_pipeline.transform(X_test)
            
            # Convert best_params to model config format
            model_config = self._convert_params_to_config(best_params)
            
            # Create time series features for LSTM
            X_train_lstm = self._prepare_lstm_features(X_train_processed)
            X_val_lstm = self._prepare_lstm_features(X_val_processed)
            X_test_lstm = self._prepare_lstm_features(X_test_processed)
            
            # Update LSTM input size based on actual features
            model_config['lstm_config'].input_size = X_train_lstm.shape[2]
            
            # Train model
            model = SupplyChainRiskEnsemble(
                lstm_config=model_config['lstm_config'],
                rf_params=model_config['rf_params'],
                xgb_params=model_config['xgb_params']
            )
            
            # Train the model
            metrics = model.train(
                X_lstm=torch.FloatTensor(X_train_lstm),
                X_tabular=X_train_processed.values,
                y=y_train.values
            )
            
            # Log training metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            
            # Evaluate on validation set
            val_predictions = model.predict(
                X_lstm=torch.FloatTensor(X_val_lstm),
                X_tabular=X_val_processed.values
            )
            val_metrics = self._calculate_metrics(y_val, val_predictions)
            
            # Evaluate on test set
            test_predictions = model.predict(
                X_lstm=torch.FloatTensor(X_test_lstm),
                X_tabular=X_test_processed.values
            )
            test_metrics = self._calculate_metrics(y_test, test_predictions)
            
            # Log metrics
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value)
            
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # Feature importance
            feature_importance = self.feature_pipeline.get_feature_importance()
            if feature_importance:
                # Log top 10 most important features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (feature, importance) in enumerate(sorted_features):
                    mlflow.log_metric(f"feature_importance_{i+1}_{feature}", importance)
            
            # Save model artifacts
            model_path = self.model_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path.mkdir(exist_ok=True, parents=True)
            
            # Save ensemble model
            model.save(str(model_path / "ensemble_model"))
            
            # Save feature pipeline
            pipeline_path = model_path / "feature_pipeline.joblib"
            self.feature_pipeline.save_pipeline(str(pipeline_path))
            
            # Save model metadata
            metadata = {
                'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'training_date': datetime.now().isoformat(),
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
                'n_test_samples': len(X_test),
                'feature_names': self.feature_pipeline.feature_names,
                'hyperparameters': best_params,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            }
            
            metadata_path = model_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model training completed.")
            
            return model, test_predictions, metadata
            
    def _prepare_lstm_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare time series features for LSTM model"""
        # In a real implementation, this would reshape the data properly
        # For now, create a simple mock time series
        n_samples = len(X)
        seq_length = 30  # 30 time steps
        n_features = 10  # 10 features per time step
        
        # Create mock time series data
        return np.random.randn(n_samples, seq_length, n_features)
    
    def _convert_params_to_config(self, params: Dict) -> Dict:
        """Convert Optuna parameters to model configuration format"""
        # Extract weights
        weights = {
            'xgb': params.get('xgb_weight', 0.4),
            'rf': params.get('rf_weight', 0.3),
            'lstm': params.get('lstm_weight', 0.3)
        }
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v/weight_sum for k, v in weights.items()}
            
        # Create LSTM config
        lstm_config = LSTMConfig(
            input_size=params.get('input_size', 10),  # Default to 10 features
            hidden_size=params.get('hidden_size', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            bidirectional=params.get('bidirectional', True),
            batch_first=True
        )
        
        # Create RF params
        rf_params = {
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', 10),
            'min_samples_split': params.get('min_samples_split', 5),
            'min_samples_leaf': params.get('min_samples_leaf', 2),
            'random_state': 42
        }
        
        # Create XGBoost params
        xgb_params = {
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', 6),
            'learning_rate': params.get('learning_rate', 0.1),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'random_state': 42
        }
        
        return {
            'lstm_config': lstm_config,
            'rf_params': rf_params,
            'xgb_params': xgb_params,
            'weights': weights
        }
    
    def _calculate_metrics(self, y_true: pd.Series, predictions: np.ndarray) -> Dict:
        """Calculate evaluation metrics"""
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, precision_score, 
            recall_score, f1_score, precision_recall_curve
        )
        
        # Convert predictions to binary
        y_pred_binary = (predictions > 0.5).astype(int)
        
        metrics = {}
        
        try:
            metrics['auc'] = roc_auc_score(y_true, predictions)
        except:
            metrics['auc'] = 0.0
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred_binary, zero_division=0)
        metrics['mse'] = np.mean((predictions - y_true) ** 2)
        
        # Additional metrics
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, predictions)
            metrics['avg_precision'] = np.mean(precision_curve)
        except:
            metrics['avg_precision'] = 0.0
        
        return metrics
    
    def run_full_training_pipeline(self) -> Dict:
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline...")
        
        try:
            # Load and prepare data
            X, y, metadata = self.load_and_prepare_data()
            
            # Hyperparameter optimization
            best_params = self.train_with_hyperparameter_optimization(X, y)
            
            # Train final model
            model, predictions, model_metadata = self.train_final_model(X, y, best_params)
            
            # Comprehensive evaluation
            evaluation_results = self.evaluator.evaluate_model(model, X, y, metadata)
            
            logger.info("Training pipeline completed successfully!")
            
            return {
                'model': model,
                'predictions': predictions,
                'metadata': model_metadata,
                'evaluation': evaluation_results,
                'best_params': best_params
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def retrain_model(self, model_path: str = None) -> Dict:
        """Retrain existing model with new data"""
        logger.info("Retraining model...")
        
        # Load existing model if provided
        if model_path:
            # Load previous best parameters
            metadata_path = Path(model_path) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    import json
                    previous_metadata = json.load(f)
                    best_params = previous_metadata.get('hyperparameters', {})
            else:
                best_params = None
        else:
            best_params = None
        
        # Load fresh data
        X, y, metadata = self.load_and_prepare_data()
        
        # Use previous hyperparameters or optimize new ones
        if best_params is None:
            best_params = self.train_with_hyperparameter_optimization(X, y)
        
        # Train model
        model, predictions, model_metadata = self.train_final_model(X, y, best_params)
        
        return {
            'model': model,
            'predictions': predictions,
            'metadata': model_metadata,
            'best_params': best_params
        }


if __name__ == "__main__":
    # Example usage
    config_path = "python/config/config.yaml"
    trainer = ModelTrainer(config_path)
    
    # Run full training pipeline
    results = trainer.run_full_training_pipeline()
    
    print("Training completed!")
    print(f"Test metrics: {results['metadata']['test_metrics']}") 