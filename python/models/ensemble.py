import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import joblib
import logging
from pathlib import Path

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score

from .lstm_model import LSTMModel, LSTMConfig

logger = logging.getLogger(__name__)

class SupplyChainRiskEnsemble:
    """
    Ensemble model combining LSTM, Random Forest, and XGBoost for supply chain risk prediction.
    
    This is the core prediction model that estimates the probability of a company
    missing quarterly earnings guidance due to supply chain issues.
    """
    
    def __init__(self, 
                 lstm_config: LSTMConfig,
                 rf_params: Optional[Dict[str, Any]] = None,
                 xgb_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the ensemble model
        
        Args:
            lstm_config: Configuration for LSTM model
            rf_params: Parameters for Random Forest model
            xgb_params: Parameters for XGBoost model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize LSTM model
        self.lstm_model = LSTMModel(lstm_config)
        
        # Initialize Random Forest model
        self.rf_params = rf_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        self.rf_model = RandomForestRegressor(**self.rf_params)
        
        # Initialize XGBoost model
        self.xgb_params = xgb_params or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        
        # Model weights for ensemble
        self.weights = {
            'lstm': 0.4,
            'rf': 0.3,
            'xgb': 0.3
        }
        
        self.is_fitted = False
        self.feature_names = None
        self.feature_pipeline = None
        
    def train(self, 
              X_lstm: torch.Tensor,
              X_tabular: np.ndarray,
              y: np.ndarray) -> Dict[str, float]:
        """
        Train all models in the ensemble
        
        Args:
            X_lstm: Input tensor for LSTM model
            X_tabular: Input array for tree-based models
            y: Target values
            
        Returns:
            Dictionary containing training metrics for each model
        """
        metrics = {}
        
        # Train LSTM
        self.lstm_model.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters())
        
        # Convert targets to tensor
        y_tensor = torch.FloatTensor(y).to(self.device)
        X_lstm = X_lstm.to(self.device)
        
        # LSTM training loop
        for epoch in range(100):  # You might want to make this configurable
            optimizer.zero_grad()
            outputs = self.lstm_model(X_lstm)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
        
        # Calculate LSTM training metrics
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_preds = self.lstm_model(X_lstm).cpu().numpy().squeeze()
            lstm_mse = np.mean((lstm_preds - y) ** 2)
            metrics['lstm_mse'] = lstm_mse
        
        # Train Random Forest
        self.rf_model.fit(X_tabular, y)
        rf_preds = self.rf_model.predict(X_tabular)
        rf_mse = np.mean((rf_preds - y) ** 2)
        metrics['rf_mse'] = rf_mse
        
        # Train XGBoost
        self.xgb_model.fit(X_tabular, y)
        xgb_preds = self.xgb_model.predict(X_tabular)
        xgb_mse = np.mean((xgb_preds - y) ** 2)
        metrics['xgb_mse'] = xgb_mse
        
        self.is_fitted = True
        
        return metrics
    
    def predict(self, 
                X_lstm: torch.Tensor,
                X_tabular: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble
        
        Args:
            X_lstm: Input tensor for LSTM model
            X_tabular: Input array for tree-based models
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from each model
        self.lstm_model.eval()
        X_lstm = X_lstm.to(self.device)
        with torch.no_grad():
            lstm_preds = self.lstm_model(X_lstm).cpu().numpy().squeeze()
        
        rf_preds = self.rf_model.predict(X_tabular)
        xgb_preds = self.xgb_model.predict(X_tabular)
        
        # Combine predictions using weights
        ensemble_preds = (
            self.weights['lstm'] * lstm_preds +
            self.weights['rf'] * rf_preds +
            self.weights['xgb'] * xgb_preds
        )
        
        return ensemble_preds
    
    def save(self, path: str):
        """Save all models in the ensemble"""
        import joblib
        
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save LSTM model
        self.lstm_model.save(f"{path}_lstm.pt")
        
        # Save tree-based models
        joblib.dump({
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'weights': self.weights,
            'rf_params': self.rf_params,
            'xgb_params': self.xgb_params
        }, f"{path}_tree.pkl")
    
    @classmethod
    def load(cls, path: str, lstm_config: LSTMConfig) -> 'SupplyChainRiskEnsemble':
        """Load ensemble from saved state"""
        import joblib
        
        # Load LSTM model
        lstm_model = LSTMModel.load(f"{path}_lstm.pt")
        
        # Load tree-based models
        tree_data = joblib.load(f"{path}_tree.pkl")
        
        # Create new ensemble instance
        ensemble = cls(lstm_config, tree_data['rf_params'], tree_data['xgb_params'])
        ensemble.lstm_model = lstm_model
        ensemble.rf_model = tree_data['rf_model']
        ensemble.xgb_model = tree_data['xgb_model']
        ensemble.weights = tree_data['weights']
        ensemble.is_fitted = True
        
        return ensemble 