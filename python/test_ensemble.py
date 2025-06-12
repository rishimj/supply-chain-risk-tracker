#!/usr/bin/env python3
"""
Test script for the Supply Chain Risk Ensemble model
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMModel, LSTMConfig
from models.ensemble import SupplyChainRiskEnsemble
from data.data_loader import DataLoader
from features.feature_pipeline import FeaturePipeline


def generate_mock_data(n_samples=100, n_features=10, n_time_steps=30):
    """Generate mock data for testing the ensemble model"""
    
    # Generate tabular features
    X_tabular = np.random.randn(n_samples, n_features)
    
    # Generate time series features
    X_lstm = np.random.randn(n_samples, n_time_steps, n_features)
    
    # Generate target values (0-1 range)
    y = np.random.uniform(0, 1, n_samples)
    
    return X_tabular, X_lstm, y


def test_ensemble_training():
    """Test the ensemble model training process"""
    print("Testing ensemble model training...")
    
    # Generate mock data
    X_tabular, X_lstm, y = generate_mock_data()
    
    # Convert to torch tensor
    X_lstm_tensor = torch.FloatTensor(X_lstm)
    
    # Create LSTM config
    lstm_config = LSTMConfig(
        input_size=X_lstm.shape[2],
        hidden_size=32,
        num_layers=1,
        dropout=0.1
    )
    
    # Create ensemble model
    ensemble = SupplyChainRiskEnsemble(lstm_config=lstm_config)
    
    # Train the model
    metrics = ensemble.train(X_lstm_tensor, X_tabular, y)
    
    # Print metrics
    print("Training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return ensemble


def test_ensemble_prediction(ensemble):
    """Test the ensemble model prediction process"""
    print("\nTesting ensemble model prediction...")
    
    # Generate new mock data
    X_tabular, X_lstm, _ = generate_mock_data(n_samples=10)
    
    # Convert to torch tensor
    X_lstm_tensor = torch.FloatTensor(X_lstm)
    
    # Make predictions
    predictions = ensemble.predict(X_lstm_tensor, X_tabular)
    
    # Print predictions
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"First 5 predictions: {predictions[:5]}")
    
    return predictions


def test_ensemble_save_load(ensemble):
    """Test saving and loading the ensemble model"""
    print("\nTesting ensemble model save/load...")
    
    # Create a temporary directory for the model
    model_dir = Path("./temp_model")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "ensemble_model"
    
    # Save the model
    print(f"Saving model to {model_path}...")
    ensemble.save(str(model_path))
    
    # Load the model
    print(f"Loading model from {model_path}...")
    loaded_ensemble = SupplyChainRiskEnsemble.load(
        str(model_path), 
        ensemble.lstm_model.config
    )
    
    # Generate new mock data
    X_tabular, X_lstm, _ = generate_mock_data(n_samples=5)
    X_lstm_tensor = torch.FloatTensor(X_lstm)
    
    # Compare predictions from original and loaded models
    original_preds = ensemble.predict(X_lstm_tensor, X_tabular)
    loaded_preds = loaded_ensemble.predict(X_lstm_tensor, X_tabular)
    
    # Print comparison
    print("Original vs Loaded model predictions:")
    for i in range(len(original_preds)):
        print(f"  Sample {i}: {original_preds[i]:.4f} vs {loaded_preds[i]:.4f}")
    
    # Cleanup
    import shutil
    shutil.rmtree(model_dir)
    
    return loaded_ensemble


def test_with_real_data(use_mock=True):
    """Test the ensemble with real data from the database"""
    if use_mock:
        print("\nUsing mock data instead of real database data")
        return test_ensemble_training()
    
    print("\nTesting ensemble with real data from database...")
    
    # Load data from database
    data_loader = DataLoader()
    data = data_loader.load_training_data()
    
    # Process features
    feature_pipeline = FeaturePipeline()
    X, y = feature_pipeline.process(data)
    
    # Split features for different models
    X_tabular = X[feature_pipeline.get_tabular_feature_names()]
    X_time_series = feature_pipeline.get_time_series_features(data)
    
    # Convert to torch tensor
    X_lstm_tensor = torch.FloatTensor(X_time_series)
    
    # Create LSTM config
    lstm_config = LSTMConfig(
        input_size=X_time_series.shape[2],
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )
    
    # Create and train ensemble model
    ensemble = SupplyChainRiskEnsemble(lstm_config=lstm_config)
    metrics = ensemble.train(X_lstm_tensor, X_tabular.values, y.values)
    
    # Print metrics
    print("Training metrics with real data:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return ensemble


def main():
    """Main function to run all tests"""
    parser = argparse.ArgumentParser(description='Test the Supply Chain Risk Ensemble model')
    parser.add_argument('--use-real-data', action='store_true', help='Use real data from database instead of mock data')
    args = parser.parse_args()
    
    # Test with real or mock data
    ensemble = test_with_real_data(use_mock=not args.use_real_data)
    
    # Test prediction
    test_ensemble_prediction(ensemble)
    
    # Test save/load
    test_ensemble_save_load(ensemble)
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main() 