#!/usr/bin/env python3
"""
Supply Chain Risk Model Training Script

This script trains the ensemble ML model for supply chain risk prediction.
It includes data loading, feature engineering, hyperparameter optimization,
model training, and evaluation.

Usage:
    python train_model.py [--config CONFIG_PATH] [--quick] [--retrain MODEL_PATH]
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import traceback
from datetime import datetime

# Add the python directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.trainer import ModelTrainer

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def create_quick_config():
    """Create a quick training configuration for testing"""
    return {
        'data': {
            'database': {
                'host': 'localhost',
                'port': 5433,
                'database': 'supply_chain_ml',
                'username': 'postgres',
                'password': 'password'
            }
        },
        'features': {
            'max_features': 50,
            'feature_selection_method': 'mutual_info',
            'scaler_type': 'robust'
        },
        'training': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'cv_folds': 3,
            'n_trials': 10,  # Reduced for quick training
            'optimization_metric': 'auc'
        },
        'models': {
            'ensemble': {
                'weights': {'xgboost': 0.4, 'lstm': 0.35, 'gnn': 0.25},
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'lstm': {
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.3
                },
                'gnn': {
                    'hidden_dim': 32,
                    'num_layers': 2,
                    'dropout': 0.2
                }
            }
        },
        'evaluation': {
            'metrics': ['auc', 'precision', 'recall', 'f1_score'],
            'default_threshold': 0.5
        },
        'mlflow': {
            'tracking_uri': 'file:./mlruns',
            'experiment_name': 'supply_chain_risk_quick',
            'model_registry_name': 'supply_chain_risk_model'
        }
    }

def save_quick_config(config_path: str):
    """Save quick configuration to file"""
    import yaml
    
    config = create_quick_config()
    
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return config_path

def validate_environment():
    """Validate that the environment is set up correctly"""
    logger = logging.getLogger(__name__)
    
    # Check if required directories exist
    required_dirs = ['models', 'logs']
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Check database connection (optional)
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5433,
            database='supply_chain_ml',
            user='postgres',
            password='password'
        )
        conn.close()
        logger.info("Database connection successful")
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
        logger.warning("Will use mock data for training")
    
    # Check MLflow
    try:
        import mlflow
        logger.info("MLflow available")
    except ImportError:
        logger.error("MLflow not installed. Please install with: pip install mlflow")
        return False
    
    return True

def create_mock_data():
    """Create mock data for training when database is not available"""
    import pandas as pd
    import numpy as np
    
    logger = logging.getLogger(__name__)
    logger.info("Creating mock training data...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'id': range(n_samples),
        'symbol': [f'COMP{i:04d}' for i in range(n_samples)],
        'sector': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail'], n_samples),
        
        # Financial features
        'financial_inventory_turnover': np.random.normal(5.0, 2.0, n_samples),
        'financial_gross_margin': np.random.normal(0.3, 0.1, n_samples),
        'financial_debt_to_equity': np.random.normal(1.5, 0.8, n_samples),
        'financial_current_ratio': np.random.normal(1.2, 0.4, n_samples),
        'financial_roa': np.random.normal(0.05, 0.03, n_samples),
        'financial_roe': np.random.normal(0.12, 0.06, n_samples),
        
        # Network features
        'network_supplier_concentration': np.random.beta(2, 3, n_samples),
        'network_supplier_risk_score': np.random.normal(0.4, 0.2, n_samples),
        'network_geographic_diversity': np.random.randint(1, 20, n_samples),
        'network_tier1_suppliers': np.random.randint(5, 100, n_samples),
        'network_critical_suppliers': np.random.randint(1, 10, n_samples),
        
        # Time series features
        'ts_volatility_30d': np.random.exponential(0.2, n_samples),
        'ts_momentum_10d': np.random.normal(0.0, 0.05, n_samples),
        'ts_trend_strength': np.random.normal(0.0, 0.1, n_samples),
        
        # NLP features
        'nlp_sentiment_score': np.random.beta(2, 2, n_samples),
        'nlp_risk_keywords_count': np.random.poisson(3, n_samples),
        'nlp_positive_sentiment': np.random.beta(3, 2, n_samples),
        'nlp_negative_sentiment': np.random.beta(2, 3, n_samples),
        'nlp_news_volume': np.random.poisson(10, n_samples),
    }
    
    # Ensure positive values where needed
    data['financial_inventory_turnover'] = np.abs(data['financial_inventory_turnover'])
    data['financial_gross_margin'] = np.clip(data['financial_gross_margin'], 0, 1)
    data['financial_current_ratio'] = np.abs(data['financial_current_ratio'])
    data['network_supplier_risk_score'] = np.clip(data['network_supplier_risk_score'], 0, 1)
    data['nlp_sentiment_score'] = np.clip(data['nlp_sentiment_score'], 0, 1)
    
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description='Train Supply Chain Risk Model')
    parser.add_argument('--config', type=str, default='python/config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick training with reduced parameters')
    parser.add_argument('--retrain', type=str, default=None,
                       help='Path to existing model to retrain')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--mock-data', action='store_true',
                       help='Use mock data instead of database')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        logger.info("Starting Supply Chain Risk Model Training")
        logger.info(f"Arguments: {vars(args)}")
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            return 1
        
        # Handle quick training
        if args.quick:
            logger.info("Quick training mode enabled")
            config_path = 'python/config/quick_config.yaml'
            save_quick_config(config_path)
            args.config = config_path
        
        # Check if config file exists
        if not os.path.exists(args.config):
            if args.quick:
                logger.info(f"Created quick config at {args.config}")
            else:
                logger.error(f"Configuration file not found: {args.config}")
                return 1
        
        # Initialize trainer
        logger.info(f"Loading configuration from {args.config}")
        trainer = ModelTrainer(args.config)
        
        # Handle mock data
        if args.mock_data:
            logger.info("Using mock data for training")
            # Monkey patch the data loader to use mock data
            original_load_method = trainer.data_loader.load_training_data
            trainer.data_loader.load_training_data = lambda *args, **kwargs: create_mock_data()
        
        # Run training
        if args.retrain:
            logger.info(f"Retraining existing model from {args.retrain}")
            results = trainer.retrain_model(args.retrain)
        else:
            logger.info("Starting full training pipeline")
            results = trainer.run_full_training_pipeline()
        
        # Print results
        logger.info("Training completed successfully!")
        
        if 'metadata' in results and 'test_metrics' in results['metadata']:
            test_metrics = results['metadata']['test_metrics']
            logger.info("Test Set Performance:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Generate evaluation report
        if 'evaluation' in results:
            from training.evaluator import ModelEvaluator
            evaluator = ModelEvaluator({'metrics': ['auc', 'precision', 'recall', 'f1_score']})
            report = evaluator.generate_evaluation_report(results['evaluation'])
            
            # Save report
            report_path = Path("logs") / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Evaluation report saved to {report_path}")
            print("\n" + report)
        
        logger.info("Training pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 