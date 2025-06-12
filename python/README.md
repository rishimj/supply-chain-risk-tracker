# Supply Chain Risk ML Pipeline

This directory contains the complete machine learning pipeline for supply chain risk prediction. The system uses an ensemble of models to predict the probability of companies missing earnings guidance based on financial, network, temporal, and sentiment features.

## 🏗️ Architecture

### Model Architecture

The ML pipeline uses an **Ensemble Model** combining three specialized models:

1. **XGBoost (40% weight)**: Handles financial and structured features
2. **LSTM (35% weight)**: Processes time series and temporal patterns
3. **GNN (25% weight)**: Analyzes supplier network relationships

### Pipeline Components

```
Data Sources → Feature Engineering → Model Training → Model Serving
     ↓              ↓                    ↓              ↓
- PostgreSQL    - Financial ratios   - Hyperparameter  - FastAPI
- Redis         - Technical indicators  optimization    - Real-time
- Neo4j         - NLP sentiment      - Cross-validation  predictions
- APIs          - Network metrics    - MLflow tracking - Batch processing
```

## 📁 Directory Structure

```
python/
├── config/                 # Configuration files
│   └── config.yaml        # Main ML pipeline configuration
├── data/                  # Data loading and processing
│   └── data_loader.py     # Database connections and data loading
├── features/              # Feature engineering
│   └── feature_pipeline.py # Feature processing and selection
├── models/                # Model definitions
│   └── ensemble.py        # Ensemble model implementation
├── training/              # Training pipeline
│   ├── trainer.py         # Main training orchestrator
│   └── evaluator.py       # Model evaluation and metrics
├── inference/             # Model serving
│   └── server.py          # FastAPI inference server
├── logs/                  # Training and inference logs
├── models/artifacts/      # Trained model artifacts
├── mlruns/               # MLflow experiment tracking
├── train_model.py        # Main training script
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
cd python
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the Pipeline

```bash
# Test the entire pipeline
make test-ml

# Quick training (2-5 minutes)
make train-model-quick

# Full training with real data
make train-model

# Start inference server
make model-serve

# View experiment tracking
make mlflow-ui
```

### 3. Train a Model (Quick)

```bash
# Quick training with mock data
make train-model-quick

# Or manually:
cd python
python train_model.py --quick --mock-data
```

### 4. Start Inference Server

```bash
# Start the model server
make model-serve

# Or manually:
cd python
python inference/server.py
```

### 5. Make Predictions

```bash
# Test prediction endpoint
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "AAPL",
    "features": {
      "financial_inventory_turnover": 5.2,
      "financial_gross_margin": 0.3,
      "financial_debt_to_equity": 1.5,
      "financial_current_ratio": 1.2,
      "network_supplier_concentration": 0.7,
      "network_supplier_risk_score": 0.4,
      "ts_volatility_30d": 0.15,
      "ts_momentum_10d": 0.02,
      "nlp_sentiment_score": 0.6,
      "nlp_risk_keywords_count": 3
    }
  }'
```

## 🎯 Features

### Input Features

The model processes four categories of features:

#### Financial Features

- `financial_inventory_turnover`: Inventory turnover ratio
- `financial_gross_margin`: Gross profit margin
- `financial_debt_to_equity`: Debt-to-equity ratio
- `financial_current_ratio`: Current ratio (liquidity)
- `financial_roa`: Return on assets
- `financial_roe`: Return on equity

#### Network Features

- `network_supplier_concentration`: Supplier concentration index
- `network_supplier_risk_score`: Average supplier risk score
- `network_geographic_diversity`: Geographic diversity of suppliers
- `network_tier1_suppliers`: Number of tier-1 suppliers
- `network_critical_suppliers`: Number of critical suppliers

#### Time Series Features

- `ts_volatility_30d`: 30-day price volatility
- `ts_momentum_10d`: 10-day price momentum
- `ts_trend_strength`: Price trend strength
- `ts_seasonality`: Seasonal patterns

#### NLP Features

- `nlp_sentiment_score`: News sentiment score (0-1)
- `nlp_risk_keywords_count`: Count of risk-related keywords
- `nlp_positive_sentiment`: Positive sentiment ratio
- `nlp_negative_sentiment`: Negative sentiment ratio
- `nlp_news_volume`: Volume of news articles

### Output Predictions

Each prediction returns:

```json
{
  "company_id": "AAPL",
  "risk_score": 65.4,
  "guidance_miss_probability": 0.654,
  "confidence": 0.87,
  "component_risks": {
    "financial_risk": 45.2,
    "network_risk": 72.1,
    "temporal_risk": 58.9,
    "sentiment_risk": 81.3
  },
  "feature_importance": {
    "financial_debt_to_equity": 0.15,
    "network_supplier_concentration": 0.12,
    "ts_volatility_30d": 0.1,
    "nlp_sentiment_score": 0.08
  },
  "model_version": "v1.0.0",
  "prediction_id": "pred_1234567890",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🔧 Configuration

### Main Configuration (`config/config.yaml`)

```yaml
# Data sources
data:
  database:
    host: "localhost"
    port: 5433
    database: "supply_chain_ml"
    username: "postgres"
    password: "password"

# Feature engineering
features:
  max_features: 100
  feature_selection_method: "mutual_info"
  scaler_type: "robust"

# Model training
training:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  cv_folds: 5
  n_trials: 100
  optimization_metric: "auc"

# Model parameters
models:
  ensemble:
    weights:
      xgboost: 0.4
      lstm: 0.35
      gnn: 0.25
    xgboost:
      n_estimators: 500
      max_depth: 6
      learning_rate: 0.1
    lstm:
      hidden_size: 128
      num_layers: 2
      dropout: 0.3
    gnn:
      hidden_dim: 64
      num_layers: 3
      dropout: 0.2

# MLflow tracking
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "supply_chain_risk_prediction"
```

## 🎓 Training Pipeline

### Full Training Process

1. **Data Loading**: Load data from PostgreSQL, Redis, and Neo4j
2. **Feature Engineering**: Create derived features and apply transformations
3. **Feature Selection**: Select most important features using mutual information
4. **Hyperparameter Optimization**: Use Optuna for automated hyperparameter tuning
5. **Model Training**: Train ensemble model with cross-validation
6. **Model Evaluation**: Comprehensive evaluation with multiple metrics
7. **Model Registration**: Save model artifacts and register in MLflow

### Training Commands

```bash
# Full training with real data
make train-model

# Quick training for testing
make train-model-quick

# Retrain existing model
make train-model-retrain

# View training progress
make mlflow-ui
```

### Training Options

```bash
# Training script options
python train_model.py --help

Options:
  --config PATH         Configuration file path
  --quick              Quick training mode (reduced parameters)
  --retrain PATH       Retrain existing model
  --mock-data          Use synthetic data instead of database
  --log-level LEVEL    Logging level (DEBUG, INFO, WARNING, ERROR)
```

## 📊 Model Evaluation

The pipeline provides comprehensive model evaluation:

### Metrics

- **AUC Score**: Area under ROC curve
- **Precision/Recall**: Classification performance
- **F1 Score**: Harmonic mean of precision and recall
- **Calibration**: How well predicted probabilities match actual outcomes
- **Feature Importance**: Which features drive predictions

### Evaluation Reports

- Automatic evaluation report generation
- Performance by sector/industry
- Confidence-based performance analysis
- Risk distribution analysis
- Model stability metrics

### Visualization

- ROC curves
- Precision-recall curves
- Feature importance plots
- Risk score distributions
- Calibration plots

## 🔄 Model Serving

### FastAPI Server

The inference server provides:

- **Real-time predictions**: Single company risk assessment
- **Batch predictions**: Multiple companies at once
- **Health monitoring**: Server and model status
- **Model management**: Reload models without restart
- **Performance metrics**: Response times and throughput

### API Endpoints

```bash
# Health check
GET /health

# Single prediction
POST /predict

# Batch predictions
POST /predict/batch

# Model status
GET /model/status

# Model metrics
GET /model/metrics

# Feature importance
GET /features/importance

# Reload model
POST /model/reload
```

### Server Configuration

```bash
# Start server
python inference/server.py

# Or with custom settings
uvicorn inference.server:app --host 0.0.0.0 --port 8001
```

## 🔍 Monitoring & Debugging

### Logging

- Structured logging with timestamps
- Separate log files for training and inference
- Configurable log levels
- Performance metrics logging

### MLflow Tracking

- Experiment tracking and comparison
- Model versioning and registry
- Hyperparameter optimization history
- Model performance metrics

### Health Checks

- Model loading status
- Feature pipeline status
- Prediction performance monitoring
- Resource usage tracking

## 🧪 Testing

### Test Suite

```bash
# Run all ML tests
make test-ml

# Run specific tests
cd python
python -m pytest tests/ -v
```

### Test Coverage

- Unit tests for all components
- Integration tests for end-to-end pipeline
- Performance benchmarks
- Data validation tests

## 🚀 Production Deployment

### Docker Deployment

```bash
# Build ML container
docker build -t supply-chain-ml .

# Run inference server
docker run -p 8001:8001 supply-chain-ml
```

### Scaling Considerations

- Horizontal scaling with load balancers
- Model caching for improved performance
- Batch processing for high-throughput scenarios
- A/B testing for model updates

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Failed**

   ```bash
   # Use mock data for testing
   python train_model.py --mock-data
   ```

2. **Model Loading Error**

   ```bash
   # Check model artifacts
   ls -la models/artifacts/

   # Retrain if needed
   make train-model-quick
   ```

3. **Memory Issues**

   ```bash
   # Reduce batch size in config
   # Use feature selection to reduce dimensionality
   ```

4. **Slow Training**

   ```bash
   # Use quick mode for testing
   python train_model.py --quick

   # Reduce n_trials in config
   ```

### Debug Mode

```bash
# Enable debug logging
python train_model.py --log-level DEBUG

# Check detailed logs
tail -f logs/training_*.log
```

## 📈 Performance Benchmarks

### Training Performance

- **Quick Training**: ~2-5 minutes with mock data
- **Full Training**: ~30-60 minutes with real data
- **Hyperparameter Optimization**: ~2-4 hours

### Inference Performance

- **Single Prediction**: <100ms
- **Batch Predictions**: ~10ms per prediction
- **Throughput**: >100 predictions/second

### Model Accuracy

- **Target AUC**: >0.75
- **Precision**: >0.70
- **Recall**: >0.65
- **F1 Score**: >0.67

## 🤝 Contributing

### Development Workflow

1. Create feature branch
2. Implement changes
3. Run tests: `make test-ml`
4. Update documentation
5. Submit pull request

### Code Standards

- Follow PEP 8 for Python code
- Add type hints for all functions
- Include docstrings for all classes/methods
- Write tests for new features

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

For questions or support, please check the main project README or create an issue in the repository.
