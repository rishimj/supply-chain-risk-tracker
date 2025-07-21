# Supply Chain Risk Tracker

A sophisticated machine learning system that predicts the probability of public companies missing quarterly earnings guidance due to supply chain disruptions.

## 🎯 Project Overview

This system combines real-time data ingestion, advanced machine learning, and network analysis to predict supply chain risks.

### Key Prediction Target

The probability that a public company will miss quarterly earnings guidance by >5% due to supply chain issues.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Data Pipeline  │───▶│   ML Pipeline   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  SEC Filings    │    │ Stream Processor│    │ Feature Engine  │
│  Earnings Calls │    │ Batch Processor │    │ Model Serving   │
│  Supplier Data  │    │ Graph Processor │    │ Online Learning │
│  Market Data    │    │ NLP Processor   │    │ A/B Testing     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### Backend Services (Go)

- **API Server**: High-performance REST API
- **Stream Processor**: Real-time data processing with Kafka + Flink
- **Batch Processor**: Historical data processing
- **Feature Engine**: Real-time feature computation

### Machine Learning (Python)

- **Ensemble Model**: XGBoost + LSTM + GNN
- **Deep Learning**: PyTorch for neural networks
- **Feature Engineering**: Custom pipeline with 1000+ features
- **Model Serving**: FastAPI with Redis caching

### Databases

- **PostgreSQL**: Relational data (financial, earnings, features)
- **Redis**: Caching and feature store

### Infrastructure

- **Streaming**: Apache Kafka + Apache Flink
- **Frontend**: React + TypeScript + Tailwind CSS
- **Monitoring**: Prometheus + Grafana
- **Deployment**: Docker + Kubernetes

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Git
- 16GB+ RAM recommended
- API keys for external data sources (optional for demo)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd supply-chain-tracker

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys (optional for demo)
```

### 2. Start the System

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps
```

### 3. Access the System

- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8080
- **Model Server**: http://localhost:8001
- **Grafana**: http://localhost:3001 (admin/ikWqB8PQ6B3.2kW)


## 📊 Data Sources

### Financial Data

- **SEC EDGAR**: 10-K, 10-Q, 8-K filings
- **Earnings Transcripts**: Quarterly calls and guidance
- **Market Data**: Stock prices, trading volumes
- **Financial APIs**: Alpha Vantage, Finnhub

### Supplier Networks

- **Public Disclosures**: SEC filings, annual reports
- **Commercial Databases**: FactSet, Bloomberg
- **Alternative Data**: Web scraping, satellite data

### Alternative Data

- **News Articles**: Financial news, trade publications
- **Social Media**: Twitter sentiment, LinkedIn activity
- **Economic Indicators**: GDP, inflation, trade data

## 🔬 Machine Learning Models

### Ensemble Architecture

1. **XGBoost**: Financial ratios and fundamental analysis
2. **LSTM**: Time series patterns and volatility
3. **GNN**: Supplier network relationships and contagion
4. **Meta-Model**: Random Forest combining base predictions

### Key Features (1000+)

- **Financial Features**: Ratios, growth rates, liquidity metrics
- **Network Features**: Centrality, clustering, risk propagation
- **Time Series Features**: Volatility, trends, seasonality
- **NLP Features**: Sentiment, topic modeling, entity extraction

### Performance Metrics

- **AUC**: 0.70+ on validation set

## 🏃‍♂️ Development

### Backend Development (Go)

```bash
# Install dependencies
go mod download

# Run API server locally
go run cmd/api-server/main.go

# Run tests
go test ./...

# Build
go build -o bin/api-server cmd/api-server/main.go
```

### ML Development (Python)

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r python/requirements.txt

# Train model
python python/training/train_model.py --config configs/model_config.yaml

# Run model server
cd python && uvicorn inference.server:app --host 0.0.0.0 --port 8001
```

### Frontend Development (React)

```bash
cd web

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## 📈 Monitoring and Observability

### Metrics Dashboard

- **Model Performance**: AUC, precision, recall over time
- **System Health**: Latency, throughput, error rates
- **Data Quality**: Completeness, freshness, accuracy
- **Business Metrics**: Prediction accuracy, alert volumes

### Alerting

- Model performance degradation
- Data pipeline failures
- High prediction volumes
- System resource usage

## 🔄 Data Pipeline

### Real-time Pipeline

1. **Ingestion**: Kafka consumers for market data, news
2. **Processing**: Flink jobs for feature extraction
3. **Storage**: Redis for features, PostgreSQL for persistence
4. **Serving**: Sub-100ms feature retrieval

### Batch Pipeline

1. **SEC Filings**: Daily batch processing
2. **Financial Data**: Quarterly updates
3. **Supplier Networks**: Monthly updates
4. **Model Training**: Weekly retraining

## 🧪 Model Training

### Training Process

1. **Data Preparation**: Feature engineering and cleaning
2. **Hyperparameter Optimization**: Optuna for automated tuning
3. **Cross-Validation**: Time series split for temporal validation
4. **Model Selection**: Performance-based selection
5. **Deployment**: Automated model deployment


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

