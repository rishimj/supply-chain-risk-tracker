# Supply Chain ML Data Pipeline

The data pipeline service is a comprehensive system for ingesting, processing, and transforming supply chain data into machine learning features. It handles real-time streaming, batch processing, and data quality monitoring.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Data Pipeline  │    │  Feature Store  │
│                 │    │                  │    │                 │
│ • SEC Filings   │───▶│ • Stream Proc.   │───▶│ • Redis Cache   │
│ • Financial API │    │ • Batch Proc.    │    │ • PostgreSQL    │
│ • News Feeds    │    │ • Quality Mon.   │    │ • Features      │
│ • Market Data   │    │ • SEC Analyzer   │    │                 │
│ • Supplier Info │    │ • News Analyzer  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Core Components

### 1. Pipeline Orchestrator (`pkg/pipeline/orchestrator.go`)

- Manages all data processing services
- Coordinates processor lifecycle
- Provides health monitoring and metrics
- Handles graceful shutdown

### 2. Data Processors

#### SEC Filings Processor (`pkg/pipeline/processors/sec_processor.go`)

- Downloads and analyzes SEC filings (10-K, 10-Q)
- Extracts supply chain risk indicators
- Performs sentiment analysis on filing content
- Generates features:
  - `sec_supply_chain_mentions`: Keyword count
  - `sec_inventory_sentiment`: Sentiment score
  - `sec_supplier_dependency`: Dependency risk
  - `sec_geographic_risk`: Geographic exposure

#### Financial Data Ingester (`pkg/pipeline/processors/financial_ingester.go`)

- Ingests financial metrics and market data
- Calculates financial ratios and trends
- Generates time series features
- Features generated:
  - `financial_inventory_turnover`: Cost of Revenue / Inventory
  - `financial_gross_margin`: Gross Profit / Revenue
  - `financial_current_ratio`: Current Assets / Current Liabilities
  - `ts_volatility_30d`: 30-day price volatility
  - `ts_momentum_10d`: 10-day price momentum

#### Stream Processor (`pkg/pipeline/processors/stream_processor.go`)

- Processes real-time data streams using Redis Streams
- Handles market data, news, financial updates, supplier events
- Generates real-time features with TTL
- Features generated:
  - `rt_price_volatility`: Real-time price volatility
  - `rt_volume_surge`: Volume surge detection
  - `rt_news_sentiment`: Real-time news sentiment
  - `rt_supplier_event_risk`: Supplier event risk score

#### Batch Processor (`pkg/pipeline/processors/batch_processor.go`)

- Scheduled batch processing jobs (hourly, daily, weekly)
- Comprehensive feature engineering
- Historical trend analysis
- Features generated:
  - `batch_revenue_growth_trend`: Revenue growth over time
  - `batch_inventory_trend`: Inventory level trends
  - `batch_supplier_concentration_risk`: Supplier concentration
  - `batch_geographic_risk_exposure`: Geographic risk

#### Feature Monitor (`pkg/pipeline/processors/feature_monitor.go`)

- Data quality monitoring and alerting
- Anomaly detection using statistical methods
- Feature completeness and freshness checks
- Alert generation for quality issues

## Data Streams

### Redis Streams

The pipeline monitors four main Redis streams:

1. **market_data_stream**: Real-time market price and volume data
2. **news_stream**: News articles and sentiment data
3. **financial_updates_stream**: Financial guidance changes and reports
4. **supplier_events_stream**: Supplier-related events and disruptions

### Stream Message Format

```json
{
  "company_id": "AAPL",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "price": "150.25",
    "volume": "1000000"
  }
}
```

## Feature Engineering

### Feature Categories

1. **Financial Features**: Ratios, trends, and financial health indicators
2. **Market Features**: Price movements, volatility, trading patterns
3. **Supply Chain Features**: Supplier risks, geographic exposure, dependencies
4. **Sentiment Features**: News sentiment, SEC filing sentiment
5. **Real-time Features**: Live market data, breaking news impact

### Feature Metadata

Each feature includes comprehensive metadata:

```go
type Feature struct {
    CompanyID string                 `json:"company_id"`
    Name      string                 `json:"name"`
    Value     float64                `json:"value"`
    Type      string                 `json:"type"`
    Source    string                 `json:"source"`
    Timestamp time.Time              `json:"timestamp"`
    TTL       time.Duration          `json:"ttl,omitempty"`
    Metadata  map[string]interface{} `json:"metadata"`
}
```

## Configuration

### Environment Variables

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=supply_chain_ml
DB_USER=postgres
DB_PASSWORD=password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### Configuration File (`configs/config.yaml`)

```yaml
pipeline:
  batch_schedule:
    hourly: "0 * * * *"
    daily: "0 2 * * *"
    weekly: "0 2 * * 0"

  processors:
    sec_processor:
      enabled: true
      batch_size: 100
      rate_limit: "10/minute"

    financial_ingester:
      enabled: true
      apis:
        - alpha_vantage
        - yahoo_finance

    stream_processor:
      enabled: true
      consumer_groups:
        - market_data_group
        - news_group
```

## Usage

### Building and Running

```bash
# Build the data pipeline
make build-data-pipeline

# Run the data pipeline
make run-data-pipeline

# Test the data pipeline
make test-data-pipeline

# Run the test script
./test_data_pipeline.sh
```

### Docker Deployment

```bash
# Start all services including data pipeline
make docker-up

# View data pipeline logs
docker logs supply-chain-data-pipeline

# Monitor processing metrics
docker logs supply-chain-data-pipeline | grep "metrics"
```

### Manual Triggers

The pipeline supports manual job triggers through the orchestrator:

```go
// Trigger feature engineering
params := map[string]interface{}{
    "lookback_days": 30,
    "companies": []string{"AAPL", "MSFT"},
}
err := orchestrator.TriggerBatchJob(ctx, "feature_engineering", params)

// Trigger quality check
err := orchestrator.TriggerBatchJob(ctx, "data_quality_check", params)
```

## Monitoring and Metrics

### Health Checks

- Individual processor health status
- Database connectivity checks
- Redis stream health
- Feature store accessibility

### Metrics Collection

- Processing throughput and latency
- Error rates and counts
- Feature generation statistics
- Data quality scores

### Alerting

- Low data completeness alerts
- Stale feature warnings
- Anomaly detection alerts
- High error rate notifications

## Performance Characteristics

### Processing Capacity

- **SEC Filings**: 100+ filings per hour
- **Financial Data**: 50+ companies every 4 hours
- **Real-time Streams**: 1000+ messages per minute
- **Batch Features**: 1000+ features per daily run

### Latency Targets

- Real-time features: < 100ms
- Batch processing: < 30 minutes
- Quality checks: < 5 minutes
- Alert generation: < 1 minute

## Data Quality

### Quality Checks

1. **Completeness**: All expected features present
2. **Freshness**: Features updated within SLA
3. **Validity**: Values within expected ranges
4. **Consistency**: Related features are consistent
5. **Anomaly Detection**: Statistical outlier detection

### Quality Thresholds

- Completeness: > 80%
- Freshness: > 90%
- Validity: > 95%
- Consistency: > 90%
- Anomaly Rate: < 5%

## Integration Points

### Feature Store

- Redis for real-time feature caching
- PostgreSQL for persistent feature storage
- Automatic TTL management for real-time features

### ML Model Server

- Features available via REST API
- Batch feature retrieval
- Real-time feature streaming

### External APIs

- SEC EDGAR for filing data
- Financial data providers (Alpha Vantage, Yahoo Finance)
- News APIs for sentiment analysis
- Supplier databases for network analysis

## Development

### Adding New Processors

1. Create processor in `pkg/pipeline/processors/`
2. Implement required interfaces:
   ```go
   type Processor interface {
       Start(ctx context.Context) error
       Stop(ctx context.Context) error
       HealthCheck(ctx context.Context) bool
       GetMetrics(ctx context.Context) interface{}
   }
   ```
3. Register in orchestrator
4. Add configuration options
5. Write tests

### Adding New Features

1. Define feature in processor
2. Specify metadata and TTL
3. Store using feature store
4. Add quality checks
5. Update documentation

### Testing

```bash
# Unit tests
go test ./pkg/pipeline/processors/... -v

# Integration tests
go test ./pkg/pipeline/... -integration

# End-to-end tests
./test_data_pipeline.sh
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**

   - Check Redis service status
   - Verify connection parameters
   - Check network connectivity

2. **Database Connection Timeout**

   - Verify PostgreSQL is running
   - Check connection pool settings
   - Review database load

3. **SEC API Rate Limiting**

   - Implement exponential backoff
   - Add request delay
   - Use multiple API keys

4. **High Memory Usage**
   - Check batch sizes
   - Monitor goroutine leaks
   - Review caching strategies

### Logging

Log levels and categories:

- `ERROR`: Critical failures
- `WARN`: Quality issues, timeouts
- `INFO`: Processing status, metrics
- `DEBUG`: Detailed processing info

### Performance Tuning

1. Adjust batch sizes based on memory
2. Tune Redis connection pool
3. Optimize database queries
4. Configure appropriate timeouts

## Security Considerations

- API key management for external services
- Database credential rotation
- Network security for inter-service communication
- Data encryption for sensitive information

## Future Enhancements

- Machine learning-based anomaly detection
- Advanced supplier network analysis
- Real-time dashboard for pipeline monitoring
- Auto-scaling based on processing load
- Enhanced data quality reporting
