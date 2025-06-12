# Supply Chain ML System - Database Guide

This guide covers the complete database setup, management, and monitoring for the Supply Chain ML System.

## 🗄️ Database Architecture

The system uses a multi-database architecture optimized for different data types and access patterns:

### Primary Databases

- **PostgreSQL**: Primary relational database (21 tables)
- **Redis**: High-speed caching and real-time data
- **Neo4j**: Graph database for supplier relationships

### Database Schema Overview

#### Core Tables (001_init.sql)

- `company_info` - Company master data
- `financial_data` - Financial statements and ratios
- `earnings_calls` - Earnings guidance and transcripts
- `sec_filings` - SEC filing data and analysis
- `supplier_relationships` - Company-supplier connections
- `supplier_companies` - Supplier company details
- `market_data` - Stock price and trading data
- `news_articles` - News articles and sentiment
- `features` - ML feature store
- `predictions` - Model predictions and results
- `model_training_logs` - Model training history

#### Data Pipeline Tables (002_pipeline_tables.sql)

- `companies` - Simplified company data for pipeline
- `feature_store` - Enhanced feature store with TTL
- `processed_filings` - Filing processing tracking
- `market_data_daily` - Daily market data
- `realtime_market_data` - Real-time market feeds
- `detected_anomalies` - Anomaly detection results
- `pipeline_jobs` - Job execution logs
- `data_quality_metrics` - Data quality monitoring
- `system_alerts` - System alert management

## 🚀 Quick Start

### 1. Start Database Services

```bash
# Start all database services
make db-start

# Or start all services
make docker-run
```

### 2. Initialize Database

```bash
# Initialize with schema and seed data
make db-init

# Alternative: Reset database (WARNING: destructive)
make db-reset
```

### 3. Verify Setup

```bash
# Check database health
make db-health

# Verify table creation
make db-verify
```

## 📋 Available Commands

### Database Management

- `make db-init` - Initialize database with schema and seed data
- `make db-reset` - Reset database (WARNING: destructive)
- `make db-start` - Start database services only
- `make db-stop` - Stop database services
- `make db-logs` - Show database logs

### Monitoring & Maintenance

- `make db-health` - Comprehensive health check
- `make db-backup` - Create database backup
- `make db-verify` - Verify database setup

### Direct Access

```bash
# PostgreSQL shell
make shell-db

# Or direct access
docker-compose exec postgres psql -U postgres -d supply_chain_ml
```

## 🔧 Manual Setup (Without Docker)

If you prefer to run PostgreSQL locally:

### Prerequisites

- PostgreSQL 15+
- Redis 7+
- Neo4j 5+

### Environment Variables

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=supply_chain_ml
export DB_USER=postgres
export DB_PASSWORD=password
```

### Initialize Database

```bash
# Make scripts executable
chmod +x scripts/init_database.sh scripts/db_health_check.sh

# Run initialization
./scripts/init_database.sh

# Check health
./scripts/db_health_check.sh
```

## 📊 Database Schema Details

### Key Design Principles

1. **Normalized Structure**: Proper foreign key relationships
2. **Performance Optimized**: 40+ indexes for query optimization
3. **Temporal Data**: Comprehensive timestamp tracking
4. **JSONB Storage**: Flexible metadata and configuration storage
5. **Data Quality**: Built-in validation and monitoring

### Sample Data Included

The seed data includes:

- **20 Major Companies**: AAPL, MSFT, GOOGL, AMZN, TSLA, etc.
- **Financial Data**: Last quarter financial statements
- **Market Data**: 30 days of market data
- **SEC Filings**: Sample 10-Q and 10-K filings
- **News Articles**: Supply chain related news
- **Features**: 8 types × 5 companies × 3 days = 120 feature records
- **Real-time Data**: 24 hours of market data

### Database Size Estimates

| Component      | Records  | Size     |
| -------------- | -------- | -------- |
| Companies      | 20       | < 1MB    |
| Financial Data | 5        | < 1MB    |
| Market Data    | 150      | < 1MB    |
| Feature Store  | 120      | < 1MB    |
| News Articles  | 5        | < 1MB    |
| **Total**      | **~300** | **~5MB** |

## 🔍 Monitoring & Health Checks

### Health Check Features

The `db_health_check.sh` script monitors:

- **Connectivity**: Database availability
- **Connections**: Active connection count
- **Performance**: Slow queries and cache hit ratio
- **Storage**: Database and table sizes
- **Index Usage**: Unused indexes detection
- **Data Quality**: NULL values and freshness
- **System Resources**: Locks and long transactions

### Health Check Commands

```bash
# Full health check
make db-health

# Specific checks
./scripts/db_health_check.sh --connections
./scripts/db_health_check.sh --slow-queries
./scripts/db_health_check.sh --data-quality
```

### Performance Thresholds

- **Max Connections**: 150 (warning at 150+)
- **Slow Queries**: 5 seconds (monitored)
- **Table Bloat**: 20% dead tuples (warning)
- **Cache Hit Ratio**: >95% (optimal)

## 🔐 Security & Access Control

### Default Credentials (Development)

- **PostgreSQL**: `postgres / password`
- **Redis**: No authentication (development)
- **Neo4j**: `neo4j / password`

### Production Recommendations

1. Change default passwords
2. Enable SSL/TLS encryption
3. Restrict network access
4. Set up read-only users for reporting
5. Enable audit logging

## 🗂️ Data Pipeline Integration

### Feature Store

- **TTL Support**: Automatic expiration for real-time features
- **Batch Operations**: Efficient bulk inserts
- **Versioning**: Feature version tracking
- **Metadata**: JSONB for feature configuration

### Job Tracking

- All pipeline jobs logged in `pipeline_jobs` table
- Status tracking: running, completed, failed
- Performance metrics: records processed, duration
- Error handling and retry logic

### Data Quality Monitoring

- Automated quality checks
- Configurable thresholds
- Alert generation for quality issues
- Historical quality trends

## 📈 Scaling Considerations

### Current Capacity

- **Concurrent Connections**: 200 max
- **Storage**: Optimized for < 1GB initial data
- **Query Performance**: Sub-100ms for API calls

### Scaling Options

1. **Read Replicas**: For reporting and analytics
2. **Connection Pooling**: PgBouncer for connection management
3. **Partitioning**: Time-based partitioning for large tables
4. **Indexing**: Additional indexes for specific queries

## 🚨 Backup & Recovery

### Automated Backups

```bash
# Create backup
make db-backup

# Manual backup with custom name
PGPASSWORD=password pg_dump -h localhost -p 5432 -U postgres supply_chain_ml > backup_$(date +%Y%m%d).sql
```

### Restore Process

```bash
# Stop services
make db-stop

# Start database
make db-start

# Restore from backup
PGPASSWORD=password psql -h localhost -p 5432 -U postgres -d supply_chain_ml < backup_20231201.sql
```

### Backup Schedule Recommendations

- **Development**: Daily backups
- **Production**: Hourly incrementals, daily full backups
- **Long-term**: Weekly/monthly archives

## 🐛 Troubleshooting

### Common Issues

#### 1. Connection Refused

```bash
# Check if PostgreSQL is running
make db-start
pg_isready -h localhost -p 5432 -U postgres
```

#### 2. Permission Denied

```bash
# Make scripts executable
chmod +x scripts/*.sh
```

#### 3. Table Already Exists

```bash
# Reset database if needed
make db-reset
```

#### 4. Out of Disk Space

```bash
# Check database size
./scripts/db_health_check.sh --size

# Clean up old data
docker system prune -f
```

### Log Analysis

```bash
# Database logs
make db-logs

# Container logs
docker-compose logs postgres

# Health check logs
./scripts/db_health_check.sh > health_report.txt
```

## 📞 Support & Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Run health checks, review slow queries
2. **Monthly**: Analyze table bloat, update statistics
3. **Quarterly**: Review indexes, optimize performance

### Performance Tuning

1. Monitor `pg_stat_statements` for slow queries
2. Check index usage with `pg_stat_user_indexes`
3. Analyze table bloat and run VACUUM as needed
4. Update table statistics with ANALYZE

### Monitoring Setup

- Set up automated health checks (cron job)
- Configure alerting for critical issues
- Monitor disk space and connection counts
- Track query performance trends

## 🔗 Integration Points

### API Server

- Connection pooling via Go database/sql
- Health checks at `/health` endpoint
- Automatic reconnection on failures

### Data Pipeline

- Batch processing with transactions
- Feature store integration
- Job tracking and monitoring

### ML Models

- Feature extraction from PostgreSQL
- Real-time predictions stored back
- Model performance tracking

---

**Database Status**: ✅ Ready for development
**Last Updated**: December 2024
**Version**: 1.0.0
