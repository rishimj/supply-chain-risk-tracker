-- Additional tables for Data Pipeline Service
-- These tables support the various data processors

-- Companies table (simplified for data pipeline)
CREATE TABLE IF NOT EXISTS companies (
    symbol VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    cik VARCHAR(20),
    active BOOLEAN DEFAULT TRUE,
    sector VARCHAR(100),
    industry VARCHAR(150),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature Store table (enhanced version for data pipeline)
CREATE TABLE IF NOT EXISTS feature_store (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) NOT NULL,
    name VARCHAR(100) NOT NULL,
    value NUMERIC(15,6),
    type VARCHAR(20) NOT NULL DEFAULT 'numerical', -- numerical, categorical, text
    source VARCHAR(50) NOT NULL, -- sec_filings, financial_data, real_time_market, etc.
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ttl INTERVAL, -- Time to live for real-time features
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Composite index for fast lookups
    UNIQUE(company_id, name, timestamp)
);

-- Processed filings tracking
CREATE TABLE IF NOT EXISTS processed_filings (
    accession_no VARCHAR(50) PRIMARY KEY,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processor_version VARCHAR(20),
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

-- SEC filings table (enhanced for pipeline)
CREATE TABLE IF NOT EXISTS sec_filings (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) NOT NULL,
    cik VARCHAR(20) NOT NULL,
    form_type VARCHAR(10) NOT NULL,
    filing_date DATE NOT NULL,
    accession_no VARCHAR(50) UNIQUE NOT NULL,
    url VARCHAR(500),
    processed_at TIMESTAMP,
    
    -- Content analysis
    content_hash VARCHAR(64), -- SHA256 hash of content
    supply_chain_mentions INTEGER DEFAULT 0,
    risk_keywords_count INTEGER DEFAULT 0,
    sentiment_score NUMERIC(6,4), -- -1 to 1
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (accession_no) REFERENCES processed_filings(accession_no)
);

-- Financial data table (for data pipeline)
CREATE TABLE IF NOT EXISTS financial_data (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) NOT NULL,
    report_date DATE NOT NULL,
    
    -- Income Statement
    revenue NUMERIC(15,2),
    cost_of_revenue NUMERIC(15,2),
    gross_profit NUMERIC(15,2),
    operating_income NUMERIC(15,2),
    net_income NUMERIC(15,2),
    
    -- Balance Sheet
    total_assets NUMERIC(15,2),
    total_liabilities NUMERIC(15,2),
    inventory NUMERIC(15,2),
    accounts_payable NUMERIC(15,2),
    working_capital NUMERIC(15,2),
    cash_and_equivalents NUMERIC(15,2),
    total_debt NUMERIC(15,2),
    
    -- Market Data
    shares_outstanding NUMERIC(15,2),
    market_cap NUMERIC(18,2),
    stock_price NUMERIC(10,4),
    
    -- Calculated Ratios
    debt_to_equity_ratio NUMERIC(8,4),
    current_ratio NUMERIC(8,4),
    inventory_turnover NUMERIC(8,4),
    gross_margin NUMERIC(8,4),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(company_id, report_date)
);

-- Market data for time series analysis
CREATE TABLE IF NOT EXISTS market_data_daily (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    
    open_price NUMERIC(10,4),
    high_price NUMERIC(10,4),
    low_price NUMERIC(10,4),
    close_price NUMERIC(10,4),
    volume BIGINT,
    adjusted_close NUMERIC(10,4),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(company_id, date)
);

-- Real-time market data cache
CREATE TABLE IF NOT EXISTS realtime_market_data (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) NOT NULL,
    price NUMERIC(10,4),
    volume BIGINT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Auto-delete old data (older than 1 day)
    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '1 day')
);

-- News articles for sentiment analysis
CREATE TABLE IF NOT EXISTS news_articles (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) NOT NULL,
    headline VARCHAR(500) NOT NULL,
    content TEXT,
    url VARCHAR(1000),
    source VARCHAR(100),
    published_at TIMESTAMP NOT NULL,
    
    -- Sentiment analysis
    sentiment_score NUMERIC(6,4), -- -1 to 1
    supply_chain_relevance NUMERIC(4,3), -- 0-1
    confidence NUMERIC(4,3), -- 0-1
    
    processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detected anomalies from feature monitoring
CREATE TABLE IF NOT EXISTS detected_anomalies (
    id VARCHAR(100) PRIMARY KEY,
    type VARCHAR(50) NOT NULL, -- statistical_outlier, trend_break, etc.
    severity VARCHAR(20) NOT NULL, -- low, medium, high
    company_id VARCHAR(50) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    
    -- Anomaly details
    value NUMERIC(15,6),
    expected_min NUMERIC(15,6),
    expected_max NUMERIC(15,6),
    z_score NUMERIC(8,4),
    
    detected_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP,
    details JSONB DEFAULT '{}',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline job logs
CREATE TABLE IF NOT EXISTS pipeline_jobs (
    id SERIAL PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL, -- sec_processing, financial_ingestion, etc.
    job_id VARCHAR(100) UNIQUE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'running', -- running, completed, failed
    
    start_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    duration INTERVAL,
    
    -- Job details
    parameters JSONB DEFAULT '{}',
    results JSONB DEFAULT '{}',
    error_message TEXT,
    
    -- Metrics
    records_processed INTEGER DEFAULT 0,
    features_generated INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data quality metrics
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id SERIAL PRIMARY KEY,
    metric_type VARCHAR(50) NOT NULL, -- completeness, freshness, validity, etc.
    target VARCHAR(100) NOT NULL, -- table name, feature name, etc.
    
    score NUMERIC(6,4) NOT NULL, -- 0-1
    threshold NUMERIC(6,4) NOT NULL,
    status VARCHAR(20) NOT NULL, -- pass, warn, fail
    
    measured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    details JSONB DEFAULT '{}',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System alerts
CREATE TABLE IF NOT EXISTS system_alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- low, medium, high, critical
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    
    triggered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    
    details JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_companies_symbol ON companies(symbol);
CREATE INDEX IF NOT EXISTS idx_companies_active ON companies(active);

CREATE INDEX IF NOT EXISTS idx_feature_store_company_name ON feature_store(company_id, name);
CREATE INDEX IF NOT EXISTS idx_feature_store_timestamp ON feature_store(timestamp);
CREATE INDEX IF NOT EXISTS idx_feature_store_source ON feature_store(source);
CREATE INDEX IF NOT EXISTS idx_feature_store_ttl ON feature_store(timestamp) WHERE ttl IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_processed_filings_processed_at ON processed_filings(processed_at);

CREATE INDEX IF NOT EXISTS idx_sec_filings_company_date ON sec_filings(company_id, filing_date);
CREATE INDEX IF NOT EXISTS idx_sec_filings_form_type ON sec_filings(form_type);
CREATE INDEX IF NOT EXISTS idx_sec_filings_processed ON sec_filings(processed_at);

CREATE INDEX IF NOT EXISTS idx_financial_data_company_date ON financial_data(company_id, report_date);
CREATE INDEX IF NOT EXISTS idx_financial_data_updated ON financial_data(updated_at);

CREATE INDEX IF NOT EXISTS idx_market_data_daily_company_date ON market_data_daily(company_id, date);
CREATE INDEX IF NOT EXISTS idx_market_data_daily_date ON market_data_daily(date);

CREATE INDEX IF NOT EXISTS idx_realtime_market_data_company ON realtime_market_data(company_id);
CREATE INDEX IF NOT EXISTS idx_realtime_market_data_timestamp ON realtime_market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_realtime_market_data_expires ON realtime_market_data(expires_at);

CREATE INDEX IF NOT EXISTS idx_news_articles_company_date ON news_articles(company_id, published_at);
CREATE INDEX IF NOT EXISTS idx_news_articles_processed ON news_articles(processed_at);
CREATE INDEX IF NOT EXISTS idx_news_articles_sentiment ON news_articles(sentiment_score);

CREATE INDEX IF NOT EXISTS idx_detected_anomalies_company ON detected_anomalies(company_id);
CREATE INDEX IF NOT EXISTS idx_detected_anomalies_detected_at ON detected_anomalies(detected_at);
CREATE INDEX IF NOT EXISTS idx_detected_anomalies_severity ON detected_anomalies(severity);
CREATE INDEX IF NOT EXISTS idx_detected_anomalies_resolved ON detected_anomalies(resolved_at);

CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_type ON pipeline_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_status ON pipeline_jobs(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_start_time ON pipeline_jobs(start_time);

CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_type ON data_quality_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_measured_at ON data_quality_metrics(measured_at);
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_status ON data_quality_metrics(status);

CREATE INDEX IF NOT EXISTS idx_system_alerts_type ON system_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_system_alerts_severity ON system_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_system_alerts_triggered_at ON system_alerts(triggered_at);
CREATE INDEX IF NOT EXISTS idx_system_alerts_resolved ON system_alerts(resolved_at);

-- Create automatic cleanup job for expired real-time data
CREATE OR REPLACE FUNCTION cleanup_expired_realtime_data()
RETURNS VOID AS $$
BEGIN
    DELETE FROM realtime_market_data WHERE expires_at < CURRENT_TIMESTAMP;
    DELETE FROM feature_store WHERE ttl IS NOT NULL AND (timestamp + ttl) < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to update the updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = CURRENT_TIMESTAMP;
   RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply the trigger to relevant tables
CREATE TRIGGER update_companies_updated_at BEFORE UPDATE ON companies
   FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_feature_store_updated_at BEFORE UPDATE ON feature_store
   FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_financial_data_updated_at BEFORE UPDATE ON financial_data
   FOR EACH ROW EXECUTE FUNCTION update_updated_at_column(); 