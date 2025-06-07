-- Supply Chain ML System Database Schema

-- Company Information
CREATE TABLE company_info (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(150),
    market_cap BIGINT,
    employees INTEGER,
    founded_year INTEGER,
    headquarters_country VARCHAR(3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Financial Data
CREATE TABLE financial_data (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) REFERENCES company_info(id),
    filing_date DATE NOT NULL,
    period_end_date DATE NOT NULL,
    filing_type VARCHAR(10) NOT NULL, -- 10-K, 10-Q, 8-K
    quarter VARCHAR(6), -- 2023Q1, 2023Q2, etc.
    fiscal_year INTEGER,
    
    -- Income Statement
    revenue NUMERIC(15,2),
    cost_of_goods_sold NUMERIC(15,2),
    gross_profit NUMERIC(15,2),
    operating_income NUMERIC(15,2),
    net_income NUMERIC(15,2),
    
    -- Balance Sheet
    total_assets NUMERIC(15,2),
    current_assets NUMERIC(15,2),
    inventory NUMERIC(15,2),
    total_liabilities NUMERIC(15,2),
    current_liabilities NUMERIC(15,2),
    shareholders_equity NUMERIC(15,2),
    
    -- Cash Flow
    operating_cash_flow NUMERIC(15,2),
    investing_cash_flow NUMERIC(15,2),
    financing_cash_flow NUMERIC(15,2),
    free_cash_flow NUMERIC(15,2),
    
    -- Calculated Ratios
    gross_margin NUMERIC(8,4),
    operating_margin NUMERIC(8,4),
    net_margin NUMERIC(8,4),
    current_ratio NUMERIC(8,4),
    inventory_turnover NUMERIC(8,4),
    days_inventory_outstanding NUMERIC(8,2),
    working_capital_ratio NUMERIC(8,4),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, filing_date, filing_type)
);

-- Earnings Calls and Transcripts
CREATE TABLE earnings_calls (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) REFERENCES company_info(id),
    quarter VARCHAR(6) NOT NULL,
    call_date TIMESTAMP NOT NULL,
    fiscal_year INTEGER,
    
    -- Guidance and Estimates
    guidance_eps_low NUMERIC(8,4),
    guidance_eps_high NUMERIC(8,4),
    guidance_revenue_low NUMERIC(15,2),
    guidance_revenue_high NUMERIC(15,2),
    
    -- Actual Results
    actual_eps NUMERIC(8,4),
    actual_revenue NUMERIC(15,2),
    
    -- Transcript Data
    transcript TEXT,
    management_section TEXT,
    qa_section TEXT,
    
    -- Derived Features
    guidance_miss_flag BOOLEAN, -- True if actual < guidance by >5%
    eps_surprise NUMERIC(8,4), -- (actual - consensus) / |consensus|
    revenue_surprise NUMERIC(8,4),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, quarter, fiscal_year)
);

-- SEC Filings
CREATE TABLE sec_filings (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) REFERENCES company_info(id),
    filing_type VARCHAR(10) NOT NULL,
    filing_date DATE NOT NULL,
    period_end_date DATE,
    accession_number VARCHAR(25) UNIQUE NOT NULL,
    
    -- Content
    content TEXT,
    parsed_sections JSONB, -- Structured sections
    
    -- Supply Chain Mentions
    supplier_mentions INTEGER DEFAULT 0,
    risk_factor_mentions INTEGER DEFAULT 0,
    supply_chain_sentiment NUMERIC(6,4), -- -1 to 1
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Supplier Relationships
CREATE TABLE supplier_relationships (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) REFERENCES company_info(id),
    supplier_id VARCHAR(50) NOT NULL,
    
    -- Relationship Details
    relationship_type VARCHAR(20) NOT NULL, -- direct, tier2, tier3
    tier INTEGER NOT NULL DEFAULT 1,
    estimated_spend NUMERIC(15,2),
    spend_percentage NUMERIC(6,4), -- Percentage of total procurement
    criticality_score NUMERIC(4,3), -- 0-1, how critical this supplier is
    
    -- Contract Information
    contract_type VARCHAR(30),
    start_date DATE,
    end_date DATE,
    renewable BOOLEAN DEFAULT TRUE,
    
    -- Risk Factors
    geographic_risk_score NUMERIC(4,3),
    financial_risk_score NUMERIC(4,3),
    operational_risk_score NUMERIC(4,3),
    concentration_risk_score NUMERIC(4,3),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Supplier Companies
CREATE TABLE supplier_companies (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    country VARCHAR(3) NOT NULL,
    region VARCHAR(50),
    industry VARCHAR(150),
    
    -- Financial Health
    financial_health_score NUMERIC(4,3), -- 0-1
    revenue NUMERIC(15,2),
    employees INTEGER,
    credit_rating VARCHAR(10),
    
    -- Risk Factors
    political_risk_score NUMERIC(4,3),
    natural_disaster_risk NUMERIC(4,3),
    cyber_risk_score NUMERIC(4,3),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market Data
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) REFERENCES company_info(id),
    date DATE NOT NULL,
    
    -- Stock Price Data
    open_price NUMERIC(10,4),
    high_price NUMERIC(10,4),
    low_price NUMERIC(10,4),
    close_price NUMERIC(10,4),
    volume BIGINT,
    adjusted_close NUMERIC(10,4),
    
    -- Market Metrics
    market_cap NUMERIC(18,2),
    shares_outstanding BIGINT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, date)
);

-- News Articles
CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) REFERENCES company_info(id),
    
    -- Article Details
    title VARCHAR(500) NOT NULL,
    content TEXT,
    url VARCHAR(1000),
    source VARCHAR(100),
    author VARCHAR(200),
    published_date TIMESTAMP NOT NULL,
    
    -- Analysis
    sentiment_score NUMERIC(6,4), -- -1 to 1
    supply_chain_relevance NUMERIC(4,3), -- 0-1
    impact_score NUMERIC(4,3), -- 0-1
    
    -- NLP Features
    entities JSONB, -- Named entities extracted
    topics JSONB, -- Topic classification
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Features Store
CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) REFERENCES company_info(id),
    feature_name VARCHAR(100) NOT NULL,
    feature_value NUMERIC(12,6),
    feature_type VARCHAR(20) NOT NULL, -- numerical, categorical, text, vector
    source VARCHAR(50) NOT NULL, -- financial, nlp, network, market, etc.
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, feature_name, timestamp)
);

-- Model Predictions
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    company_id VARCHAR(50) REFERENCES company_info(id),
    
    -- Prediction Results
    guidance_miss_probability NUMERIC(6,4) NOT NULL,
    risk_score NUMERIC(6,2) NOT NULL, -- 0-100
    confidence NUMERIC(4,3) NOT NULL, -- 0-1
    
    -- Component Risks
    financial_risk NUMERIC(6,4),
    network_risk NUMERIC(6,4),
    temporal_risk NUMERIC(6,4),
    sentiment_risk NUMERIC(6,4),
    
    -- Model Information
    model_version VARCHAR(20) NOT NULL,
    prediction_id UUID UNIQUE NOT NULL,
    
    -- Timing
    prediction_timestamp TIMESTAMP NOT NULL,
    prediction_horizon_days INTEGER, -- How many days ahead
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Training Logs
CREATE TABLE model_training_logs (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(20) NOT NULL,
    training_start TIMESTAMP NOT NULL,
    training_end TIMESTAMP,
    
    -- Training Data
    training_samples INTEGER,
    validation_samples INTEGER,
    test_samples INTEGER,
    
    -- Performance Metrics
    train_auc NUMERIC(6,4),
    validation_auc NUMERIC(6,4),
    test_auc NUMERIC(6,4),
    precision NUMERIC(6,4),
    recall NUMERIC(6,4),
    f1_score NUMERIC(6,4),
    
    -- Configuration
    hyperparameters JSONB,
    feature_count INTEGER,
    
    -- Status
    status VARCHAR(20) DEFAULT 'training', -- training, completed, failed
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_company_info_ticker ON company_info(ticker);
CREATE INDEX idx_company_info_sector ON company_info(sector);

CREATE INDEX idx_financial_data_company_date ON financial_data(company_id, filing_date);
CREATE INDEX idx_financial_data_quarter ON financial_data(quarter);

CREATE INDEX idx_earnings_calls_company_quarter ON earnings_calls(company_id, quarter);
CREATE INDEX idx_earnings_calls_date ON earnings_calls(call_date);
CREATE INDEX idx_earnings_calls_guidance_miss ON earnings_calls(guidance_miss_flag);

CREATE INDEX idx_sec_filings_company_date ON sec_filings(company_id, filing_date);
CREATE INDEX idx_sec_filings_type ON sec_filings(filing_type);

CREATE INDEX idx_supplier_relationships_company ON supplier_relationships(company_id);
CREATE INDEX idx_supplier_relationships_supplier ON supplier_relationships(supplier_id);
CREATE INDEX idx_supplier_relationships_tier ON supplier_relationships(tier);

CREATE INDEX idx_market_data_company_date ON market_data(company_id, date);
CREATE INDEX idx_market_data_date ON market_data(date);

CREATE INDEX idx_news_articles_company_date ON news_articles(company_id, published_date);
CREATE INDEX idx_news_articles_date ON news_articles(published_date);
CREATE INDEX idx_news_articles_sentiment ON news_articles(sentiment_score);

CREATE INDEX idx_features_company_name_time ON features(company_id, feature_name, timestamp);
CREATE INDEX idx_features_source ON features(source);

CREATE INDEX idx_predictions_company_time ON predictions(company_id, prediction_timestamp);
CREATE INDEX idx_predictions_model_version ON predictions(model_version);
CREATE INDEX idx_predictions_risk_score ON predictions(risk_score);

-- Add some sample data
INSERT INTO company_info (id, name, ticker, sector, industry, market_cap, employees) VALUES
('AAPL', 'Apple Inc.', 'AAPL', 'Technology', 'Consumer Electronics', 3000000000000, 164000),
('MSFT', 'Microsoft Corporation', 'MSFT', 'Technology', 'Software', 2800000000000, 221000),
('TSLA', 'Tesla, Inc.', 'TSLA', 'Consumer Cyclical', 'Auto Manufacturers', 800000000000, 127855),
('AMZN', 'Amazon.com, Inc.', 'AMZN', 'Consumer Cyclical', 'Internet Retail', 1500000000000, 1541000),
('GOOGL', 'Alphabet Inc.', 'GOOGL', 'Communication Services', 'Internet Content & Information', 1700000000000, 190000); 