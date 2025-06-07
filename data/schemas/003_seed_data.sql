-- Seed Data for Supply Chain ML System
-- This file contains test data to populate the database for development and testing

-- Insert sample companies (top 20 by market cap with supply chain complexity)
INSERT INTO companies (symbol, name, cik, active, sector, industry) VALUES
('AAPL', 'Apple Inc.', '0000320193', TRUE, 'Technology', 'Consumer Electronics'),
('MSFT', 'Microsoft Corporation', '0000789019', TRUE, 'Technology', 'Software'),
('GOOGL', 'Alphabet Inc.', '0001652044', TRUE, 'Communication Services', 'Internet Content & Information'),
('AMZN', 'Amazon.com, Inc.', '0001018724', TRUE, 'Consumer Cyclical', 'Internet Retail'),
('TSLA', 'Tesla, Inc.', '0001318605', TRUE, 'Consumer Cyclical', 'Auto Manufacturers'),
('META', 'Meta Platforms, Inc.', '0001326801', TRUE, 'Communication Services', 'Internet Content & Information'),
('NVDA', 'NVIDIA Corporation', '0001045810', TRUE, 'Technology', 'Semiconductors'),
('BRK.B', 'Berkshire Hathaway Inc.', '0001067983', TRUE, 'Financial Services', 'Insurance - Diversified'),
('JPM', 'JPMorgan Chase & Co.', '0000019617', TRUE, 'Financial Services', 'Banks - Diversified'),
('JNJ', 'Johnson & Johnson', '0000200406', TRUE, 'Healthcare', 'Drug Manufacturers - General'),
('PG', 'Procter & Gamble Company', '0000080424', TRUE, 'Consumer Defensive', 'Household & Personal Products'),
('V', 'Visa Inc.', '0001403161', TRUE, 'Financial Services', 'Credit Services'),
('UNH', 'UnitedHealth Group Incorporated', '0000731766', TRUE, 'Healthcare', 'Healthcare Plans'),
('HD', 'Home Depot, Inc.', '0000354950', TRUE, 'Consumer Cyclical', 'Home Improvement Retail'),
('BAC', 'Bank of America Corporation', '0000070858', TRUE, 'Financial Services', 'Banks - Diversified'),
('MA', 'Mastercard Incorporated', '0001141391', TRUE, 'Financial Services', 'Credit Services'),
('WMT', 'Walmart Inc.', '0000104169', TRUE, 'Consumer Defensive', 'Discount Stores'),
('PFE', 'Pfizer Inc.', '0000078003', TRUE, 'Healthcare', 'Drug Manufacturers - General'),
('KO', 'Coca-Cola Company', '0000021344', TRUE, 'Consumer Defensive', 'Beverages - Non-Alcoholic'),
('INTC', 'Intel Corporation', '0000050863', TRUE, 'Technology', 'Semiconductors')
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    cik = EXCLUDED.cik,
    active = EXCLUDED.active,
    sector = EXCLUDED.sector,
    industry = EXCLUDED.industry,
    updated_at = CURRENT_TIMESTAMP;

-- Insert sample market data for the last 30 days
INSERT INTO market_data_daily (company_id, date, open_price, high_price, low_price, close_price, volume, adjusted_close)
SELECT 
    symbol,
    CURRENT_DATE - INTERVAL '1 day' * generate_series(0, 29),
    150.00 + random() * 50, -- open_price
    160.00 + random() * 40, -- high_price
    140.00 + random() * 30, -- low_price
    155.00 + random() * 45, -- close_price
    1000000 + floor(random() * 9000000)::bigint, -- volume
    155.00 + random() * 45 -- adjusted_close
FROM companies 
WHERE symbol IN ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA')
ON CONFLICT (company_id, date) DO NOTHING;

-- Insert sample financial data
INSERT INTO financial_data (company_id, report_date, revenue, cost_of_revenue, gross_profit, operating_income, net_income, 
                          total_assets, total_liabilities, inventory, accounts_payable, working_capital, 
                          cash_and_equivalents, total_debt, shares_outstanding, market_cap, stock_price,
                          debt_to_equity_ratio, current_ratio, inventory_turnover, gross_margin)
VALUES
('AAPL', '2024-06-30', 394328000000, 214137000000, 180191000000, 114301000000, 93736000000,
 352755000000, 290437000000, 6511000000, 64115000000, 9355000000, 67150000000, 95281000000,
 15204100000, 3500000000000, 230.15, 1.53, 1.13, 52.3, 0.457),
('MSFT', '2024-06-30', 245122000000, 65525000000, 179597000000, 109410000000, 88136000000,
 512236000000, 198298000000, 3742000000, 25066000000, 75298000000, 75000000000, 47032000000,
 7430000000, 3100000000000, 417.25, 0.24, 2.37, 17.5, 0.733),
('GOOGL', '2024-06-30', 307394000000, 145971000000, 161423000000, 84299000000, 73795000000,
 402392000000, 92963000000, 1506000000, 12047000000, 155000000000, 110916000000, 13253000000,
 12700000000, 2100000000000, 165.32, 0.04, 5.19, 96.8, 0.525),
('AMZN', '2024-06-30', 574785000000, 446369000000, 128416000000, 17766000000, 30425000000,
 527854000000, 339706000000, 34405000000, 84946000000, -17000000000, 88307000000, 67150000000,
 10757000000, 1800000000000, 167.33, 0.36, 1.01, 12.98, 0.223),
('TSLA', '2024-06-30', 96773000000, 79113000000, 17660000000, 8891000000, 14997000000,
 106618000000, 43009000000, 15211000000, 15255000000, 6267000000, 17000000000, 7872000000,
 3169000000, 800000000000, 252.18, 0.12, 1.32, 5.2, 0.182);

-- Insert sample SEC filings data
INSERT INTO processed_filings (accession_no, processor_version, success) VALUES
('0000320193-24-000019', 'v1.2.0', TRUE),
('0000789019-24-000016', 'v1.2.0', TRUE),
('0001652044-24-000018', 'v1.2.0', TRUE),
('0001018724-24-000023', 'v1.2.0', TRUE),
('0001318605-24-000012', 'v1.2.0', TRUE);

INSERT INTO sec_filings (company_id, cik, form_type, filing_date, accession_no, url, processed_at,
                        content_hash, supply_chain_mentions, risk_keywords_count, sentiment_score)
VALUES
('AAPL', '0000320193', '10-Q', '2024-08-01', '0000320193-24-000019', 
 'https://www.sec.gov/Archives/edgar/data/320193/000032019324000019/aapl-20240630.htm',
 CURRENT_TIMESTAMP - INTERVAL '5 days',
 'a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456', 47, 23, -0.15),

('MSFT', '0000789019', '10-Q', '2024-07-25', '0000789019-24-000016',
 'https://www.sec.gov/Archives/edgar/data/789019/000078901924000016/msft-20240630.htm',
 CURRENT_TIMESTAMP - INTERVAL '3 days',
 'b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456a1', 32, 18, 0.12),

('GOOGL', '0001652044', '10-K', '2024-02-02', '0001652044-24-000018',
 'https://www.sec.gov/Archives/edgar/data/1652044/000165204424000018/goog-20231231.htm',
 CURRENT_TIMESTAMP - INTERVAL '7 days',
 'c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456a1b2', 28, 14, 0.08),

('AMZN', '0001018724', '10-Q', '2024-08-01', '0001018724-24-000023',
 'https://www.sec.gov/Archives/edgar/data/1018724/000101872424000023/amzn-20240630.htm',
 CURRENT_TIMESTAMP - INTERVAL '2 days',
 'd4e5f6789012345678901234567890abcdef1234567890abcdef123456a1b2c3', 65, 31, -0.22),

('TSLA', '0001318605', '10-Q', '2024-07-24', '0001318605-24-000012',
 'https://www.sec.gov/Archives/edgar/data/1318605/000131860524000012/tsla-20240630.htm',
 CURRENT_TIMESTAMP - INTERVAL '4 days',
 'e5f6789012345678901234567890abcdef1234567890abcdef123456a1b2c3d4', 89, 42, -0.35);

-- Insert sample news articles
INSERT INTO news_articles (company_id, headline, content, url, source, published_at,
                          sentiment_score, supply_chain_relevance, confidence, processed_at)
VALUES
('AAPL', 'Apple Faces Supply Chain Disruptions in Asia', 
 'Apple Inc. is experiencing supply chain challenges affecting iPhone production...', 
 'https://example.com/news/apple-supply-chain-1', 'TechNews', 
 CURRENT_TIMESTAMP - INTERVAL '2 days', -0.65, 0.92, 0.87, CURRENT_TIMESTAMP - INTERVAL '1 day'),

('TSLA', 'Tesla Reports Semiconductor Shortage Impact', 
 'Tesla acknowledges ongoing semiconductor shortages affecting vehicle production...', 
 'https://example.com/news/tesla-shortage-1', 'AutoNews', 
 CURRENT_TIMESTAMP - INTERVAL '3 days', -0.78, 0.89, 0.91, CURRENT_TIMESTAMP - INTERVAL '2 days'),

('AMZN', 'Amazon Expands Logistics Network to Mitigate Supply Chain Risks', 
 'Amazon announces new fulfillment centers to improve supply chain resilience...', 
 'https://example.com/news/amazon-logistics-1', 'BusinessDaily', 
 CURRENT_TIMESTAMP - INTERVAL '1 day', 0.45, 0.84, 0.82, CURRENT_TIMESTAMP - INTERVAL '12 hours'),

('NVDA', 'NVIDIA Secures New Supplier Partnerships for AI Chips', 
 'NVIDIA announces strategic partnerships to secure chip supply for AI applications...', 
 'https://example.com/news/nvidia-partnerships-1', 'TechReporter', 
 CURRENT_TIMESTAMP - INTERVAL '4 days', 0.72, 0.91, 0.88, CURRENT_TIMESTAMP - INTERVAL '3 days'),

('WMT', 'Walmart Invests in Supply Chain Technology', 
 'Walmart announces major investment in supply chain automation and tracking...', 
 'https://example.com/news/walmart-tech-1', 'RetailToday', 
 CURRENT_TIMESTAMP - INTERVAL '5 days', 0.58, 0.76, 0.79, CURRENT_TIMESTAMP - INTERVAL '4 days');

-- Insert sample feature store data for the last 7 days
INSERT INTO feature_store (company_id, name, value, type, source, timestamp, metadata)
SELECT 
    c.symbol,
    feature_name,
    (random() * 100)::numeric(15,6),
    'numerical',
    'financial_data',
    CURRENT_TIMESTAMP - INTERVAL '1 day' * day_offset,
    '{"calculated_by": "data_pipeline", "version": "1.0"}'::jsonb
FROM companies c,
     (VALUES 
         ('gross_margin_trend', 0), ('gross_margin_trend', 1), ('gross_margin_trend', 2),
         ('inventory_turnover_ratio', 0), ('inventory_turnover_ratio', 1), ('inventory_turnover_ratio', 2),
         ('working_capital_ratio', 0), ('working_capital_ratio', 1), ('working_capital_ratio', 2),
         ('revenue_growth_rate', 0), ('revenue_growth_rate', 1), ('revenue_growth_rate', 2),
         ('debt_to_equity_ratio', 0), ('debt_to_equity_ratio', 1), ('debt_to_equity_ratio', 2),
         ('days_payable_outstanding', 0), ('days_payable_outstanding', 1), ('days_payable_outstanding', 2),
         ('supply_chain_sentiment', 0), ('supply_chain_sentiment', 1), ('supply_chain_sentiment', 2),
         ('market_volatility', 0), ('market_volatility', 1), ('market_volatility', 2)
     ) AS features(feature_name, day_offset)
WHERE c.symbol IN ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA')
ON CONFLICT (company_id, name, timestamp) DO NOTHING;

-- Insert real-time market data for the last few hours
INSERT INTO realtime_market_data (company_id, price, volume, timestamp, expires_at)
SELECT 
    symbol,
    (150.00 + random() * 50)::numeric(10,4),
    (1000000 + floor(random() * 2000000))::bigint,
    CURRENT_TIMESTAMP - INTERVAL '1 hour' * generate_series(0, 23),
    CURRENT_TIMESTAMP + INTERVAL '1 day'
FROM companies 
WHERE symbol IN ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA');

-- Insert sample detected anomalies
INSERT INTO detected_anomalies (id, type, severity, company_id, feature_name, value, expected_min, expected_max, z_score, detected_at, details)
VALUES
('anom_001', 'statistical_outlier', 'high', 'AAPL', 'gross_margin_trend', 12.5, 35.0, 65.0, -3.2, 
 CURRENT_TIMESTAMP - INTERVAL '6 hours', '{"description": "Gross margin significantly below expected range", "threshold": 2.5}'),

('anom_002', 'trend_break', 'medium', 'TSLA', 'inventory_turnover_ratio', 2.1, 4.0, 8.0, -2.8, 
 CURRENT_TIMESTAMP - INTERVAL '12 hours', '{"description": "Significant drop in inventory turnover", "trend_change": "-45%"}'),

('anom_003', 'statistical_outlier', 'high', 'AMZN', 'supply_chain_sentiment', -0.89, -0.3, 0.3, -4.1, 
 CURRENT_TIMESTAMP - INTERVAL '18 hours', '{"description": "Extremely negative supply chain sentiment detected", "news_sources": 5}');

-- Insert sample pipeline job logs
INSERT INTO pipeline_jobs (job_type, job_id, status, start_time, end_time, duration, parameters, results, records_processed, features_generated)
VALUES
('sec_processing', 'sec_job_20241201_001', 'completed', 
 CURRENT_TIMESTAMP - INTERVAL '2 hours', CURRENT_TIMESTAMP - INTERVAL '90 minutes', INTERVAL '30 minutes',
 '{"companies": ["AAPL", "MSFT"], "forms": ["10-Q", "10-K"]}'::jsonb,
 '{"filings_processed": 12, "features_extracted": 156}'::jsonb, 12, 156),

('financial_ingestion', 'fin_job_20241201_001', 'completed',
 CURRENT_TIMESTAMP - INTERVAL '4 hours', CURRENT_TIMESTAMP - INTERVAL '3 hours', INTERVAL '1 hour',
 '{"data_source": "yahoo_finance", "companies": ["AAPL", "MSFT", "GOOGL"]}'::jsonb,
 '{"records_updated": 45, "ratios_calculated": 135}'::jsonb, 45, 135),

('real_time_market', 'rt_job_20241201_001', 'running',
 CURRENT_TIMESTAMP - INTERVAL '30 minutes', NULL, NULL,
 '{"stream": "market_data_stream", "batch_size": 1000}'::jsonb,
 '{}'::jsonb, 2450, 0),

('feature_monitor', 'monitor_job_20241201_001', 'completed',
 CURRENT_TIMESTAMP - INTERVAL '1 hour', CURRENT_TIMESTAMP - INTERVAL '45 minutes', INTERVAL '15 minutes',
 '{"check_type": "anomaly_detection", "lookback_hours": 24}'::jsonb,
 '{"anomalies_detected": 3, "features_monitored": 89}'::jsonb, 89, 0);

-- Insert sample data quality metrics
INSERT INTO data_quality_metrics (metric_type, target, score, threshold, status, measured_at, details)
VALUES
('completeness', 'financial_data', 0.94, 0.90, 'pass', CURRENT_TIMESTAMP - INTERVAL '1 hour',
 '{"missing_records": 12, "total_records": 200}'::jsonb),

('freshness', 'market_data_daily', 0.98, 0.95, 'pass', CURRENT_TIMESTAMP - INTERVAL '2 hours',
 '{"latest_data_age_hours": 0.5, "threshold_hours": 24}'::jsonb),

('validity', 'feature_store', 0.87, 0.85, 'pass', CURRENT_TIMESTAMP - INTERVAL '30 minutes',
 '{"invalid_values": 23, "total_values": 1789}'::jsonb),

('completeness', 'news_articles', 0.82, 0.85, 'warn', CURRENT_TIMESTAMP - INTERVAL '45 minutes',
 '{"missing_sentiment": 18, "total_articles": 100}'::jsonb);

-- Insert sample system alerts
INSERT INTO system_alerts (alert_type, severity, title, message, triggered_at, details)
VALUES
('data_quality', 'medium', 'News Article Completeness Below Threshold', 
 'News article sentiment analysis completeness is 82%, below the 85% threshold',
 CURRENT_TIMESTAMP - INTERVAL '45 minutes',
 '{"metric": "completeness", "score": 0.82, "threshold": 0.85}'::jsonb),

('anomaly_detection', 'high', 'Supply Chain Sentiment Anomaly Detected', 
 'Extremely negative supply chain sentiment detected for AMZN (-0.89)',
 CURRENT_TIMESTAMP - INTERVAL '18 hours',
 '{"company": "AMZN", "feature": "supply_chain_sentiment", "value": -0.89, "z_score": -4.1}'::jsonb),

('system_performance', 'low', 'High Memory Usage on Data Pipeline', 
 'Data pipeline service is using 78% of available memory',
 CURRENT_TIMESTAMP - INTERVAL '2 hours',
 '{"service": "data_pipeline", "memory_usage_percent": 78, "threshold_percent": 75}'::jsonb);

-- Update statistics for query optimization
ANALYZE companies;
ANALYZE financial_data;
ANALYZE market_data_daily;
ANALYZE feature_store;
ANALYZE sec_filings;
ANALYZE news_articles;
ANALYZE detected_anomalies;
ANALYZE pipeline_jobs;
ANALYZE data_quality_metrics;
ANALYZE system_alerts; 