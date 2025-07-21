import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: Dict):
        self.config = config
        self.db_config = config['database']
        self.engine = self._create_db_connection()
        
    def _create_db_connection(self):
        """Create database connection"""
        connection_string = (
            f"postgresql://{self.db_config['username']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(connection_string)
    
    def load_training_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Load comprehensive training data from database"""
        logger.info("Loading training data from database...")
        
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        # Load different data sources
        companies_data = self._load_companies_data(start_date, end_date)
        financial_data = self._load_financial_data(start_date, end_date)
        market_data = self._load_market_data(start_date, end_date)
        news_data = self._load_news_data(start_date, end_date)
        supplier_data = self._load_supplier_network_data(start_date, end_date)
        
        # Merge all data sources
        training_data = self._merge_data_sources(
            companies_data, financial_data, market_data, news_data, supplier_data
        )
        
        # Clean and preprocess
        clean_data = self._clean_merged_data(training_data)
        
        logger.info(f"Loaded {len(clean_data)} training samples")
        return clean_data
    

    
    def _load_companies_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load basic company information"""
        query = """
        SELECT 
            c.id as company_id,
            c.ticker as symbol,
            c.name,
            c.sector,
            c.industry,
            c.market_cap,
            c.employees,
            c.founded_year,
            c.headquarters_country,
            c.created_at,
            c.updated_at
        FROM company_info c
        WHERE c.created_at BETWEEN %s AND %s
        ORDER BY c.ticker
        """
        
        return pd.read_sql(query, self.engine, params=(start_date, end_date))
    
    def _load_financial_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load financial metrics data"""
        query = """
        SELECT 
            fd.company_id,
            fd.filing_date as report_date,
            fd.revenue,
            fd.gross_profit,
            fd.operating_income,
            fd.net_income,
            fd.total_assets,
            fd.total_liabilities,
            fd.shareholders_equity,
            fd.current_assets,
            fd.inventory,
            fd.current_liabilities,
            
            -- Calculate financial ratios
            CASE WHEN fd.total_assets > 0 THEN fd.net_income / fd.total_assets ELSE 0 END as roa,
            CASE WHEN fd.shareholders_equity > 0 THEN fd.net_income / fd.shareholders_equity ELSE 0 END as roe,
            CASE WHEN fd.current_liabilities > 0 THEN fd.current_assets / fd.current_liabilities ELSE 0 END as current_ratio,
            CASE WHEN fd.shareholders_equity > 0 THEN fd.total_liabilities / fd.shareholders_equity ELSE 0 END as debt_to_equity,
            CASE WHEN fd.revenue > 0 THEN fd.gross_profit / fd.revenue ELSE 0 END as gross_margin,
            CASE WHEN fd.inventory > 0 AND fd.revenue > 0 THEN fd.revenue / fd.inventory ELSE 0 END as inventory_turnover,
            CASE WHEN (fd.current_assets - fd.inventory) > 0 AND fd.current_liabilities > 0 
                 THEN (fd.current_assets - fd.inventory) / fd.current_liabilities ELSE 0 END as quick_ratio,
            CASE WHEN fd.total_assets > 0 THEN fd.revenue / fd.total_assets ELSE 0 END as asset_turnover
            
        FROM financial_data fd
        WHERE fd.filing_date BETWEEN %s AND %s
        ORDER BY fd.company_id, fd.filing_date
        """
        
        return pd.read_sql(query, self.engine, params=(start_date, end_date))
    
    def _load_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load market data and calculate technical indicators"""
        query = """
        WITH daily_returns AS (
            SELECT 
                md.company_id,
                md.date,
                md.close_price,
                md.volume,
                md.high_price,
                md.low_price,
                md.open_price,
                LAG(md.close_price) OVER (PARTITION BY md.company_id ORDER BY md.date) as prev_close,
                
                -- Calculate daily return
                CASE WHEN LAG(md.close_price) OVER (PARTITION BY md.company_id ORDER BY md.date) > 0
                     THEN (md.close_price - LAG(md.close_price) OVER (PARTITION BY md.company_id ORDER BY md.date)) 
                          / LAG(md.close_price) OVER (PARTITION BY md.company_id ORDER BY md.date)
                     ELSE 0 END as daily_return
            FROM market_data md
            WHERE md.date BETWEEN %s AND %s
        ),
        volatility_metrics AS (
            SELECT 
                company_id,
                date,
                close_price,
                volume,
                daily_return,
                
                -- 30-day volatility
                STDDEV(daily_return) OVER (
                    PARTITION BY company_id 
                    ORDER BY date 
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as volatility_30d,
                
                -- 10-day momentum
                AVG(daily_return) OVER (
                    PARTITION BY company_id 
                    ORDER BY date 
                    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
                ) as momentum_10d,
                
                -- Price trend (20-day moving average slope)
                AVG(close_price) OVER (
                    PARTITION BY company_id 
                    ORDER BY date 
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) as ma_20d
                
            FROM daily_returns
        )
        SELECT 
            company_id,
            date,
            close_price,
            volume,
            daily_return,
            volatility_30d,
            momentum_10d,
            ma_20d,
            
            -- Additional technical indicators
            CASE WHEN LAG(ma_20d, 5) OVER (PARTITION BY company_id ORDER BY date) > 0
                 THEN (ma_20d - LAG(ma_20d, 5) OVER (PARTITION BY company_id ORDER BY date)) 
                      / LAG(ma_20d, 5) OVER (PARTITION BY company_id ORDER BY date)
                 ELSE 0 END as trend_strength,
                 
            -- Volume trend
            AVG(volume) OVER (
                PARTITION BY company_id 
                ORDER BY date 
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) as avg_volume_20d
            
        FROM volatility_metrics
        ORDER BY company_id, date
        """
        
        return pd.read_sql(query, self.engine, params=(start_date, end_date))
    
    def _load_news_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load news sentiment data"""
        query = """
        WITH daily_sentiment AS (
            SELECT 
                na.company_id,
                DATE(na.published_date) as date,
                AVG(na.sentiment_score) as avg_sentiment,
                COUNT(*) as news_count,
                SUM(CASE WHEN na.sentiment_score > 0.6 THEN 1 ELSE 0 END) as positive_news,
                SUM(CASE WHEN na.sentiment_score < 0.4 THEN 1 ELSE 0 END) as negative_news,
                SUM(COALESCE(na.supply_chain_mentions, 0)) as total_risk_keywords
            FROM news_articles na
            WHERE DATE(na.published_date) BETWEEN %s AND %s
            GROUP BY na.company_id, DATE(na.published_date)
        )
        SELECT 
            company_id,
            date,
            avg_sentiment as sentiment_score,
            news_count as news_volume,
            CASE WHEN news_count > 0 THEN positive_news::float / news_count ELSE 0 END as positive_sentiment_ratio,
            CASE WHEN news_count > 0 THEN negative_news::float / news_count ELSE 0 END as negative_sentiment_ratio,
            total_risk_keywords as risk_keywords_count
        FROM daily_sentiment
        ORDER BY company_id, date
        """
        
        return pd.read_sql(query, self.engine, params=(start_date, end_date))
    
    def _load_supplier_network_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load supplier network and relationship data"""
        query = """
        WITH supplier_metrics AS (
            SELECT 
                sr.company_id,
                COUNT(DISTINCT sr.supplier_id) as total_suppliers,
                AVG(COALESCE(sr.criticality_score, 0.5)) as avg_relationship_strength,
                MAX(COALESCE(sr.criticality_score, 0.5)) as max_supplier_dependency,
                
                -- Supplier concentration (Herfindahl index)
                SUM(POWER(COALESCE(sr.spend_percentage, 0.1), 2)) as supplier_concentration_index,
                
                -- Geographic diversity
                COUNT(DISTINCT sc.country) as supplier_countries,
                
                -- Risk metrics
                AVG(COALESCE((sr.geographic_risk_score + sr.financial_risk_score + sr.operational_risk_score)/3, 0.5)) as avg_supplier_risk,
                MAX(COALESCE((sr.geographic_risk_score + sr.financial_risk_score + sr.operational_risk_score)/3, 0.5)) as max_supplier_risk,
                
                -- Critical suppliers (assuming tier 1 are critical)
                SUM(CASE WHEN sr.tier = 1 THEN 1 ELSE 0 END) as critical_suppliers_count
                
            FROM supplier_relationships sr
            LEFT JOIN supplier_companies sc ON sr.supplier_id = sc.id
            WHERE sr.created_at BETWEEN %s AND %s
            GROUP BY sr.company_id
        )
        SELECT 
            company_id,
            total_suppliers,
            avg_relationship_strength,
            max_supplier_dependency,
            supplier_concentration_index,
            supplier_countries,
            avg_supplier_risk as supplier_risk_score,
            max_supplier_risk,
            critical_suppliers_count,
            
            -- Normalized metrics
            CASE WHEN total_suppliers > 0 
                 THEN supplier_concentration_index / total_suppliers 
                 ELSE 1.0 END as normalized_supplier_concentration,
                 
            CASE WHEN total_suppliers > 0 
                 THEN critical_suppliers_count::float / total_suppliers 
                 ELSE 0.0 END as critical_supplier_ratio
                 
        FROM supplier_metrics
        ORDER BY company_id
        """
        
        return pd.read_sql(query, self.engine, params=(start_date, end_date))
    
    def _merge_data_sources(self, companies_data: pd.DataFrame, financial_data: pd.DataFrame, 
                           market_data: pd.DataFrame, news_data: pd.DataFrame, 
                           supplier_data: pd.DataFrame) -> pd.DataFrame:
        """Merge all data sources into a single training dataset"""
        
        # Start with companies as base
        merged_data = companies_data.copy()
        
        # Merge financial data (latest available for each company)
        if not financial_data.empty:
            latest_financial = financial_data.groupby('company_id').last().reset_index()
            merged_data = merged_data.merge(
                latest_financial, 
                left_on='id', 
                right_on='company_id', 
                how='left',
                suffixes=('', '_financial')
            )
        
        # Merge market data (latest available for each company)
        if not market_data.empty:
            latest_market = market_data.groupby('company_id').last().reset_index()
            merged_data = merged_data.merge(
                latest_market, 
                left_on='id', 
                right_on='company_id', 
                how='left',
                suffixes=('', '_market')
            )
        
        # Merge news data (latest available for each company)
        if not news_data.empty:
            latest_news = news_data.groupby('company_id').last().reset_index()
            merged_data = merged_data.merge(
                latest_news, 
                left_on='id', 
                right_on='company_id', 
                how='left',
                suffixes=('', '_news')
            )
        
        # Merge supplier data
        if not supplier_data.empty:
            merged_data = merged_data.merge(
                supplier_data, 
                left_on='id', 
                right_on='company_id', 
                how='left',
                suffixes=('', '_supplier')
            )
        
        # Clean up duplicate columns
        merged_data = self._clean_merged_data(merged_data)
        
        return merged_data
    
    def _clean_merged_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the merged dataset"""
        
        # Remove duplicate company_id columns
        company_id_cols = [col for col in data.columns if col.startswith('company_id')]
        for col in company_id_cols[1:]:  # Keep the first one
            if col in data.columns:
                data = data.drop(columns=[col])
        
        # Rename columns to match feature naming convention
        column_mapping = {
            'roa': 'financial_roa',
            'roe': 'financial_roe',
            'current_ratio': 'financial_current_ratio',
            'debt_to_equity': 'financial_debt_to_equity',
            'gross_margin': 'financial_gross_margin',
            'inventory_turnover': 'financial_inventory_turnover',
            'quick_ratio': 'financial_quick_ratio',
            'asset_turnover': 'financial_asset_turnover',
            'volatility_30d': 'ts_volatility_30d',
            'momentum_10d': 'ts_momentum_10d',
            'trend_strength': 'ts_trend_strength',
            'sentiment_score': 'nlp_sentiment_score',
            'risk_keywords_count': 'nlp_risk_keywords_count',
            'positive_sentiment_ratio': 'nlp_positive_sentiment',
            'negative_sentiment_ratio': 'nlp_negative_sentiment',
            'news_volume': 'nlp_news_volume',
            'normalized_supplier_concentration': 'network_supplier_concentration',
            'supplier_risk_score': 'network_supplier_risk_score',
            'supplier_countries': 'network_geographic_diversity',
            'total_suppliers': 'network_tier1_suppliers',
            'critical_suppliers_count': 'network_critical_suppliers'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Fill missing values with appropriate defaults
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(0)
        
        return data
    
    def create_labels(self, data: pd.DataFrame, target_column: str = 'guidance_miss') -> pd.Series:
        """Create target labels for training"""
        
        # For now, create synthetic labels based on risk indicators
        # In production, this would be based on actual guidance miss events
        
        risk_indicators = []
        
        # Financial risk indicators
        if 'financial_debt_to_equity' in data.columns:
            risk_indicators.append(data['financial_debt_to_equity'] > 2.0)
        if 'financial_current_ratio' in data.columns:
            risk_indicators.append(data['financial_current_ratio'] < 1.0)
        if 'financial_roa' in data.columns:
            risk_indicators.append(data['financial_roa'] < 0)
        
        # Market risk indicators
        if 'ts_volatility_30d' in data.columns:
            risk_indicators.append(data['ts_volatility_30d'] > data['ts_volatility_30d'].quantile(0.8))
        if 'ts_momentum_10d' in data.columns:
            risk_indicators.append(data['ts_momentum_10d'] < -0.05)
        
        # Sentiment risk indicators
        if 'nlp_sentiment_score' in data.columns:
            risk_indicators.append(data['nlp_sentiment_score'] < 0.3)
        if 'nlp_risk_keywords_count' in data.columns:
            risk_indicators.append(data['nlp_risk_keywords_count'] > data['nlp_risk_keywords_count'].quantile(0.7))
        
        # Network risk indicators
        if 'network_supplier_concentration' in data.columns:
            risk_indicators.append(data['network_supplier_concentration'] > 0.7)
        if 'network_supplier_risk_score' in data.columns:
            risk_indicators.append(data['network_supplier_risk_score'] > 0.6)
        
        # Combine risk indicators
        if risk_indicators:
            risk_score = sum(risk_indicators) / len(risk_indicators)
            # Convert to binary labels with some noise
            labels = (risk_score > 0.3).astype(int)
            
            # Add some randomness to make it more realistic
            np.random.seed(42)
            noise = np.random.random(len(labels)) < 0.1  # 10% label noise
            labels = labels ^ noise  # XOR to flip some labels
        else:
            # Fallback: random labels
            np.random.seed(42)
            labels = np.random.choice([0, 1], size=len(data), p=[0.7, 0.3])
        
        return pd.Series(labels, index=data.index, name=target_column)
    
    def get_feature_columns(self, config: Dict) -> List[str]:
        """Get list of feature columns based on configuration"""
        feature_columns = []
        
        for category in ['financial_features', 'network_features', 'temporal_features', 'nlp_features']:
            if category in config:
                feature_columns.extend(config[category])
        
        return feature_columns 