package database

import (
	"database/sql"
	"fmt"
	"time"

	"supply-chain-ml/pkg/config"

	_ "github.com/lib/pq"
)

// NewPostgresDB creates a new PostgreSQL database connection
func NewPostgresDB(cfg config.PostgresConfig) (*sql.DB, error) {
	dsn := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		cfg.Host, cfg.Port, cfg.Username, cfg.Password, cfg.Database, cfg.SSLMode,
	)

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open database connection: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(cfg.MaxConnections)
	db.SetMaxIdleConns(cfg.MaxConnections / 2)
	db.SetConnMaxLifetime(5 * time.Minute)

	// Test the connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return db, nil
}

// Store represents the database storage layer
type Store struct {
	db *sql.DB
}

// NewStore creates a new database store
func NewStore(db *sql.DB) *Store {
	return &Store{db: db}
}

// Close closes the database connection
func (s *Store) Close() error {
	return s.db.Close()
}

// GetCompanyData retrieves company data for a specific company and date
func (s *Store) GetCompanyData(companyID string, asOfDate time.Time) (*CompanyData, error) {
	query := `
		SELECT 
			ci.id, ci.name, ci.ticker, ci.sector, ci.industry,
			ci.market_cap, ci.employees, ci.founded_year
		FROM company_info ci
		WHERE ci.id = $1
	`
	
	var company CompanyData
	err := s.db.QueryRow(query, companyID).Scan(
		&company.ID, &company.Name, &company.Ticker,
		&company.Sector, &company.Industry, &company.MarketCap,
		&company.Employees, &company.FoundedYear,
	)
	
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("company not found: %s", companyID)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to query company data: %w", err)
	}
	
	return &company, nil
}

// StoreFeature stores a feature in the database
func (s *Store) StoreFeature(feature Feature) error {
	query := `
		INSERT INTO features (
			company_id, feature_name, feature_value, feature_type,
			source, timestamp, metadata
		) VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (company_id, feature_name, timestamp) 
		DO UPDATE SET 
			feature_value = EXCLUDED.feature_value,
			metadata = EXCLUDED.metadata
	`
	
	_, err := s.db.Exec(
		query,
		feature.CompanyID, feature.Name, feature.Value,
		feature.Type, feature.Source, feature.Timestamp,
		feature.Metadata,
	)
	
	if err != nil {
		return fmt.Errorf("failed to store feature: %w", err)
	}
	
	return nil
}

// GetHistoricalFeatures retrieves historical features for a company
func (s *Store) GetHistoricalFeatures(companyID string, featureName string, fromDate, toDate time.Time) ([]Feature, error) {
	query := `
		SELECT 
			company_id, feature_name, feature_value, feature_type,
			source, timestamp, metadata
		FROM features
		WHERE company_id = $1 
			AND feature_name = $2
			AND timestamp BETWEEN $3 AND $4
		ORDER BY timestamp ASC
	`
	
	rows, err := s.db.Query(query, companyID, featureName, fromDate, toDate)
	if err != nil {
		return nil, fmt.Errorf("failed to query historical features: %w", err)
	}
	defer rows.Close()
	
	var features []Feature
	for rows.Next() {
		var feature Feature
		err := rows.Scan(
			&feature.CompanyID, &feature.Name, &feature.Value,
			&feature.Type, &feature.Source, &feature.Timestamp,
			&feature.Metadata,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan feature row: %w", err)
		}
		features = append(features, feature)
	}
	
	return features, nil
}

// Data models
type CompanyData struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Ticker      string    `json:"ticker"`
	Sector      string    `json:"sector"`
	Industry    string    `json:"industry"`
	MarketCap   float64   `json:"market_cap"`
	Employees   int       `json:"employees"`
	FoundedYear int       `json:"founded_year"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type Feature struct {
	CompanyID string                 `json:"company_id"`
	Name      string                 `json:"name"`
	Value     interface{}            `json:"value"`
	Type      string                 `json:"type"`
	Source    string                 `json:"source"`
	Timestamp time.Time              `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata"`
} 