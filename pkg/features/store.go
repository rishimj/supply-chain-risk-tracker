package features

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// Store represents the feature store with Redis caching and PostgreSQL persistence
type Store struct {
	redisClient *redis.Client
	postgres    *sql.DB
}

// NewStore creates a new feature store instance
func NewStore(redisClient *redis.Client, postgres *sql.DB) *Store {
	return &Store{
		redisClient: redisClient,
		postgres:    postgres,
	}
}

// Feature represents a single feature
type Feature struct {
	CompanyID string                 `json:"company_id"`
	Name      string                 `json:"name"`
	Value     interface{}            `json:"value"`
	Type      string                 `json:"type"` // numerical, categorical, text, vector
	Source    string                 `json:"source"`
	Timestamp time.Time              `json:"timestamp"`
	TTL       time.Duration          `json:"ttl"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// FeatureVector represents a collection of features for a company at a specific time
type FeatureVector struct {
	CompanyID string             `json:"company_id"`
	Features  map[string]float64 `json:"features"`
	Timestamp time.Time          `json:"timestamp"`
	Version   string             `json:"version"`
}

// Store stores a feature in both Redis and PostgreSQL
func (fs *Store) Store(ctx context.Context, feature Feature) error {
	// Store in Redis for fast access
	if err := fs.storeInRedis(ctx, feature); err != nil {
		return fmt.Errorf("failed to store in Redis: %w", err)
	}
	
	// Store in PostgreSQL for persistence
	if err := fs.storeInPostgres(ctx, feature); err != nil {
		return fmt.Errorf("failed to store in PostgreSQL: %w", err)
	}
	
	return nil
}

// GetFeatureVector retrieves a complete feature vector for a company at a specific timestamp
func (fs *Store) GetFeatureVector(ctx context.Context, companyID string, timestamp time.Time) (*FeatureVector, error) {
	// Try Redis first
	fv, err := fs.getFeatureVectorFromRedis(ctx, companyID, timestamp)
	if err == nil {
		return fv, nil
	}
	
	// Fallback to PostgreSQL
	return fs.getFeatureVectorFromPostgres(ctx, companyID, timestamp)
}

// GetFeatures retrieves features with optional filtering
func (fs *Store) GetFeatures(ctx context.Context, companyID, featureName, source string, fromDate, toDate time.Time) ([]Feature, error) {
	// For historical data, query PostgreSQL directly
	return fs.getFeaturesFromPostgres(ctx, companyID, featureName, source, fromDate, toDate)
}

// HealthCheck verifies the feature store is working
func (fs *Store) HealthCheck(ctx context.Context) error {
	// Check Redis connection
	if err := fs.redisClient.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("Redis health check failed: %w", err)
	}
	
	// Check PostgreSQL connection
	if err := fs.postgres.PingContext(ctx); err != nil {
		return fmt.Errorf("PostgreSQL health check failed: %w", err)
	}
	
	return nil
}

// Redis operations
func (fs *Store) storeInRedis(ctx context.Context, feature Feature) error {
	key := fmt.Sprintf("feature:%s:%s:%s", 
		feature.CompanyID, 
		feature.Name, 
		feature.Timestamp.Format("2006-01-02"))
	
	data, err := json.Marshal(feature)
	if err != nil {
		return err
	}
	
	ttl := feature.TTL
	if ttl == 0 {
		ttl = 24 * time.Hour // Default TTL
	}
	
	return fs.redisClient.Set(ctx, key, data, ttl).Err()
}

func (fs *Store) getFeatureVectorFromRedis(ctx context.Context, companyID string, timestamp time.Time) (*FeatureVector, error) {
	// Get all features for the company on the specified date
	pattern := fmt.Sprintf("feature:%s:*:%s", companyID, timestamp.Format("2006-01-02"))
	keys, err := fs.redisClient.Keys(ctx, pattern).Result()
	if err != nil {
		return nil, err
	}
	
	if len(keys) == 0 {
		return nil, fmt.Errorf("no features found in Redis")
	}
	
	features := make(map[string]float64)
	
	for _, key := range keys {
		data, err := fs.redisClient.Get(ctx, key).Result()
		if err != nil {
			continue // Skip failed retrievals
		}
		
		var feature Feature
		if err := json.Unmarshal([]byte(data), &feature); err != nil {
			continue
		}
		
		// Convert to numerical value
		if val, ok := feature.Value.(float64); ok {
			features[feature.Name] = val
		}
	}
	
	return &FeatureVector{
		CompanyID: companyID,
		Features:  features,
		Timestamp: timestamp,
		Version:   "v1",
	}, nil
}

// PostgreSQL operations
func (fs *Store) storeInPostgres(ctx context.Context, feature Feature) error {
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
	
	// Convert feature value to float64 for storage
	var numericValue float64
	switch v := feature.Value.(type) {
	case float64:
		numericValue = v
	case float32:
		numericValue = float64(v)
	case int:
		numericValue = float64(v)
	case int64:
		numericValue = float64(v)
	default:
		numericValue = 0.0 // Default for non-numeric values
	}
	
	// Convert metadata to JSON
	metadataJSON, err := json.Marshal(feature.Metadata)
	if err != nil {
		metadataJSON = []byte("{}")
	}
	
	_, err = fs.postgres.ExecContext(
		ctx,
		query,
		feature.CompanyID,
		feature.Name,
		numericValue,
		feature.Type,
		feature.Source,
		feature.Timestamp,
		string(metadataJSON),
	)
	
	return err
}

func (fs *Store) getFeatureVectorFromPostgres(ctx context.Context, companyID string, timestamp time.Time) (*FeatureVector, error) {
	query := `
		SELECT feature_name, feature_value
		FROM features
		WHERE company_id = $1 
		AND DATE(timestamp) = DATE($2)
		ORDER BY timestamp DESC
	`
	
	rows, err := fs.postgres.QueryContext(ctx, query, companyID, timestamp)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	
	features := make(map[string]float64)
	
	for rows.Next() {
		var name string
		var value float64
		
		if err := rows.Scan(&name, &value); err != nil {
			continue
		}
		
		features[name] = value
	}
	
	if len(features) == 0 {
		return nil, fmt.Errorf("no features found in PostgreSQL")
	}
	
	return &FeatureVector{
		CompanyID: companyID,
		Features:  features,
		Timestamp: timestamp,
		Version:   "v1",
	}, nil
}

func (fs *Store) getFeaturesFromPostgres(ctx context.Context, companyID, featureName, source string, fromDate, toDate time.Time) ([]Feature, error) {
	query := `
		SELECT 
			company_id, feature_name, feature_value, feature_type,
			source, timestamp, metadata
		FROM features
		WHERE company_id = $1 
		AND timestamp BETWEEN $2 AND $3
	`
	
	args := []interface{}{companyID, fromDate, toDate}
	argIndex := 4
	
	// Add optional filters
	if featureName != "" {
		query += fmt.Sprintf(" AND feature_name = $%d", argIndex)
		args = append(args, featureName)
		argIndex++
	}
	
	if source != "" {
		query += fmt.Sprintf(" AND source = $%d", argIndex)
		args = append(args, source)
		argIndex++
	}
	
	query += " ORDER BY timestamp DESC LIMIT 1000"
	
	rows, err := fs.postgres.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	
	var features []Feature
	
	for rows.Next() {
		var feature Feature
		var metadataJSON string
		
		err := rows.Scan(
			&feature.CompanyID,
			&feature.Name,
			&feature.Value,
			&feature.Type,
			&feature.Source,
			&feature.Timestamp,
			&metadataJSON,
		)
		if err != nil {
			continue
		}
		
		// Parse metadata
		if metadataJSON != "" {
			if err := json.Unmarshal([]byte(metadataJSON), &feature.Metadata); err != nil {
				feature.Metadata = make(map[string]interface{})
			}
		} else {
			feature.Metadata = make(map[string]interface{})
		}
		
		features = append(features, feature)
	}
	
	return features, nil
}

// Batch operations for efficiency
func (fs *Store) StoreBatch(ctx context.Context, features []Feature) error {
	// Store in Redis concurrently
	for _, feature := range features {
		go func(f Feature) {
			_ = fs.storeInRedis(ctx, f)
		}(feature)
	}
	
	// Batch insert into PostgreSQL
	return fs.storeBatchInPostgres(ctx, features)
}

func (fs *Store) storeBatchInPostgres(ctx context.Context, features []Feature) error {
	if len(features) == 0 {
		return nil
	}
	
	tx, err := fs.postgres.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	
	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO features (
			company_id, feature_name, feature_value, feature_type,
			source, timestamp, metadata
		) VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (company_id, feature_name, timestamp) 
		DO UPDATE SET 
			feature_value = EXCLUDED.feature_value,
			metadata = EXCLUDED.metadata
	`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	
	for _, feature := range features {
		// Convert feature value to float64
		var numericValue float64
		switch v := feature.Value.(type) {
		case float64:
			numericValue = v
		case float32:
			numericValue = float64(v)
		case int:
			numericValue = float64(v)
		case int64:
			numericValue = float64(v)
		default:
			numericValue = 0.0
		}
		
		// Convert metadata to JSON
		metadataJSON, err := json.Marshal(feature.Metadata)
		if err != nil {
			metadataJSON = []byte("{}")
		}
		
		_, err = stmt.ExecContext(
			ctx,
			feature.CompanyID,
			feature.Name,
			numericValue,
			feature.Type,
			feature.Source,
			feature.Timestamp,
			string(metadataJSON),
		)
		if err != nil {
			return err
		}
	}
	
	return tx.Commit()
}

// Utility functions
func (fs *Store) GetLatestFeatures(ctx context.Context, companyID string, count int) ([]Feature, error) {
	query := `
		SELECT 
			company_id, feature_name, feature_value, feature_type,
			source, timestamp, metadata
		FROM features
		WHERE company_id = $1 
		ORDER BY timestamp DESC 
		LIMIT $2
	`
	
	rows, err := fs.postgres.QueryContext(ctx, query, companyID, count)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	
	var features []Feature
	
	for rows.Next() {
		var feature Feature
		var metadataJSON string
		
		err := rows.Scan(
			&feature.CompanyID,
			&feature.Name,
			&feature.Value,
			&feature.Type,
			&feature.Source,
			&feature.Timestamp,
			&metadataJSON,
		)
		if err != nil {
			continue
		}
		
		// Parse metadata
		if metadataJSON != "" {
			json.Unmarshal([]byte(metadataJSON), &feature.Metadata)
		}
		
		features = append(features, feature)
	}
	
	return features, nil
}

func (fs *Store) DeleteExpiredFeatures(ctx context.Context, olderThan time.Time) error {
	query := `DELETE FROM features WHERE timestamp < $1`
	
	_, err := fs.postgres.ExecContext(ctx, query, olderThan)
	return err
} 