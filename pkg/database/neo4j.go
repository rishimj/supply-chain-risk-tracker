package database

import (
	"context"
	"fmt"

	"supply-chain-ml/pkg/config"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// NewNeo4jDriver creates a new Neo4j driver connection
func NewNeo4jDriver(cfg config.Neo4jConfig) (neo4j.DriverWithContext, error) {
	driver, err := neo4j.NewDriverWithContext(
		cfg.URI,
		neo4j.BasicAuth(cfg.Username, cfg.Password, ""),
		func(config *neo4j.Config) {
			config.MaxConnectionPoolSize = cfg.MaxConnections
		},
	)

	if err != nil {
		return nil, fmt.Errorf("failed to create Neo4j driver: %w", err)
	}

	// Test the connection
	ctx := context.Background()
	if err := driver.VerifyConnectivity(ctx); err != nil {
		return nil, fmt.Errorf("failed to verify Neo4j connectivity: %w", err)
	}

	return driver, nil
} 