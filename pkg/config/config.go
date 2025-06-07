package config

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Config represents the application configuration
type Config struct {
	Server      ServerConfig      `yaml:"server"`
	Database    DatabaseConfig    `yaml:"database"`
	Streaming   StreamingConfig   `yaml:"streaming"`
	Features    FeaturesConfig    `yaml:"features"`
	ML          MLConfig          `yaml:"ml"`
	Monitoring  MonitoringConfig  `yaml:"monitoring"`
	ExternalAPIs ExternalAPIsConfig `yaml:"external_apis"`
	Environment string            `yaml:"environment"`
}

type ServerConfig struct {
	Host    string        `yaml:"host"`
	Port    int           `yaml:"port"`
	Timeout time.Duration `yaml:"timeout"`
}

type DatabaseConfig struct {
	Postgres PostgresConfig `yaml:"postgres"`
	Redis    RedisConfig    `yaml:"redis"`
	Neo4j    Neo4jConfig    `yaml:"neo4j"`
}

type PostgresConfig struct {
	Host           string `yaml:"host"`
	Port           int    `yaml:"port"`
	Database       string `yaml:"database"`
	Username       string `yaml:"username"`
	Password       string `yaml:"password"`
	SSLMode        string `yaml:"ssl_mode"`
	MaxConnections int    `yaml:"max_connections"`
}

type RedisConfig struct {
	Host           string `yaml:"host"`
	Port           int    `yaml:"port"`
	Password       string `yaml:"password"`
	DB             int    `yaml:"db"`
	MaxConnections int    `yaml:"max_connections"`
}

type Neo4jConfig struct {
	URI            string `yaml:"uri"`
	Username       string `yaml:"username"`
	Password       string `yaml:"password"`
	MaxConnections int    `yaml:"max_connections"`
}

type StreamingConfig struct {
	Kafka        KafkaConfig   `yaml:"kafka"`
	BatchSize    int           `yaml:"batch_size"`
	FlushInterval time.Duration `yaml:"flush_interval"`
}

type KafkaConfig struct {
	Brokers       []string          `yaml:"brokers"`
	ConsumerGroup string            `yaml:"consumer_group"`
	Topics        map[string]string `yaml:"topics"`
}

type FeaturesConfig struct {
	CacheTTL             time.Duration `yaml:"cache_ttl"`
	MaxFeatures          int           `yaml:"max_features"`
	FeatureStoreEndpoint string        `yaml:"feature_store_endpoint"`
}

type MLConfig struct {
	ModelServerEndpoint string        `yaml:"model_server_endpoint"`
	ModelVersion        string        `yaml:"model_version"`
	PredictionCacheTTL  time.Duration `yaml:"prediction_cache_ttl"`
	RetrainInterval     time.Duration `yaml:"retrain_interval"`
}

type MonitoringConfig struct {
	Prometheus PrometheusConfig `yaml:"prometheus"`
	Logging    LoggingConfig    `yaml:"logging"`
}

type PrometheusConfig struct {
	Endpoint       string        `yaml:"endpoint"`
	ScrapeInterval time.Duration `yaml:"scrape_interval"`
}

type LoggingConfig struct {
	Level  string `yaml:"level"`
	Format string `yaml:"format"`
	Output string `yaml:"output"`
}

type ExternalAPIsConfig struct {
	SECEDGAR      SECEDGARConfig      `yaml:"sec_edgar"`
	FinancialData FinancialDataConfig `yaml:"financial_data"`
	NewsAPI       NewsAPIConfig       `yaml:"news_api"`
}

type SECEDGARConfig struct {
	BaseURL   string `yaml:"base_url"`
	RateLimit int    `yaml:"rate_limit"`
}

type FinancialDataConfig struct {
	AlphaVantageKey string `yaml:"alpha_vantage_key"`
	FinnhubKey      string `yaml:"finnhub_key"`
}

type NewsAPIConfig struct {
	NewsAPIKey string `yaml:"newsapi_key"`
}

// Load reads and parses the configuration file
func Load(configPath string) (*Config, error) {
	config := &Config{}
	
	// Read config file
	file, err := os.Open(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()
	
	// Parse YAML
	decoder := yaml.NewDecoder(file)
	if err := decoder.Decode(config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}
	
	// Expand environment variables
	if err := expandEnvVars(config); err != nil {
		return nil, fmt.Errorf("failed to expand environment variables: %w", err)
	}
	
	// Validate configuration
	if err := validate(config); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}
	
	return config, nil
}

// expandEnvVars expands environment variables in string fields
func expandEnvVars(config *Config) error {
	config.ExternalAPIs.FinancialData.AlphaVantageKey = os.ExpandEnv(config.ExternalAPIs.FinancialData.AlphaVantageKey)
	config.ExternalAPIs.FinancialData.FinnhubKey = os.ExpandEnv(config.ExternalAPIs.FinancialData.FinnhubKey)
	config.ExternalAPIs.NewsAPI.NewsAPIKey = os.ExpandEnv(config.ExternalAPIs.NewsAPI.NewsAPIKey)
	
	return nil
}

// validate ensures the configuration is valid
func validate(config *Config) error {
	if config.Server.Port <= 0 || config.Server.Port > 65535 {
		return fmt.Errorf("invalid server port: %d", config.Server.Port)
	}
	
	if config.Database.Postgres.Host == "" {
		return fmt.Errorf("postgres host is required")
	}
	
	if config.Database.Redis.Host == "" {
		return fmt.Errorf("redis host is required")
	}
	
	if config.Database.Neo4j.URI == "" {
		return fmt.Errorf("neo4j URI is required")
	}
	
	if len(config.Streaming.Kafka.Brokers) == 0 {
		return fmt.Errorf("at least one kafka broker is required")
	}
	
	return nil
} 