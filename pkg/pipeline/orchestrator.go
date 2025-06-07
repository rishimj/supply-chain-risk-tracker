package pipeline

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"supply-chain-ml/pkg/config"
	"supply-chain-ml/pkg/features"
	"supply-chain-ml/pkg/pipeline/processors"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/redis/go-redis/v9"
)

// Orchestrator manages all data pipeline services
type Orchestrator struct {
	postgres     *sql.DB
	redis        *redis.Client
	neo4j        neo4j.DriverWithContext
	featureStore *features.Store
	config       *config.Config

	// Individual processors
	secProcessor       *processors.SECProcessor
	earningsAnalyzer   *processors.EarningsAnalyzer
	financialIngester  *processors.FinancialIngester
	newsAnalyzer       *processors.NewsAnalyzer
	networkAnalyzer    *processors.NetworkAnalyzer
	streamProcessor    *processors.StreamProcessor
	batchProcessor     *processors.BatchProcessor
	featureMonitor     *processors.FeatureMonitor
}

// NewOrchestrator creates a new pipeline orchestrator
func NewOrchestrator(
	postgres *sql.DB,
	redis *redis.Client,
	neo4j neo4j.DriverWithContext,
	featureStore *features.Store,
	cfg *config.Config,
) *Orchestrator {
	return &Orchestrator{
		postgres:           postgres,
		redis:              redis,
		neo4j:              neo4j,
		featureStore:       featureStore,
		config:             cfg,
		secProcessor:       processors.NewSECProcessor(postgres, featureStore, cfg),
		earningsAnalyzer:   processors.NewEarningsAnalyzer(postgres, featureStore, cfg),
		financialIngester:  processors.NewFinancialIngester(postgres, featureStore, cfg),
		newsAnalyzer:       processors.NewNewsAnalyzer(postgres, featureStore, cfg),
		networkAnalyzer:    processors.NewNetworkAnalyzer(postgres, neo4j, featureStore, cfg),
		streamProcessor:    processors.NewStreamProcessor(redis, featureStore, cfg),
		batchProcessor:     processors.NewBatchProcessor(postgres, featureStore, cfg),
		featureMonitor:     processors.NewFeatureMonitor(postgres, redis, featureStore, cfg),
	}
}

// StartSECProcessor starts the SEC filings processor
func (o *Orchestrator) StartSECProcessor(ctx context.Context) error {
	return o.secProcessor.Start(ctx)
}

// StartEarningsAnalyzer starts the earnings call analyzer
func (o *Orchestrator) StartEarningsAnalyzer(ctx context.Context) error {
	return o.earningsAnalyzer.Start(ctx)
}

// StartFinancialDataIngestion starts financial data ingestion
func (o *Orchestrator) StartFinancialDataIngestion(ctx context.Context) error {
	return o.financialIngester.Start(ctx)
}

// StartNewsSentimentAnalysis starts news sentiment analysis
func (o *Orchestrator) StartNewsSentimentAnalysis(ctx context.Context) error {
	return o.newsAnalyzer.Start(ctx)
}

// StartSupplierNetworkAnalysis starts supplier network analysis
func (o *Orchestrator) StartSupplierNetworkAnalysis(ctx context.Context) error {
	return o.networkAnalyzer.Start(ctx)
}

// StartStreamProcessor starts real-time stream processing
func (o *Orchestrator) StartStreamProcessor(ctx context.Context) error {
	return o.streamProcessor.Start(ctx)
}

// StartBatchProcessor starts batch feature engineering
func (o *Orchestrator) StartBatchProcessor(ctx context.Context) error {
	return o.batchProcessor.Start(ctx)
}

// StartFeatureMonitoring starts feature quality monitoring
func (o *Orchestrator) StartFeatureMonitoring(ctx context.Context) error {
	return o.featureMonitor.Start(ctx)
}

// HealthCheck verifies all processors are healthy
func (o *Orchestrator) HealthCheck(ctx context.Context) map[string]bool {
	health := make(map[string]bool)
	
	health["sec_processor"] = o.secProcessor.HealthCheck(ctx)
	health["earnings_analyzer"] = o.earningsAnalyzer.HealthCheck(ctx)
	health["financial_ingester"] = o.financialIngester.HealthCheck(ctx)
	health["news_analyzer"] = o.newsAnalyzer.HealthCheck(ctx)
	health["network_analyzer"] = o.networkAnalyzer.HealthCheck(ctx)
	health["stream_processor"] = o.streamProcessor.HealthCheck(ctx)
	health["batch_processor"] = o.batchProcessor.HealthCheck(ctx)
	health["feature_monitor"] = o.featureMonitor.HealthCheck(ctx)
	
	return health
}

// GetMetrics returns metrics from all processors
func (o *Orchestrator) GetMetrics(ctx context.Context) map[string]interface{} {
	metrics := make(map[string]interface{})
	
	metrics["sec_processor"] = o.secProcessor.GetMetrics(ctx)
	metrics["earnings_analyzer"] = o.earningsAnalyzer.GetMetrics(ctx)
	metrics["financial_ingester"] = o.financialIngester.GetMetrics(ctx)
	metrics["news_analyzer"] = o.newsAnalyzer.GetMetrics(ctx)
	metrics["network_analyzer"] = o.networkAnalyzer.GetMetrics(ctx)
	metrics["stream_processor"] = o.streamProcessor.GetMetrics(ctx)
	metrics["batch_processor"] = o.batchProcessor.GetMetrics(ctx)
	metrics["feature_monitor"] = o.featureMonitor.GetMetrics(ctx)
	metrics["timestamp"] = time.Now()
	
	return metrics
}

// Stop gracefully stops all processors
func (o *Orchestrator) Stop(ctx context.Context) error {
	// Stop all processors concurrently
	done := make(chan error, 8)
	
	go func() { done <- o.secProcessor.Stop(ctx) }()
	go func() { done <- o.earningsAnalyzer.Stop(ctx) }()
	go func() { done <- o.financialIngester.Stop(ctx) }()
	go func() { done <- o.newsAnalyzer.Stop(ctx) }()
	go func() { done <- o.networkAnalyzer.Stop(ctx) }()
	go func() { done <- o.streamProcessor.Stop(ctx) }()
	go func() { done <- o.batchProcessor.Stop(ctx) }()
	go func() { done <- o.featureMonitor.Stop(ctx) }()
	
	// Wait for all to complete
	for i := 0; i < 8; i++ {
		if err := <-done; err != nil {
			return err
		}
	}
	
	return nil
}

// TriggerBatchJob manually triggers a batch processing job
func (o *Orchestrator) TriggerBatchJob(ctx context.Context, jobType string, params map[string]interface{}) error {
	switch jobType {
	case "feature_engineering":
		return o.batchProcessor.TriggerFeatureEngineering(ctx, params)
	case "model_training_data":
		return o.batchProcessor.TriggerModelTrainingData(ctx, params)
	case "data_quality_check":
		return o.featureMonitor.TriggerQualityCheck(ctx, params)
	default:
		return ErrUnknownJobType
	}
}

// PipelineStatus represents the overall pipeline status
type PipelineStatus struct {
	Status       string                 `json:"status"`
	Timestamp    time.Time              `json:"timestamp"`
	Processors   map[string]bool        `json:"processors"`
	Metrics      map[string]interface{} `json:"metrics"`
	LastUpdated  time.Time              `json:"last_updated"`
}

// GetStatus returns comprehensive pipeline status
func (o *Orchestrator) GetStatus(ctx context.Context) *PipelineStatus {
	health := o.HealthCheck(ctx)
	metrics := o.GetMetrics(ctx)
	
	// Determine overall status
	allHealthy := true
	for _, healthy := range health {
		if !healthy {
			allHealthy = false
			break
		}
	}
	
	status := "healthy"
	if !allHealthy {
		status = "degraded"
	}
	
	return &PipelineStatus{
		Status:      status,
		Timestamp:   time.Now(),
		Processors:  health,
		Metrics:     metrics,
		LastUpdated: time.Now(),
	}
}

// Error definitions
var (
	ErrUnknownJobType = fmt.Errorf("unknown job type")
) 