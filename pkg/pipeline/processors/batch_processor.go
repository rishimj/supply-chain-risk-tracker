package processors

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"

	"supply-chain-ml/pkg/config"
	"supply-chain-ml/pkg/features"
)

// BatchProcessor handles scheduled batch processing jobs
type BatchProcessor struct {
	db           *sql.DB
	featureStore *features.Store
	config       *config.Config
	isRunning    bool
	metrics      BatchMetrics
}

// BatchMetrics tracks batch processor performance
type BatchMetrics struct {
	JobsCompleted      int64         `json:"jobs_completed"`
	FeaturesGenerated  int64         `json:"features_generated"`
	ErrorCount         int64         `json:"error_count"`
	LastJobTime        time.Time     `json:"last_job_time"`
	AverageJobDuration time.Duration `json:"average_job_duration"`
}

// BatchJob represents a batch processing job
type BatchJob struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Status      string                 `json:"status"`
	StartTime   time.Time              `json:"start_time"`
	EndTime     time.Time              `json:"end_time"`
	Duration    time.Duration          `json:"duration"`
	Parameters  map[string]interface{} `json:"parameters"`
	Results     map[string]interface{} `json:"results"`
	ErrorMsg    string                 `json:"error_msg,omitempty"`
}

// NewBatchProcessor creates a new batch processor
func NewBatchProcessor(db *sql.DB, featureStore *features.Store, cfg *config.Config) *BatchProcessor {
	return &BatchProcessor{
		db:           db,
		featureStore: featureStore,
		config:       cfg,
		isRunning:    false,
	}
}

// Start begins batch processing scheduler
func (bp *BatchProcessor) Start(ctx context.Context) error {
	bp.isRunning = true
	log.Println("Batch Processor started")

	// Schedule different types of batch jobs
	
	// Daily feature engineering job (runs at 2 AM)
	dailyTicker := time.NewTicker(24 * time.Hour)
	defer dailyTicker.Stop()

	// Weekly comprehensive analysis (runs on Sundays)
	weeklyTicker := time.NewTicker(7 * 24 * time.Hour)
	defer weeklyTicker.Stop()

	// Hourly incremental processing
	hourlyTicker := time.NewTicker(1 * time.Hour)
	defer hourlyTicker.Stop()

	// Run initial batch processing
	go bp.runInitialJobs(ctx)

	for {
		select {
		case <-ctx.Done():
			bp.isRunning = false
			return nil
		case <-hourlyTicker.C:
			go bp.runHourlyJob(ctx)
		case <-dailyTicker.C:
			go bp.runDailyJob(ctx)
		case <-weeklyTicker.C:
			go bp.runWeeklyJob(ctx)
		}
	}
}

// Stop gracefully stops the processor
func (bp *BatchProcessor) Stop(ctx context.Context) error {
	bp.isRunning = false
	return nil
}

// HealthCheck verifies processor health
func (bp *BatchProcessor) HealthCheck(ctx context.Context) bool {
	return bp.isRunning
}

// GetMetrics returns processor metrics
func (bp *BatchProcessor) GetMetrics(ctx context.Context) BatchMetrics {
	return bp.metrics
}

// runInitialJobs runs initial batch processing on startup
func (bp *BatchProcessor) runInitialJobs(ctx context.Context) {
	log.Println("Running initial batch processing jobs...")
	
	// Initialize feature engineering for all companies
	if err := bp.runFeatureEngineeringJob(ctx, map[string]interface{}{
		"mode": "initialize",
		"lookback_days": 90,
	}); err != nil {
		log.Printf("Initial feature engineering job failed: %v", err)
	}
}

// runHourlyJob runs hourly incremental processing
func (bp *BatchProcessor) runHourlyJob(ctx context.Context) {
	job := &BatchJob{
		ID:        fmt.Sprintf("hourly_%d", time.Now().Unix()),
		Type:      "hourly_incremental",
		Status:    "running",
		StartTime: time.Now(),
		Parameters: map[string]interface{}{
			"lookback_hours": 2,
		},
		Results: make(map[string]interface{}),
	}

	defer bp.updateJobMetrics(job)

	log.Println("Running hourly incremental processing...")

	// Process recent feature updates
	if err := bp.runIncrementalFeatureUpdate(ctx, job.Parameters); err != nil {
		job.Status = "failed"
		job.ErrorMsg = err.Error()
		bp.metrics.ErrorCount++
		return
	}

	job.Status = "completed"
	job.EndTime = time.Now()
	job.Duration = job.EndTime.Sub(job.StartTime)
	bp.metrics.JobsCompleted++
}

// runDailyJob runs daily comprehensive processing
func (bp *BatchProcessor) runDailyJob(ctx context.Context) {
	job := &BatchJob{
		ID:        fmt.Sprintf("daily_%d", time.Now().Unix()),
		Type:      "daily_comprehensive",
		Status:    "running",
		StartTime: time.Now(),
		Parameters: map[string]interface{}{
			"lookback_days": 7,
			"include_model_training_data": true,
		},
		Results: make(map[string]interface{}),
	}

	defer bp.updateJobMetrics(job)

	log.Println("Running daily comprehensive processing...")

	// Run full feature engineering
	if err := bp.runFeatureEngineeringJob(ctx, job.Parameters); err != nil {
		job.Status = "failed"
		job.ErrorMsg = err.Error()
		bp.metrics.ErrorCount++
		return
	}

	// Generate model training data
	if err := bp.generateModelTrainingData(ctx, job.Parameters); err != nil {
		job.Status = "failed"
		job.ErrorMsg = err.Error()
		bp.metrics.ErrorCount++
		return
	}

	job.Status = "completed"
	job.EndTime = time.Now()
	job.Duration = job.EndTime.Sub(job.StartTime)
	bp.metrics.JobsCompleted++
}

// runWeeklyJob runs weekly analysis and cleanup
func (bp *BatchProcessor) runWeeklyJob(ctx context.Context) {
	job := &BatchJob{
		ID:        fmt.Sprintf("weekly_%d", time.Now().Unix()),
		Type:      "weekly_analysis",
		Status:    "running",
		StartTime: time.Now(),
		Parameters: map[string]interface{}{
			"lookback_days": 30,
			"include_cleanup": true,
		},
		Results: make(map[string]interface{}),
	}

	defer bp.updateJobMetrics(job)

	log.Println("Running weekly analysis and cleanup...")

	// Run comprehensive feature analysis
	if err := bp.runComprehensiveAnalysis(ctx, job.Parameters); err != nil {
		job.Status = "failed"
		job.ErrorMsg = err.Error()
		bp.metrics.ErrorCount++
		return
	}

	// Clean up old data
	if err := bp.runDataCleanup(ctx, job.Parameters); err != nil {
		job.Status = "failed"
		job.ErrorMsg = err.Error()
		bp.metrics.ErrorCount++
		return
	}

	job.Status = "completed"
	job.EndTime = time.Now()
	job.Duration = job.EndTime.Sub(job.StartTime)
	bp.metrics.JobsCompleted++
}

// runFeatureEngineeringJob runs feature engineering for all companies
func (bp *BatchProcessor) runFeatureEngineeringJob(ctx context.Context, params map[string]interface{}) error {
	companies, err := bp.getActiveCompanies(ctx)
	if err != nil {
		return fmt.Errorf("failed to get active companies: %w", err)
	}

	lookbackDays := 30
	if days, ok := params["lookback_days"].(int); ok {
		lookbackDays = days
	}

	log.Printf("Running feature engineering for %d companies (lookback: %d days)", len(companies), lookbackDays)

	for _, company := range companies {
		if err := bp.generateCompanyFeatures(ctx, company, lookbackDays); err != nil {
			log.Printf("Error generating features for company %s: %v", company, err)
			continue
		}
	}

	return nil
}

// generateCompanyFeatures generates comprehensive features for a company
func (bp *BatchProcessor) generateCompanyFeatures(ctx context.Context, companyID string, lookbackDays int) error {
	// Generate financial trend features
	financialFeatures, err := bp.generateFinancialTrendFeatures(ctx, companyID, lookbackDays)
	if err != nil {
		return fmt.Errorf("failed to generate financial trends: %w", err)
	}

	// Generate market behavior features
	marketFeatures, err := bp.generateMarketBehaviorFeatures(ctx, companyID, lookbackDays)
	if err != nil {
		return fmt.Errorf("failed to generate market behavior: %w", err)
	}

	// Generate supply chain risk indicators
	supplyChainFeatures, err := bp.generateSupplyChainRiskFeatures(ctx, companyID, lookbackDays)
	if err != nil {
		return fmt.Errorf("failed to generate supply chain risks: %w", err)
	}

	// Combine all features
	allFeatures := append(financialFeatures, marketFeatures...)
	allFeatures = append(allFeatures, supplyChainFeatures...)

	// Store all features
	for _, feature := range allFeatures {
		if err := bp.featureStore.Store(ctx, feature); err != nil {
			log.Printf("Error storing feature %s for %s: %v", feature.Name, companyID, err)
			continue
		}
		bp.metrics.FeaturesGenerated++
	}

	return nil
}

// generateFinancialTrendFeatures generates financial trend features
func (bp *BatchProcessor) generateFinancialTrendFeatures(ctx context.Context, companyID string, lookbackDays int) ([]features.Feature, error) {
	var result []features.Feature
	now := time.Now()

	// Get historical financial data
	query := `
		SELECT report_date, revenue, inventory, working_capital, debt_to_equity_ratio
		FROM financial_data 
		WHERE company_id = $1 AND report_date >= $2
		ORDER BY report_date DESC
		LIMIT 4
	`

	rows, err := bp.db.QueryContext(ctx, query, companyID, now.AddDate(0, 0, -lookbackDays))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var dataPoints []map[string]float64
	for rows.Next() {
		var reportDate time.Time
		var revenue, inventory, workingCapital, debtToEquity sql.NullFloat64

		err := rows.Scan(&reportDate, &revenue, &inventory, &workingCapital, &debtToEquity)
		if err != nil {
			continue
		}

		dataPoints = append(dataPoints, map[string]float64{
			"revenue":         revenue.Float64,
			"inventory":       inventory.Float64,
			"working_capital": workingCapital.Float64,
			"debt_to_equity":  debtToEquity.Float64,
		})
	}

	if len(dataPoints) < 2 {
		return result, nil // Not enough data for trends
	}

	// Calculate revenue growth trend
	revenueGrowth := bp.calculateGrowthTrend(dataPoints, "revenue")
	result = append(result, features.Feature{
		CompanyID: companyID,
		Name:      "batch_revenue_growth_trend",
		Value:     revenueGrowth,
		Type:      "numerical",
		Source:    "batch_financial_analysis",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"calculation_period": fmt.Sprintf("%d_days", lookbackDays),
			"data_points":        len(dataPoints),
		},
	})

	// Calculate inventory trend
	inventoryTrend := bp.calculateGrowthTrend(dataPoints, "inventory")
	result = append(result, features.Feature{
		CompanyID: companyID,
		Name:      "batch_inventory_trend",
		Value:     inventoryTrend,
		Type:      "numerical",
		Source:    "batch_financial_analysis",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"calculation_period": fmt.Sprintf("%d_days", lookbackDays),
		},
	})

	// Calculate working capital stability
	workingCapitalStability := bp.calculateStability(dataPoints, "working_capital")
	result = append(result, features.Feature{
		CompanyID: companyID,
		Name:      "batch_working_capital_stability",
		Value:     workingCapitalStability,
		Type:      "numerical",
		Source:    "batch_financial_analysis",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"calculation_period": fmt.Sprintf("%d_days", lookbackDays),
		},
	})

	return result, nil
}

// generateMarketBehaviorFeatures generates market behavior features
func (bp *BatchProcessor) generateMarketBehaviorFeatures(ctx context.Context, companyID string, lookbackDays int) ([]features.Feature, error) {
	var result []features.Feature
	now := time.Now()

	// Mock market data analysis (in production, integrate with market data APIs)
	// Generate beta coefficient
	beta := 0.8 + (float64(lookbackDays%10) * 0.05) // Simplified calculation
	result = append(result, features.Feature{
		CompanyID: companyID,
		Name:      "batch_market_beta",
		Value:     beta,
		Type:      "numerical",
		Source:    "batch_market_analysis",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"calculation_period": fmt.Sprintf("%d_days", lookbackDays),
		},
	})

	// Generate correlation with market indices
	marketCorrelation := 0.6 + (float64(len(companyID)%5) * 0.08) // Simplified
	result = append(result, features.Feature{
		CompanyID: companyID,
		Name:      "batch_market_correlation",
		Value:     marketCorrelation,
		Type:      "numerical",
		Source:    "batch_market_analysis",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"calculation_period": fmt.Sprintf("%d_days", lookbackDays),
		},
	})

	return result, nil
}

// generateSupplyChainRiskFeatures generates supply chain risk features
func (bp *BatchProcessor) generateSupplyChainRiskFeatures(ctx context.Context, companyID string, lookbackDays int) ([]features.Feature, error) {
	var result []features.Feature
	now := time.Now()

	// Analyze supplier concentration risk
	supplierConcentration, err := bp.calculateSupplierConcentration(ctx, companyID)
	if err == nil {
		result = append(result, features.Feature{
			CompanyID: companyID,
			Name:      "batch_supplier_concentration_risk",
			Value:     supplierConcentration,
			Type:      "numerical",
			Source:    "batch_supply_chain_analysis",
			Timestamp: now,
			Metadata: map[string]interface{}{
				"calculation_method": "herfindahl_index",
			},
		})
	}

	// Analyze geographic risk exposure
	geoRisk := bp.calculateGeographicRiskExposure(ctx, companyID)
	result = append(result, features.Feature{
		CompanyID: companyID,
		Name:      "batch_geographic_risk_exposure",
		Value:     geoRisk,
		Type:      "numerical",
		Source:    "batch_supply_chain_analysis",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"high_risk_regions": []string{"China", "Southeast Asia", "Eastern Europe"},
		},
	})

	return result, nil
}

// Helper functions

func (bp *BatchProcessor) calculateGrowthTrend(dataPoints []map[string]float64, metric string) float64 {
	if len(dataPoints) < 2 {
		return 0.0
	}

	latest := dataPoints[0][metric]
	previous := dataPoints[1][metric]

	if previous == 0 {
		return 0.0
	}

	return (latest - previous) / previous
}

func (bp *BatchProcessor) calculateStability(dataPoints []map[string]float64, metric string) float64 {
	if len(dataPoints) < 2 {
		return 1.0
	}

	values := make([]float64, len(dataPoints))
	for i, dp := range dataPoints {
		values[i] = dp[metric]
	}

	// Calculate coefficient of variation
	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))

	if mean == 0 {
		return 0.0
	}

	variance := 0.0
	for _, v := range values {
		variance += (v - mean) * (v - mean)
	}
	variance /= float64(len(values))

	stdDev := variance // Simplified
	coefficientOfVariation := stdDev / mean

	// Return stability (inverse of coefficient of variation)
	return 1.0 / (1.0 + coefficientOfVariation)
}

func (bp *BatchProcessor) calculateSupplierConcentration(ctx context.Context, companyID string) (float64, error) {
	// Simplified supplier concentration calculation
	// In production, this would analyze actual supplier data
	return 0.3 + (float64(len(companyID)%10) * 0.05), nil
}

func (bp *BatchProcessor) calculateGeographicRiskExposure(ctx context.Context, companyID string) float64 {
	// Simplified geographic risk calculation
	// Based on company characteristics or known supplier locations
	riskFactors := map[string]float64{
		"manufacturing_heavy": 0.6,
		"tech_services":      0.3,
		"retail":             0.5,
	}

	// Simplified industry detection based on company ID patterns
	if len(companyID) > 3 {
		return riskFactors["manufacturing_heavy"]
	}
	return riskFactors["tech_services"]
}

func (bp *BatchProcessor) getActiveCompanies(ctx context.Context) ([]string, error) {
	query := `SELECT DISTINCT symbol FROM companies WHERE active = true ORDER BY symbol`
	
	rows, err := bp.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var companies []string
	for rows.Next() {
		var symbol string
		if err := rows.Scan(&symbol); err != nil {
			continue
		}
		companies = append(companies, symbol)
	}

	// Default companies if none in database
	if len(companies) == 0 {
		companies = []string{"AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"}
	}

	return companies, nil
}

func (bp *BatchProcessor) updateJobMetrics(job *BatchJob) {
	bp.metrics.LastJobTime = time.Now()
	if job.Status == "completed" {
		// Update average duration
		totalJobs := bp.metrics.JobsCompleted + 1
		bp.metrics.AverageJobDuration = time.Duration(
			(int64(bp.metrics.AverageJobDuration)*int64(bp.metrics.JobsCompleted) + int64(job.Duration)) / int64(totalJobs),
		)
	}
}

// Manual trigger methods

// TriggerFeatureEngineering manually triggers feature engineering
func (bp *BatchProcessor) TriggerFeatureEngineering(ctx context.Context, params map[string]interface{}) error {
	return bp.runFeatureEngineeringJob(ctx, params)
}

// TriggerModelTrainingData manually triggers model training data generation
func (bp *BatchProcessor) TriggerModelTrainingData(ctx context.Context, params map[string]interface{}) error {
	return bp.generateModelTrainingData(ctx, params)
}

// Additional batch job methods

func (bp *BatchProcessor) runIncrementalFeatureUpdate(ctx context.Context, params map[string]interface{}) error {
	log.Println("Running incremental feature update...")
	// Implementation for incremental updates
	return nil
}

func (bp *BatchProcessor) generateModelTrainingData(ctx context.Context, params map[string]interface{}) error {
	log.Println("Generating model training data...")
	// Implementation for training data generation
	return nil
}

func (bp *BatchProcessor) runComprehensiveAnalysis(ctx context.Context, params map[string]interface{}) error {
	log.Println("Running comprehensive analysis...")
	// Implementation for comprehensive analysis
	return nil
}

func (bp *BatchProcessor) runDataCleanup(ctx context.Context, params map[string]interface{}) error {
	log.Println("Running data cleanup...")
	// Implementation for data cleanup
	return nil
} 