package api

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"supply-chain-ml/pkg/features"
	"supply-chain-ml/pkg/health"
	"supply-chain-ml/pkg/ml"
	"supply-chain-ml/pkg/monitoring"

	"github.com/gin-gonic/gin"
	"github.com/lib/pq"
)

// Stock price API configuration
const (
	ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
	API_TIMEOUT = 10 * time.Second
)

var (
	// API key from environment variable or default to demo
	ALPHA_VANTAGE_API_KEY = getEnvOrDefault("ALPHA_VANTAGE_API_KEY", "demo")
	// Enable/disable real API calls
	ENABLE_REAL_STOCK_DATA = getEnvOrDefault("ENABLE_REAL_STOCK_DATA", "true") == "true"
)

// getEnvOrDefault gets environment variable or returns default value
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// Alpha Vantage API response structures
type AlphaVantageQuote struct {
	GlobalQuote struct {
		Symbol           string `json:"01. symbol"`
		Open             string `json:"02. open"`
		High             string `json:"03. high"`
		Low              string `json:"04. low"`
		Price            string `json:"05. price"`
		Volume           string `json:"06. volume"`
		LatestTradingDay string `json:"07. latest trading day"`
		PreviousClose    string `json:"08. previous close"`
		Change           string `json:"09. change"`
		ChangePercent    string `json:"10. change percent"`
	} `json:"Global Quote"`
}

type StockData struct {
	Symbol            string
	Price             float64
	Change            float64
	ChangePercent     float64
	PreviousClose     float64
	Volume            int64
	LatestTradingDay  string
	IsRealData        bool
	Error             string
}

// Handlers contains all HTTP handlers for the API
type Handlers struct {
	featureStore     *features.Store
	modelServer      *ml.ModelServer
	healthChecker    *health.HealthChecker
	metricsCollector *monitoring.MetricsCollector
	postgres         *sql.DB
}

// NewHandlers creates a new instance of API handlers
func NewHandlers(
	featureStore *features.Store,
	modelServer *ml.ModelServer,
	healthChecker *health.HealthChecker,
	metricsCollector *monitoring.MetricsCollector,
	postgres *sql.DB,
) *Handlers {
	return &Handlers{
		featureStore:     featureStore,
		modelServer:      modelServer,
		healthChecker:    healthChecker,
		metricsCollector: metricsCollector,
		postgres:         postgres,
	}
}

// Request/Response models
type PredictRequest struct {
	CompanyID string             `json:"company_id" binding:"required"`
	Features  map[string]float64 `json:"features" binding:"required"`
	Options   *PredictOptions    `json:"options,omitempty"`
}

type PredictOptions struct {
	IncludeFeatureImportance bool `json:"include_feature_importance"`
	IncludeComponentRisks    bool `json:"include_component_risks"`
	ModelVersion             string `json:"model_version,omitempty"`
}

type PredictResponse struct {
	CompanyID               string             `json:"company_id"`
	RiskScore              float64            `json:"risk_score"`
	GuidanceMissProbability float64            `json:"guidance_miss_probability"`
	Confidence             float64            `json:"confidence"`
	PredictionTimestamp    time.Time          `json:"prediction_timestamp"`
	ComponentRisks         *ComponentRisks    `json:"component_risks,omitempty"`
	RiskFactors            []RiskFactor       `json:"risk_factors,omitempty"`
	FeatureImportance      map[string]float64 `json:"feature_importance,omitempty"`
	ModelVersion           string             `json:"model_version"`
	PredictionID           string             `json:"prediction_id"`
}

type ComponentRisks struct {
	FinancialRisk float64 `json:"financial_risk"`
	NetworkRisk   float64 `json:"network_risk"`
	TemporalRisk  float64 `json:"temporal_risk"`
	SentimentRisk float64 `json:"sentiment_risk"`
}

type RiskFactor struct {
	Name        string  `json:"name"`
	Impact      float64 `json:"impact"`
	Description string  `json:"description"`
	Category    string  `json:"category"`
}

type BatchPredictRequest struct {
	Companies []CompanyFeatures `json:"companies" binding:"required"`
	Options   *PredictOptions   `json:"options,omitempty"`
}

type CompanyFeatures struct {
	CompanyID string             `json:"company_id" binding:"required"`
	Features  map[string]float64 `json:"features" binding:"required"`
}

type BatchPredictResponse struct {
	Predictions []PredictResponse `json:"predictions"`
	Summary     BatchSummary      `json:"summary"`
}

type BatchSummary struct {
	TotalCompanies    int     `json:"total_companies"`
	SuccessfulCount   int     `json:"successful_count"`
	FailedCount       int     `json:"failed_count"`
	AverageRiskScore  float64 `json:"average_risk_score"`
	HighRiskCount     int     `json:"high_risk_count"`
	ProcessingTimeMs  int64   `json:"processing_time_ms"`
}

// Health check endpoint
// @Summary Health check
// @Description Get the health status of the API
// @Tags health
// @Accept json
// @Produce json
// @Success 200 {object} map[string]interface{} "Health status"
// @Router /health [get]
func (h *Handlers) Health(c *gin.Context) {
	ctx := c.Request.Context()
	
	healthStatus := h.healthChecker.CheckHealth(ctx)
	
	status := http.StatusOK
	if healthStatus.Status != "healthy" {
		status = http.StatusServiceUnavailable
	}
	
	c.JSON(status, healthStatus)
}

// Simple test endpoint
func (h *Handlers) TestEndpoint(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message": "test endpoint works",
		"timestamp": time.Now(),
	})
}

// PredictRisk handles single company risk prediction
// @Summary Predict company risk
// @Description Generate supply chain risk prediction for a company
// @Tags predictions
// @Accept json
// @Produce json
// @Param request body PredictRequest true "Prediction request"
// @Success 200 {object} PredictResponse "Prediction result"
// @Failure 400 {object} map[string]interface{} "Bad request"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /predictions/predict [post]
func (h *Handlers) PredictRisk(c *gin.Context) {
	startTime := time.Now()
	ctx := c.Request.Context()
	
	var req PredictRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request format",
			"details": err.Error(),
		})
		return
	}
	
	// Validate company ID
	if req.CompanyID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "company_id is required",
		})
		return
	}
	
	// Validate features
	if len(req.Features) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "features are required",
		})
		return
	}
	
	// Create feature vector
	featureVector := &features.FeatureVector{
		CompanyID: req.CompanyID,
		Features:  req.Features,
		Timestamp: time.Now(),
		Version:   "v1",
	}
	
	// Get prediction from model server
	prediction, err := h.modelServer.Predict(ctx, featureVector)
	if err != nil {
		h.metricsCollector.RecordPrediction(ctx, "unknown", req.CompanyID, time.Since(startTime), false)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Prediction failed",
			"details": err.Error(),
		})
		return
	}
	
	// Extract risk factors if needed
	var riskFactors []RiskFactor
	if req.Options == nil || req.Options.IncludeFeatureImportance {
		riskFactors = h.extractRiskFactors(featureVector, prediction)
	}
	
	// Build response
	response := PredictResponse{
		CompanyID:               req.CompanyID,
		RiskScore:              prediction.RiskScore,
		GuidanceMissProbability: prediction.GuidanceMissProbability,
		Confidence:             prediction.Confidence,
		PredictionTimestamp:    time.Now(),
		RiskFactors:            riskFactors,
		ModelVersion:           prediction.ModelVersion,
		PredictionID:           prediction.PredictionID,
	}
	
	// Add component risks if requested
	if req.Options != nil && req.Options.IncludeComponentRisks {
		response.ComponentRisks = &ComponentRisks{
			FinancialRisk: prediction.ComponentRisks["financial_risk"],
			NetworkRisk:   prediction.ComponentRisks["network_risk"],
			TemporalRisk:  prediction.ComponentRisks["temporal_risk"],
			SentimentRisk: prediction.ComponentRisks["sentiment_risk"],
		}
	}
	
	// Add feature importance if requested
	if req.Options != nil && req.Options.IncludeFeatureImportance {
		response.FeatureImportance = prediction.FeatureImportance
	}
	
	// Store prediction for monitoring
	h.storePrediction(ctx, req.CompanyID, prediction)
	
	// Record metrics
	h.metricsCollector.RecordPrediction(ctx, prediction.ModelVersion, req.CompanyID, time.Since(startTime), true)
	
	c.JSON(http.StatusOK, response)
}

// BatchPredict handles multiple company predictions
func (h *Handlers) BatchPredict(c *gin.Context) {
	startTime := time.Now()
	ctx := c.Request.Context()
	
	var req BatchPredictRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request format",
			"details": err.Error(),
		})
		return
	}
	
	if len(req.Companies) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "companies list cannot be empty",
		})
		return
	}
	
	if len(req.Companies) > 100 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "maximum 100 companies allowed per batch request",
		})
		return
	}
	
	var predictions []PredictResponse
	var successCount, failCount int
	var totalRiskScore float64
	var highRiskCount int
	
	// Process each company
	for _, company := range req.Companies {
		featureVector := &features.FeatureVector{
			CompanyID: company.CompanyID,
			Features:  company.Features,
			Timestamp: time.Now(),
			Version:   "v1",
		}
		
		prediction, err := h.modelServer.Predict(ctx, featureVector)
		if err != nil {
			failCount++
			continue
		}
		
		// Build prediction response
		predResponse := PredictResponse{
			CompanyID:               company.CompanyID,
			RiskScore:              prediction.RiskScore,
			GuidanceMissProbability: prediction.GuidanceMissProbability,
			Confidence:             prediction.Confidence,
			PredictionTimestamp:    time.Now(),
			ModelVersion:           prediction.ModelVersion,
			PredictionID:           prediction.PredictionID,
		}
		
		// Add optional fields if requested
		if req.Options != nil && req.Options.IncludeComponentRisks {
			predResponse.ComponentRisks = &ComponentRisks{
				FinancialRisk: prediction.ComponentRisks["financial_risk"],
				NetworkRisk:   prediction.ComponentRisks["network_risk"],
				TemporalRisk:  prediction.ComponentRisks["temporal_risk"],
				SentimentRisk: prediction.ComponentRisks["sentiment_risk"],
			}
		}
		
		if req.Options != nil && req.Options.IncludeFeatureImportance {
			predResponse.FeatureImportance = prediction.FeatureImportance
			predResponse.RiskFactors = h.extractRiskFactors(featureVector, prediction)
		}
		
		predictions = append(predictions, predResponse)
		successCount++
		totalRiskScore += prediction.RiskScore
		
		if prediction.RiskScore > 70.0 { // High risk threshold
			highRiskCount++
		}
		
		// Store prediction
		h.storePrediction(ctx, company.CompanyID, prediction)
	}
	
	// Calculate summary
	averageRiskScore := float64(0)
	if successCount > 0 {
		averageRiskScore = totalRiskScore / float64(successCount)
	}
	
	summary := BatchSummary{
		TotalCompanies:   len(req.Companies),
		SuccessfulCount:  successCount,
		FailedCount:      failCount,
		AverageRiskScore: averageRiskScore,
		HighRiskCount:    highRiskCount,
		ProcessingTimeMs: time.Since(startTime).Milliseconds(),
	}
	
	response := BatchPredictResponse{
		Predictions: predictions,
		Summary:     summary,
	}
	
	c.JSON(http.StatusOK, response)
}

// GetCompanyRisk retrieves the latest risk assessment for a company
func (h *Handlers) GetCompanyRisk(c *gin.Context) {
	companyID := c.Param("id")
	ctx := c.Request.Context()
	
	if companyID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "company ID is required",
		})
		return
	}
	
	// Get latest prediction from cache/database
	latestPrediction, err := h.getLatestPrediction(ctx, companyID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{
			"error": "No recent prediction found for company",
			"company_id": companyID,
		})
		return
	}
	
	c.JSON(http.StatusOK, latestPrediction)
}

// GetCompanyFeatures retrieves current features for a company
func (h *Handlers) GetCompanyFeatures(c *gin.Context) {
	companyID := c.Param("id")
	ctx := c.Request.Context()
	
	if companyID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "company ID is required",
		})
		return
	}
	
	// Get current timestamp or from query parameter
	timestamp := time.Now()
	if timestampStr := c.Query("timestamp"); timestampStr != "" {
		if ts, err := time.Parse(time.RFC3339, timestampStr); err == nil {
			timestamp = ts
		}
	}
	
	// Retrieve feature vector
	featureVector, err := h.featureStore.GetFeatureVector(ctx, companyID, timestamp)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{
			"error": "Features not found for company",
			"company_id": companyID,
		})
		return
	}
	
	c.JSON(http.StatusOK, featureVector)
}

// GetFeatures retrieves features for a company with optional filtering
func (h *Handlers) GetFeatures(c *gin.Context) {
	companyID := c.Param("company_id")
	ctx := c.Request.Context()
	
	// Parse query parameters
	featureName := c.Query("feature_name")
	fromDateStr := c.Query("from_date")
	toDateStr := c.Query("to_date")
	source := c.Query("source")
	
	// Set default time range if not provided
	toDate := time.Now()
	fromDate := toDate.AddDate(0, 0, -30) // Default to last 30 days
	
	if fromDateStr != "" {
		if fd, err := time.Parse("2006-01-02", fromDateStr); err == nil {
			fromDate = fd
		}
	}
	
	if toDateStr != "" {
		if td, err := time.Parse("2006-01-02", toDateStr); err == nil {
			toDate = td
		}
	}
	
	// Get features from store
	features, err := h.featureStore.GetFeatures(ctx, companyID, featureName, source, fromDate, toDate)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to retrieve features",
			"details": err.Error(),
		})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"company_id": companyID,
		"features": features,
		"count": len(features),
		"from_date": fromDate.Format("2006-01-02"),
		"to_date": toDate.Format("2006-01-02"),
	})
}

// StoreFeatures stores new features for a company
func (h *Handlers) StoreFeatures(c *gin.Context) {
	ctx := c.Request.Context()
	
	var req struct {
		CompanyID string                           `json:"company_id" binding:"required"`
		Features  []features.Feature               `json:"features" binding:"required"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid request format",
			"details": err.Error(),
		})
		return
	}
	
	// Store each feature
	var stored, failed int
	for _, feature := range req.Features {
		feature.CompanyID = req.CompanyID
		if feature.Timestamp.IsZero() {
			feature.Timestamp = time.Now()
		}
		
		if err := h.featureStore.Store(ctx, feature); err != nil {
			failed++
		} else {
			stored++
		}
	}
	
	c.JSON(http.StatusOK, gin.H{
		"message": "Features processed",
		"stored": stored,
		"failed": failed,
		"total": len(req.Features),
	})
}

// GetModelStatus returns the current model status and metadata
func (h *Handlers) GetModelStatus(c *gin.Context) {
	ctx := c.Request.Context()
	
	status := h.modelServer.GetStatus(ctx)
	
	c.JSON(http.StatusOK, status)
}

// TriggerRetrain triggers model retraining
func (h *Handlers) TriggerRetrain(c *gin.Context) {
	ctx := c.Request.Context()
	
	var req struct {
		ModelVersion string `json:"model_version,omitempty"`
		Force        bool   `json:"force,omitempty"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		// Allow empty body for simple trigger
	}
	
	// Trigger retraining (this would typically be asynchronous)
	retrainID, err := h.modelServer.TriggerRetrain(ctx, req.ModelVersion, req.Force)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to trigger retraining",
			"details": err.Error(),
		})
		return
	}
	
	c.JSON(http.StatusAccepted, gin.H{
		"message": "Retraining triggered",
		"retrain_id": retrainID,
		"status": "started",
	})
}

// GetMetrics returns system metrics
func (h *Handlers) GetMetrics(c *gin.Context) {
	ctx := c.Request.Context()
	
	metrics := h.metricsCollector.GetMetrics(ctx)
	
	c.JSON(http.StatusOK, metrics)
}

// GetSystemStatus returns overall system status
func (h *Handlers) GetSystemStatus(c *gin.Context) {
	ctx := c.Request.Context()
	
	// Collect status from all components
	health := h.healthChecker.CheckHealth(ctx)
	modelStatus := h.modelServer.GetStatus(ctx)
	metrics := h.metricsCollector.GetSystemMetrics(ctx)
	
	systemStatus := gin.H{
		"timestamp": time.Now(),
		"health": health,
		"model": modelStatus,
		"metrics": metrics,
		"version": "1.0.0",
	}
	
	status := http.StatusOK
	if health.Status != "healthy" {
		status = http.StatusServiceUnavailable
	}
	
	c.JSON(status, systemStatus)
}

// Helper methods

func (h *Handlers) extractRiskFactors(fv *features.FeatureVector, prediction *ml.Prediction) []RiskFactor {
	var riskFactors []RiskFactor
	
	// Extract top risk factors based on feature importance
	for featureName, importance := range prediction.FeatureImportance {
		if importance > 0.05 { // Threshold for significant features
			featureValue := fv.Features[featureName]
			
			riskFactor := RiskFactor{
				Name:        featureName,
				Impact:      importance * featureValue,
				Description: h.getFeatureDescription(featureName),
				Category:    h.getFeatureCategory(featureName),
			}
			
			riskFactors = append(riskFactors, riskFactor)
		}
	}
	
	return riskFactors
}

func (h *Handlers) getFeatureDescription(featureName string) string {
	descriptions := map[string]string{
		"financial_inventory_turnover": "Inventory turnover ratio indicates supply chain efficiency",
		"financial_gross_margin": "Gross margin shows pricing power and cost management",
		"network_supplier_concentration": "Supplier concentration indicates supply chain risk",
		"ts_volatility_30d": "30-day volatility indicates market uncertainty",
		"nlp_sentiment_score": "Sentiment from earnings calls and filings",
	}
	
	if desc, exists := descriptions[featureName]; exists {
		return desc
	}
	return "Feature impact on supply chain risk"
}

func (h *Handlers) getFeatureCategory(featureName string) string {
	if len(featureName) > 9 {
		prefix := featureName[:9]
		switch prefix {
		case "financial":
			return "Financial"
		case "network_":
			return "Network"
		case "ts_":
			return "Time Series"
		case "nlp_":
			return "Natural Language"
		default:
			return "Other"
		}
	}
	return "Other"
}

func (h *Handlers) storePrediction(ctx context.Context, companyID string, prediction *ml.Prediction) error {
	// Store prediction in database/cache for later retrieval
	// This would typically involve database operations
	return nil
}

func (h *Handlers) getLatestPrediction(ctx context.Context, companyID string) (*PredictResponse, error) {
	// Retrieve latest prediction from cache/database
	// For now, return a mock response
	return &PredictResponse{
		CompanyID:               companyID,
		RiskScore:              65.5,
		GuidanceMissProbability: 0.655,
		Confidence:             0.85,
		PredictionTimestamp:    time.Now().Add(-1 * time.Hour),
		ModelVersion:           "v1.0.0",
		PredictionID:           "pred_" + companyID + "_" + strconv.FormatInt(time.Now().Unix(), 10),
	}, nil
}

// Stub handlers for frontend compatibility

// GetPredictions returns recent predictions with pagination
func (h *Handlers) GetPredictions(c *gin.Context) {
	ctx := c.Request.Context()
	
	// Parse query parameters
	page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))
	sortBy := c.DefaultQuery("sortBy", "prediction_timestamp")
	sortOrder := c.DefaultQuery("sortOrder", "desc")
	
	if page < 1 {
		page = 1
	}
	if limit < 1 || limit > 100 {
		limit = 10
	}
	
	offset := (page - 1) * limit
	
	// Query recent predictions from database (using companies as sample data for now)
	query := `
		SELECT c.symbol, c.name, c.sector, 
		       (50 + random() * 50)::numeric(5,2) as risk_score,
		       (0.3 + random() * 0.4)::numeric(4,3) as guidance_miss_probability,
		       (0.7 + random() * 0.3)::numeric(4,3) as confidence,
		       CURRENT_TIMESTAMP - INTERVAL '1 hour' * (ROW_NUMBER() OVER (ORDER BY c.symbol)) as prediction_timestamp,
		       'v1.0.0' as model_version,
		       'pred_' || c.symbol || '_' || EXTRACT(epoch FROM NOW())::text as prediction_id
		FROM companies c 
		WHERE c.active = true
		ORDER BY ` + sortBy + ` ` + sortOrder + `
		LIMIT $1 OFFSET $2
	`
	
	rows, err := h.postgres.QueryContext(ctx, query, limit, offset)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to fetch predictions",
			"details": err.Error(),
		})
		return
	}
	defer rows.Close()
	
	var predictions []gin.H
	for rows.Next() {
		var symbol, name, sector, modelVersion, predictionID string
		var riskScore, guidanceMissProbability, confidence float64
		var predictionTimestamp time.Time
		
		err := rows.Scan(&symbol, &name, &sector, &riskScore, &guidanceMissProbability, 
						&confidence, &predictionTimestamp, &modelVersion, &predictionID)
		if err != nil {
			continue
		}
		
		predictions = append(predictions, gin.H{
			"company_id":                symbol,
			"company_name":             name,
			"sector":                   sector,
			"risk_score":              riskScore,
			"guidance_miss_probability": guidanceMissProbability,
			"confidence":              confidence,
			"prediction_timestamp":    predictionTimestamp,
			"model_version":           modelVersion,
			"prediction_id":           predictionID,
		})
	}
	
	// Get total count
	var total int
	countQuery := `SELECT COUNT(*) FROM companies WHERE active = true`
	h.postgres.QueryRowContext(ctx, countQuery).Scan(&total)
	
	totalPages := (total + limit - 1) / limit
	
	c.JSON(http.StatusOK, gin.H{
		"data": gin.H{
			"predictions": predictions,
			"total":       total,
			"page":        page,
			"totalPages":  totalPages,
		},
		"success": true,
	})
}

// GetAlerts returns system alerts based on anomalies
func (h *Handlers) GetAlerts(c *gin.Context) {
	ctx := c.Request.Context()
	
	// Parse query parameters
	severity := c.QueryArray("severity[]")
	resolved := c.DefaultQuery("resolved", "false")
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))
	
	if limit < 1 || limit > 100 {
		limit = 10
	}
	
	// Build where clause
	whereClause := "WHERE 1=1"
	args := []interface{}{}
	argCount := 0
	
	if len(severity) > 0 {
		argCount++
		whereClause += fmt.Sprintf(" AND severity = ANY($%d)", argCount)
		args = append(args, pq.Array(severity))
	}
	
	if resolved == "false" {
		whereClause += " AND resolved_at IS NULL"
	} else if resolved == "true" {
		whereClause += " AND resolved_at IS NOT NULL"
	}
	
	// Query alerts from detected_anomalies table
	query := `
		SELECT da.id, da.type, da.severity, da.company_id, da.feature_name, 
		       da.value, da.detected_at, da.details, c.name as company_name
		FROM detected_anomalies da
		LEFT JOIN companies c ON da.company_id = c.symbol
		` + whereClause + `
		ORDER BY da.detected_at DESC
		LIMIT ` + strconv.Itoa(limit)
	
	rows, err := h.postgres.QueryContext(ctx, query, args...)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to fetch alerts",
			"details": err.Error(),
		})
		return
	}
	defer rows.Close()
	
	var alerts []gin.H
	for rows.Next() {
		var id, alertType, severity, companyID, featureName, details, companyName string
		var value float64
		var detectedAt time.Time
		
		err := rows.Scan(&id, &alertType, &severity, &companyID, &featureName, 
						&value, &detectedAt, &details, &companyName)
		if err != nil {
			continue
		}
		
		alerts = append(alerts, gin.H{
			"id":           id,
			"type":         alertType,
			"severity":     severity,
			"company_id":   companyID,
			"company_name": companyName,
			"feature_name": featureName,
			"value":        value,
			"detected_at":  detectedAt,
			"details":      details,
		})
	}
	
	c.JSON(http.StatusOK, gin.H{
		"data":    alerts,
		"success": true,
	})
}

// GetRiskTrends returns historical risk trend data
func (h *Handlers) GetRiskTrends(c *gin.Context) {
	ctx := c.Request.Context()
	
	// Parse query parameters
	_ = c.DefaultQuery("period", "daily") // period parameter for future use
	days, _ := strconv.Atoi(c.DefaultQuery("days", "30"))
	
	if days < 1 || days > 365 {
		days = 30
	}
	
	// Query financial data to generate trend data
	query := `
		SELECT 
			c.symbol,
			c.name,
			c.sector,
			DATE(fs.timestamp) as date,
			AVG(CASE 
				WHEN fs.name = 'gross_margin_trend' THEN fs.value::numeric
				ELSE NULL 
			END) as gross_margin,
			AVG(CASE 
				WHEN fs.name = 'inventory_turnover_ratio' THEN fs.value::numeric
				ELSE NULL 
			END) as inventory_turnover,
			AVG(CASE 
				WHEN fs.name = 'supply_chain_sentiment' THEN fs.value::numeric
				ELSE NULL 
			END) as sentiment,
			(25 + random() * 50)::numeric(5,2) as risk_score
		FROM companies c
		LEFT JOIN feature_store fs ON c.symbol = fs.company_id
		WHERE c.active = true 
		  AND fs.timestamp >= CURRENT_DATE - INTERVAL '%d days'
		GROUP BY c.symbol, c.name, c.sector, DATE(fs.timestamp)
		ORDER BY date DESC, c.symbol
		LIMIT 200
	`
	
	formattedQuery := fmt.Sprintf(query, days)
	rows, err := h.postgres.QueryContext(ctx, formattedQuery)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to fetch risk trends",
			"details": err.Error(),
		})
		return
	}
	defer rows.Close()
	
	var trends []gin.H
	for rows.Next() {
		var symbol, name, sector string
		var date time.Time
		var grossMargin, inventoryTurnover, sentiment, riskScore sql.NullFloat64
		
		err := rows.Scan(&symbol, &name, &sector, &date, &grossMargin, 
						&inventoryTurnover, &sentiment, &riskScore)
		if err != nil {
			continue
		}
		
		trends = append(trends, gin.H{
			"company_id":         symbol,
			"company_name":       name,
			"sector":             sector,
			"date":               date.Format("2006-01-02"),
			"risk_score":         riskScore.Float64,
			"gross_margin":       grossMargin.Float64,
			"inventory_turnover": inventoryTurnover.Float64,
			"sentiment":          sentiment.Float64,
		})
	}
	
	c.JSON(http.StatusOK, gin.H{
		"data":    trends,
		"success": true,
	})
}

// GetSectorAnalysis returns risk analysis by sector
func (h *Handlers) GetSectorAnalysis(c *gin.Context) {
	ctx := c.Request.Context()
	
	// Query sector risk metrics
	query := `
		SELECT 
			c.sector,
			COUNT(c.symbol) as company_count,
			AVG((50 + random() * 40)::numeric(5,2)) as avg_risk_score,
			STDDEV((50 + random() * 40)::numeric(5,2)) as risk_volatility,
			COUNT(CASE WHEN (50 + random() * 50) > 70 THEN 1 END) as high_risk_count,
			AVG(CASE 
				WHEN fs.name = 'supply_chain_sentiment' THEN fs.value::numeric
				ELSE NULL 
			END) as avg_sentiment
		FROM companies c
		LEFT JOIN feature_store fs ON c.symbol = fs.company_id 
			AND fs.timestamp >= CURRENT_DATE - INTERVAL '7 days'
		WHERE c.active = true
		GROUP BY c.sector
		ORDER BY avg_risk_score DESC
	`
	
	rows, err := h.postgres.QueryContext(ctx, query)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to fetch sector analysis",
			"details": err.Error(),
		})
		return
	}
	defer rows.Close()
	
	var sectors []gin.H
	for rows.Next() {
		var sector string
		var companyCount, highRiskCount int
		var avgRiskScore, riskVolatility, avgSentiment sql.NullFloat64
		
		err := rows.Scan(&sector, &companyCount, &avgRiskScore, 
						&riskVolatility, &highRiskCount, &avgSentiment)
		if err != nil {
			continue
		}
		
		sectors = append(sectors, gin.H{
			"sector":           sector,
			"company_count":    companyCount,
			"avg_risk_score":   avgRiskScore.Float64,
			"risk_volatility":  riskVolatility.Float64,
			"high_risk_count":  highRiskCount,
			"avg_sentiment":    avgSentiment.Float64,
		})
	}
	
	c.JSON(http.StatusOK, gin.H{
		"data":    sectors,
		"success": true,
	})
}

// GetDisruptionRisk handles disruption risk analysis requests
// @Summary Get disruption risk analysis
// @Description Get supply chain disruption risk analysis for one or more companies
// @Tags disruption
// @Accept json
// @Produce json
// @Param symbol query string true "Company symbol(s), comma-separated for multiple"
// @Success 200 {object} map[string]interface{} "Single company risk data or array of company data"
// @Failure 400 {object} map[string]interface{} "Bad request"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /disruption-risk [get]
func (h *Handlers) GetDisruptionRisk(c *gin.Context) {
	symbols := c.Query("symbol")
	if symbols == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "symbol parameter is required",
			"message": "Please provide at least one company symbol",
		})
		return
	}

	// Log the request for debugging
	log.Printf("GetDisruptionRisk called with symbols: %s", symbols)

	defer func() {
		if r := recover(); r != nil {
			log.Printf("Panic in GetDisruptionRisk: %v", r)
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Internal server error",
				"message": "An unexpected error occurred while processing the request",
			})
		}
	}()

	// For single symbol requests, return detailed data structure for DisruptionCard
	if !strings.Contains(symbols, ",") {
		riskData, err := generateSingleCompanyRiskData(symbols)
		if err != nil {
			log.Printf("Error generating single company risk data for %s: %v", symbols, err)
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to generate risk data",
				"message": fmt.Sprintf("Could not generate risk data for symbol %s", symbols),
			})
			return
		}
		c.JSON(http.StatusOK, riskData)
		return
	}

	// For multiple symbols, return array of company data for CompanyList
	symbolList := splitSymbols(symbols)
	if len(symbolList) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid symbols",
			"message": "No valid symbols found in the request",
		})
		return
	}

	var companies []gin.H
	var errors []string

	// Batch fetch stock data for all symbols to reduce API calls
	stockDataMap := batchFetchStockData(symbolList)

	for _, symbol := range symbolList {
		companyData, err := generateCompanyListDataWithStockData(symbol, stockDataMap[symbol])
		if err != nil {
			log.Printf("Error generating company data for %s: %v", symbol, err)
			errors = append(errors, fmt.Sprintf("Failed to generate data for %s: %v", symbol, err))
			continue
		}
		companies = append(companies, companyData)
	}

	// If we have some successful results, return them with warnings
	if len(companies) > 0 {
		response := gin.H{
			"data": companies,
		}
		if len(errors) > 0 {
			response["warnings"] = errors
		}
		c.JSON(http.StatusOK, companies) // Return just the array for frontend compatibility
		return
	}

	// If no companies were processed successfully
	c.JSON(http.StatusInternalServerError, gin.H{
		"error": "Failed to process any symbols",
		"message": "Could not generate data for any of the requested symbols",
		"details": errors,
	})
}

// Helper function to generate single company risk data for DisruptionCard
func generateSingleCompanyRiskData(symbol string) (gin.H, error) {
	if symbol == "" {
		return nil, fmt.Errorf("symbol cannot be empty")
	}

	// Validate symbol format (basic validation)
	if len(symbol) > 10 || len(symbol) < 1 {
		return nil, fmt.Errorf("invalid symbol format: %s", symbol)
	}

	// Generate mock risk data based on symbol
	baseRisk, err := getBaseRiskForSymbol(symbol)
	if err != nil {
		return nil, fmt.Errorf("failed to get base risk for %s: %w", symbol, err)
	}

	// Generate historical data for chart (6 months)
	chartData, err := generateHistoricalRiskData(symbol, baseRisk)
	if err != nil {
		return nil, fmt.Errorf("failed to generate historical data for %s: %w", symbol, err)
	}

	// Generate risk factors
	factors, err := generateRiskFactors(symbol)
	if err != nil {
		return nil, fmt.Errorf("failed to generate risk factors for %s: %w", symbol, err)
	}

	// Get trend data
	trend, err := getTrendForSymbol(symbol)
	if err != nil {
		return nil, fmt.Errorf("failed to get trend for %s: %w", symbol, err)
	}

	return gin.H{
		"currentRisk": baseRisk,
		"trend":       trend,
		"data":        chartData,
		"factors":     factors,
	}, nil
}

// Helper function to generate company data for CompanyList
func generateCompanyListData(symbol string) (gin.H, error) {
	if symbol == "" {
		return nil, fmt.Errorf("symbol cannot be empty")
	}

	baseRisk, err := getBaseRiskForSymbol(symbol)
	if err != nil {
		return nil, fmt.Errorf("failed to get base risk: %w", err)
	}

	companyInfo, err := getCompanyInfo(symbol)
	if err != nil {
		return nil, fmt.Errorf("failed to get company info: %w", err)
	}

	// Fetch real stock data with fallback to mock data
	stockData := getStockDataWithFallback(symbol)

	response := gin.H{
		"name":                companyInfo["name"],
		"symbol":              symbol,
		"logo":                fmt.Sprintf("https://logo.clearbit.com/%s.com", getCompanyDomain(symbol)),
		"stockPrice":          stockData.Price,
		"priceChange":         stockData.Change,
		"priceChangePercent":  stockData.ChangePercent,
		"riskLevel":           baseRisk,
		"riskTrend":           "stable",
		"sector":              companyInfo["sector"],
		"isRealData":          stockData.IsRealData,
	}

	// Add additional fields if we have real data
	if stockData.IsRealData {
		response["previousClose"] = stockData.PreviousClose
		response["volume"] = stockData.Volume
		response["latestTradingDay"] = stockData.LatestTradingDay
	}

	// Add error information if data fetch failed
	if stockData.Error != "" {
		response["dataSource"] = "mock"
		response["apiError"] = stockData.Error
	} else if stockData.IsRealData {
		response["dataSource"] = "alpha_vantage"
	} else {
		response["dataSource"] = "mock"
	}

	return response, nil
}

// Helper function to generate historical risk data for charts
func generateHistoricalRiskData(symbol string, baseRisk float64) ([]gin.H, error) {
	if baseRisk < 0 || baseRisk > 100 {
		return nil, fmt.Errorf("invalid base risk value: %f", baseRisk)
	}

	var chartData []gin.H
	months := []string{"Aug", "Sep", "Oct", "Nov", "Dec", "Jan"}
	
	for i, month := range months {
		// Add some realistic variation to the risk over time
		variation := (float64(i)*1.5) + (float64(i%3)*2.5) - 3.0 // Range: -3 to +6
		risk := baseRisk + variation
		
		// Ensure risk stays within reasonable bounds
		if risk < 0 {
			risk = 0
		}
		if risk > 100 {
			risk = 100
		}
		
		chartData = append(chartData, gin.H{
			"date": month,
			"risk": math.Round(risk*10)/10, // Round to 1 decimal place
		})
	}
	
	return chartData, nil
}

// Helper functions for data generation with error handling
func getBaseRiskForSymbol(symbol string) (float64, error) {
	if symbol == "" {
		return 0, fmt.Errorf("symbol cannot be empty")
	}

	riskMap := map[string]float64{
		"AAPL":  35.5,
		"TSLA":  72.3,
		"MSFT":  28.1,
		"AMZN":  45.7,
		"WMT":   22.8,
		"GOOGL": 31.2,
		"META":  48.9,
		"NVDA":  67.4,
		"NFLX":  41.2,
		"CRM":   39.8,
	}
	
	if risk, exists := riskMap[symbol]; exists {
		return risk, nil
	}
	
	// Generate a pseudo-random but consistent risk for unknown symbols
	hash := 0
	for _, char := range symbol {
		hash = hash*31 + int(char)
	}
	risk := float64(20 + (hash%60)) // Range: 20-80%
	return math.Round(risk*10)/10, nil
}

func getTrendForSymbol(symbol string) (float64, error) {
	if symbol == "" {
		return 0, fmt.Errorf("symbol cannot be empty")
	}

	trendMap := map[string]float64{
		"AAPL":  2.3,
		"TSLA":  -5.1,
		"MSFT":  1.8,
		"AMZN":  3.2,
		"WMT":   -0.5,
		"GOOGL": 1.1,
		"META":  4.7,
		"NVDA":  -2.8,
		"NFLX":  -1.2,
		"CRM":   2.1,
	}
	
	if trend, exists := trendMap[symbol]; exists {
		return trend, nil
	}
	
	// Generate a pseudo-random but consistent trend for unknown symbols
	hash := 0
	for _, char := range symbol {
		hash = hash*31 + int(char)
	}
	trend := float64(-5 + (hash%11)) // Range: -5 to +5
	return math.Round(trend*10)/10, nil
}

func generateRiskFactors(symbol string) ([]gin.H, error) {
	if symbol == "" {
		return nil, fmt.Errorf("symbol cannot be empty")
	}

	allFactors := []gin.H{
		{"name": "Supply Chain Concentration", "level": "High"},
		{"name": "Geopolitical Risk", "level": "Medium"},
		{"name": "Supplier Financial Health", "level": "Low"},
		{"name": "Transportation Disruption", "level": "Medium"},
		{"name": "Raw Material Availability", "level": "High"},
		{"name": "Regulatory Compliance", "level": "Low"},
		{"name": "Cyber Security Risk", "level": "Medium"},
		{"name": "Climate Change Impact", "level": "High"},
		{"name": "Labor Availability", "level": "Medium"},
		{"name": "Currency Fluctuation", "level": "Low"},
	}
	
	// Return 3-6 factors based on symbol to provide variety
	hash := 0
	for _, char := range symbol {
		hash = hash*31 + int(char)
	}
	factorCount := 3 + (hash % 4) // 3-6 factors
	
	if factorCount > len(allFactors) {
		factorCount = len(allFactors)
	}
	
	// Use hash to determine which factors to include for consistency
	selectedFactors := make([]gin.H, factorCount)
	for i := 0; i < factorCount; i++ {
		selectedFactors[i] = allFactors[(hash+i)%len(allFactors)]
	}
	
	return selectedFactors, nil
}

func getCompanyInfo(symbol string) (map[string]string, error) {
	if symbol == "" {
		return nil, fmt.Errorf("symbol cannot be empty")
	}

	companyMap := map[string]map[string]string{
		"AAPL":  {"name": "Apple Inc.", "sector": "Technology"},
		"TSLA":  {"name": "Tesla Inc.", "sector": "Automotive"},
		"MSFT":  {"name": "Microsoft Corp.", "sector": "Technology"},
		"AMZN":  {"name": "Amazon.com Inc.", "sector": "E-commerce"},
		"WMT":   {"name": "Walmart Inc.", "sector": "Retail"},
		"GOOGL": {"name": "Alphabet Inc.", "sector": "Technology"},
		"META":  {"name": "Meta Platforms Inc.", "sector": "Technology"},
		"NVDA":  {"name": "NVIDIA Corp.", "sector": "Technology"},
		"NFLX":  {"name": "Netflix Inc.", "sector": "Entertainment"},
		"CRM":   {"name": "Salesforce Inc.", "sector": "Technology"},
	}
	
	if info, exists := companyMap[symbol]; exists {
		return info, nil
	}
	
	// Generate default info for unknown symbols
	return map[string]string{
		"name":   symbol + " Corp.",
		"sector": "Unknown",
	}, nil
}

func getCompanyDomain(symbol string) string {
	domainMap := map[string]string{
		"AAPL":  "apple",
		"TSLA":  "tesla",
		"MSFT":  "microsoft",
		"AMZN":  "amazon",
		"WMT":   "walmart",
		"GOOGL": "google",
		"META":  "meta",
		"NVDA":  "nvidia",
		"NFLX":  "netflix",
		"CRM":   "salesforce",
	}
	
	if domain, exists := domainMap[symbol]; exists {
		return domain
	}
	return "example"
}

// fetchRealStockData fetches real stock data from Alpha Vantage API
func fetchRealStockData(symbol string) (*StockData, error) {
	if symbol == "" {
		return nil, fmt.Errorf("symbol cannot be empty")
	}

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: API_TIMEOUT,
	}

	// Build API URL
	url := fmt.Sprintf("%s?function=GLOBAL_QUOTE&symbol=%s&apikey=%s", 
		ALPHA_VANTAGE_BASE_URL, symbol, ALPHA_VANTAGE_API_KEY)

	log.Printf("Fetching stock data for %s from Alpha Vantage", symbol)

	// Make HTTP request
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to make API request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status: %d", resp.StatusCode)
	}

	// Parse JSON response
	var quote AlphaVantageQuote
	if err := json.NewDecoder(resp.Body).Decode(&quote); err != nil {
		return nil, fmt.Errorf("failed to parse API response: %w", err)
	}

	// Check if we got valid data
	if quote.GlobalQuote.Symbol == "" || quote.GlobalQuote.Price == "" {
		return nil, fmt.Errorf("no data returned for symbol %s", symbol)
	}

	// Parse numeric values
	price, err := strconv.ParseFloat(quote.GlobalQuote.Price, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to parse price: %w", err)
	}

	change, err := strconv.ParseFloat(quote.GlobalQuote.Change, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to parse change: %w", err)
	}

	// Parse change percent (remove % sign)
	changePercentStr := strings.TrimSuffix(quote.GlobalQuote.ChangePercent, "%")
	changePercent, err := strconv.ParseFloat(changePercentStr, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to parse change percent: %w", err)
	}

	previousClose, err := strconv.ParseFloat(quote.GlobalQuote.PreviousClose, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to parse previous close: %w", err)
	}

	volume, err := strconv.ParseInt(quote.GlobalQuote.Volume, 10, 64)
	if err != nil {
		log.Printf("Warning: failed to parse volume for %s: %v", symbol, err)
		volume = 0 // Set to 0 if parsing fails
	}

	return &StockData{
		Symbol:            quote.GlobalQuote.Symbol,
		Price:             math.Round(price*100)/100,
		Change:            math.Round(change*100)/100,
		ChangePercent:     math.Round(changePercent*100)/100,
		PreviousClose:     math.Round(previousClose*100)/100,
		Volume:            volume,
		LatestTradingDay:  quote.GlobalQuote.LatestTradingDay,
		IsRealData:        true,
	}, nil
}

// getStockDataWithFallback tries to fetch real data, falls back to mock data if API fails
func getStockDataWithFallback(symbol string) *StockData {
	// Check if real API calls are enabled
	if !ENABLE_REAL_STOCK_DATA {
		log.Printf("Real stock data disabled, using mock data for %s", symbol)
		mockPrice, _ := generateMockStockPrice(symbol)
		mockChange, _ := generateMockPriceChange()
		mockChangePercent, _ := generateMockPriceChangePercent()
		
		return &StockData{
			Symbol:        symbol,
			Price:         mockPrice,
			Change:        mockChange,
			ChangePercent: mockChangePercent,
			PreviousClose: mockPrice - mockChange,
			Volume:        0,
			IsRealData:    false,
			Error:         "Real stock data disabled",
		}
	}

	// Try to fetch real data first
	realData, err := fetchRealStockData(symbol)
	if err != nil {
		log.Printf("Failed to fetch real stock data for %s: %v, using mock data", symbol, err)
		
		// Fall back to mock data
		mockPrice, _ := generateMockStockPrice(symbol)
		mockChange, _ := generateMockPriceChange()
		mockChangePercent, _ := generateMockPriceChangePercent()
		
		return &StockData{
			Symbol:        symbol,
			Price:         mockPrice,
			Change:        mockChange,
			ChangePercent: mockChangePercent,
			PreviousClose: mockPrice - mockChange,
			Volume:        0,
			IsRealData:    false,
			Error:         err.Error(),
		}
	}

	log.Printf("Successfully fetched real stock data for %s: $%.2f (%.2f%%)", 
		symbol, realData.Price, realData.ChangePercent)
	return realData
}

// batchFetchStockData fetches stock data for multiple symbols
func batchFetchStockData(symbols []string) map[string]*StockData {
	stockDataMap := make(map[string]*StockData)
	
	// Use goroutines to fetch data concurrently (but limit concurrency to avoid rate limits)
	const maxConcurrency = 5
	semaphore := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup
	var mu sync.Mutex
	
	for _, symbol := range symbols {
		wg.Add(1)
		go func(sym string) {
			defer wg.Done()
			semaphore <- struct{}{} // Acquire semaphore
			defer func() { <-semaphore }() // Release semaphore
			
			stockData := getStockDataWithFallback(sym)
			
			mu.Lock()
			stockDataMap[sym] = stockData
			mu.Unlock()
		}(symbol)
	}
	
	wg.Wait()
	return stockDataMap
}

// generateCompanyListDataWithStockData generates company data using pre-fetched stock data
func generateCompanyListDataWithStockData(symbol string, stockData *StockData) (gin.H, error) {
	if symbol == "" {
		return nil, fmt.Errorf("symbol cannot be empty")
	}

	baseRisk, err := getBaseRiskForSymbol(symbol)
	if err != nil {
		return nil, fmt.Errorf("failed to get base risk: %w", err)
	}

	companyInfo, err := getCompanyInfo(symbol)
	if err != nil {
		return nil, fmt.Errorf("failed to get company info: %w", err)
	}

	// Use provided stock data or fetch it if not provided
	if stockData == nil {
		stockData = getStockDataWithFallback(symbol)
	}

	response := gin.H{
		"name":                companyInfo["name"],
		"symbol":              symbol,
		"logo":                fmt.Sprintf("https://logo.clearbit.com/%s.com", getCompanyDomain(symbol)),
		"stockPrice":          stockData.Price,
		"priceChange":         stockData.Change,
		"priceChangePercent":  stockData.ChangePercent,
		"riskLevel":           baseRisk,
		"riskTrend":           "stable",
		"sector":              companyInfo["sector"],
		"isRealData":          stockData.IsRealData,
	}

	// Add additional fields if we have real data
	if stockData.IsRealData {
		response["previousClose"] = stockData.PreviousClose
		response["volume"] = stockData.Volume
		response["latestTradingDay"] = stockData.LatestTradingDay
	}

	// Add error information if data fetch failed
	if stockData.Error != "" {
		response["dataSource"] = "mock"
		response["apiError"] = stockData.Error
	} else if stockData.IsRealData {
		response["dataSource"] = "alpha_vantage"
	} else {
		response["dataSource"] = "mock"
	}

	return response, nil
}

func generateMockStockPrice(symbol string) (float64, error) {
	if symbol == "" {
		return 0, fmt.Errorf("symbol cannot be empty")
	}

	basePrices := map[string]float64{
		"AAPL":  175.50,
		"TSLA":  245.80,
		"MSFT":  378.25,
		"AMZN":  145.90,
		"WMT":   165.75,
		"GOOGL": 142.30,
		"META":  485.60,
		"NVDA":  875.25,
		"NFLX":  425.30,
		"CRM":   215.40,
	}
	
	if price, exists := basePrices[symbol]; exists {
		return price, nil
	}
	
	// Generate a pseudo-random but consistent price for unknown symbols
	hash := 0
	for _, char := range symbol {
		hash = hash*31 + int(char)
	}
	price := float64(50 + (hash%500)) // Range: $50-$550
	return math.Round(price*100)/100, nil
}

func generateMockPriceChange() (float64, error) {
	// Generate a time-based pseudo-random price change for consistency during the same day
	now := time.Now()
	seed := now.Year()*10000 + int(now.YearDay())*100 + now.Hour()
	
	// Generate change between -15 and +15
	change := float64(-15 + (seed%31))
	return math.Round(change*100)/100, nil
}

func generateMockPriceChangePercent() (float64, error) {
	// Generate a time-based pseudo-random percentage change
	now := time.Now()
	seed := now.Year()*10000 + int(now.YearDay())*100 + now.Hour() + 1 // +1 to differentiate from price change
	
	// Generate percentage change between -8% and +8%
	changePercent := float64(-8 + (seed%17))
	return math.Round(changePercent*100)/100, nil
}

// Legacy functions for backward compatibility
func generateStockPrice(symbol string) (float64, error) {
	stockData := getStockDataWithFallback(symbol)
	return stockData.Price, nil
}

func generatePriceChange() (float64, error) {
	return generateMockPriceChange()
}

func generatePriceChangePercent() (float64, error) {
	return generateMockPriceChangePercent()
}

func splitSymbols(symbols string) []string {
	if symbols == "" {
		return []string{}
	}
	
	// Split by comma and trim spaces
	parts := strings.Split(symbols, ",")
	result := make([]string, 0, len(parts))
	
	for _, part := range parts {
		trimmed := strings.TrimSpace(strings.ToUpper(part)) // Convert to uppercase for consistency
		if trimmed != "" && len(trimmed) <= 10 { // Basic validation
			result = append(result, trimmed)
		}
	}
	
	return result
} 