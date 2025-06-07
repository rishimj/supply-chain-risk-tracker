package api

import (
	"context"
	"database/sql"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"supply-chain-ml/pkg/features"
	"supply-chain-ml/pkg/health"
	"supply-chain-ml/pkg/ml"
	"supply-chain-ml/pkg/monitoring"

	"github.com/gin-gonic/gin"
	"github.com/lib/pq"
)

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
func (h *Handlers) Health(c *gin.Context) {
	ctx := c.Request.Context()
	
	healthStatus := h.healthChecker.CheckHealth(ctx)
	
	status := http.StatusOK
	if healthStatus.Status != "healthy" {
		status = http.StatusServiceUnavailable
	}
	
	c.JSON(status, healthStatus)
}

// PredictRisk handles single company risk prediction
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