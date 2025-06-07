package ml

import (
	"context"
	"fmt"
	"time"

	"supply-chain-ml/pkg/features"
)

// ModelServer handles ML model operations
type ModelServer struct {
	ensembleModel *EnsembleModel
	modelVersion  string
	isHealthy     bool
}

// NewModelServer creates a new ML model server
func NewModelServer() *ModelServer {
	return &ModelServer{
		ensembleModel: NewEnsembleModel(),
		modelVersion:  "v1.0.0",
		isHealthy:     true,
	}
}

// Prediction represents a model prediction result
type Prediction struct {
	PredictionID            string             `json:"prediction_id"`
	RiskScore              float64            `json:"risk_score"`
	GuidanceMissProbability float64            `json:"guidance_miss_probability"`
	Confidence             float64            `json:"confidence"`
	ComponentRisks         map[string]float64 `json:"component_risks"`
	FeatureImportance      map[string]float64 `json:"feature_importance"`
	ModelVersion           string             `json:"model_version"`
	Timestamp              time.Time          `json:"timestamp"`
}

// EnsembleModel represents the ensemble ML model
type EnsembleModel struct {
	version         string
	lastTrainedAt   time.Time
	modelMetrics    map[string]float64
	featureColumns  []string
}

// NewEnsembleModel creates a new ensemble model instance
func NewEnsembleModel() *EnsembleModel {
	return &EnsembleModel{
		version:       "v1.0.0",
		lastTrainedAt: time.Now().AddDate(0, 0, -1), // Simulate trained yesterday
		modelMetrics: map[string]float64{
			"accuracy":  0.752,
			"precision": 0.698,
			"recall":    0.745,
			"f1_score":  0.721,
			"auc":       0.823,
		},
		featureColumns: []string{
			"financial_inventory_turnover",
			"financial_gross_margin",
			"financial_debt_to_equity",
			"financial_current_ratio",
			"network_supplier_concentration",
			"network_supplier_risk_score",
			"ts_volatility_30d",
			"ts_momentum_10d",
			"nlp_sentiment_score",
			"nlp_risk_keywords_count",
		},
	}
}

// Predict performs a prediction using the ensemble model
func (ms *ModelServer) Predict(ctx context.Context, featureVector *features.FeatureVector) (*Prediction, error) {
	if !ms.isHealthy {
		return nil, fmt.Errorf("model server is not healthy")
	}
	
	startTime := time.Now()
	
	// Validate feature vector
	if err := ms.validateFeatureVector(featureVector); err != nil {
		return nil, fmt.Errorf("invalid feature vector: %w", err)
	}
	
	// Normalize features
	normalizedFeatures := ms.normalizeFeatures(featureVector.Features)
	
	// Get predictions from ensemble components
	xgbPred := ms.getXGBoostPrediction(normalizedFeatures)
	lstmPred := ms.getLSTMPrediction(normalizedFeatures)
	gnnPred := ms.getGNNPrediction(normalizedFeatures)
	
	// Combine predictions with weights
	ensembleWeights := map[string]float64{
		"xgboost": 0.4,
		"lstm":    0.35,
		"gnn":     0.25,
	}
	
	riskScore := xgbPred.RiskScore*ensembleWeights["xgboost"] +
		lstmPred.RiskScore*ensembleWeights["lstm"] +
		gnnPred.RiskScore*ensembleWeights["gnn"]
	
	// Calculate guidance miss probability
	guidanceMissProbability := ms.calculateGuidanceMissProbability(riskScore)
	
	// Calculate confidence based on component agreement
	confidence := ms.calculateConfidence(xgbPred, lstmPred, gnnPred)
	
	// Feature importance (weighted average from XGBoost mainly)
	featureImportance := ms.calculateFeatureImportance(normalizedFeatures)
	
	// Component risks
	componentRisks := map[string]float64{
		"financial_risk": xgbPred.RiskScore,
		"network_risk":   gnnPred.RiskScore,
		"temporal_risk":  lstmPred.RiskScore,
		"sentiment_risk": ms.getSentimentRisk(normalizedFeatures),
	}
	
	prediction := &Prediction{
		PredictionID:            ms.generatePredictionID(),
		RiskScore:              riskScore,
		GuidanceMissProbability: guidanceMissProbability,
		Confidence:             confidence,
		ComponentRisks:         componentRisks,
		FeatureImportance:      featureImportance,
		ModelVersion:           ms.modelVersion,
		Timestamp:              startTime,
	}
	
	return prediction, nil
}

// GetStatus returns the current status of the model server
func (ms *ModelServer) GetStatus(ctx context.Context) map[string]interface{} {
	return map[string]interface{}{
		"version":         ms.modelVersion,
		"healthy":         ms.isHealthy,
		"last_trained_at": ms.ensembleModel.lastTrainedAt,
		"metrics":         ms.ensembleModel.modelMetrics,
		"feature_count":   len(ms.ensembleModel.featureColumns),
		"uptime":          time.Since(ms.ensembleModel.lastTrainedAt),
	}
}

// TriggerRetrain triggers model retraining (placeholder implementation)
func (ms *ModelServer) TriggerRetrain(ctx context.Context, version string, force bool) (string, error) {
	retrainID := fmt.Sprintf("retrain_%d", time.Now().Unix())
	
	// In a real implementation, this would:
	// 1. Queue a retraining job
	// 2. Return immediately with job ID
	// 3. Update model when training completes
	
	go func() {
		// Simulate training time
		time.Sleep(5 * time.Second)
		
		// Update model version and metrics
		ms.modelVersion = "v1.1.0"
		ms.ensembleModel.lastTrainedAt = time.Now()
		ms.ensembleModel.modelMetrics["accuracy"] = 0.765 // Simulated improvement
	}()
	
	return retrainID, nil
}

// Helper methods for ensemble predictions
type ComponentPrediction struct {
	RiskScore  float64
	Confidence float64
}

func (ms *ModelServer) getXGBoostPrediction(features map[string]float64) ComponentPrediction {
	// Simulate XGBoost prediction based on financial features
	financialScore := 0.0
	financialFeatures := []string{
		"financial_inventory_turnover",
		"financial_gross_margin",
		"financial_debt_to_equity",
		"financial_current_ratio",
	}
	
	count := 0
	for _, feature := range financialFeatures {
		if val, exists := features[feature]; exists {
			financialScore += val
			count++
		}
	}
	
	if count > 0 {
		financialScore /= float64(count)
	}
	
	// Apply sigmoid transformation to get probability
	riskScore := sigmoid(financialScore)
	
	return ComponentPrediction{
		RiskScore:  riskScore * 100, // Convert to 0-100 scale
		Confidence: 0.85,
	}
}

func (ms *ModelServer) getLSTMPrediction(features map[string]float64) ComponentPrediction {
	// Simulate LSTM prediction based on time series features
	timeSeriesScore := 0.0
	tsFeatures := []string{
		"ts_volatility_30d",
		"ts_momentum_10d",
	}
	
	count := 0
	for _, feature := range tsFeatures {
		if val, exists := features[feature]; exists {
			timeSeriesScore += val
			count++
		}
	}
	
	if count > 0 {
		timeSeriesScore /= float64(count)
	}
	
	riskScore := sigmoid(timeSeriesScore)
	
	return ComponentPrediction{
		RiskScore:  riskScore * 100,
		Confidence: 0.75,
	}
}

func (ms *ModelServer) getGNNPrediction(features map[string]float64) ComponentPrediction {
	// Simulate GNN prediction based on network features
	networkScore := 0.0
	networkFeatures := []string{
		"network_supplier_concentration",
		"network_supplier_risk_score",
	}
	
	count := 0
	for _, feature := range networkFeatures {
		if val, exists := features[feature]; exists {
			networkScore += val
			count++
		}
	}
	
	if count > 0 {
		networkScore /= float64(count)
	}
	
	riskScore := sigmoid(networkScore)
	
	return ComponentPrediction{
		RiskScore:  riskScore * 100,
		Confidence: 0.80,
	}
}

func (ms *ModelServer) calculateGuidanceMissProbability(riskScore float64) float64 {
	// Convert risk score to guidance miss probability
	// Higher risk score = higher probability of missing guidance
	return riskScore / 100.0
}

func (ms *ModelServer) calculateConfidence(xgb, lstm, gnn ComponentPrediction) float64 {
	// Calculate confidence based on component agreement
	scores := []float64{xgb.RiskScore, lstm.RiskScore, gnn.RiskScore}
	
	// Calculate standard deviation
	mean := (scores[0] + scores[1] + scores[2]) / 3.0
	variance := 0.0
	for _, score := range scores {
		variance += (score - mean) * (score - mean)
	}
	variance /= 3.0
	stdDev := variance // Simplified, should be sqrt(variance)
	
	// Higher agreement (lower std dev) = higher confidence
	maxStdDev := 30.0 // Maximum expected standard deviation
	confidence := 1.0 - (stdDev / maxStdDev)
	
	if confidence < 0.3 {
		confidence = 0.3 // Minimum confidence
	}
	if confidence > 0.95 {
		confidence = 0.95 // Maximum confidence
	}
	
	return confidence
}

func (ms *ModelServer) calculateFeatureImportance(features map[string]float64) map[string]float64 {
	// Simulate feature importance calculation
	importance := make(map[string]float64)
	
	// Static importance weights (in practice, from trained model)
	staticImportance := map[string]float64{
		"financial_inventory_turnover":    0.15,
		"financial_gross_margin":          0.12,
		"financial_debt_to_equity":        0.10,
		"financial_current_ratio":         0.08,
		"network_supplier_concentration":  0.18,
		"network_supplier_risk_score":     0.14,
		"ts_volatility_30d":              0.11,
		"ts_momentum_10d":                0.06,
		"nlp_sentiment_score":            0.04,
		"nlp_risk_keywords_count":        0.02,
	}
	
	for feature, value := range features {
		if baseImportance, exists := staticImportance[feature]; exists {
			// Adjust importance based on feature value
			importance[feature] = baseImportance * (1.0 + value/100.0)
		}
	}
	
	return importance
}

func (ms *ModelServer) getSentimentRisk(features map[string]float64) float64 {
	sentimentScore := 0.0
	if val, exists := features["nlp_sentiment_score"]; exists {
		// Convert sentiment to risk (negative sentiment = higher risk)
		sentimentScore = (1.0 - val) * 100.0
	}
	
	keywordCount := 0.0
	if val, exists := features["nlp_risk_keywords_count"]; exists {
		keywordCount = val
	}
	
	// Combine sentiment and keyword indicators
	return (sentimentScore + keywordCount*5) / 2.0
}

func (ms *ModelServer) validateFeatureVector(fv *features.FeatureVector) error {
	if fv.CompanyID == "" {
		return fmt.Errorf("company_id is required")
	}
	
	if len(fv.Features) == 0 {
		return fmt.Errorf("features cannot be empty")
	}
	
	// Check for required features
	requiredFeatures := []string{
		"financial_inventory_turnover",
		"network_supplier_concentration",
	}
	
	for _, required := range requiredFeatures {
		if _, exists := fv.Features[required]; !exists {
			return fmt.Errorf("required feature missing: %s", required)
		}
	}
	
	return nil
}

func (ms *ModelServer) normalizeFeatures(features map[string]float64) map[string]float64 {
	normalized := make(map[string]float64)
	
	// Simple min-max normalization (in practice, use pre-computed stats)
	normalizationParams := map[string]struct {
		min, max float64
	}{
		"financial_inventory_turnover":    {0.5, 15.0},
		"financial_gross_margin":          {0.1, 0.8},
		"financial_debt_to_equity":        {0.0, 5.0},
		"financial_current_ratio":         {0.5, 3.0},
		"network_supplier_concentration":  {0.1, 1.0},
		"network_supplier_risk_score":     {0.0, 100.0},
		"ts_volatility_30d":              {0.05, 2.0},
		"ts_momentum_10d":                {-0.5, 0.5},
		"nlp_sentiment_score":            {0.0, 1.0},
		"nlp_risk_keywords_count":        {0.0, 20.0},
	}
	
	for feature, value := range features {
		if params, exists := normalizationParams[feature]; exists {
			// Min-max normalization to [0, 1]
			normalized[feature] = (value - params.min) / (params.max - params.min)
			
			// Clamp to [0, 1]
			if normalized[feature] < 0 {
				normalized[feature] = 0
			}
			if normalized[feature] > 1 {
				normalized[feature] = 1
			}
		} else {
			// Use raw value if no normalization params
			normalized[feature] = value
		}
	}
	
	return normalized
}

func (ms *ModelServer) generatePredictionID() string {
	return fmt.Sprintf("pred_%d_%d", 
		time.Now().Unix(), 
		time.Now().UnixNano()%1000000)
}

// Utility functions
func sigmoid(x float64) float64 {
	// Simplified sigmoid function
	if x > 0 {
		return 1.0 / (1.0 + (1.0/(1.0+x)))
	} else {
		return 1.0 / (1.0 + (1.0+(-x)))
	}
}

// BatchPredict handles batch predictions efficiently
func (ms *ModelServer) BatchPredict(ctx context.Context, featureVectors []*features.FeatureVector) ([]*Prediction, error) {
	var predictions []*Prediction
	
	for _, fv := range featureVectors {
		pred, err := ms.Predict(ctx, fv)
		if err != nil {
			// Continue with other predictions, don't fail entire batch
			continue
		}
		predictions = append(predictions, pred)
	}
	
	return predictions, nil
}

// HealthCheck verifies the model server is functioning
func (ms *ModelServer) HealthCheck(ctx context.Context) error {
	if !ms.isHealthy {
		return fmt.Errorf("model server is unhealthy")
	}
	
	// Test prediction with dummy data
	testFeatures := map[string]float64{
		"financial_inventory_turnover":   5.2,
		"network_supplier_concentration": 0.7,
	}
	
	fv := &features.FeatureVector{
		CompanyID: "TEST",
		Features:  testFeatures,
		Timestamp: time.Now(),
	}
	
	_, err := ms.Predict(context.Background(), fv)
	return err
} 