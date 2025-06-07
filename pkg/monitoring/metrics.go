package monitoring

import (
	"context"
	"sync"
	"time"
)

// MetricsCollector handles collecting and storing system metrics
type MetricsCollector struct {
	mutex           sync.RWMutex
	predictionCount map[string]int64
	responseTime    map[string][]time.Duration
	errorCount      map[string]int64
	systemMetrics   SystemMetrics
	startTime       time.Time
}

// SystemMetrics represents overall system performance metrics
type SystemMetrics struct {
	TotalRequests     int64             `json:"total_requests"`
	SuccessfulRequests int64             `json:"successful_requests"`
	FailedRequests    int64             `json:"failed_requests"`
	AverageLatency    time.Duration     `json:"average_latency"`
	ErrorRate         float64           `json:"error_rate"`
	ThroughputRPS     float64           `json:"throughput_rps"`
	Uptime            time.Duration     `json:"uptime"`
	ComponentMetrics  map[string]interface{} `json:"component_metrics"`
}

// PredictionMetrics tracks ML prediction performance
type PredictionMetrics struct {
	TotalPredictions      int64         `json:"total_predictions"`
	AveragePredictionTime time.Duration `json:"average_prediction_time"`
	PredictionsByModel    map[string]int64 `json:"predictions_by_model"`
	SuccessRate           float64       `json:"success_rate"`
	LastPredictionTime    time.Time     `json:"last_prediction_time"`
}

// NewMetricsCollector creates a new metrics collector instance
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		predictionCount: make(map[string]int64),
		responseTime:    make(map[string][]time.Duration),
		errorCount:      make(map[string]int64),
		systemMetrics: SystemMetrics{
			ComponentMetrics: make(map[string]interface{}),
		},
		startTime: time.Now(),
	}
}

// RecordPrediction records metrics for a prediction request
func (mc *MetricsCollector) RecordPrediction(ctx context.Context, modelVersion, companyID string, duration time.Duration, success bool) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	
	// Update prediction counts
	mc.predictionCount[modelVersion]++
	mc.systemMetrics.TotalRequests++
	
	// Record response time
	if mc.responseTime[modelVersion] == nil {
		mc.responseTime[modelVersion] = make([]time.Duration, 0)
	}
	mc.responseTime[modelVersion] = append(mc.responseTime[modelVersion], duration)
	
	// Limit response time history to last 1000 entries
	if len(mc.responseTime[modelVersion]) > 1000 {
		mc.responseTime[modelVersion] = mc.responseTime[modelVersion][1:]
	}
	
	// Update success/failure counts
	if success {
		mc.systemMetrics.SuccessfulRequests++
	} else {
		mc.systemMetrics.FailedRequests++
		mc.errorCount[modelVersion]++
	}
	
	// Update derived metrics
	mc.updateDerivedMetrics()
}

// RecordAPIRequest records metrics for general API requests
func (mc *MetricsCollector) RecordAPIRequest(ctx context.Context, endpoint, method string, duration time.Duration, statusCode int) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	
	key := method + "_" + endpoint
	
	mc.systemMetrics.TotalRequests++
	
	// Record response time
	if mc.responseTime[key] == nil {
		mc.responseTime[key] = make([]time.Duration, 0)
	}
	mc.responseTime[key] = append(mc.responseTime[key], duration)
	
	// Limit history
	if len(mc.responseTime[key]) > 1000 {
		mc.responseTime[key] = mc.responseTime[key][1:]
	}
	
	// Record success/failure based on status code
	if statusCode < 400 {
		mc.systemMetrics.SuccessfulRequests++
	} else {
		mc.systemMetrics.FailedRequests++
		mc.errorCount[key]++
	}
	
	mc.updateDerivedMetrics()
}

// GetMetrics returns current system metrics
func (mc *MetricsCollector) GetMetrics(ctx context.Context) map[string]interface{} {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()
	
	return map[string]interface{}{
		"system_metrics":     mc.systemMetrics,
		"prediction_metrics": mc.getPredictionMetrics(),
		"endpoint_metrics":   mc.getEndpointMetrics(),
		"timestamp":          time.Now(),
	}
}

// GetSystemMetrics returns system-level metrics
func (mc *MetricsCollector) GetSystemMetrics(ctx context.Context) SystemMetrics {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()
	
	// Update uptime
	metrics := mc.systemMetrics
	metrics.Uptime = time.Since(mc.startTime)
	
	return metrics
}

// GetPredictionMetrics returns prediction-specific metrics
func (mc *MetricsCollector) GetPredictionMetrics(ctx context.Context) PredictionMetrics {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()
	
	return mc.getPredictionMetrics()
}

// updateDerivedMetrics calculates derived metrics (called with lock held)
func (mc *MetricsCollector) updateDerivedMetrics() {
	// Calculate error rate
	if mc.systemMetrics.TotalRequests > 0 {
		mc.systemMetrics.ErrorRate = float64(mc.systemMetrics.FailedRequests) / float64(mc.systemMetrics.TotalRequests)
	}
	
	// Calculate average latency across all endpoints
	var totalDuration time.Duration
	var totalCount int64
	
	for _, durations := range mc.responseTime {
		for _, duration := range durations {
			totalDuration += duration
			totalCount++
		}
	}
	
	if totalCount > 0 {
		mc.systemMetrics.AverageLatency = totalDuration / time.Duration(totalCount)
	}
	
	// Calculate throughput (requests per second)
	uptime := time.Since(mc.startTime)
	if uptime.Seconds() > 0 {
		mc.systemMetrics.ThroughputRPS = float64(mc.systemMetrics.TotalRequests) / uptime.Seconds()
	}
}

// getPredictionMetrics calculates prediction metrics (called with lock held)
func (mc *MetricsCollector) getPredictionMetrics() PredictionMetrics {
	var totalPredictions int64
	var totalDuration time.Duration
	var totalResponseCount int64
	var lastPredictionTime time.Time
	
	// Calculate totals across all models
	for model, count := range mc.predictionCount {
		totalPredictions += count
		
		if durations, exists := mc.responseTime[model]; exists {
			for _, duration := range durations {
				totalDuration += duration
				totalResponseCount++
			}
		}
	}
	
	// Calculate average prediction time
	avgPredictionTime := time.Duration(0)
	if totalResponseCount > 0 {
		avgPredictionTime = totalDuration / time.Duration(totalResponseCount)
	}
	
	// Calculate success rate for predictions
	var predictionErrors int64
	for model := range mc.predictionCount {
		predictionErrors += mc.errorCount[model]
	}
	
	successRate := 1.0
	if totalPredictions > 0 {
		successRate = float64(totalPredictions-predictionErrors) / float64(totalPredictions)
	}
	
	// Set last prediction time (simplified - would track actual last prediction in real implementation)
	if totalPredictions > 0 {
		lastPredictionTime = time.Now()
	}
	
	return PredictionMetrics{
		TotalPredictions:      totalPredictions,
		AveragePredictionTime: avgPredictionTime,
		PredictionsByModel:    copyMap(mc.predictionCount),
		SuccessRate:           successRate,
		LastPredictionTime:    lastPredictionTime,
	}
}

// getEndpointMetrics returns metrics for API endpoints
func (mc *MetricsCollector) getEndpointMetrics() map[string]interface{} {
	endpointMetrics := make(map[string]interface{})
	
	for endpoint := range mc.responseTime {
		if len(mc.responseTime[endpoint]) > 0 {
			// Calculate metrics for this endpoint
			durations := mc.responseTime[endpoint]
			var total time.Duration
			for _, d := range durations {
				total += d
			}
			
			avgLatency := total / time.Duration(len(durations))
			requestCount := int64(len(durations))
			errorCount := mc.errorCount[endpoint]
			
			endpointMetrics[endpoint] = map[string]interface{}{
				"request_count":   requestCount,
				"average_latency": avgLatency,
				"error_count":     errorCount,
				"error_rate":      float64(errorCount) / float64(requestCount),
			}
		}
	}
	
	return endpointMetrics
}

// Reset resets all metrics (useful for testing)
func (mc *MetricsCollector) Reset() {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	
	mc.predictionCount = make(map[string]int64)
	mc.responseTime = make(map[string][]time.Duration)
	mc.errorCount = make(map[string]int64)
	mc.systemMetrics = SystemMetrics{
		ComponentMetrics: make(map[string]interface{}),
	}
	mc.startTime = time.Now()
}

// RecordCustomMetric records a custom metric
func (mc *MetricsCollector) RecordCustomMetric(name string, value interface{}) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	
	mc.systemMetrics.ComponentMetrics[name] = value
}

// GetMetricsForPrometheus returns metrics in a format suitable for Prometheus
func (mc *MetricsCollector) GetMetricsForPrometheus(ctx context.Context) map[string]float64 {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()
	
	metrics := make(map[string]float64)
	
	// System metrics
	metrics["total_requests"] = float64(mc.systemMetrics.TotalRequests)
	metrics["successful_requests"] = float64(mc.systemMetrics.SuccessfulRequests)
	metrics["failed_requests"] = float64(mc.systemMetrics.FailedRequests)
	metrics["error_rate"] = mc.systemMetrics.ErrorRate
	metrics["throughput_rps"] = mc.systemMetrics.ThroughputRPS
	metrics["average_latency_ms"] = float64(mc.systemMetrics.AverageLatency.Milliseconds())
	metrics["uptime_seconds"] = time.Since(mc.startTime).Seconds()
	
	// Prediction metrics by model
	for model, count := range mc.predictionCount {
		metrics["predictions_total_"+model] = float64(count)
		metrics["prediction_errors_total_"+model] = float64(mc.errorCount[model])
		
		if durations, exists := mc.responseTime[model]; exists && len(durations) > 0 {
			var total time.Duration
			for _, d := range durations {
				total += d
			}
			avgLatency := total / time.Duration(len(durations))
			metrics["prediction_latency_ms_"+model] = float64(avgLatency.Milliseconds())
		}
	}
	
	return metrics
}

// StartMetricsCollection starts background metrics collection
func (mc *MetricsCollector) StartMetricsCollection(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Perform periodic metrics calculations
			mc.mutex.Lock()
			mc.updateDerivedMetrics()
			mc.mutex.Unlock()
			
			// In production, this would export metrics to monitoring systems
			// like Prometheus, StatsD, etc.
		}
	}
}

// HealthScore returns a health score based on metrics
func (mc *MetricsCollector) HealthScore() float64 {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()
	
	// Simple health score calculation
	baseScore := 1.0
	
	// Penalize high error rate
	if mc.systemMetrics.ErrorRate > 0.05 { // 5% error rate threshold
		baseScore -= mc.systemMetrics.ErrorRate
	}
	
	// Penalize high latency
	if mc.systemMetrics.AverageLatency > 1*time.Second {
		latencyPenalty := float64(mc.systemMetrics.AverageLatency.Milliseconds()) / 5000.0 // 5 second max
		baseScore -= latencyPenalty
	}
	
	// Ensure score is between 0 and 1
	if baseScore < 0 {
		baseScore = 0
	}
	if baseScore > 1 {
		baseScore = 1
	}
	
	return baseScore
}

// AlertRule represents a monitoring alert rule
type AlertRule struct {
	Name        string
	Condition   func(*SystemMetrics) bool
	Message     string
	Severity    string
	Cooldown    time.Duration
	LastFired   time.Time
}

// CheckAlerts checks if any alert conditions are met
func (mc *MetricsCollector) CheckAlerts(rules []AlertRule) []AlertRule {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()
	
	var firedAlerts []AlertRule
	now := time.Now()
	
	for _, rule := range rules {
		// Check cooldown
		if now.Sub(rule.LastFired) < rule.Cooldown {
			continue
		}
		
		// Check condition
		if rule.Condition(&mc.systemMetrics) {
			rule.LastFired = now
			firedAlerts = append(firedAlerts, rule)
		}
	}
	
	return firedAlerts
}

// Helper function to copy map
func copyMap(original map[string]int64) map[string]int64 {
	copy := make(map[string]int64)
	for k, v := range original {
		copy[k] = v
	}
	return copy
} 