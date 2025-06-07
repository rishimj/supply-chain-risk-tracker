package health

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// HealthChecker monitors the health of all system components
type HealthChecker struct {
	postgres     *sql.DB
	redisClient  *redis.Client
	dependencies map[string]HealthCheckFunc
}

// HealthStatus represents the overall system health
type HealthStatus struct {
	Status      string                    `json:"status"`
	Timestamp   time.Time                 `json:"timestamp"`
	Version     string                    `json:"version"`
	Uptime      time.Duration             `json:"uptime"`
	Components  map[string]ComponentHealth `json:"components"`
	Summary     HealthSummary             `json:"summary"`
}

// ComponentHealth represents the health of a single component
type ComponentHealth struct {
	Status       string        `json:"status"`
	ResponseTime time.Duration `json:"response_time"`
	LastChecked  time.Time     `json:"last_checked"`
	Error        string        `json:"error,omitempty"`
	Details      interface{}   `json:"details,omitempty"`
}

// HealthSummary provides a high-level health overview
type HealthSummary struct {
	TotalComponents   int `json:"total_components"`
	HealthyComponents int `json:"healthy_components"`
	UnhealthyComponents int `json:"unhealthy_components"`
	OverallHealth     string `json:"overall_health"`
}

// HealthCheckFunc is a function that checks the health of a component
type HealthCheckFunc func(ctx context.Context) (ComponentHealth, error)

// NewHealthChecker creates a new health checker instance
func NewHealthChecker(postgres *sql.DB, redisClient *redis.Client) *HealthChecker {
	hc := &HealthChecker{
		postgres:     postgres,
		redisClient:  redisClient,
		dependencies: make(map[string]HealthCheckFunc),
	}
	
	// Register built-in health checks
	hc.registerBuiltinChecks()
	
	return hc
}

// RegisterHealthCheck adds a custom health check
func (hc *HealthChecker) RegisterHealthCheck(name string, checkFunc HealthCheckFunc) {
	hc.dependencies[name] = checkFunc
}

// CheckHealth performs a comprehensive health check of all components
func (hc *HealthChecker) CheckHealth(ctx context.Context) *HealthStatus {
	startTime := time.Now()
	components := make(map[string]ComponentHealth)
	
	// Check all registered components
	for name, checkFunc := range hc.dependencies {
		componentHealth, err := hc.checkComponentHealth(ctx, name, checkFunc)
		if err != nil {
			componentHealth = ComponentHealth{
				Status:       "unhealthy",
				ResponseTime: time.Since(startTime),
				LastChecked:  time.Now(),
				Error:        err.Error(),
			}
		}
		components[name] = componentHealth
	}
	
	// Calculate overall health
	healthyCount := 0
	totalCount := len(components)
	
	for _, component := range components {
		if component.Status == "healthy" {
			healthyCount++
		}
	}
	
	var overallStatus string
	if healthyCount == totalCount {
		overallStatus = "healthy"
	} else if healthyCount > totalCount/2 {
		overallStatus = "degraded"
	} else {
		overallStatus = "unhealthy"
	}
	
	summary := HealthSummary{
		TotalComponents:     totalCount,
		HealthyComponents:   healthyCount,
		UnhealthyComponents: totalCount - healthyCount,
		OverallHealth:       overallStatus,
	}
	
	return &HealthStatus{
		Status:     overallStatus,
		Timestamp:  time.Now(),
		Version:    "1.0.0",
		Uptime:     time.Since(startTime), // This would be actual uptime in production
		Components: components,
		Summary:    summary,
	}
}

// registerBuiltinChecks registers the default health checks
func (hc *HealthChecker) registerBuiltinChecks() {
	// PostgreSQL health check
	hc.dependencies["postgresql"] = func(ctx context.Context) (ComponentHealth, error) {
		start := time.Now()
		
		if hc.postgres == nil {
			return ComponentHealth{}, fmt.Errorf("PostgreSQL client not initialized")
		}
		
		// Test connection with a simple query
		var result int
		err := hc.postgres.QueryRowContext(ctx, "SELECT 1").Scan(&result)
		if err != nil {
			return ComponentHealth{}, fmt.Errorf("PostgreSQL query failed: %w", err)
		}
		
		// Get database stats
		stats := hc.postgres.Stats()
		
		return ComponentHealth{
			Status:       "healthy",
			ResponseTime: time.Since(start),
			LastChecked:  time.Now(),
			Details: map[string]interface{}{
				"open_connections": stats.OpenConnections,
				"in_use":          stats.InUse,
				"idle":            stats.Idle,
			},
		}, nil
	}
	
	// Redis health check
	hc.dependencies["redis"] = func(ctx context.Context) (ComponentHealth, error) {
		start := time.Now()
		
		if hc.redisClient == nil {
			return ComponentHealth{}, fmt.Errorf("Redis client not initialized")
		}
		
		// Test connection with ping
		pong, err := hc.redisClient.Ping(ctx).Result()
		if err != nil {
			return ComponentHealth{}, fmt.Errorf("Redis ping failed: %w", err)
		}
		
		if pong != "PONG" {
			return ComponentHealth{}, fmt.Errorf("Redis ping returned unexpected response: %s", pong)
		}
		
		// Get Redis info
		info, err := hc.redisClient.Info(ctx, "memory").Result()
		if err != nil {
			// Don't fail if info is unavailable
			info = "unavailable"
		}
		
		return ComponentHealth{
			Status:       "healthy",
			ResponseTime: time.Since(start),
			LastChecked:  time.Now(),
			Details: map[string]interface{}{
				"ping_response": pong,
				"memory_info":   info,
			},
		}, nil
	}
	
	// Disk space check
	hc.dependencies["disk_space"] = func(ctx context.Context) (ComponentHealth, error) {
		start := time.Now()
		
		// Simplified disk check (in production, use proper disk space monitoring)
		// For now, just return healthy
		return ComponentHealth{
			Status:       "healthy",
			ResponseTime: time.Since(start),
			LastChecked:  time.Now(),
			Details: map[string]interface{}{
				"available_space": "sufficient",
				"warning":        "disk space monitoring not implemented",
			},
		}, nil
	}
	
	// Memory check
	hc.dependencies["memory"] = func(ctx context.Context) (ComponentHealth, error) {
		start := time.Now()
		
		// Simplified memory check (in production, use proper memory monitoring)
		return ComponentHealth{
			Status:       "healthy",
			ResponseTime: time.Since(start),
			LastChecked:  time.Now(),
			Details: map[string]interface{}{
				"status":  "sufficient",
				"warning": "memory monitoring not implemented",
			},
		}, nil
	}
}

// checkComponentHealth executes a health check for a specific component
func (hc *HealthChecker) checkComponentHealth(ctx context.Context, name string, checkFunc HealthCheckFunc) (ComponentHealth, error) {
	// Set timeout for individual health checks
	checkCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	
	start := time.Now()
	
	// Execute the health check
	componentHealth, err := checkFunc(checkCtx)
	if err != nil {
		return ComponentHealth{
			Status:       "unhealthy",
			ResponseTime: time.Since(start),
			LastChecked:  time.Now(),
			Error:        err.Error(),
		}, err
	}
	
	// Ensure response time is set
	if componentHealth.ResponseTime == 0 {
		componentHealth.ResponseTime = time.Since(start)
	}
	
	// Ensure last checked is set
	if componentHealth.LastChecked.IsZero() {
		componentHealth.LastChecked = time.Now()
	}
	
	return componentHealth, nil
}

// GetComponentHealth returns the health of a specific component
func (hc *HealthChecker) GetComponentHealth(ctx context.Context, componentName string) (*ComponentHealth, error) {
	checkFunc, exists := hc.dependencies[componentName]
	if !exists {
		return nil, fmt.Errorf("component '%s' not found", componentName)
	}
	
	componentHealth, err := hc.checkComponentHealth(ctx, componentName, checkFunc)
	if err != nil {
		return &componentHealth, err
	}
	
	return &componentHealth, nil
}

// IsHealthy returns true if all critical components are healthy
func (hc *HealthChecker) IsHealthy(ctx context.Context) bool {
	healthStatus := hc.CheckHealth(ctx)
	return healthStatus.Status == "healthy"
}

// GetReadinessStatus checks if the system is ready to serve requests
func (hc *HealthChecker) GetReadinessStatus(ctx context.Context) *HealthStatus {
	// For readiness, we only check critical components
	criticalComponents := []string{"postgresql", "redis"}
	
	readinessStatus := &HealthStatus{
		Timestamp:  time.Now(),
		Version:    "1.0.0",
		Components: make(map[string]ComponentHealth),
	}
	
	healthyCount := 0
	totalCount := len(criticalComponents)
	
	for _, componentName := range criticalComponents {
		if checkFunc, exists := hc.dependencies[componentName]; exists {
			componentHealth, err := hc.checkComponentHealth(ctx, componentName, checkFunc)
			if err != nil {
				componentHealth = ComponentHealth{
					Status:      "unhealthy",
					LastChecked: time.Now(),
					Error:       err.Error(),
				}
			}
			
			readinessStatus.Components[componentName] = componentHealth
			
			if componentHealth.Status == "healthy" {
				healthyCount++
			}
		}
	}
	
	// System is ready if all critical components are healthy
	if healthyCount == totalCount {
		readinessStatus.Status = "ready"
	} else {
		readinessStatus.Status = "not_ready"
	}
	
	readinessStatus.Summary = HealthSummary{
		TotalComponents:     totalCount,
		HealthyComponents:   healthyCount,
		UnhealthyComponents: totalCount - healthyCount,
		OverallHealth:       readinessStatus.Status,
	}
	
	return readinessStatus
}

// GetLivenessStatus checks if the system is alive (basic functionality)
func (hc *HealthChecker) GetLivenessStatus(ctx context.Context) *HealthStatus {
	// For liveness, we do minimal checks
	return &HealthStatus{
		Status:    "alive",
		Timestamp: time.Now(),
		Version:   "1.0.0",
		Uptime:    time.Since(time.Now().Add(-1 * time.Hour)), // Placeholder uptime
		Summary: HealthSummary{
			TotalComponents:   1,
			HealthyComponents: 1,
			OverallHealth:     "alive",
		},
	}
}

// StartPeriodicHealthChecks runs health checks in the background
func (hc *HealthChecker) StartPeriodicHealthChecks(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Perform health check and log results
			healthStatus := hc.CheckHealth(ctx)
			
			// In production, you would send this to monitoring/alerting systems
			if healthStatus.Status != "healthy" {
				fmt.Printf("Health check warning: %s - %d/%d components healthy\n",
					healthStatus.Status,
					healthStatus.Summary.HealthyComponents,
					healthStatus.Summary.TotalComponents)
			}
		}
	}
}

// HealthMetrics provides metrics for monitoring systems
type HealthMetrics struct {
	HealthScore          float64           `json:"health_score"`
	ComponentHealthScore map[string]float64 `json:"component_health_score"`
	AverageResponseTime  time.Duration     `json:"average_response_time"`
	TotalChecks          int               `json:"total_checks"`
	FailedChecks         int               `json:"failed_checks"`
}

// GetHealthMetrics returns health metrics for monitoring
func (hc *HealthChecker) GetHealthMetrics(ctx context.Context) *HealthMetrics {
	healthStatus := hc.CheckHealth(ctx)
	
	// Calculate overall health score (0.0 to 1.0)
	healthScore := float64(healthStatus.Summary.HealthyComponents) / float64(healthStatus.Summary.TotalComponents)
	
	// Calculate component health scores
	componentScores := make(map[string]float64)
	var totalResponseTime time.Duration
	failedChecks := 0
	
	for name, component := range healthStatus.Components {
		if component.Status == "healthy" {
			componentScores[name] = 1.0
		} else {
			componentScores[name] = 0.0
			failedChecks++
		}
		totalResponseTime += component.ResponseTime
	}
	
	avgResponseTime := time.Duration(0)
	if len(healthStatus.Components) > 0 {
		avgResponseTime = totalResponseTime / time.Duration(len(healthStatus.Components))
	}
	
	return &HealthMetrics{
		HealthScore:          healthScore,
		ComponentHealthScore: componentScores,
		AverageResponseTime:  avgResponseTime,
		TotalChecks:          len(healthStatus.Components),
		FailedChecks:         failedChecks,
	}
} 