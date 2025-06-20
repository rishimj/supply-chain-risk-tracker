package api

import (
	"github.com/gin-gonic/gin"
)

// SetupRoutes configures all API routes
func SetupRoutes(router *gin.Engine, handlers *Handlers) {
	// Add global middleware
	router.Use(CORSMiddleware())
	router.Use(RequestIDMiddleware())
	router.Use(MetricsMiddleware())
	router.Use(ErrorHandlerMiddleware())
	
	// Apply rate limiting (100 requests per minute)
	router.Use(RateLimitMiddleware(100))
	
	// Health check endpoints (no auth required)
	router.GET("/health", handlers.Health)
	router.GET("/health/live", handlers.Health)
	router.GET("/health/ready", handlers.Health)
	
	// API v1 routes
	v1 := router.Group("/api/v1")
	{
		// Disable auth for now to test the frontend
		// v1.Use(AuthMiddleware())
		
		// Health endpoints (alias to /health)
		v1.GET("/health", handlers.Health)
		
		// Simple test endpoint using handler method
		v1.GET("/test-handler", handlers.TestEndpoint)
		
		// Disruption risk endpoint
		v1.GET("/disruption-risk", handlers.GetDisruptionRisk)
		
		// Prediction endpoints
		predictions := v1.Group("/predictions")
		{
			predictions.POST("/predict", handlers.PredictRisk)
			predictions.POST("/batch", handlers.BatchPredict)
		}
		
		// Company endpoints
		companies := v1.Group("/companies")
		{
			companies.GET("/:id/risk", handlers.GetCompanyRisk)
			companies.GET("/:id/features", handlers.GetCompanyFeatures)
		}
		
		// Feature endpoints
		features := v1.Group("/features")
		{
			features.GET("/:company_id", handlers.GetFeatures)
			features.POST("/", handlers.StoreFeatures)
		}
		
		// Model management endpoints
		models := v1.Group("/models")
		{
			models.GET("/status", handlers.GetModelStatus)
			models.POST("/retrain", handlers.TriggerRetrain)
		}
		
		// System monitoring endpoints
		system := v1.Group("/system")
		{
			system.GET("/health", handlers.Health) // Alias to main health endpoint
			system.GET("/metrics", handlers.GetMetrics)
			system.GET("/status", handlers.GetSystemStatus)
			system.GET("/alerts", handlers.GetAlerts)
		}
		
		// Analytics endpoints
		analytics := v1.Group("/analytics")
		{
			analytics.GET("/risk-trends", handlers.GetRiskTrends)
			analytics.GET("/sector-analysis", handlers.GetSectorAnalysis)
		}
		
		// Predictions endpoint with real data
		v1.GET("/predictions", handlers.GetPredictions)
	}
	
	// Admin endpoints (if needed in the future)
	admin := router.Group("/admin")
	{
		admin.Use(AuthMiddleware()) // Would use admin-specific auth
		// Future admin endpoints can be added here
	}
} 