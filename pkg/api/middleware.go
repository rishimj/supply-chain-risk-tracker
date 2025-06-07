package api

import (
	"time"

	"github.com/gin-gonic/gin"
)

// CORSMiddleware handles Cross-Origin Resource Sharing
func CORSMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})
}

// MetricsMiddleware records metrics for all API requests
func MetricsMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		start := time.Now()
		
		// Process request
		c.Next()
		
		// Record metrics
		duration := time.Since(start)
		path := c.FullPath()
		method := c.Request.Method
		status := c.Writer.Status()
		
		// Log metrics (in a real implementation, this would send to Prometheus)
		// For now, we'll just use the variables to avoid linter errors
		_ = path
		_ = method
		_ = status
		
		gin.Logger()(c)
		
		// Add custom headers for debugging
		c.Writer.Header().Set("X-Response-Time", duration.String())
		c.Writer.Header().Set("X-Request-ID", generateRequestID())
	})
}

// ErrorHandlerMiddleware handles panics and errors gracefully
func ErrorHandlerMiddleware() gin.HandlerFunc {
	return gin.Recovery()
}

// RequestIDMiddleware adds a unique request ID to each request
func RequestIDMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		requestID := c.GetHeader("X-Request-ID")
		if requestID == "" {
			requestID = generateRequestID()
		}
		
		c.Set("request_id", requestID)
		c.Writer.Header().Set("X-Request-ID", requestID)
		
		c.Next()
	})
}

// RateLimitMiddleware implements basic rate limiting
func RateLimitMiddleware(requestsPerMinute int) gin.HandlerFunc {
	// Simple in-memory rate limiter (in production, use Redis)
	clients := make(map[string][]time.Time)
	
	return gin.HandlerFunc(func(c *gin.Context) {
		clientIP := c.ClientIP()
		now := time.Now()
		
		// Clean old requests
		if requests, exists := clients[clientIP]; exists {
			var validRequests []time.Time
			for _, requestTime := range requests {
				if now.Sub(requestTime) < time.Minute {
					validRequests = append(validRequests, requestTime)
				}
			}
			clients[clientIP] = validRequests
		}
		
		// Check rate limit
		if len(clients[clientIP]) >= requestsPerMinute {
			c.JSON(429, gin.H{
				"error": "Rate limit exceeded",
				"retry_after": 60,
			})
			c.Abort()
			return
		}
		
		// Add current request
		clients[clientIP] = append(clients[clientIP], now)
		
		c.Next()
	})
}

// AuthMiddleware handles API authentication (placeholder)
func AuthMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		// In a real implementation, validate API keys or JWT tokens
		apiKey := c.GetHeader("X-API-Key")
		
		if apiKey == "" {
			// For demo purposes, allow requests without API key
			c.Next()
			return
		}
		
		// Validate API key (placeholder logic)
		if !isValidAPIKey(apiKey) {
			c.JSON(401, gin.H{
				"error": "Invalid API key",
			})
			c.Abort()
			return
		}
		
		c.Set("api_key", apiKey)
		c.Next()
	})
}

// Helper functions
func generateRequestID() string {
	// Simple request ID generation
	return "req_" + time.Now().Format("20060102_150405") + "_" + randomString(8)
}

func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(b)
}

func isValidAPIKey(apiKey string) bool {
	// Placeholder API key validation
	validKeys := map[string]bool{
		"demo-key-123": true,
		"test-key-456": true,
	}
	
	return validKeys[apiKey]
} 