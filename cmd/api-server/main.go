package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"supply-chain-ml/pkg/api"
	"supply-chain-ml/pkg/config"
	"supply-chain-ml/pkg/database"
	"supply-chain-ml/pkg/features"
	"supply-chain-ml/pkg/health"
	"supply-chain-ml/pkg/ml"
	"supply-chain-ml/pkg/monitoring"

	"github.com/gin-gonic/gin"
)

func main() {
	// Load configuration
	cfg, err := config.Load("configs/config.yaml")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize database connections
	postgres, err := database.NewPostgresDB(cfg.Database.Postgres)
	if err != nil {
		log.Fatalf("Failed to connect to PostgreSQL: %v", err)
	}
	defer postgres.Close()

	redis, err := database.NewRedisClient(cfg.Database.Redis)
	if err != nil {
		log.Fatalf("Failed to connect to Redis: %v", err)
	}
	defer redis.Close()

	neo4j, err := database.NewNeo4jDriver(cfg.Database.Neo4j)
	if err != nil {
		log.Fatalf("Failed to connect to Neo4j: %v", err)
	}
	defer neo4j.Close(context.Background())

	// Initialize components
	featureStore := features.NewStore(redis, postgres)
	modelServer := ml.NewModelServer()
	healthChecker := health.NewHealthChecker(postgres, redis)
	metricsCollector := monitoring.NewMetricsCollector()

	// Initialize API handlers
	handlers := api.NewHandlers(featureStore, modelServer, healthChecker, metricsCollector, postgres)

	// Setup Gin router
	router := gin.Default()
	
	// Setup routes using the existing route configuration
	api.SetupRoutes(router, handlers)

	// Create HTTP server
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.Port),
		Handler:      router,
		ReadTimeout:  cfg.Server.Timeout,
		WriteTimeout: cfg.Server.Timeout,
	}

	// Start server in goroutine
	go func() {
		log.Printf("Starting API server on port %d", cfg.Server.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	// Give outstanding requests 30 seconds to complete
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	if err := server.Shutdown(ctx); err != nil {
		log.Printf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited")
} 