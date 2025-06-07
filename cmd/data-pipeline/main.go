package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"supply-chain-ml/pkg/config"
	"supply-chain-ml/pkg/database"
	"supply-chain-ml/pkg/features"
	"supply-chain-ml/pkg/pipeline"
)

func main() {
	log.Println("Starting Supply Chain Data Pipeline Service...")

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

	// Initialize feature store
	featureStore := features.NewStore(redis, postgres)

	// Initialize pipeline orchestrator
	orchestrator := pipeline.NewOrchestrator(postgres, redis, neo4j, featureStore, cfg)

	// Create context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start all pipeline services
	var wg sync.WaitGroup

	// Start SEC filings processor
	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Println("Starting SEC filings processor...")
		if err := orchestrator.StartSECProcessor(ctx); err != nil {
			log.Printf("SEC processor error: %v", err)
		}
	}()

	// Start earnings call analyzer
	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Println("Starting earnings call analyzer...")
		if err := orchestrator.StartEarningsAnalyzer(ctx); err != nil {
			log.Printf("Earnings analyzer error: %v", err)
		}
	}()

	// Start financial data ingestion
	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Println("Starting financial data ingestion...")
		if err := orchestrator.StartFinancialDataIngestion(ctx); err != nil {
			log.Printf("Financial data ingestion error: %v", err)
		}
	}()

	// Start news sentiment analysis
	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Println("Starting news sentiment analysis...")
		if err := orchestrator.StartNewsSentimentAnalysis(ctx); err != nil {
			log.Printf("News sentiment analysis error: %v", err)
		}
	}()

	// Start supplier network analysis
	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Println("Starting supplier network analysis...")
		if err := orchestrator.StartSupplierNetworkAnalysis(ctx); err != nil {
			log.Printf("Supplier network analysis error: %v", err)
		}
	}()

	// Start real-time stream processor
	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Println("Starting real-time stream processor...")
		if err := orchestrator.StartStreamProcessor(ctx); err != nil {
			log.Printf("Stream processor error: %v", err)
		}
	}()

	// Start batch feature engineering
	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Println("Starting batch feature engineering...")
		if err := orchestrator.StartBatchProcessor(ctx); err != nil {
			log.Printf("Batch processor error: %v", err)
		}
	}()

	// Start feature quality monitoring
	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Println("Starting feature quality monitoring...")
		if err := orchestrator.StartFeatureMonitoring(ctx); err != nil {
			log.Printf("Feature monitoring error: %v", err)
		}
	}()

	log.Println("All pipeline services started successfully")

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down data pipeline...")

	// Cancel context to stop all services
	cancel()

	// Wait for all services to stop gracefully (with timeout)
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		log.Println("All services stopped gracefully")
	case <-time.After(30 * time.Second):
		log.Println("Timeout waiting for services to stop")
	}

	log.Println("Data pipeline service exited")
} 