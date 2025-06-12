package processors

import (
	"context"
	"database/sql"
	"log"
	"strings"
	"time"

	"supply-chain-ml/pkg/config"
	"supply-chain-ml/pkg/external"
	"supply-chain-ml/pkg/features"
	"supply-chain-ml/pkg/pipeline/types"
)

// NewsAnalyzer processes news sentiment analysis
type NewsAnalyzer struct {
	db           *sql.DB
	featureStore *features.Store
	config       *config.Config
	newsAPI      *external.NewsAPIClient
	isRunning    bool
	metrics      NewsMetrics
}

// NewsMetrics tracks news analyzer performance
type NewsMetrics struct {
	ArticlesProcessed int64     `json:"articles_processed"`
	FeaturesExtracted int64     `json:"features_extracted"`
	ErrorCount        int64     `json:"error_count"`
	LastProcessedTime time.Time `json:"last_processed_time"`
}

// NewNewsAnalyzer creates a new news analyzer
func NewNewsAnalyzer(db *sql.DB, featureStore *features.Store, cfg *config.Config) *NewsAnalyzer {
	var newsAPIClient *external.NewsAPIClient
	
	// Debug logging
	log.Printf("DEBUG: News API Key from config: '%s'", cfg.ExternalAPIs.NewsAPI.NewsAPIKey)
	
	if cfg.ExternalAPIs.NewsAPI.NewsAPIKey != "" {
		newsAPIClient = external.NewNewsAPIClient(cfg.ExternalAPIs.NewsAPI.NewsAPIKey)
		log.Printf("DEBUG: News API client initialized successfully")
	} else {
		log.Printf("DEBUG: News API key is empty, using mock data")
	}

	return &NewsAnalyzer{
		db:           db,
		featureStore: featureStore,
		config:       cfg,
		newsAPI:      newsAPIClient,
		isRunning:    false,
	}
}

// Start begins news analysis
func (na *NewsAnalyzer) Start(ctx context.Context) error {
	na.isRunning = true
	log.Println("News Analyzer started")

	// Process initial batch immediately
	log.Println("Processing initial news batch...")
	if err := na.processBatch(ctx); err != nil {
		log.Printf("Initial news processing error: %v", err)
		na.metrics.ErrorCount++
	}

	ticker := time.NewTicker(30 * time.Minute) // Changed from 2 hours to 30 minutes for testing
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			na.isRunning = false
			return nil
		case <-ticker.C:
			log.Println("Processing scheduled news batch...")
			if err := na.processBatch(ctx); err != nil {
				log.Printf("News analyzer error: %v", err)
				na.metrics.ErrorCount++
			}
		}
	}
}

func (na *NewsAnalyzer) Stop(ctx context.Context) error {
	na.isRunning = false
	return nil
}

func (na *NewsAnalyzer) HealthCheck(ctx context.Context) bool {
	return na.isRunning
}

func (na *NewsAnalyzer) GetMetrics(ctx context.Context) NewsMetrics {
	return na.metrics
}

func (na *NewsAnalyzer) processBatch(ctx context.Context) error {
	if na.newsAPI == nil {
		// Simplified news processing (mock mode)
		log.Println("News analyzer running in mock mode (no API key)")
		na.metrics.ArticlesProcessed++
		na.metrics.LastProcessedTime = time.Now()
		return nil
	}

	log.Println("Starting news processing with real API...")

	// Get companies to analyze
	companies, err := na.getMonitoredCompanies(ctx)
	if err != nil {
		log.Printf("Error getting monitored companies: %v", err)
		return err
	}

	log.Printf("Processing news for %d companies: %v", len(companies), func() []string {
		var symbols []string
		for _, c := range companies {
			symbols = append(symbols, c.Symbol)
		}
		return symbols
	}())

	// Process company-specific news
	for _, company := range companies {
		log.Printf("Processing news for %s (%s)...", company.Symbol, company.Name)
		if err := na.processCompanyNews(ctx, company.Symbol, company.Name); err != nil {
			log.Printf("Error processing news for %s: %v", company.Symbol, err)
			na.metrics.ErrorCount++
			continue
		}
	}

	// Process general supply chain news
	log.Println("Processing general supply chain news...")
	if err := na.processSupplyChainNews(ctx); err != nil {
		log.Printf("Error processing supply chain news: %v", err)
		na.metrics.ErrorCount++
	}

	log.Printf("News processing completed. Articles processed: %d, Features extracted: %d", 
		na.metrics.ArticlesProcessed, na.metrics.FeaturesExtracted)
	na.metrics.LastProcessedTime = time.Now()
	return nil
}

// getMonitoredCompanies gets companies to monitor for news
func (na *NewsAnalyzer) getMonitoredCompanies(ctx context.Context) ([]types.Company, error) {
	// Use company_info table since that's what exists
	query := `SELECT ticker, name, COALESCE(sector, '') FROM company_info ORDER BY ticker LIMIT 20`
	
	rows, err := na.db.QueryContext(ctx, query)
	if err != nil {
		// If query fails, use fallback companies
		return []types.Company{
			{Symbol: "AAPL", Name: "Apple Inc.", Sector: "Technology"},
			{Symbol: "AMZN", Name: "Amazon.com, Inc.", Sector: "Consumer Cyclical"},
			{Symbol: "GOOGL", Name: "Alphabet Inc.", Sector: "Communication Services"},
			{Symbol: "MSFT", Name: "Microsoft Corporation", Sector: "Technology"},
			{Symbol: "TSLA", Name: "Tesla, Inc.", Sector: "Consumer Cyclical"},
		}, nil
	}
	defer rows.Close()

	var companies []types.Company
	for rows.Next() {
		var company types.Company
		if err := rows.Scan(&company.Symbol, &company.Name, &company.Sector); err != nil {
			continue
		}
		companies = append(companies, company)
	}

	// If no companies found, use fallback
	if len(companies) == 0 {
		companies = []types.Company{
			{Symbol: "AAPL", Name: "Apple Inc.", Sector: "Technology"},
			{Symbol: "AMZN", Name: "Amazon.com, Inc.", Sector: "Consumer Cyclical"},
			{Symbol: "GOOGL", Name: "Alphabet Inc.", Sector: "Communication Services"},
			{Symbol: "MSFT", Name: "Microsoft Corporation", Sector: "Technology"},
			{Symbol: "TSLA", Name: "Tesla, Inc.", Sector: "Consumer Cyclical"},
		}
	}

	return companies, nil
}

// processCompanyNews processes news for a specific company
func (na *NewsAnalyzer) processCompanyNews(ctx context.Context, symbol, name string) error {
	// Get recent news for this company (last 3 days)
	newsResponse, err := na.newsAPI.GetCompanyNews(ctx, name, symbol, 3)
	if err != nil {
		return err
	}

	for _, article := range newsResponse.Articles {
		// Calculate sentiment score
		sentiment := na.calculateSentiment(article.Title + " " + article.Description)
		
		// Calculate supply chain relevance
		relevance := na.calculateSupplyChainRelevance(article.Title + " " + article.Description)
		
		// Store news article in database
		if err := na.storeNewsArticle(ctx, symbol, article, sentiment, relevance); err != nil {
			log.Printf("Error storing news article: %v", err)
			continue
		}

		// Generate news sentiment feature
		feature := features.Feature{
			CompanyID: symbol,
			Name:      "news_sentiment_score",
			Value:     sentiment,
			Type:      "numerical",
			Source:    "news_analysis",
			Timestamp: article.PublishedAt,
			Metadata: map[string]interface{}{
				"headline":              article.Title,
				"source":                article.Source.Name,
				"supply_chain_relevance": relevance,
				"url":                   article.URL,
			},
		}

		if err := na.featureStore.Store(ctx, feature); err != nil {
			log.Printf("Error storing news feature: %v", err)
			continue
		}

		na.metrics.FeaturesExtracted++
		na.metrics.ArticlesProcessed++
	}

	return nil
}

// processSupplyChainNews processes general supply chain news
func (na *NewsAnalyzer) processSupplyChainNews(ctx context.Context) error {
	// Get supply chain related news (last 2 days)
	newsResponse, err := na.newsAPI.GetSupplyChainNews(ctx, 2)
	if err != nil {
		return err
	}

	for _, article := range newsResponse.Articles {
		// Calculate sentiment score
		sentiment := na.calculateSentiment(article.Title + " " + article.Description)
		
		// Store as general supply chain news
		if err := na.storeNewsArticle(ctx, "GLOBAL", article, sentiment, 1.0); err != nil {
			log.Printf("Error storing supply chain news: %v", err)
			continue
		}

		na.metrics.ArticlesProcessed++
	}

	return nil
}

// calculateSentiment calculates sentiment score for text
func (na *NewsAnalyzer) calculateSentiment(text string) float64 {
	if text == "" {
		return 0.5 // Neutral
	}

	textLower := strings.ToLower(text)
	
	// Positive keywords
	positiveWords := []string{
		"growth", "increase", "positive", "strong", "good", "excellent", "up",
		"rise", "gain", "success", "improve", "better", "optimistic", "boost",
		"expand", "profit", "revenue", "efficient", "stable", "recovery",
	}
	
	// Negative keywords
	negativeWords := []string{
		"decline", "decrease", "negative", "weak", "bad", "poor", "down",
		"fall", "loss", "failure", "worse", "concern", "risk", "problem",
		"shortage", "disruption", "delay", "crisis", "challenge", "cut",
	}

	positiveCount := 0
	negativeCount := 0

	for _, word := range positiveWords {
		positiveCount += strings.Count(textLower, word)
	}

	for _, word := range negativeWords {
		negativeCount += strings.Count(textLower, word)
	}

	if positiveCount == 0 && negativeCount == 0 {
		return 0.5 // Neutral
	}

	total := positiveCount + negativeCount
	return float64(positiveCount) / float64(total)
}

// calculateSupplyChainRelevance calculates how relevant the article is to supply chain
func (na *NewsAnalyzer) calculateSupplyChainRelevance(text string) float64 {
	if text == "" {
		return 0.0
	}

	textLower := strings.ToLower(text)
	
	supplyChainKeywords := []string{
		"supply chain", "supplier", "logistics", "shipping", "manufacturing",
		"inventory", "procurement", "distribution", "warehouse", "factory",
		"production", "delivery", "transportation", "raw materials", "components",
	}

	relevanceScore := 0.0
	for _, keyword := range supplyChainKeywords {
		if strings.Contains(textLower, keyword) {
			relevanceScore += 0.2
		}
	}

	// Cap at 1.0
	if relevanceScore > 1.0 {
		relevanceScore = 1.0
	}

	return relevanceScore
}

// storeNewsArticle stores a news article in the database
func (na *NewsAnalyzer) storeNewsArticle(ctx context.Context, companyID string, article external.NewsArticle, sentiment, relevance float64) error {
	query := `
		INSERT INTO news_articles (
			company_id, title, content, url, source, author, published_date,
			sentiment_score, supply_chain_relevance
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`
	
	_, err := na.db.ExecContext(ctx, query,
		companyID,
		article.Title,
		article.Content,
		article.URL,
		article.Source.Name,
		article.Author,
		article.PublishedAt,
		sentiment,
		relevance,
	)
	
	return err
} 