package processors

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"regexp"
	"strings"
	"time"

	"supply-chain-ml/pkg/config"
	"supply-chain-ml/pkg/features"
	"supply-chain-ml/pkg/pipeline/types"
)

// SECProcessor handles SEC filings ingestion and analysis
type SECProcessor struct {
	db           *sql.DB
	featureStore *features.Store
	config       *config.Config
	client       *http.Client
	isRunning    bool
	metrics      SECMetrics
}

// SECMetrics tracks SEC processor performance
type SECMetrics struct {
	FilingsProcessed   int64     `json:"filings_processed"`
	FeaturesExtracted  int64     `json:"features_extracted"`
	ErrorCount         int64     `json:"error_count"`
	LastProcessedTime  time.Time `json:"last_processed_time"`
	ProcessingDuration time.Duration `json:"processing_duration"`
}

// SECFiling represents a SEC filing document
type SECFiling struct {
	CIK          string    `json:"cik"`
	CompanyName  string    `json:"company_name"`
	FormType     string    `json:"form_type"`
	FilingDate   time.Time `json:"filing_date"`
	AccessionNo  string    `json:"accession_no"`
	Content      string    `json:"content"`
	URL          string    `json:"url"`
}

// NewSECProcessor creates a new SEC processor
func NewSECProcessor(db *sql.DB, featureStore *features.Store, cfg *config.Config) *SECProcessor {
	return &SECProcessor{
		db:           db,
		featureStore: featureStore,
		config:       cfg,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		isRunning: false,
	}
}

// Start begins SEC filings processing
func (sp *SECProcessor) Start(ctx context.Context) error {
	sp.isRunning = true
	log.Println("SEC Processor started")

	// Process filings every hour
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	// Initial processing
	if err := sp.processBatch(ctx); err != nil {
		log.Printf("SEC processor initial batch error: %v", err)
	}

	for {
		select {
		case <-ctx.Done():
			sp.isRunning = false
			return nil
		case <-ticker.C:
			if err := sp.processBatch(ctx); err != nil {
				log.Printf("SEC processor batch error: %v", err)
				sp.metrics.ErrorCount++
			}
		}
	}
}

// Stop gracefully stops the processor
func (sp *SECProcessor) Stop(ctx context.Context) error {
	sp.isRunning = false
	return nil
}

// HealthCheck verifies processor health
func (sp *SECProcessor) HealthCheck(ctx context.Context) bool {
	return sp.isRunning
}

// GetMetrics returns processor metrics
func (sp *SECProcessor) GetMetrics(ctx context.Context) SECMetrics {
	return sp.metrics
}

// processBatch processes a batch of SEC filings
func (sp *SECProcessor) processBatch(ctx context.Context) error {
	startTime := time.Now()
	
	// Get companies to monitor
	companies, err := sp.getMonitoredCompanies(ctx)
	if err != nil {
		return fmt.Errorf("failed to get monitored companies: %w", err)
	}

	log.Printf("Processing SEC filings for %d companies", len(companies))

	for _, company := range companies {
		if err := sp.processCompanyFilings(ctx, company); err != nil {
			log.Printf("Error processing filings for company %s: %v", company.Symbol, err)
			sp.metrics.ErrorCount++
			continue
		}
	}

	sp.metrics.ProcessingDuration = time.Since(startTime)
	sp.metrics.LastProcessedTime = time.Now()
	
	return nil
}

// getMonitoredCompanies retrieves companies to monitor for SEC filings
func (sp *SECProcessor) getMonitoredCompanies(ctx context.Context) ([]types.Company, error) {
	query := `
		SELECT symbol, cik, name 
		FROM companies 
		WHERE active = true 
		ORDER BY symbol
		LIMIT 100
	`

	rows, err := sp.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var companies []types.Company
	for rows.Next() {
		var company types.Company
		err := rows.Scan(&company.Symbol, &company.CIK, &company.Name)
		if err != nil {
			continue
		}
		companies = append(companies, company)
	}

	// If no companies in database, use default set that matches existing companies
	if len(companies) == 0 {
		companies = []types.Company{
			{Symbol: "AAPL", CIK: "0000320193", Name: "Apple Inc."},
			{Symbol: "AMZN", CIK: "0001018724", Name: "Amazon.com Inc."},
			{Symbol: "GOOGL", CIK: "0001652044", Name: "Alphabet Inc."},
			{Symbol: "INTC", CIK: "0000050863", Name: "Intel Corporation"},
			{Symbol: "JNJ", CIK: "0000200406", Name: "Johnson & Johnson"},
		}
	}

	return companies, nil
}

// processCompanyFilings processes recent filings for a company
func (sp *SECProcessor) processCompanyFilings(ctx context.Context, company types.Company) error {
	// Get recent filings from SEC EDGAR API
	filings, err := sp.getRecentFilings(ctx, company.CIK)
	if err != nil {
		return fmt.Errorf("failed to get recent filings: %w", err)
	}

	for _, filing := range filings {
		// Check if already processed
		if sp.isFilingProcessed(ctx, filing.AccessionNo) {
			continue
		}

		// Download and analyze filing content
		if err := sp.analyzeFilingContent(ctx, filing, company.Symbol); err != nil {
			log.Printf("Error analyzing filing %s: %v", filing.AccessionNo, err)
			continue
		}

		// Mark as processed
		sp.markFilingProcessed(ctx, filing.AccessionNo)
		sp.metrics.FilingsProcessed++
	}

	return nil
}

// getRecentFilings retrieves recent SEC filings for a CIK
func (sp *SECProcessor) getRecentFilings(ctx context.Context, cik string) ([]SECFiling, error) {
	// Simulate SEC EDGAR API call (in production, use real SEC API)
	// For demo, return mock filings
	mockFilings := []SECFiling{
		{
			CIK:         cik,
			FormType:    "10-K",
			FilingDate:  time.Now().AddDate(0, 0, -30),
			AccessionNo: fmt.Sprintf("%s-22-000001", cik),
			URL:         fmt.Sprintf("https://www.sec.gov/Archives/edgar/data/%s/000000000022000001.txt", cik),
		},
		{
			CIK:         cik,
			FormType:    "10-Q",
			FilingDate:  time.Now().AddDate(0, 0, -90),
			AccessionNo: fmt.Sprintf("%s-22-000002", cik),
			URL:         fmt.Sprintf("https://www.sec.gov/Archives/edgar/data/%s/000000000022000002.txt", cik),
		},
	}

	return mockFilings, nil
}

// analyzeFilingContent downloads and analyzes SEC filing content
func (sp *SECProcessor) analyzeFilingContent(ctx context.Context, filing SECFiling, companySymbol string) error {
	// Download filing content (simplified - would use real SEC URLs)
	content := sp.getMockFilingContent(filing.FormType)
	
	// Extract supply chain related features
	features := sp.extractSupplyChainFeatures(content, companySymbol)
	
	// Store features in feature store
	for _, feature := range features {
		if err := sp.featureStore.Store(ctx, feature); err != nil {
			log.Printf("Error storing feature %s: %v", feature.Name, err)
			continue
		}
		sp.metrics.FeaturesExtracted++
	}

	// Store filing metadata
	return sp.storeFilingMetadata(ctx, filing, companySymbol)
}

// extractSupplyChainFeatures extracts supply chain features from SEC filing text
func (sp *SECProcessor) extractSupplyChainFeatures(content, companySymbol string) []features.Feature {
	now := time.Now()
	var extractedFeatures []features.Feature

	// Supply chain risk keywords
	riskKeywords := []string{
		"supply chain", "supplier", "shortage", "disruption", "inventory",
		"raw materials", "manufacturing", "logistics", "procurement",
	}

	// Count risk keyword mentions
	riskCount := 0
	contentLower := strings.ToLower(content)
	for _, keyword := range riskKeywords {
		riskCount += strings.Count(contentLower, keyword)
	}

	extractedFeatures = append(extractedFeatures, features.Feature{
		CompanyID: companySymbol,
		Name:      "sec_supply_chain_mentions",
		Value:     float64(riskCount),
		Type:      "numerical",
		Source:    "sec_filings",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"form_type": "10-K",
			"keywords":  riskKeywords,
		},
	})

	// Extract inventory mention sentiment
	inventoryMentions := sp.extractInventoryMentions(content)
	inventorySentiment := sp.calculateSentiment(inventoryMentions)

	extractedFeatures = append(extractedFeatures, features.Feature{
		CompanyID: companySymbol,
		Name:      "sec_inventory_sentiment",
		Value:     inventorySentiment,
		Type:      "numerical",
		Source:    "sec_filings",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"mentions_count": len(inventoryMentions),
		},
	})

	// Extract supplier dependency indicators
	supplierDependency := sp.extractSupplierDependency(content)
	extractedFeatures = append(extractedFeatures, features.Feature{
		CompanyID: companySymbol,
		Name:      "sec_supplier_dependency",
		Value:     supplierDependency,
		Type:      "numerical",
		Source:    "sec_filings",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"extraction_method": "keyword_analysis",
		},
	})

	// Extract geographic risk exposure
	geoRisk := sp.extractGeographicRisk(content)
	extractedFeatures = append(extractedFeatures, features.Feature{
		CompanyID: companySymbol,
		Name:      "sec_geographic_risk",
		Value:     geoRisk,
		Type:      "numerical",
		Source:    "sec_filings",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"risk_regions": []string{"China", "Southeast Asia", "Eastern Europe"},
		},
	})

	return extractedFeatures
}

// extractInventoryMentions finds sentences mentioning inventory
func (sp *SECProcessor) extractInventoryMentions(content string) []string {
	// Simple regex to find sentences with inventory mentions
	inventoryRegex := regexp.MustCompile(`[^.!?]*\binventory\b[^.!?]*[.!?]`)
	return inventoryRegex.FindAllString(content, -1)
}

// calculateSentiment calculates sentiment score for text snippets
func (sp *SECProcessor) calculateSentiment(texts []string) float64 {
	if len(texts) == 0 {
		return 0.5 // Neutral
	}

	// Simple sentiment analysis based on positive/negative words
	positiveWords := []string{"increase", "growth", "improve", "strong", "stable", "adequate"}
	negativeWords := []string{"decrease", "decline", "shortage", "problem", "risk", "challenge", "difficult"}

	totalSentiment := 0.0
	for _, text := range texts {
		textLower := strings.ToLower(text)
		sentiment := 0.5 // Start neutral

		positiveCount := 0
		negativeCount := 0

		for _, word := range positiveWords {
			positiveCount += strings.Count(textLower, word)
		}

		for _, word := range negativeWords {
			negativeCount += strings.Count(textLower, word)
		}

		if positiveCount > negativeCount {
			sentiment = 0.7
		} else if negativeCount > positiveCount {
			sentiment = 0.3
		}

		totalSentiment += sentiment
	}

	return totalSentiment / float64(len(texts))
}

// extractSupplierDependency analyzes supplier dependency indicators
func (sp *SECProcessor) extractSupplierDependency(content string) float64 {
	// Look for concentration risk indicators
	concentrationKeywords := []string{
		"single supplier", "limited suppliers", "key supplier", "sole source",
		"supplier concentration", "depend on", "reliance on",
	}

	contentLower := strings.ToLower(content)
	dependencyScore := 0.0

	for _, keyword := range concentrationKeywords {
		if strings.Contains(contentLower, keyword) {
			dependencyScore += 0.2
		}
	}

	// Cap at 1.0
	if dependencyScore > 1.0 {
		dependencyScore = 1.0
	}

	return dependencyScore
}

// extractGeographicRisk analyzes geographic concentration risk
func (sp *SECProcessor) extractGeographicRisk(content string) float64 {
	// High-risk regions and keywords
	riskRegions := map[string]float64{
		"china":          0.3,
		"southeast asia": 0.2,
		"eastern europe": 0.15,
		"middle east":    0.1,
	}

	contentLower := strings.ToLower(content)
	totalRisk := 0.0

	for region, riskWeight := range riskRegions {
		if strings.Contains(contentLower, region) {
			totalRisk += riskWeight
		}
	}

	// Cap at 1.0
	if totalRisk > 1.0 {
		totalRisk = 1.0
	}

	return totalRisk
}

// getMockFilingContent returns mock SEC filing content for demonstration
func (sp *SECProcessor) getMockFilingContent(formType string) string {
	if formType == "10-K" {
		return `
		ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
		
		Our business depends on our ability to manage our supply chain effectively. We rely on suppliers located
		primarily in China and Southeast Asia for key components. Any disruption to our supply chain could
		materially impact our operations. We maintain adequate inventory levels to mitigate short-term disruptions,
		but prolonged shortages could affect our ability to meet customer demand.
		
		The Company sources raw materials from multiple suppliers to reduce concentration risk. However, certain
		specialized components are available from a limited number of suppliers, creating potential supply chain
		vulnerabilities. We continue to monitor our supplier base and work to diversify our sourcing strategy.
		`
	}
	
	return `
	QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
	
	During the quarter, we experienced some supply chain challenges related to global logistics disruptions.
	Our inventory turnover improved compared to the previous quarter. We are working closely with our key
	suppliers to ensure continuity of supply for critical components.
	`
}

// isFilingProcessed checks if a filing has already been processed
func (sp *SECProcessor) isFilingProcessed(ctx context.Context, accessionNo string) bool {
	query := `SELECT COUNT(*) FROM processed_filings WHERE accession_no = $1`
	
	var count int
	err := sp.db.QueryRowContext(ctx, query, accessionNo).Scan(&count)
	if err != nil {
		return false
	}
	
	return count > 0
}

// markFilingProcessed marks a filing as processed
func (sp *SECProcessor) markFilingProcessed(ctx context.Context, accessionNo string) error {
	query := `
		INSERT INTO processed_filings (accession_no, processed_at) 
		VALUES ($1, $2)
		ON CONFLICT (accession_no) DO NOTHING
	`
	
	_, err := sp.db.ExecContext(ctx, query, accessionNo, time.Now())
	return err
}

// storeFilingMetadata stores SEC filing metadata
func (sp *SECProcessor) storeFilingMetadata(ctx context.Context, filing SECFiling, companySymbol string) error {
	query := `
		INSERT INTO sec_filings (
			company_id, cik, form_type, filing_date, accession_no, url, processed_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (accession_no) DO UPDATE SET
			processed_at = EXCLUDED.processed_at
	`
	
	_, err := sp.db.ExecContext(
		ctx, query,
		companySymbol, filing.CIK, filing.FormType, filing.FilingDate,
		filing.AccessionNo, filing.URL, time.Now(),
	)
	
	return err
} 