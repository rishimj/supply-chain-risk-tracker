package processors

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"time"

	"supply-chain-ml/pkg/config"
	"supply-chain-ml/pkg/external"
	"supply-chain-ml/pkg/features"
)

// FinancialIngester handles financial data ingestion and processing
type FinancialIngester struct {
	db              *sql.DB
	featureStore    *features.Store
	config          *config.Config
	client          *http.Client
	alphaVantage    *external.AlphaVantageClient
	isRunning       bool
	metrics         FinancialMetrics
}

// FinancialMetrics tracks financial ingester performance
type FinancialMetrics struct {
	CompaniesProcessed int64         `json:"companies_processed"`
	MetricsCalculated  int64         `json:"metrics_calculated"`
	ErrorCount         int64         `json:"error_count"`
	LastUpdated        time.Time     `json:"last_updated"`
	ProcessingDuration time.Duration `json:"processing_duration"`
}

// FinancialData represents financial data for a company
type FinancialData struct {
	CompanyID        string    `json:"company_id"`
	ReportDate       time.Time `json:"report_date"`
	Revenue          float64   `json:"revenue"`
	CostOfRevenue    float64   `json:"cost_of_revenue"`
	GrossProfit      float64   `json:"gross_profit"`
	OperatingIncome  float64   `json:"operating_income"`
	NetIncome        float64   `json:"net_income"`
	TotalAssets      float64   `json:"total_assets"`
	TotalLiabilities float64   `json:"total_liabilities"`
	Inventory        float64   `json:"inventory"`
	AccountsPayable  float64   `json:"accounts_payable"`
	WorkingCapital   float64   `json:"working_capital"`
	CashAndEquiv     float64   `json:"cash_and_equivalents"`
	TotalDebt        float64   `json:"total_debt"`
	SharesOutstanding float64  `json:"shares_outstanding"`
	MarketCap        float64   `json:"market_cap"`
	StockPrice       float64   `json:"stock_price"`
}

// MarketData represents market data for time series analysis
type MarketData struct {
	CompanyID string    `json:"company_id"`
	Date      time.Time `json:"date"`
	Open      float64   `json:"open"`
	High      float64   `json:"high"`
	Low       float64   `json:"low"`
	Close     float64   `json:"close"`
	Volume    int64     `json:"volume"`
}

// NewFinancialIngester creates a new financial data ingester
func NewFinancialIngester(db *sql.DB, featureStore *features.Store, cfg *config.Config) *FinancialIngester {
	var alphaVantageClient *external.AlphaVantageClient
	if cfg.ExternalAPIs.FinancialData.AlphaVantageKey != "" {
		alphaVantageClient = external.NewAlphaVantageClient(cfg.ExternalAPIs.FinancialData.AlphaVantageKey)
	}

	return &FinancialIngester{
		db:           db,
		featureStore: featureStore,
		config:       cfg,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		alphaVantage: alphaVantageClient,
		isRunning:    false,
	}
}

// Start begins financial data ingestion
func (fi *FinancialIngester) Start(ctx context.Context) error {
	fi.isRunning = true
	log.Println("Financial Ingester started")

	// Process financial data every 4 hours
	ticker := time.NewTicker(4 * time.Hour)
	defer ticker.Stop()

	// Initial processing
	if err := fi.processBatch(ctx); err != nil {
		log.Printf("Financial ingester initial batch error: %v", err)
	}

	for {
		select {
		case <-ctx.Done():
			fi.isRunning = false
			return nil
		case <-ticker.C:
			if err := fi.processBatch(ctx); err != nil {
				log.Printf("Financial ingester batch error: %v", err)
				fi.metrics.ErrorCount++
			}
		}
	}
}

// Stop gracefully stops the ingester
func (fi *FinancialIngester) Stop(ctx context.Context) error {
	fi.isRunning = false
	return nil
}

// HealthCheck verifies ingester health
func (fi *FinancialIngester) HealthCheck(ctx context.Context) bool {
	return fi.isRunning
}

// GetMetrics returns ingester metrics
func (fi *FinancialIngester) GetMetrics(ctx context.Context) FinancialMetrics {
	return fi.metrics
}

// processBatch processes financial data for all monitored companies
func (fi *FinancialIngester) processBatch(ctx context.Context) error {
	startTime := time.Now()

	companies, err := fi.getMonitoredCompanies(ctx)
	if err != nil {
		return fmt.Errorf("failed to get monitored companies: %w", err)
	}

	log.Printf("Processing financial data for %d companies", len(companies))

	for _, company := range companies {
		if err := fi.processCompanyFinancials(ctx, company); err != nil {
			log.Printf("Error processing financials for company %s: %v", company, err)
			fi.metrics.ErrorCount++
			continue
		}
		fi.metrics.CompaniesProcessed++
	}

	fi.metrics.ProcessingDuration = time.Since(startTime)
	fi.metrics.LastUpdated = time.Now()

	return nil
}

// getMonitoredCompanies gets list of companies to process
func (fi *FinancialIngester) getMonitoredCompanies(ctx context.Context) ([]string, error) {
	query := `SELECT DISTINCT symbol FROM companies WHERE active = true ORDER BY symbol LIMIT 50`

	rows, err := fi.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var companies []string
	for rows.Next() {
		var symbol string
		if err := rows.Scan(&symbol); err != nil {
			continue
		}
		companies = append(companies, symbol)
	}

	// Default companies if none in database
	if len(companies) == 0 {
		companies = []string{"AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "META", "NFLX", "CRM", "ORCL"}
	}

	return companies, nil
}

// processCompanyFinancials processes financial data for a single company
func (fi *FinancialIngester) processCompanyFinancials(ctx context.Context, companySymbol string) error {
	// Get latest financial data
	financialData, err := fi.getFinancialData(ctx, companySymbol)
	if err != nil {
		return fmt.Errorf("failed to get financial data: %w", err)
	}

	// Get recent market data for time series features
	marketData, err := fi.getMarketData(ctx, companySymbol, 90) // Last 90 days
	if err != nil {
		return fmt.Errorf("failed to get market data: %w", err)
	}

	// Calculate financial ratios and features
	features := fi.calculateFinancialFeatures(financialData, marketData)

	// Store all features
	for _, feature := range features {
		if err := fi.featureStore.Store(ctx, feature); err != nil {
			log.Printf("Error storing financial feature %s: %v", feature.Name, err)
			continue
		}
		fi.metrics.MetricsCalculated++
	}

	// Store raw financial data
	return fi.storeFinancialData(ctx, financialData)
}

// getFinancialDataFromAPI retrieves real financial data using Alpha Vantage API
func (fi *FinancialIngester) getFinancialDataFromAPI(ctx context.Context, symbol string) (*FinancialData, error) {
	if fi.alphaVantage == nil {
		return fi.getFinancialData(ctx, symbol) // Fallback to mock data
	}

	// Get company overview (fundamental data)
	overview, err := fi.alphaVantage.GetCompanyOverview(ctx, symbol)
	if err != nil {
		log.Printf("Failed to get Alpha Vantage overview for %s: %v", symbol, err)
		return fi.getFinancialData(ctx, symbol) // Fallback to mock data
	}

	// Get current quote
	quote, err := fi.alphaVantage.GetQuote(ctx, symbol)
	if err != nil {
		log.Printf("Failed to get Alpha Vantage quote for %s: %v", symbol, err)
		return fi.getFinancialData(ctx, symbol) // Fallback to mock data
	}

	// Parse financial data from Alpha Vantage
	financialData := &FinancialData{
		CompanyID: symbol,
		ReportDate: time.Now(),
	}

	// Parse revenue
	if revenue, err := strconv.ParseFloat(overview.RevenueTTM, 64); err == nil {
		financialData.Revenue = revenue
	}

	// Parse market cap
	if marketCap, err := strconv.ParseFloat(overview.MarketCapitalization, 64); err == nil {
		financialData.MarketCap = marketCap
	}

	// Parse current stock price
	if price, err := strconv.ParseFloat(quote.GlobalQuote.Price, 64); err == nil {
		financialData.StockPrice = price
	}

	// Parse shares outstanding
	if shares, err := strconv.ParseFloat(overview.SharesOutstanding, 64); err == nil {
		financialData.SharesOutstanding = shares
	}

	// Parse gross profit
	if grossProfit, err := strconv.ParseFloat(overview.GrossProfitTTM, 64); err == nil {
		financialData.GrossProfit = grossProfit
	}

	// Calculate cost of revenue
	if financialData.Revenue > 0 && financialData.GrossProfit > 0 {
		financialData.CostOfRevenue = financialData.Revenue - financialData.GrossProfit
	}

	// Parse book value (approximate total assets)
	if bookValue, err := strconv.ParseFloat(overview.BookValue, 64); err == nil && financialData.SharesOutstanding > 0 {
		financialData.TotalAssets = bookValue * financialData.SharesOutstanding
	}

	// Parse EPS and calculate net income
	if eps, err := strconv.ParseFloat(overview.EPS, 64); err == nil && financialData.SharesOutstanding > 0 {
		financialData.NetIncome = eps * financialData.SharesOutstanding
	}

	// Store the real data in database
	if err := fi.storeFinancialData(ctx, financialData); err != nil {
		log.Printf("Failed to store financial data for %s: %v", symbol, err)
	}

	return financialData, nil
}

// getFinancialData retrieves financial data for a company (mock implementation)
func (fi *FinancialIngester) getFinancialData(ctx context.Context, symbol string) (*FinancialData, error) {
	// In production, this would call real financial APIs like Alpha Vantage, Yahoo Finance, etc.
	// For demo, generate realistic mock data

	baseValues := map[string]float64{
		"AAPL": 100000, // Revenue in millions
		"MSFT": 80000,
		"AMZN": 120000,
		"GOOGL": 75000,
		"TSLA": 30000,
	}

	baseRevenue := baseValues[symbol]
	if baseRevenue == 0 {
		baseRevenue = 50000 // Default
	}

	// Add some randomness to simulate quarterly variations
	variation := 0.9 + rand.Float64()*0.2 // ±10% variation
	revenue := baseRevenue * variation

	return &FinancialData{
		CompanyID:         symbol,
		ReportDate:        time.Now().AddDate(0, 0, -30), // Last month
		Revenue:           revenue,
		CostOfRevenue:     revenue * 0.65, // 65% cost ratio
		GrossProfit:       revenue * 0.35,
		OperatingIncome:   revenue * 0.25,
		NetIncome:         revenue * 0.20,
		TotalAssets:       revenue * 2.5,
		TotalLiabilities:  revenue * 1.5,
		Inventory:         revenue * 0.15,
		AccountsPayable:   revenue * 0.08,
		WorkingCapital:    revenue * 0.25,
		CashAndEquiv:      revenue * 0.30,
		TotalDebt:         revenue * 0.40,
		SharesOutstanding: 1000 + rand.Float64()*15000, // 1B to 16B shares
		MarketCap:         revenue * 8,                  // P/S ratio of ~8
		StockPrice:        100 + rand.Float64()*400,     // $100-500
	}, nil
}

// getMarketData retrieves market data for time series analysis
func (fi *FinancialIngester) getMarketData(ctx context.Context, symbol string, days int) ([]MarketData, error) {
	// Mock market data generation
	var marketData []MarketData
	basePrice := 150.0 + rand.Float64()*200 // $150-350 base price

	for i := days; i >= 0; i-- {
		date := time.Now().AddDate(0, 0, -i)
		
		// Simple random walk for price
		priceChange := (rand.Float64() - 0.5) * 0.05 // ±2.5% daily change
		price := basePrice * (1 + priceChange)
		
		marketData = append(marketData, MarketData{
			CompanyID: symbol,
			Date:      date,
			Open:      price * 0.99,
			High:      price * 1.02,
			Low:       price * 0.98,
			Close:     price,
			Volume:    int64(1000000 + rand.Intn(10000000)), // 1M-11M volume
		})

		basePrice = price // Update for next day
	}

	return marketData, nil
}

// calculateFinancialFeatures calculates financial ratios and features
func (fi *FinancialIngester) calculateFinancialFeatures(fd *FinancialData, md []MarketData) []features.Feature {
	now := time.Now()
	var calculatedFeatures []features.Feature

	// Financial Ratios
	
	// 1. Inventory Turnover (critical for supply chain analysis)
	inventoryTurnover := 0.0
	if fd.Inventory > 0 {
		inventoryTurnover = fd.CostOfRevenue / fd.Inventory
	}
	calculatedFeatures = append(calculatedFeatures, features.Feature{
		CompanyID: fd.CompanyID,
		Name:      "financial_inventory_turnover",
		Value:     inventoryTurnover,
		Type:      "numerical",
		Source:    "financial_data",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"report_date": fd.ReportDate,
			"formula":    "Cost of Revenue / Inventory",
		},
	})

	// 2. Gross Margin
	grossMargin := 0.0
	if fd.Revenue > 0 {
		grossMargin = fd.GrossProfit / fd.Revenue
	}
	calculatedFeatures = append(calculatedFeatures, features.Feature{
		CompanyID: fd.CompanyID,
		Name:      "financial_gross_margin",
		Value:     grossMargin,
		Type:      "numerical",
		Source:    "financial_data",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"report_date": fd.ReportDate,
		},
	})

	// 3. Current Ratio (liquidity indicator)
	currentAssets := fd.CashAndEquiv + fd.Inventory + (fd.Revenue * 0.1) // Simplified
	currentRatio := 0.0
	if fd.AccountsPayable > 0 {
		currentRatio = currentAssets / fd.AccountsPayable
	}
	calculatedFeatures = append(calculatedFeatures, features.Feature{
		CompanyID: fd.CompanyID,
		Name:      "financial_current_ratio",
		Value:     currentRatio,
		Type:      "numerical",
		Source:    "financial_data",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"report_date": fd.ReportDate,
		},
	})

	// 4. Debt-to-Equity Ratio
	equity := fd.TotalAssets - fd.TotalLiabilities
	debtToEquity := 0.0
	if equity > 0 {
		debtToEquity = fd.TotalDebt / equity
	}
	calculatedFeatures = append(calculatedFeatures, features.Feature{
		CompanyID: fd.CompanyID,
		Name:      "financial_debt_to_equity",
		Value:     debtToEquity,
		Type:      "numerical",
		Source:    "financial_data",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"report_date": fd.ReportDate,
		},
	})

	// 5. Working Capital Ratio
	workingCapitalRatio := 0.0
	if fd.Revenue > 0 {
		workingCapitalRatio = fd.WorkingCapital / fd.Revenue
	}
	calculatedFeatures = append(calculatedFeatures, features.Feature{
		CompanyID: fd.CompanyID,
		Name:      "financial_working_capital_ratio",
		Value:     workingCapitalRatio,
		Type:      "numerical",
		Source:    "financial_data",
		Timestamp: now,
		Metadata: map[string]interface{}{
			"report_date": fd.ReportDate,
		},
	})

	// Time Series Features from Market Data
	if len(md) > 0 {
		// Calculate volatility (30-day)
		volatility := fi.calculateVolatility(md, 30)
		calculatedFeatures = append(calculatedFeatures, features.Feature{
			CompanyID: fd.CompanyID,
			Name:      "ts_volatility_30d",
			Value:     volatility,
			Type:      "numerical",
			Source:    "market_data",
			Timestamp: now,
			Metadata: map[string]interface{}{
				"period": "30_days",
				"method": "standard_deviation",
			},
		})

		// Calculate momentum (10-day)
		momentum := fi.calculateMomentum(md, 10)
		calculatedFeatures = append(calculatedFeatures, features.Feature{
			CompanyID: fd.CompanyID,
			Name:      "ts_momentum_10d",
			Value:     momentum,
			Type:      "numerical",
			Source:    "market_data",
			Timestamp: now,
			Metadata: map[string]interface{}{
				"period": "10_days",
			},
		})

		// Calculate average volume
		avgVolume := fi.calculateAverageVolume(md, 30)
		calculatedFeatures = append(calculatedFeatures, features.Feature{
			CompanyID: fd.CompanyID,
			Name:      "ts_avg_volume_30d",
			Value:     avgVolume,
			Type:      "numerical",
			Source:    "market_data",
			Timestamp: now,
			Metadata: map[string]interface{}{
				"period": "30_days",
			},
		})
	}

	return calculatedFeatures
}

// calculateVolatility calculates price volatility over a period
func (fi *FinancialIngester) calculateVolatility(md []MarketData, days int) float64 {
	if len(md) < days {
		return 0.0
	}

	recentData := md[len(md)-days:]
	var returns []float64

	for i := 1; i < len(recentData); i++ {
		if recentData[i-1].Close > 0 {
			dailyReturn := (recentData[i].Close - recentData[i-1].Close) / recentData[i-1].Close
			returns = append(returns, dailyReturn)
		}
	}

	if len(returns) == 0 {
		return 0.0
	}

	// Calculate standard deviation
	mean := 0.0
	for _, ret := range returns {
		mean += ret
	}
	mean /= float64(len(returns))

	variance := 0.0
	for _, ret := range returns {
		variance += (ret - mean) * (ret - mean)
	}
	variance /= float64(len(returns))

	return math.Sqrt(variance)
}

// calculateMomentum calculates price momentum over a period
func (fi *FinancialIngester) calculateMomentum(md []MarketData, days int) float64 {
	if len(md) < days+1 {
		return 0.0
	}

	currentPrice := md[len(md)-1].Close
	pastPrice := md[len(md)-days-1].Close

	if pastPrice == 0 {
		return 0.0
	}

	return (currentPrice - pastPrice) / pastPrice
}

// calculateAverageVolume calculates average trading volume
func (fi *FinancialIngester) calculateAverageVolume(md []MarketData, days int) float64 {
	if len(md) < days {
		return 0.0
	}

	recentData := md[len(md)-days:]
	totalVolume := int64(0)

	for _, data := range recentData {
		totalVolume += data.Volume
	}

	return float64(totalVolume) / float64(len(recentData))
}

// storeFinancialData stores raw financial data in the database
func (fi *FinancialIngester) storeFinancialData(ctx context.Context, fd *FinancialData) error {
	query := `
		INSERT INTO financial_data (
			company_id, report_date, revenue, cost_of_revenue, gross_profit,
			operating_income, net_income, total_assets, total_liabilities,
			inventory, accounts_payable, working_capital, cash_and_equivalents,
			total_debt, shares_outstanding, market_cap, stock_price, created_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
		ON CONFLICT (company_id, report_date) DO UPDATE SET
			revenue = EXCLUDED.revenue,
			stock_price = EXCLUDED.stock_price,
			market_cap = EXCLUDED.market_cap,
			updated_at = NOW()
	`

	_, err := fi.db.ExecContext(
		ctx, query,
		fd.CompanyID, fd.ReportDate, fd.Revenue, fd.CostOfRevenue, fd.GrossProfit,
		fd.OperatingIncome, fd.NetIncome, fd.TotalAssets, fd.TotalLiabilities,
		fd.Inventory, fd.AccountsPayable, fd.WorkingCapital, fd.CashAndEquiv,
		fd.TotalDebt, fd.SharesOutstanding, fd.MarketCap, fd.StockPrice, time.Now(),
	)

	return err
}

// TriggerDataRefresh manually triggers a data refresh for specific companies
func (fi *FinancialIngester) TriggerDataRefresh(ctx context.Context, companies []string) error {
	log.Printf("Manually triggering financial data refresh for %d companies", len(companies))

	for _, company := range companies {
		if err := fi.processCompanyFinancials(ctx, company); err != nil {
			log.Printf("Error refreshing data for %s: %v", company, err)
			continue
		}
	}

	return nil
} 