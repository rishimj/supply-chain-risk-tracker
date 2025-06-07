package processors

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"supply-chain-ml/pkg/config"
	"supply-chain-ml/pkg/features"

	"github.com/redis/go-redis/v9"
)

// StreamProcessor handles real-time data stream processing
type StreamProcessor struct {
	redis        *redis.Client
	featureStore *features.Store
	config       *config.Config
	isRunning    bool
	metrics      StreamMetrics
}

// StreamMetrics tracks stream processor performance
type StreamMetrics struct {
	MessagesProcessed  int64         `json:"messages_processed"`
	FeaturesGenerated  int64         `json:"features_generated"`
	ErrorCount         int64         `json:"error_count"`
	LastProcessedTime  time.Time     `json:"last_processed_time"`
	ProcessingLatency  time.Duration `json:"processing_latency"`
	StreamHealth       map[string]bool `json:"stream_health"`
}

// StreamMessage represents a message from the stream
type StreamMessage struct {
	ID        string                 `json:"id"`
	Stream    string                 `json:"stream"`
	Data      map[string]interface{} `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
}

// NewStreamProcessor creates a new stream processor
func NewStreamProcessor(redis *redis.Client, featureStore *features.Store, cfg *config.Config) *StreamProcessor {
	return &StreamProcessor{
		redis:        redis,
		featureStore: featureStore,
		config:       cfg,
		isRunning:    false,
		metrics: StreamMetrics{
			StreamHealth: make(map[string]bool),
		},
	}
}

// Start begins stream processing
func (sp *StreamProcessor) Start(ctx context.Context) error {
	sp.isRunning = true
	log.Println("Stream Processor started")

	// Define the streams we're monitoring
	streams := []string{
		"market_data_stream",
		"news_stream",
		"financial_updates_stream",
		"supplier_events_stream",
	}

	// Start monitoring each stream in separate goroutines
	for _, stream := range streams {
		go sp.monitorStream(ctx, stream)
	}

	// Keep running until context is cancelled
	<-ctx.Done()
	sp.isRunning = false
	return nil
}

// Stop gracefully stops the processor
func (sp *StreamProcessor) Stop(ctx context.Context) error {
	sp.isRunning = false
	return nil
}

// HealthCheck verifies processor health
func (sp *StreamProcessor) HealthCheck(ctx context.Context) bool {
	return sp.isRunning
}

// GetMetrics returns processor metrics
func (sp *StreamProcessor) GetMetrics(ctx context.Context) StreamMetrics {
	return sp.metrics
}

// monitorStream monitors a specific Redis stream
func (sp *StreamProcessor) monitorStream(ctx context.Context, streamName string) {
	log.Printf("Starting to monitor stream: %s", streamName)
	
	lastID := "0" // Start from beginning
	
	for {
		select {
		case <-ctx.Done():
			return
		default:
			// Read from stream
			streams, err := sp.redis.XRead(ctx, &redis.XReadArgs{
				Streams: []string{streamName, lastID},
				Count:   10,
				Block:   1 * time.Second,
			}).Result()
			
			if err != nil {
				if err != redis.Nil {
					log.Printf("Error reading from stream %s: %v", streamName, err)
					sp.metrics.ErrorCount++
					sp.metrics.StreamHealth[streamName] = false
				}
				continue
			}
			
			sp.metrics.StreamHealth[streamName] = true
			
			// Process messages
			for _, stream := range streams {
				for _, message := range stream.Messages {
					startTime := time.Now()
					
					if err := sp.processMessage(ctx, streamName, message); err != nil {
						log.Printf("Error processing message from %s: %v", streamName, err)
						sp.metrics.ErrorCount++
					} else {
						sp.metrics.MessagesProcessed++
						sp.metrics.ProcessingLatency = time.Since(startTime)
						sp.metrics.LastProcessedTime = time.Now()
					}
					
					lastID = message.ID
				}
			}
		}
	}
}

// processMessage processes a single stream message
func (sp *StreamProcessor) processMessage(ctx context.Context, streamName string, message redis.XMessage) error {
	switch streamName {
	case "market_data_stream":
		return sp.processMarketDataMessage(ctx, message)
	case "news_stream":
		return sp.processNewsMessage(ctx, message)
	case "financial_updates_stream":
		return sp.processFinancialUpdateMessage(ctx, message)
	case "supplier_events_stream":
		return sp.processSupplierEventMessage(ctx, message)
	default:
		return fmt.Errorf("unknown stream: %s", streamName)
	}
}

// processMarketDataMessage processes real-time market data
func (sp *StreamProcessor) processMarketDataMessage(ctx context.Context, message redis.XMessage) error {
	// Extract market data from message
	companyID, ok := message.Values["company_id"].(string)
	if !ok {
		return fmt.Errorf("missing company_id in market data message")
	}
	
	// Parse price and volume
	price, ok := message.Values["price"].(string)
	if !ok {
		return fmt.Errorf("missing price in market data message")
	}
	
	volume, ok := message.Values["volume"].(string)
	if !ok {
		return fmt.Errorf("missing volume in market data message")
	}
	
	// Convert to numbers and generate real-time features
	priceFloat := parseFloat(price, 0.0)
	volumeFloat := parseFloat(volume, 0.0)
	
	now := time.Now()
	
	// Generate real-time volatility feature
	volatilityFeature := features.Feature{
		CompanyID: companyID,
		Name:      "rt_price_volatility",
		Value:     sp.calculateRealTimeVolatility(ctx, companyID, priceFloat),
		Type:      "numerical",
		Source:    "real_time_market",
		Timestamp: now,
		TTL:       1 * time.Hour, // Real-time features have shorter TTL
		Metadata: map[string]interface{}{
			"stream_id": message.ID,
			"price":     priceFloat,
		},
	}
	
	// Generate volume surge detection
	volumeSurge := sp.detectVolumeSurge(ctx, companyID, volumeFloat)
	volumeFeature := features.Feature{
		CompanyID: companyID,
		Name:      "rt_volume_surge",
		Value:     volumeSurge,
		Type:      "numerical",
		Source:    "real_time_market",
		Timestamp: now,
		TTL:       1 * time.Hour,
		Metadata: map[string]interface{}{
			"stream_id":     message.ID,
			"volume":        volumeFloat,
			"surge_factor":  volumeSurge,
		},
	}
	
	// Store features
	if err := sp.featureStore.Store(ctx, volatilityFeature); err != nil {
		return err
	}
	
	if err := sp.featureStore.Store(ctx, volumeFeature); err != nil {
		return err
	}
	
	sp.metrics.FeaturesGenerated += 2
	return nil
}

// processNewsMessage processes real-time news sentiment
func (sp *StreamProcessor) processNewsMessage(ctx context.Context, message redis.XMessage) error {
	companyID, ok := message.Values["company_id"].(string)
	if !ok {
		return fmt.Errorf("missing company_id in news message")
	}
	
	headline, ok := message.Values["headline"].(string)
	if !ok {
		return fmt.Errorf("missing headline in news message")
	}
	
	content, ok := message.Values["content"].(string)
	if !ok {
		content = headline // Use headline if no content
	}
	
	// Calculate sentiment score
	sentiment := sp.calculateSentiment(content)
	
	// Extract supply chain keywords
	supplyChainScore := sp.extractSupplyChainScore(content)
	
	now := time.Now()
	
	// Generate sentiment feature
	sentimentFeature := features.Feature{
		CompanyID: companyID,
		Name:      "rt_news_sentiment",
		Value:     sentiment,
		Type:      "numerical",
		Source:    "real_time_news",
		Timestamp: now,
		TTL:       4 * time.Hour,
		Metadata: map[string]interface{}{
			"stream_id": message.ID,
			"headline":  headline,
		},
	}
	
	// Generate supply chain relevance feature
	supplyChainFeature := features.Feature{
		CompanyID: companyID,
		Name:      "rt_supply_chain_relevance",
		Value:     supplyChainScore,
		Type:      "numerical",
		Source:    "real_time_news",
		Timestamp: now,
		TTL:       4 * time.Hour,
		Metadata: map[string]interface{}{
			"stream_id": message.ID,
			"headline":  headline,
		},
	}
	
	// Store features
	if err := sp.featureStore.Store(ctx, sentimentFeature); err != nil {
		return err
	}
	
	if err := sp.featureStore.Store(ctx, supplyChainFeature); err != nil {
		return err
	}
	
	sp.metrics.FeaturesGenerated += 2
	return nil
}

// processFinancialUpdateMessage processes real-time financial updates
func (sp *StreamProcessor) processFinancialUpdateMessage(ctx context.Context, message redis.XMessage) error {
	companyID, ok := message.Values["company_id"].(string)
	if !ok {
		return fmt.Errorf("missing company_id in financial update message")
	}
	
	updateType, ok := message.Values["update_type"].(string)
	if !ok {
		return fmt.Errorf("missing update_type in financial update message")
	}
	
	now := time.Now()
	
	// Handle different types of financial updates
	switch updateType {
	case "earnings_guidance_change":
		guidance := parseFloat(message.Values["new_guidance"].(string), 0.0)
		previous := parseFloat(message.Values["previous_guidance"].(string), 0.0)
		
		changePercent := 0.0
		if previous != 0 {
			changePercent = (guidance - previous) / previous
		}
		
		feature := features.Feature{
			CompanyID: companyID,
			Name:      "rt_guidance_change",
			Value:     changePercent,
			Type:      "numerical",
			Source:    "real_time_financial",
			Timestamp: now,
			TTL:       24 * time.Hour,
			Metadata: map[string]interface{}{
				"stream_id":         message.ID,
				"new_guidance":      guidance,
				"previous_guidance": previous,
			},
		}
		
		if err := sp.featureStore.Store(ctx, feature); err != nil {
			return err
		}
		sp.metrics.FeaturesGenerated++
		
	case "inventory_report":
		inventoryLevel := parseFloat(message.Values["inventory_level"].(string), 0.0)
		
		feature := features.Feature{
			CompanyID: companyID,
			Name:      "rt_inventory_level",
			Value:     inventoryLevel,
			Type:      "numerical",
			Source:    "real_time_financial",
			Timestamp: now,
			TTL:       8 * time.Hour,
			Metadata: map[string]interface{}{
				"stream_id": message.ID,
			},
		}
		
		if err := sp.featureStore.Store(ctx, feature); err != nil {
			return err
		}
		sp.metrics.FeaturesGenerated++
	}
	
	return nil
}

// processSupplierEventMessage processes supplier-related events
func (sp *StreamProcessor) processSupplierEventMessage(ctx context.Context, message redis.XMessage) error {
	companyID, ok := message.Values["company_id"].(string)
	if !ok {
		return fmt.Errorf("missing company_id in supplier event message")
	}
	
	eventType, ok := message.Values["event_type"].(string)
	if !ok {
		return fmt.Errorf("missing event_type in supplier event message")
	}
	
	now := time.Now()
	
	// Calculate risk score based on event type
	riskScore := sp.calculateSupplierEventRisk(eventType)
	
	feature := features.Feature{
		CompanyID: companyID,
		Name:      "rt_supplier_event_risk",
		Value:     riskScore,
		Type:      "numerical",
		Source:    "real_time_supplier",
		Timestamp: now,
		TTL:       12 * time.Hour,
		Metadata: map[string]interface{}{
			"stream_id":  message.ID,
			"event_type": eventType,
		},
	}
	
	if err := sp.featureStore.Store(ctx, feature); err != nil {
		return err
	}
	
	sp.metrics.FeaturesGenerated++
	return nil
}

// Helper functions for real-time feature calculation

func (sp *StreamProcessor) calculateRealTimeVolatility(ctx context.Context, companyID string, currentPrice float64) float64 {
	// Get recent prices from Redis (simplified implementation)
	key := fmt.Sprintf("rt_prices:%s", companyID)
	
	// Store current price
	sp.redis.LPush(ctx, key, currentPrice)
	sp.redis.LTrim(ctx, key, 0, 19) // Keep last 20 prices
	sp.redis.Expire(ctx, key, 1*time.Hour)
	
	// Get recent prices
	prices, err := sp.redis.LRange(ctx, key, 0, -1).Result()
	if err != nil || len(prices) < 2 {
		return 0.0
	}
	
	// Calculate simple volatility
	var priceFloats []float64
	for _, priceStr := range prices {
		priceFloats = append(priceFloats, parseFloat(priceStr, 0.0))
	}
	
	if len(priceFloats) < 2 {
		return 0.0
	}
	
	// Calculate standard deviation of recent prices
	mean := 0.0
	for _, price := range priceFloats {
		mean += price
	}
	mean /= float64(len(priceFloats))
	
	variance := 0.0
	for _, price := range priceFloats {
		variance += (price - mean) * (price - mean)
	}
	variance /= float64(len(priceFloats))
	
	return variance // Simplified volatility measure
}

func (sp *StreamProcessor) detectVolumeSurge(ctx context.Context, companyID string, currentVolume float64) float64 {
	// Get recent volumes from Redis
	key := fmt.Sprintf("rt_volumes:%s", companyID)
	
	// Store current volume
	sp.redis.LPush(ctx, key, currentVolume)
	sp.redis.LTrim(ctx, key, 0, 9) // Keep last 10 volumes
	sp.redis.Expire(ctx, key, 2*time.Hour)
	
	// Get recent volumes
	volumes, err := sp.redis.LRange(ctx, key, 1, -1).Result() // Exclude current volume
	if err != nil || len(volumes) == 0 {
		return 1.0 // No comparison data
	}
	
	// Calculate average of recent volumes
	totalVolume := 0.0
	for _, volumeStr := range volumes {
		totalVolume += parseFloat(volumeStr, 0.0)
	}
	avgVolume := totalVolume / float64(len(volumes))
	
	if avgVolume == 0 {
		return 1.0
	}
	
	// Return surge factor (current volume / average volume)
	return currentVolume / avgVolume
}

func (sp *StreamProcessor) calculateSentiment(text string) float64 {
	// Simple sentiment analysis (in production, use NLP library)
	positiveWords := []string{"good", "great", "positive", "growth", "success", "strong", "improve"}
	negativeWords := []string{"bad", "negative", "decline", "loss", "weak", "concern", "risk", "problem"}
	
	textLower := strings.ToLower(text)
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

func (sp *StreamProcessor) extractSupplyChainScore(text string) float64 {
	supplyChainKeywords := []string{
		"supply chain", "supplier", "manufacturing", "logistics", "inventory",
		"shortage", "disruption", "procurement", "raw materials",
	}
	
	textLower := strings.ToLower(text)
	score := 0.0
	
	for _, keyword := range supplyChainKeywords {
		if strings.Contains(textLower, keyword) {
			score += 0.2
		}
	}
	
	// Cap at 1.0
	if score > 1.0 {
		score = 1.0
	}
	
	return score
}

func (sp *StreamProcessor) calculateSupplierEventRisk(eventType string) float64 {
	riskScores := map[string]float64{
		"supplier_bankruptcy":     0.9,
		"factory_closure":         0.8,
		"natural_disaster":        0.7,
		"trade_restriction":       0.6,
		"quality_issue":           0.5,
		"delivery_delay":          0.4,
		"price_increase":          0.3,
		"new_supplier_contract":   0.1,
	}
	
	if score, exists := riskScores[eventType]; exists {
		return score
	}
	
	return 0.5 // Default moderate risk
}

// Helper functions
func parseFloat(s string, defaultValue float64) float64 {
	// Simple string to float conversion with default
	// In production, use strconv.ParseFloat with proper error handling
	if s == "" {
		return defaultValue
	}
	
	// Simplified conversion (implement proper parsing in production)
	return defaultValue + 0.1 // Placeholder
}

// PublishToStream publishes a message to a stream (for testing)
func (sp *StreamProcessor) PublishToStream(ctx context.Context, streamName string, data map[string]interface{}) error {
	return sp.redis.XAdd(ctx, &redis.XAddArgs{
		Stream: streamName,
		Values: data,
	}).Err()
} 