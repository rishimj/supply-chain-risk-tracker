package external

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// NewsAPIClient handles NewsAPI requests
type NewsAPIClient struct {
	apiKey string
	client *http.Client
}

// NewsAPIResponse represents the response from NewsAPI
type NewsAPIResponse struct {
	Status       string        `json:"status"`
	TotalResults int           `json:"totalResults"`
	Articles     []NewsArticle `json:"articles"`
}

// NewsArticle represents a news article from NewsAPI
type NewsArticle struct {
	Source struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	} `json:"source"`
	Author      string    `json:"author"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	URL         string    `json:"url"`
	URLToImage  string    `json:"urlToImage"`
	PublishedAt time.Time `json:"publishedAt"`
	Content     string    `json:"content"`
}

// NewsAPIParams represents parameters for news search
type NewsAPIParams struct {
	Query      string
	Sources    string
	Domains    string
	From       time.Time
	To         time.Time
	Language   string
	SortBy     string // relevancy, popularity, publishedAt
	PageSize   int
	Page       int
}

// NewNewsAPIClient creates a new NewsAPI client
func NewNewsAPIClient(apiKey string) *NewsAPIClient {
	return &NewsAPIClient{
		apiKey: apiKey,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// GetEverything searches for articles using the /everything endpoint
func (na *NewsAPIClient) GetEverything(ctx context.Context, params NewsAPIParams) (*NewsAPIResponse, error) {
	baseURL := "https://newsapi.org/v2/everything"
	
	// Build query parameters
	queryParams := url.Values{}
	if params.Query != "" {
		queryParams.Set("q", params.Query)
	}
	if params.Sources != "" {
		queryParams.Set("sources", params.Sources)
	}
	if params.Domains != "" {
		queryParams.Set("domains", params.Domains)
	}
	if !params.From.IsZero() {
		queryParams.Set("from", params.From.Format("2006-01-02"))
	}
	if !params.To.IsZero() {
		queryParams.Set("to", params.To.Format("2006-01-02"))
	}
	if params.Language != "" {
		queryParams.Set("language", params.Language)
	}
	if params.SortBy != "" {
		queryParams.Set("sortBy", params.SortBy)
	}
	if params.PageSize > 0 {
		queryParams.Set("pageSize", strconv.Itoa(params.PageSize))
	}
	if params.Page > 0 {
		queryParams.Set("page", strconv.Itoa(params.Page))
	}
	
	queryParams.Set("apiKey", na.apiKey)
	
	fullURL := fmt.Sprintf("%s?%s", baseURL, queryParams.Encode())
	
	req, err := http.NewRequestWithContext(ctx, "GET", fullURL, nil)
	if err != nil {
		return nil, err
	}
	
	resp, err := na.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("NewsAPI returned status %d", resp.StatusCode)
	}
	
	var newsResponse NewsAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&newsResponse); err != nil {
		return nil, err
	}
	
	return &newsResponse, nil
}

// GetTopHeadlines retrieves top headlines using the /top-headlines endpoint
func (na *NewsAPIClient) GetTopHeadlines(ctx context.Context, country, category string, pageSize int) (*NewsAPIResponse, error) {
	baseURL := "https://newsapi.org/v2/top-headlines"
	
	queryParams := url.Values{}
	if country != "" {
		queryParams.Set("country", country)
	}
	if category != "" {
		queryParams.Set("category", category)
	}
	if pageSize > 0 {
		queryParams.Set("pageSize", strconv.Itoa(pageSize))
	}
	
	queryParams.Set("apiKey", na.apiKey)
	
	fullURL := fmt.Sprintf("%s?%s", baseURL, queryParams.Encode())
	
	req, err := http.NewRequestWithContext(ctx, "GET", fullURL, nil)
	if err != nil {
		return nil, err
	}
	
	resp, err := na.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var newsResponse NewsAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&newsResponse); err != nil {
		return nil, err
	}
	
	return &newsResponse, nil
}

// GetCompanyNews searches for news articles related to a specific company
func (na *NewsAPIClient) GetCompanyNews(ctx context.Context, companyName, companySymbol string, days int) (*NewsAPIResponse, error) {
	// Create search query for the company
	query := fmt.Sprintf(`"%s" OR "%s" OR "supply chain" OR "logistics"`, companyName, companySymbol)
	
	// Set date range
	from := time.Now().AddDate(0, 0, -days)
	
	params := NewsAPIParams{
		Query:    query,
		From:     from,
		Language: "en",
		SortBy:   "publishedAt",
		PageSize: 50,
		Page:     1,
	}
	
	return na.GetEverything(ctx, params)
}

// GetSupplyChainNews searches for supply chain related news
func (na *NewsAPIClient) GetSupplyChainNews(ctx context.Context, days int) (*NewsAPIResponse, error) {
	query := `"supply chain" OR "logistics" OR "shipping" OR "manufacturing" OR "supplier" OR "procurement" OR "inventory" OR "disruption"`
	
	from := time.Now().AddDate(0, 0, -days)
	
	params := NewsAPIParams{
		Query:    query,
		From:     from,
		Language: "en",
		SortBy:   "relevancy",
		PageSize: 100,
		Page:     1,
	}
	
	return na.GetEverything(ctx, params)
}

// GetSectorNews retrieves news for a specific sector
func (na *NewsAPIClient) GetSectorNews(ctx context.Context, sector string, days int) (*NewsAPIResponse, error) {
	var query string
	
	switch sector {
	case "Technology":
		query = `"technology" OR "tech" OR "software" OR "hardware" OR "semiconductor"`
	case "Healthcare":
		query = `"healthcare" OR "pharmaceutical" OR "biotech" OR "medical device"`
	case "Financial Services":
		query = `"banking" OR "financial services" OR "fintech" OR "insurance"`
	case "Consumer Cyclical":
		query = `"retail" OR "consumer goods" OR "automotive" OR "travel"`
	case "Consumer Defensive":
		query = `"food" OR "beverage" OR "household products" OR "consumer staples"`
	case "Communication Services":
		query = `"telecommunications" OR "media" OR "internet" OR "social media"`
	default:
		query = fmt.Sprintf(`"%s"`, sector)
	}
	
	from := time.Now().AddDate(0, 0, -days)
	
	params := NewsAPIParams{
		Query:    query,
		From:     from,
		Language: "en",
		SortBy:   "relevancy",
		PageSize: 50,
		Page:     1,
	}
	
	return na.GetEverything(ctx, params)
} 