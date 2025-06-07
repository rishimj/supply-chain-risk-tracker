package external

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// AlphaVantageClient handles Alpha Vantage API requests
type AlphaVantageClient struct {
	apiKey string
	client *http.Client
}

// AlphaVantageQuote represents a stock quote from Alpha Vantage
type AlphaVantageQuote struct {
	GlobalQuote struct {
		Symbol           string `json:"01. symbol"`
		Open             string `json:"02. open"`
		High             string `json:"03. high"`
		Low              string `json:"04. low"`
		Price            string `json:"05. price"`
		Volume           string `json:"06. volume"`
		LatestTradingDay string `json:"07. latest trading day"`
		PreviousClose    string `json:"08. previous close"`
		Change           string `json:"09. change"`
		ChangePercent    string `json:"10. change percent"`
	} `json:"Global Quote"`
}

// AlphaVantageOverview represents company overview data
type AlphaVantageOverview struct {
	Symbol                     string `json:"Symbol"`
	AssetType                  string `json:"AssetType"`
	Name                       string `json:"Name"`
	Description                string `json:"Description"`
	CIK                        string `json:"CIK"`
	Exchange                   string `json:"Exchange"`
	Currency                   string `json:"Currency"`
	Country                    string `json:"Country"`
	Sector                     string `json:"Sector"`
	Industry                   string `json:"Industry"`
	Address                    string `json:"Address"`
	FiscalYearEnd              string `json:"FiscalYearEnd"`
	LatestQuarter              string `json:"LatestQuarter"`
	MarketCapitalization       string `json:"MarketCapitalization"`
	EBITDA                     string `json:"EBITDA"`
	PERatio                    string `json:"PERatio"`
	PEGRatio                   string `json:"PEGRatio"`
	BookValue                  string `json:"BookValue"`
	DividendPerShare           string `json:"DividendPerShare"`
	DividendYield              string `json:"DividendYield"`
	EPS                        string `json:"EPS"`
	RevenuePerShareTTM         string `json:"RevenuePerShareTTM"`
	ProfitMargin               string `json:"ProfitMargin"`
	OperatingMarginTTM         string `json:"OperatingMarginTTM"`
	ReturnOnAssetsTTM          string `json:"ReturnOnAssetsTTM"`
	ReturnOnEquityTTM          string `json:"ReturnOnEquityTTM"`
	RevenueTTM                 string `json:"RevenueTTM"`
	GrossProfitTTM             string `json:"GrossProfitTTM"`
	DilutedEPSTTM              string `json:"DilutedEPSTTM"`
	QuarterlyEarningsGrowthYOY string `json:"QuarterlyEarningsGrowthYOY"`
	QuarterlyRevenueGrowthYOY  string `json:"QuarterlyRevenueGrowthYOY"`
	AnalystTargetPrice         string `json:"AnalystTargetPrice"`
	TrailingPE                 string `json:"TrailingPE"`
	ForwardPE                  string `json:"ForwardPE"`
	PriceToSalesRatioTTM       string `json:"PriceToSalesRatioTTM"`
	PriceToBookRatio           string `json:"PriceToBookRatio"`
	EVToRevenue                string `json:"EVToRevenue"`
	EVToEBITDA                 string `json:"EVToEBITDA"`
	Beta                       string `json:"Beta"`
	WeekHigh52                 string `json:"52WeekHigh"`
	WeekLow52                  string `json:"52WeekLow"`
	MovingAverage50Day         string `json:"50DayMovingAverage"`
	MovingAverage200Day        string `json:"200DayMovingAverage"`
	SharesOutstanding          string `json:"SharesOutstanding"`
	SharesFloat                string `json:"SharesFloat"`
	SharesShort                string `json:"SharesShort"`
	SharesShortPriorMonth      string `json:"SharesShortPriorMonth"`
	ShortRatio                 string `json:"ShortRatio"`
	ShortPercentOutstanding    string `json:"ShortPercentOutstanding"`
	ShortPercentFloat          string `json:"ShortPercentFloat"`
	PercentInsiders            string `json:"PercentInsiders"`
	PercentInstitutions        string `json:"PercentInstitutions"`
	ForwardAnnualDividendRate  string `json:"ForwardAnnualDividendRate"`
	ForwardAnnualDividendYield string `json:"ForwardAnnualDividendYield"`
	PayoutRatio                string `json:"PayoutRatio"`
	DividendDate               string `json:"DividendDate"`
	ExDividendDate             string `json:"ExDividendDate"`
	LastSplitFactor            string `json:"LastSplitFactor"`
	LastSplitDate              string `json:"LastSplitDate"`
}

// AlphaVantageTimeSeries represents daily time series data
type AlphaVantageTimeSeries struct {
	MetaData struct {
		Information   string `json:"1. Information"`
		Symbol        string `json:"2. Symbol"`
		LastRefreshed string `json:"3. Last Refreshed"`
		OutputSize    string `json:"4. Output Size"`
		TimeZone      string `json:"5. Time Zone"`
	} `json:"Meta Data"`
	TimeSeries map[string]struct {
		Open   string `json:"1. open"`
		High   string `json:"2. high"`
		Low    string `json:"3. low"`
		Close  string `json:"4. close"`
		Volume string `json:"5. volume"`
	} `json:"Time Series (Daily)"`
}

// NewAlphaVantageClient creates a new Alpha Vantage API client
func NewAlphaVantageClient(apiKey string) *AlphaVantageClient {
	return &AlphaVantageClient{
		apiKey: apiKey,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// GetQuote retrieves real-time quote for a symbol
func (av *AlphaVantageClient) GetQuote(ctx context.Context, symbol string) (*AlphaVantageQuote, error) {
	url := fmt.Sprintf("https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=%s&apikey=%s", symbol, av.apiKey)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	
	resp, err := av.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var quote AlphaVantageQuote
	if err := json.NewDecoder(resp.Body).Decode(&quote); err != nil {
		return nil, err
	}
	
	return &quote, nil
}

// GetCompanyOverview retrieves fundamental data for a symbol
func (av *AlphaVantageClient) GetCompanyOverview(ctx context.Context, symbol string) (*AlphaVantageOverview, error) {
	url := fmt.Sprintf("https://www.alphavantage.co/query?function=OVERVIEW&symbol=%s&apikey=%s", symbol, av.apiKey)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	
	resp, err := av.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var overview AlphaVantageOverview
	if err := json.NewDecoder(resp.Body).Decode(&overview); err != nil {
		return nil, err
	}
	
	return &overview, nil
}

// GetDailyTimeSeries retrieves daily adjusted time series data
func (av *AlphaVantageClient) GetDailyTimeSeries(ctx context.Context, symbol string, outputSize string) (*AlphaVantageTimeSeries, error) {
	if outputSize == "" {
		outputSize = "compact" // compact or full
	}
	
	url := fmt.Sprintf("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=%s&outputsize=%s&apikey=%s", symbol, outputSize, av.apiKey)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	
	resp, err := av.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var timeSeries AlphaVantageTimeSeries
	if err := json.NewDecoder(resp.Body).Decode(&timeSeries); err != nil {
		return nil, err
	}
	
	return &timeSeries, nil
} 