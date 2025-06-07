package types

// Company represents a company to monitor across different processors
type Company struct {
	Symbol string `json:"symbol"`
	CIK    string `json:"cik,omitempty"`    // SEC Central Index Key (for SEC filings)
	Name   string `json:"name"`
	Sector string `json:"sector,omitempty"` // Business sector
} 