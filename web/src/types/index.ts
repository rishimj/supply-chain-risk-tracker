// API Types
export interface ApiResponse<T> {
  data: T;
  message?: string;
  success: boolean;
}

// Company Types
export interface Company {
  id: string;
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  market_cap: number;
  employees: number;
  headquarters_country: string;
  created_at: string;
  updated_at: string;
}

// Risk Prediction Types
export interface RiskPrediction {
  id: string;
  company_id: string;
  company_name?: string;
  company_symbol?: string;
  guidance_miss_probability: number;
  risk_score: number;
  confidence: number;
  financial_risk: number;
  network_risk: number;
  temporal_risk: number;
  sentiment_risk: number;
  model_version: string;
  prediction_timestamp: string;
  prediction_horizon_days: number;
}

// Feature Types
export interface Feature {
  id: number;
  company_id: string;
  name: string;
  value: number;
  type: "numerical" | "categorical" | "text";
  source: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

// Market Data Types
export interface MarketData {
  id: number;
  company_id: string;
  date: string;
  open_price: number;
  high_price: number;
  low_price: number;
  close_price: number;
  volume: number;
  adjusted_close: number;
}

// News Article Types
export interface NewsArticle {
  id: number;
  company_id: string;
  headline: string;
  content?: string;
  url?: string;
  source: string;
  published_at: string;
  sentiment_score: number;
  supply_chain_relevance: number;
  confidence: number;
}

// System Alert Types
export interface SystemAlert {
  id: number;
  alert_type: string;
  severity: "low" | "medium" | "high" | "critical";
  title: string;
  message: string;
  triggered_at: string;
  acknowledged_at?: string;
  resolved_at?: string;
  details?: Record<string, any>;
}

// Data Quality Metrics Types
export interface DataQualityMetric {
  id: number;
  metric_type: string;
  target: string;
  score: number;
  threshold: number;
  status: "pass" | "warn" | "fail";
  measured_at: string;
  details?: Record<string, any>;
}

// Pipeline Job Types
export interface PipelineJob {
  id: number;
  job_type: string;
  job_id: string;
  status: "running" | "completed" | "failed";
  start_time: string;
  end_time?: string;
  duration?: string;
  parameters?: Record<string, any>;
  results?: Record<string, any>;
  error_message?: string;
  records_processed: number;
  features_generated: number;
}

// Health Check Types
export interface HealthCheck {
  status: "healthy" | "degraded" | "unhealthy";
  timestamp: string;
  services: {
    [serviceName: string]: {
      status: "up" | "down";
      response_time_ms?: number;
      error?: string;
    };
  };
  version: string;
  uptime_seconds: number;
}

// Dashboard Metrics Types
export interface DashboardMetrics {
  total_companies: number;
  active_predictions: number;
  high_risk_companies: number;
  medium_risk_companies: number;
  low_risk_companies: number;
  latest_update: string;
  system_status: "operational" | "degraded" | "maintenance";
  data_freshness_hours: number;
}

// Chart Data Types
export interface ChartDataPoint {
  timestamp: string;
  value: number;
  label?: string;
  metadata?: Record<string, any>;
}

export interface RiskDistributionData {
  risk_level: "Low" | "Medium" | "High" | "Critical";
  count: number;
  percentage: number;
  color: string;
}

export interface TrendData {
  period: string;
  risk_score: number;
  prediction_count: number;
  accuracy: number;
}

// Component Props Types
export interface LoadingState {
  isLoading: boolean;
  error?: string | null;
}

export interface PaginationParams {
  page: number;
  limit: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

export interface FilterParams {
  companies?: string[];
  sectors?: string[];
  risk_levels?: string[];
  date_range?: {
    start: string;
    end: string;
  };
}

// API Request Types
export interface GetPredictionsParams extends PaginationParams {
  company_id?: string;
  min_risk_score?: number;
  max_risk_score?: number;
  include_resolved?: boolean;
}

export interface PredictRiskRequest {
  company_id: string;
  features?: Record<string, number>;
  model_version?: string;
}

export interface BatchPredictRequest {
  companies: {
    company_id: string;
    features?: Record<string, number>;
  }[];
  model_version?: string;
}

// UI State Types
export interface UIState {
  sidebarOpen: boolean;
  darkMode: boolean;
  notifications: boolean;
  autoRefresh: boolean;
  refreshInterval: number;
}

// Navigation Types
export interface NavigationItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  current?: boolean;
  badge?: number;
}

// Form Types
export interface CompanySearchForm {
  query: string;
  sectors: string[];
  market_cap_min?: number;
  market_cap_max?: number;
}

export interface DateRangeForm {
  start_date: string;
  end_date: string;
}

// Error Types
export interface ApiError {
  message: string;
  code?: string;
  details?: Record<string, any>;
}

// Utility Types
export type Status = "idle" | "loading" | "success" | "error";

export type SortDirection = "asc" | "desc";

export type TimeRange = "1h" | "24h" | "7d" | "30d" | "90d" | "1y";

export type RefreshInterval = 30 | 60 | 300 | 600 | 1800; // seconds
