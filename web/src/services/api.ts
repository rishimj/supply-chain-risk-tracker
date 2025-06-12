import axios, { AxiosInstance, AxiosResponse } from "axios";
import toast from "react-hot-toast";
import {
  ApiResponse,
  Company,
  RiskPrediction,
  Feature,
  MarketData,
  NewsArticle,
  SystemAlert,
  DataQualityMetric,
  PipelineJob,
  HealthCheck,
  DashboardMetrics,
  GetPredictionsParams,
  PredictRiskRequest,
  BatchPredictRequest,
} from "../types";

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: "http://localhost:8080/api/v1",
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem("auth_token");
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        return response;
      },
      (error) => {
        if (error.response) {
          const { status, data } = error.response;

          switch (status) {
            case 401:
              toast.error("Authentication failed. Please log in again.");
              // Handle logout
              break;
            case 403:
              toast.error("Access denied.");
              break;
            case 404:
              toast.error("Resource not found.");
              break;
            case 429:
              toast.error("Too many requests. Please try again later.");
              break;
            case 500:
              toast.error("Server error. Please try again later.");
              break;
            default:
              toast.error(data?.message || "An unexpected error occurred.");
          }
        } else if (error.request) {
          toast.error("Network error. Please check your connection.");
        } else {
          toast.error("An unexpected error occurred.");
        }

        return Promise.reject(error);
      }
    );
  }

  // Health endpoints
  async getHealth(): Promise<HealthCheck> {
    const response = await axios.get<HealthCheck>(
      "http://localhost:8080/health"
    );
    return response.data;
  }

  async getSystemHealth(): Promise<HealthCheck> {
    // Use the main health endpoint since /system/health isn't working
    const response = await axios.get<HealthCheck>(
      "http://localhost:8080/health"
    );
    return response.data;
  }

  // Dashboard endpoints
  async getDashboardMetrics(): Promise<DashboardMetrics> {
    const response = await this.client.get<ApiResponse<DashboardMetrics>>(
      "/system/metrics"
    );
    return response.data.data;
  }

  // Company endpoints
  async getCompanies(params?: {
    page?: number;
    limit?: number;
    search?: string;
  }): Promise<{
    companies: Company[];
    total: number;
    page: number;
    totalPages: number;
  }> {
    const response = await this.client.get<
      ApiResponse<{
        companies: Company[];
        total: number;
        page: number;
        totalPages: number;
      }>
    >("/companies", { params });
    return response.data.data;
  }

  async getCompany(id: string): Promise<Company> {
    const response = await this.client.get<ApiResponse<Company>>(
      `/companies/${id}`
    );
    return response.data.data;
  }

  // Prediction endpoints
  async getPredictions(params?: GetPredictionsParams): Promise<{
    predictions: RiskPrediction[];
    total: number;
    page: number;
    totalPages: number;
  }> {
    const response = await this.client.get<
      ApiResponse<{
        predictions: RiskPrediction[];
        total: number;
        page: number;
        totalPages: number;
      }>
    >("/predictions", { params });
    return response.data.data;
  }

  async getCompanyRisk(companyId: string): Promise<RiskPrediction> {
    const response = await this.client.get<ApiResponse<RiskPrediction>>(
      `/companies/${companyId}/risk`
    );
    return response.data.data;
  }

  async predictRisk(request: PredictRiskRequest): Promise<RiskPrediction> {
    const response = await this.client.post<ApiResponse<RiskPrediction>>(
      "/predictions/predict",
      request
    );
    return response.data.data;
  }

  async batchPredict(request: BatchPredictRequest): Promise<RiskPrediction[]> {
    const response = await this.client.post<ApiResponse<RiskPrediction[]>>(
      "/predictions/batch",
      request
    );
    return response.data.data;
  }

  // Feature endpoints
  async getFeatures(
    companyId: string,
    params?: {
      names?: string[];
      startDate?: string;
      endDate?: string;
    }
  ): Promise<Feature[]> {
    const response = await this.client.get<ApiResponse<Feature[]>>(
      `/companies/${companyId}/features`,
      { params }
    );
    return response.data.data;
  }

  async getLatestFeatures(companyId: string): Promise<Feature[]> {
    const response = await this.client.get<ApiResponse<Feature[]>>(
      `/companies/${companyId}/features/latest`
    );
    return response.data.data;
  }

  // Market data endpoints
  async getMarketData(
    companyId: string,
    params?: {
      startDate?: string;
      endDate?: string;
      interval?: "daily" | "weekly" | "monthly";
    }
  ): Promise<MarketData[]> {
    const response = await this.client.get<ApiResponse<MarketData[]>>(
      `/companies/${companyId}/market-data`,
      { params }
    );
    return response.data.data;
  }

  // News endpoints
  async getNews(params?: {
    companyId?: string;
    limit?: number;
    sentiment?: "positive" | "negative" | "neutral";
  }): Promise<NewsArticle[]> {
    const response = await this.client.get<ApiResponse<NewsArticle[]>>(
      "/news",
      { params }
    );
    return response.data.data;
  }

  async getCompanyNews(
    companyId: string,
    params?: {
      limit?: number;
      sentiment?: "positive" | "negative" | "neutral";
    }
  ): Promise<NewsArticle[]> {
    const response = await this.client.get<ApiResponse<NewsArticle[]>>(
      `/companies/${companyId}/news`,
      { params }
    );
    return response.data.data;
  }

  // Alert endpoints
  async getAlerts(params?: {
    severity?: string[];
    resolved?: boolean;
    limit?: number;
  }): Promise<SystemAlert[]> {
    const response = await this.client.get<ApiResponse<SystemAlert[]>>(
      "/system/alerts",
      { params }
    );
    return response.data.data;
  }

  async acknowledgeAlert(alertId: number): Promise<void> {
    await this.client.post(`/system/alerts/${alertId}/acknowledge`);
  }

  async resolveAlert(alertId: number): Promise<void> {
    await this.client.post(`/system/alerts/${alertId}/resolve`);
  }

  // Data quality endpoints
  async getDataQualityMetrics(params?: {
    target?: string;
    status?: string[];
    limit?: number;
  }): Promise<DataQualityMetric[]> {
    const response = await this.client.get<ApiResponse<DataQualityMetric[]>>(
      "/system/data-quality",
      { params }
    );
    return response.data.data;
  }

  // Pipeline job endpoints
  async getPipelineJobs(params?: {
    status?: string[];
    jobType?: string[];
    limit?: number;
  }): Promise<PipelineJob[]> {
    const response = await this.client.get<ApiResponse<PipelineJob[]>>(
      "/system/jobs",
      { params }
    );
    return response.data.data;
  }

  async getPipelineJob(jobId: string): Promise<PipelineJob> {
    const response = await this.client.get<ApiResponse<PipelineJob>>(
      `/system/jobs/${jobId}`
    );
    return response.data.data;
  }

  // Model management endpoints
  async getModels(): Promise<{
    models: Array<{
      version: string;
      status: string;
      accuracy: number;
      created_at: string;
    }>;
  }> {
    const response = await this.client.get<
      ApiResponse<{
        models: Array<{
          version: string;
          status: string;
          accuracy: number;
          created_at: string;
        }>;
      }>
    >("/models");
    return response.data.data;
  }

  async setActiveModel(version: string): Promise<void> {
    await this.client.post(`/models/${version}/activate`);
  }

  // Analytics endpoints
  async getRiskTrends(params?: {
    period?: "daily" | "weekly" | "monthly";
    days?: number;
  }): Promise<
    Array<{
      date: string;
      avg_risk_score: number;
      prediction_count: number;
      high_risk_count: number;
    }>
  > {
    const response = await this.client.get<
      ApiResponse<
        Array<{
          date: string;
          avg_risk_score: number;
          prediction_count: number;
          high_risk_count: number;
        }>
      >
    >("/analytics/risk-trends", { params });
    return response.data.data;
  }

  async getSectorAnalysis(): Promise<
    Array<{
      sector: string;
      avg_risk_score: number;
      company_count: number;
      high_risk_percentage: number;
    }>
  > {
    const response = await this.client.get<
      ApiResponse<
        Array<{
          sector: string;
          avg_risk_score: number;
          company_count: number;
          high_risk_percentage: number;
        }>
      >
    >("/analytics/sector-analysis");
    return response.data.data;
  }

  async getModelPerformance(): Promise<{
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    confusion_matrix: number[][];
    feature_importance: Array<{
      feature: string;
      importance: number;
    }>;
  }> {
    const response = await this.client.get<
      ApiResponse<{
        accuracy: number;
        precision: number;
        recall: number;
        f1_score: number;
        confusion_matrix: number[][];
        feature_importance: Array<{
          feature: string;
          importance: number;
        }>;
      }>
    >("/analytics/model-performance");
    return response.data.data;
  }
}

// Create and export a singleton instance
export const apiService = new ApiService();
export default apiService;
