import React from "react";
import { useQuery } from "react-query";
import {
  BuildingOfficeIcon,
  ExclamationTriangleIcon,
  ChartBarIcon,
  ClockIcon,
} from "@heroicons/react/24/outline";
import { apiService } from "../services/api";
import { LoadingSpinner } from "../components/LoadingSpinner";
import { MetricCard } from "../components/MetricCard";
import { RiskDistributionChart } from "../components/Charts/RiskDistributionChart";
import { RiskTrendChart } from "../components/Charts/RiskTrendChart";
import { SectorAnalysisChart } from "../components/Charts/SectorAnalysisChart";
import { RecentPredictions } from "../components/RecentPredictions";
import { SystemStatus } from "../components/SystemStatus";

export function Dashboard() {
  const { data: metrics, isLoading: metricsLoading } = useQuery(
    "dashboard-metrics",
    () => apiService.getDashboardMetrics(),
    {
      refetchInterval: 60000, // 1 minute
    }
  );

  const { data: riskTrends, isLoading: trendsLoading } = useQuery(
    "risk-trends",
    () => apiService.getRiskTrends({ period: "daily", days: 30 }),
    {
      refetchInterval: 300000, // 5 minutes
    }
  );

  const { data: sectorAnalysis, isLoading: sectorLoading } = useQuery(
    "sector-analysis",
    () => apiService.getSectorAnalysis(),
    {
      refetchInterval: 300000, // 5 minutes
    }
  );

  if (metricsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">
          Supply Chain Risk Dashboard
        </h1>
        <p className="mt-1 text-sm text-gray-600">
          Real-time monitoring and prediction of supply chain risks across your
          portfolio
        </p>
      </div>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Companies"
          value={metrics?.total_companies || 0}
          icon={BuildingOfficeIcon}
          color="blue"
        />
        <MetricCard
          title="Active Predictions"
          value={metrics?.active_predictions || 0}
          icon={ExclamationTriangleIcon}
          color="yellow"
          change="+12%"
          changeType="increase"
        />
        <MetricCard
          title="High Risk Companies"
          value={metrics?.high_risk_companies || 0}
          icon={ChartBarIcon}
          color="red"
          change="-3%"
          changeType="decrease"
        />
        <MetricCard
          title="Data Freshness"
          value={`${metrics?.data_freshness_hours || 0}h`}
          icon={ClockIcon}
          color="green"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Distribution */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">
              Risk Distribution
            </h3>
            <p className="text-sm text-gray-500">
              Current risk levels across all companies
            </p>
          </div>
          <div className="card-body">
            <RiskDistributionChart
              data={[
                {
                  risk_level: "Low",
                  count:
                    (metrics?.total_companies || 0) -
                    (metrics?.medium_risk_companies || 0) -
                    (metrics?.high_risk_companies || 0),
                  percentage: 60,
                  color: "#10b981",
                },
                {
                  risk_level: "Medium",
                  count: metrics?.medium_risk_companies || 0,
                  percentage: 25,
                  color: "#f59e0b",
                },
                {
                  risk_level: "High",
                  count: metrics?.high_risk_companies || 0,
                  percentage: 15,
                  color: "#ef4444",
                },
              ]}
            />
          </div>
        </div>

        {/* Risk Trends */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Risk Trends</h3>
            <p className="text-sm text-gray-500">30-day risk score trends</p>
          </div>
          <div className="card-body">
            {trendsLoading ? (
              <div className="flex items-center justify-center h-64">
                <LoadingSpinner />
              </div>
            ) : (
              <RiskTrendChart data={riskTrends || []} />
            )}
          </div>
        </div>
      </div>

      {/* Second Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sector Analysis */}
        <div className="lg:col-span-2 card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">
              Sector Analysis
            </h3>
            <p className="text-sm text-gray-500">
              Risk analysis by industry sector
            </p>
          </div>
          <div className="card-body">
            {sectorLoading ? (
              <div className="flex items-center justify-center h-64">
                <LoadingSpinner />
              </div>
            ) : (
              <SectorAnalysisChart data={sectorAnalysis || []} />
            )}
          </div>
        </div>

        {/* System Status */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">System Status</h3>
            <p className="text-sm text-gray-500">Current system health</p>
          </div>
          <div className="card-body">
            <SystemStatus />
          </div>
        </div>
      </div>

      {/* Recent Predictions */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-medium text-gray-900">
            Recent Predictions
          </h3>
          <p className="text-sm text-gray-500">
            Latest risk predictions and alerts
          </p>
        </div>
        <div className="card-body">
          <RecentPredictions />
        </div>
      </div>
    </div>
  );
}
