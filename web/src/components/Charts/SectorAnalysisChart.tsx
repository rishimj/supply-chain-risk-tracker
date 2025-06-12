import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface SectorAnalysisChartProps {
  data: Array<{
    sector: string;
    avg_risk_score: number;
    company_count: number;
    high_risk_percentage: number;
  }>;
}

export function SectorAnalysisChart({ data }: SectorAnalysisChartProps) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="sector" />
          <YAxis />
          <Tooltip
            formatter={(value: number, name: string) => [
              typeof value === "number" ? value.toFixed(2) : value,
              name === "avg_risk_score"
                ? "Average Risk Score"
                : name === "high_risk_percentage"
                ? "High Risk %"
                : name,
            ]}
          />
          <Bar
            dataKey="avg_risk_score"
            fill="#3b82f6"
            name="Average Risk Score"
          />
          <Bar
            dataKey="high_risk_percentage"
            fill="#ef4444"
            name="High Risk %"
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
