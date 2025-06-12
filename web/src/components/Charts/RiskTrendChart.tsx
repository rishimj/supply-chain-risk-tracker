import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface RiskTrendChartProps {
  data: Array<{
    date: string;
    avg_risk_score: number;
    prediction_count: number;
    high_risk_count: number;
  }>;
}

export function RiskTrendChart({ data }: RiskTrendChartProps) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tickFormatter={(value) => new Date(value).toLocaleDateString()}
          />
          <YAxis />
          <Tooltip
            labelFormatter={(value) => new Date(value).toLocaleDateString()}
            formatter={(value: number, name: string) => [
              typeof value === "number" ? value.toFixed(2) : value,
              name === "avg_risk_score"
                ? "Average Risk Score"
                : name === "high_risk_count"
                ? "High Risk Count"
                : name,
            ]}
          />
          <Line
            type="monotone"
            dataKey="avg_risk_score"
            stroke="#ef4444"
            strokeWidth={2}
            name="Average Risk Score"
          />
          <Line
            type="monotone"
            dataKey="high_risk_count"
            stroke="#f59e0b"
            strokeWidth={2}
            name="High Risk Count"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
