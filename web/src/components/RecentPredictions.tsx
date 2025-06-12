import React from "react";
import { useQuery } from "react-query";
import { Link } from "react-router-dom";
import { ExclamationTriangleIcon, EyeIcon } from "@heroicons/react/24/outline";
import { apiService } from "../services/api";
import { LoadingSpinner } from "./LoadingSpinner";
import { clsx } from "clsx";

export function RecentPredictions() {
  const { data: predictionsData, isLoading } = useQuery(
    "recent-predictions",
    () =>
      apiService.getPredictions({
        page: 1,
        limit: 10,
        sortBy: "prediction_timestamp",
        sortOrder: "desc",
      }),
    {
      refetchInterval: 120000, // 2 minutes
    }
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <LoadingSpinner />
      </div>
    );
  }

  const predictions = predictionsData?.predictions || [];

  if (predictions.length === 0) {
    return (
      <div className="text-center text-gray-500 py-8">
        <ExclamationTriangleIcon className="h-12 w-12 mx-auto mb-3 text-gray-400" />
        <p>No recent predictions available</p>
      </div>
    );
  }

  const getRiskLevel = (score: number) => {
    if (score >= 0.7) return "High";
    if (score >= 0.4) return "Medium";
    return "Low";
  };

  const getRiskColor = (score: number) => {
    if (score >= 0.7) return "text-danger-700 bg-danger-100";
    if (score >= 0.4) return "text-warning-700 bg-warning-100";
    return "text-success-700 bg-success-100";
  };

  return (
    <div className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Company
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Risk Score
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Risk Level
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Confidence
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Predicted At
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {predictions.map((prediction) => (
              <tr key={prediction.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div>
                    <div className="text-sm font-medium text-gray-900">
                      {prediction.company_name || prediction.company_id}
                    </div>
                    {prediction.company_symbol && (
                      <div className="text-sm text-gray-500">
                        {prediction.company_symbol}
                      </div>
                    )}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-mono text-gray-900">
                    {(prediction.risk_score * 100).toFixed(1)}%
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span
                    className={clsx(
                      "inline-flex px-2 py-1 text-xs font-semibold rounded-full",
                      getRiskColor(prediction.risk_score)
                    )}
                  >
                    {getRiskLevel(prediction.risk_score)}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {new Date(
                    prediction.prediction_timestamp
                  ).toLocaleDateString()}{" "}
                  {new Date(
                    prediction.prediction_timestamp
                  ).toLocaleTimeString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  <Link
                    to={`/companies/${prediction.company_id}`}
                    className="text-primary-600 hover:text-primary-900 flex items-center space-x-1"
                  >
                    <EyeIcon className="h-4 w-4" />
                    <span>View</span>
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {predictions.length >= 10 && (
        <div className="px-6 py-3 bg-gray-50 border-t border-gray-200">
          <Link
            to="/predictions"
            className="text-sm text-primary-600 hover:text-primary-900"
          >
            View all predictions →
          </Link>
        </div>
      )}
    </div>
  );
}
