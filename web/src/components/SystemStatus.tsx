import React from "react";
import { useQuery } from "react-query";
import {
  CheckCircleIcon,
  XCircleIcon,
  ExclamationCircleIcon,
} from "@heroicons/react/24/outline";
import { apiService } from "../services/api";
import { LoadingSpinner } from "./LoadingSpinner";

export function SystemStatus() {
  const { data: health, isLoading } = useQuery(
    "system-health",
    () => apiService.getSystemHealth(),
    {
      refetchInterval: 30000, // 30 seconds
    }
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <LoadingSpinner />
      </div>
    );
  }

  if (!health) {
    return (
      <div className="text-center text-gray-500">
        Unable to load system status
      </div>
    );
  }

  const services = Object.entries(health.services || {});

  return (
    <div className="space-y-4">
      {/* Overall Status */}
      <div className="flex items-center space-x-2">
        {health.status === "healthy" ? (
          <CheckCircleIcon className="h-5 w-5 text-success-500" />
        ) : health.status === "degraded" ? (
          <ExclamationCircleIcon className="h-5 w-5 text-warning-500" />
        ) : (
          <XCircleIcon className="h-5 w-5 text-danger-500" />
        )}
        <span className="font-medium">
          {health.status === "healthy"
            ? "All Systems Operational"
            : health.status === "degraded"
            ? "Some Issues Detected"
            : "System Down"}
        </span>
      </div>

      {/* Service Status */}
      <div className="space-y-2">
        {services.map(([serviceName, serviceHealth]) => (
          <div key={serviceName} className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {serviceHealth.status === "up" ? (
                <div className="w-2 h-2 bg-success-400 rounded-full" />
              ) : (
                <div className="w-2 h-2 bg-danger-400 rounded-full" />
              )}
              <span className="text-sm text-gray-700">{serviceName}</span>
            </div>
            <div className="text-xs text-gray-500">
              {serviceHealth.response_time_ms && (
                <span>{serviceHealth.response_time_ms}ms</span>
              )}
              {serviceHealth.error && (
                <span className="text-danger-600">{serviceHealth.error}</span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* System Info */}
      <div className="pt-3 border-t border-gray-200 text-xs text-gray-500 space-y-1">
        <div>Version: {health.version}</div>
        <div>
          Uptime: {Math.floor(health.uptime_seconds / 3600)}h{" "}
          {Math.floor((health.uptime_seconds % 3600) / 60)}m
        </div>
        <div>Last Check: {new Date(health.timestamp).toLocaleTimeString()}</div>
      </div>
    </div>
  );
}
