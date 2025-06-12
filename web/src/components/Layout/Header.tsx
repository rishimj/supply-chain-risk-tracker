import React from "react";
import { useQuery } from "react-query";
import { Bars3Icon, BellIcon } from "@heroicons/react/24/outline";
import { apiService } from "../../services/api";
import { LoadingSpinner } from "../LoadingSpinner";

interface HeaderProps {
  onMenuClick: () => void;
}

export function Header({ onMenuClick }: HeaderProps) {
  const { data: health, isLoading } = useQuery(
    "health",
    () => apiService.getHealth(),
    {
      refetchInterval: 30000, // 30 seconds
      retry: false,
    }
  );

  return (
    <div className="relative z-10 flex-shrink-0 flex h-16 bg-white shadow border-b border-gray-200">
      <button
        type="button"
        className="px-4 border-r border-gray-200 text-gray-500 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500 md:hidden"
        onClick={onMenuClick}
      >
        <span className="sr-only">Open sidebar</span>
        <Bars3Icon className="h-6 w-6" aria-hidden="true" />
      </button>

      <div className="flex-1 px-4 flex justify-between items-center">
        {/* Left side - breadcrumb could go here */}
        <div className="flex-1 flex">
          <div className="w-full flex md:ml-0">
            {/* Page title or breadcrumb */}
          </div>
        </div>

        {/* Right side */}
        <div className="ml-4 flex items-center md:ml-6 space-x-4">
          {/* System Status */}
          <div className="flex items-center space-x-2">
            {isLoading ? (
              <LoadingSpinner size="sm" />
            ) : (
              <>
                <div
                  className={`w-2 h-2 rounded-full ${
                    health?.status === "healthy"
                      ? "bg-success-400"
                      : health?.status === "degraded"
                      ? "bg-warning-400"
                      : "bg-danger-400"
                  }`}
                />
                <span className="text-sm font-medium text-gray-700">
                  {health?.status === "healthy"
                    ? "All Systems Operational"
                    : health?.status === "degraded"
                    ? "Some Issues Detected"
                    : "System Issues"}
                </span>
              </>
            )}
          </div>

          {/* Notifications */}
          <button
            type="button"
            className="bg-white p-1 rounded-full text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
          >
            <span className="sr-only">View notifications</span>
            <BellIcon className="h-6 w-6" aria-hidden="true" />
          </button>

          {/* Last Updated */}
          {health?.timestamp && (
            <div className="text-sm text-gray-500">
              Updated: {new Date(health.timestamp).toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
