import React from "react";
import { useQuery } from "react-query";
import {
  XMarkIcon,
  ExclamationTriangleIcon,
} from "@heroicons/react/24/outline";
import { apiService } from "../services/api";

export function AlertBanner() {
  const { data: alerts } = useQuery(
    "critical-alerts",
    () =>
      apiService.getAlerts({
        severity: ["critical", "high"],
        resolved: false,
        limit: 1,
      }),
    {
      refetchInterval: 15000, // 15 seconds
    }
  );

  const criticalAlert = alerts?.[0];

  if (!criticalAlert) {
    return null;
  }

  const severityColor =
    criticalAlert.severity === "critical"
      ? "bg-danger-50 border-danger-200 text-danger-800"
      : "bg-warning-50 border-warning-200 text-warning-800";

  return (
    <div className={`border-l-4 p-4 ${severityColor}`}>
      <div className="flex">
        <div className="flex-shrink-0">
          <ExclamationTriangleIcon className="h-5 w-5" aria-hidden="true" />
        </div>
        <div className="ml-3 flex-1">
          <p className="text-sm font-medium">{criticalAlert.title}</p>
          <p className="mt-1 text-sm">{criticalAlert.message}</p>
        </div>
        <div className="ml-auto pl-3">
          <div className="-mx-1.5 -my-1.5">
            <button
              type="button"
              className="inline-flex rounded-md p-1.5 hover:bg-opacity-20 focus:outline-none focus:ring-2 focus:ring-offset-2"
              onClick={() => apiService.acknowledgeAlert(criticalAlert.id)}
            >
              <span className="sr-only">Dismiss</span>
              <XMarkIcon className="h-5 w-5" aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
