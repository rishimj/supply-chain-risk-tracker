import React from "react";
import { clsx } from "clsx";
import { ArrowUpIcon, ArrowDownIcon } from "@heroicons/react/24/solid";

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: React.ComponentType<{ className?: string }>;
  color: "blue" | "green" | "yellow" | "red" | "gray";
  change?: string;
  changeType?: "increase" | "decrease";
  description?: string;
}

const colorMap = {
  blue: "bg-primary-500",
  green: "bg-success-500",
  yellow: "bg-warning-500",
  red: "bg-danger-500",
  gray: "bg-gray-500",
};

export function MetricCard({
  title,
  value,
  icon: Icon,
  color,
  change,
  changeType,
  description,
}: MetricCardProps) {
  return (
    <div className="metric-card">
      <div className="flex items-center">
        <div className="flex-shrink-0">
          <div className={clsx("p-2 rounded-lg", colorMap[color])}>
            <Icon className="h-6 w-6 text-white" />
          </div>
        </div>

        <div className="ml-4 flex-1">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">{title}</p>
              <p className="text-2xl font-bold text-gray-900">
                {typeof value === "number" ? value.toLocaleString() : value}
              </p>
            </div>

            {change && (
              <div className="flex items-center">
                {changeType === "increase" ? (
                  <ArrowUpIcon className="h-4 w-4 text-success-500" />
                ) : (
                  <ArrowDownIcon className="h-4 w-4 text-danger-500" />
                )}
                <span
                  className={clsx(
                    "text-sm font-medium ml-1",
                    changeType === "increase"
                      ? "text-success-600"
                      : "text-danger-600"
                  )}
                >
                  {change}
                </span>
              </div>
            )}
          </div>

          {description && (
            <p className="text-xs text-gray-500 mt-1">{description}</p>
          )}
        </div>
      </div>
    </div>
  );
}
