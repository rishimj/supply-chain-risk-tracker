import React from "react";
import { useParams } from "react-router-dom";

export function CompanyDetails() {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Company Details</h1>
        <p className="mt-1 text-sm text-gray-600">
          Detailed risk analysis for company ID: {id}
        </p>
      </div>

      <div className="card">
        <div className="card-body">
          <p className="text-gray-500">
            Company details implementation coming soon...
          </p>
        </div>
      </div>
    </div>
  );
}
