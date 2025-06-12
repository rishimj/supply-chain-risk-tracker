import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Dashboard } from "./pages/Dashboard";
import { Companies } from "./pages/Companies";
import { CompanyDetails } from "./pages/CompanyDetails";
import { Predictions } from "./pages/Predictions";
import { Analytics } from "./pages/Analytics";
import { SystemHealth } from "./pages/SystemHealth";
import { DataQuality } from "./pages/DataQuality";
import { Models } from "./pages/Models";
import { Settings } from "./pages/Settings";
import { NotFound } from "./pages/NotFound";

function App() {
  return (
    <Layout>
      <Routes>
        {/* Dashboard */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />

        {/* Companies */}
        <Route path="/companies" element={<Companies />} />
        <Route path="/companies/:id" element={<CompanyDetails />} />

        {/* Predictions */}
        <Route path="/predictions" element={<Predictions />} />

        {/* Analytics */}
        <Route path="/analytics" element={<Analytics />} />

        {/* System */}
        <Route path="/system/health" element={<SystemHealth />} />
        <Route path="/system/data-quality" element={<DataQuality />} />
        <Route path="/system/models" element={<Models />} />

        {/* Settings */}
        <Route path="/settings" element={<Settings />} />

        {/* 404 */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Layout>
  );
}

export default App;
