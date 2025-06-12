#!/bin/bash

# Test script for Supply Chain ML API
BASE_URL="http://localhost:8080"

echo "Starting API tests..."

# Test health endpoint
echo "Testing health endpoint..."
curl -X GET "$BASE_URL/health" \
  -H "Content-Type: application/json" \
  --silent --show-error || echo "Health endpoint test failed"

echo ""

# Test prediction endpoint
echo "Testing prediction endpoint..."
curl -X POST "$BASE_URL/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "AAPL",
    "features": {
      "financial_inventory_turnover": 5.2,
      "financial_gross_margin": 0.42,
      "financial_debt_to_equity": 1.1,
      "financial_current_ratio": 1.8,
      "network_supplier_concentration": 0.65,
      "network_supplier_risk_score": 35.0,
      "ts_volatility_30d": 0.25,
      "ts_momentum_10d": 0.15,
      "nlp_sentiment_score": 0.7,
      "nlp_risk_keywords_count": 2.0
    },
    "options": {
      "include_feature_importance": true,
      "include_component_risks": true
    }
  }' \
  --silent --show-error || echo "Prediction endpoint test failed"

echo ""

# Test batch prediction endpoint
echo "Testing batch prediction endpoint..."
curl -X POST "$BASE_URL/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "companies": [
      {
        "company_id": "AAPL",
        "features": {
          "financial_inventory_turnover": 5.2,
          "network_supplier_concentration": 0.65
        }
      },
      {
        "company_id": "MSFT",
        "features": {
          "financial_inventory_turnover": 4.8,
          "network_supplier_concentration": 0.55
        }
      }
    ],
    "options": {
      "include_component_risks": true
    }
  }' \
  --silent --show-error || echo "Batch prediction endpoint test failed"

echo ""

# Test model status endpoint
echo "Testing model status endpoint..."
curl -X GET "$BASE_URL/api/v1/models/status" \
  -H "Content-Type: application/json" \
  --silent --show-error || echo "Model status endpoint test failed"

echo ""

# Test system metrics endpoint
echo "Testing system metrics endpoint..."
curl -X GET "$BASE_URL/monitoring/metrics" \
  -H "Content-Type: application/json" \
  --silent --show-error || echo "Metrics endpoint test failed"

echo ""

echo "API tests completed!" 