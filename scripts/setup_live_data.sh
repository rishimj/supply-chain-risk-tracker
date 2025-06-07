#!/bin/bash

# Supply Chain Risk Tracker - Live Data Source Setup
# This script helps you configure and test live data sources

set -e

echo "🚀 Supply Chain Risk Tracker - Live Data Setup"
echo "=============================================="
echo

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📋 Creating .env file for API keys..."
    cat > .env << 'EOF'
# Alpha Vantage API Key (for financial data)
# Get your free API key at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=

# NewsAPI Key (for news data)
# Get your free API key at: https://newsapi.org/register
NEWS_API_KEY=

# Finnhub API Key (alternative financial data source)
# Get your free API key at: https://finnhub.io/register
FINNHUB_API_KEY=

# Database Configuration
DB_HOST=localhost
DB_PORT=5433
DB_NAME=supply_chain_ml
DB_USER=postgres
DB_PASSWORD=password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
EOF
    echo "✅ Created .env file"
else
    echo "📋 .env file already exists"
fi

echo
echo "🔑 API Key Setup Instructions"
echo "=============================="
echo
echo "To use live data sources, you need to obtain free API keys:"
echo
echo "1. Alpha Vantage (Financial Data):"
echo "   - Visit: https://www.alphavantage.co/support/#api-key"
echo "   - Sign up for a free account"
echo "   - Copy your API key to ALPHA_VANTAGE_API_KEY in .env"
echo "   - Free tier: 5 API requests per minute, 500 requests per day"
echo
echo "2. NewsAPI (News Data):"
echo "   - Visit: https://newsapi.org/register"
echo "   - Sign up for a free account"
echo "   - Copy your API key to NEWS_API_KEY in .env"
echo "   - Free tier: 1,000 requests per month"
echo
echo "3. Finnhub (Alternative Financial Data):"
echo "   - Visit: https://finnhub.io/register"
echo "   - Sign up for a free account"
echo "   - Copy your API key to FINNHUB_API_KEY in .env"
echo "   - Free tier: 60 API calls per minute"
echo

# Function to test API connectivity
test_alpha_vantage() {
    echo "🧪 Testing Alpha Vantage API..."
    
    if [ -z "$ALPHA_VANTAGE_API_KEY" ]; then
        echo "❌ ALPHA_VANTAGE_API_KEY not set in .env"
        return 1
    fi
    
    response=$(curl -s "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=$ALPHA_VANTAGE_API_KEY")
    
    if echo "$response" | grep -q "Global Quote"; then
        echo "✅ Alpha Vantage API working correctly"
        return 0
    else
        echo "❌ Alpha Vantage API test failed"
        echo "Response: $response"
        return 1
    fi
}

test_newsapi() {
    echo "🧪 Testing NewsAPI..."
    
    if [ -z "$NEWS_API_KEY" ]; then
        echo "❌ NEWS_API_KEY not set in .env"
        return 1
    fi
    
    response=$(curl -s "https://newsapi.org/v2/everything?q=supply%20chain&pageSize=1&apiKey=$NEWS_API_KEY")
    
    if echo "$response" | grep -q "totalResults"; then
        echo "✅ NewsAPI working correctly"
        return 0
    else
        echo "❌ NewsAPI test failed"
        echo "Response: $response"
        return 1
    fi
}

# Source .env file if it exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
fi

echo "🔧 Testing API Connections"
echo "=========================="
echo

# Test APIs if keys are provided
if [ ! -z "$ALPHA_VANTAGE_API_KEY" ]; then
    test_alpha_vantage
else
    echo "⏭️  Skipping Alpha Vantage test (no API key)"
fi

echo

if [ ! -z "$NEWS_API_KEY" ]; then
    test_newsapi
else
    echo "⏭️  Skipping NewsAPI test (no API key)"
fi

echo
echo "🗄️  Database Setup"
echo "=================="
echo

# Check if databases are running
if command -v docker-compose &> /dev/null; then
    echo "🐳 Starting databases with docker-compose..."
    docker-compose up -d postgres redis neo4j
    echo "✅ Databases started"
    
    # Wait for databases to be ready
    echo "⏳ Waiting for databases to be ready..."
    sleep 10
    
    # Test database connections
    if pg_isready -h localhost -p 5433 -U postgres &> /dev/null; then
        echo "✅ PostgreSQL is ready"
    else
        echo "❌ PostgreSQL connection failed"
    fi
    
    if redis-cli -h localhost -p 6379 ping &> /dev/null; then
        echo "✅ Redis is ready"
    else
        echo "❌ Redis connection failed"
    fi
    
else
    echo "⚠️  docker-compose not found. Please ensure PostgreSQL, Redis, and Neo4j are running manually."
fi

echo
echo "🚦 Starting Data Pipeline"
echo "========================="
echo

# Build the data pipeline if Go is available
if command -v go &> /dev/null; then
    echo "🔨 Building data pipeline..."
    go build -o bin/data-pipeline cmd/data-pipeline/main.go
    echo "✅ Data pipeline built"
    
    echo "🚀 You can now start the data pipeline with:"
    echo "   ./bin/data-pipeline"
    echo
    echo "Or use the Makefile:"
    echo "   make run-pipeline"
else
    echo "⚠️  Go not found. Please install Go to build the data pipeline."
fi

echo
echo "🎉 Setup Complete!"
echo "=================="
echo
echo "Next steps:"
echo "1. Add your API keys to the .env file"
echo "2. Start the data pipeline: make run-pipeline"
echo "3. Start the API server: make run-api"
echo "4. Start the frontend: make run-web"
echo
echo "Your system will now ingest real-time data from:"
echo "• 📈 Financial data (Alpha Vantage)"
echo "• 📰 News sentiment (NewsAPI)"
echo "• 🏢 SEC filings (SEC EDGAR)"
echo "• 🔗 Supply chain networks"
echo
echo "Monitor your data pipeline at: http://localhost:8080/health"
echo "View your dashboard at: http://localhost:3000"
echo 