#!/bin/bash

# Supply Chain Risk Tracker - Data Pipeline Management
# This script helps manage the data pipeline and monitor live data ingestion

set -e

# Source environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
fi

PIPELINE_PID_FILE="/tmp/supply-chain-pipeline.pid"
API_PID_FILE="/tmp/supply-chain-api.pid"

show_help() {
    echo "Supply Chain Risk Tracker - Pipeline Management"
    echo "==============================================="
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  start       Start the data pipeline and API server"
    echo "  stop        Stop the data pipeline and API server"
    echo "  restart     Restart the data pipeline and API server"
    echo "  status      Show status of running services"
    echo "  logs        Show recent pipeline logs"
    echo "  test        Test API connectivity and data ingestion"
    echo "  monitor     Monitor real-time data ingestion"
    echo "  help        Show this help message"
    echo
}

check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    if ! command -v go &> /dev/null; then
        echo "❌ Go is not installed"
        exit 1
    fi
    
    if ! command -v psql &> /dev/null; then
        echo "⚠️  PostgreSQL client not found (optional)"
    fi
    
    if ! command -v redis-cli &> /dev/null; then
        echo "⚠️  Redis client not found (optional)"
    fi
    
    echo "✅ Dependencies checked"
}

build_services() {
    echo "🔨 Building services..."
    
    # Build data pipeline
    echo "Building data pipeline..."
    go build -o bin/data-pipeline cmd/data-pipeline/main.go
    
    # Build API server
    echo "Building API server..."
    go build -o bin/api-server cmd/api-server/main.go
    
    echo "✅ Services built successfully"
}

start_pipeline() {
    if [ -f "$PIPELINE_PID_FILE" ] && kill -0 $(cat "$PIPELINE_PID_FILE") 2>/dev/null; then
        echo "📊 Data pipeline is already running (PID: $(cat $PIPELINE_PID_FILE))"
    else
        echo "🚀 Starting data pipeline..."
        nohup ./bin/data-pipeline > pipeline.log 2>&1 &
        echo $! > "$PIPELINE_PID_FILE"
        echo "✅ Data pipeline started (PID: $!)"
    fi
}

start_api() {
    if [ -f "$API_PID_FILE" ] && kill -0 $(cat "$API_PID_FILE") 2>/dev/null; then
        echo "🌐 API server is already running (PID: $(cat $API_PID_FILE))"
    else
        echo "🚀 Starting API server..."
        nohup ./bin/api-server > api.log 2>&1 &
        echo $! > "$API_PID_FILE"
        echo "✅ API server started (PID: $!)"
    fi
}

stop_pipeline() {
    if [ -f "$PIPELINE_PID_FILE" ] && kill -0 $(cat "$PIPELINE_PID_FILE") 2>/dev/null; then
        echo "🛑 Stopping data pipeline..."
        kill $(cat "$PIPELINE_PID_FILE")
        rm -f "$PIPELINE_PID_FILE"
        echo "✅ Data pipeline stopped"
    else
        echo "📊 Data pipeline is not running"
    fi
}

stop_api() {
    if [ -f "$API_PID_FILE" ] && kill -0 $(cat "$API_PID_FILE") 2>/dev/null; then
        echo "🛑 Stopping API server..."
        kill $(cat "$API_PID_FILE")
        rm -f "$API_PID_FILE"
        echo "✅ API server stopped"
    else
        echo "🌐 API server is not running"
    fi
}

show_status() {
    echo "📊 Service Status"
    echo "=================="
    echo
    
    # Check data pipeline
    if [ -f "$PIPELINE_PID_FILE" ] && kill -0 $(cat "$PIPELINE_PID_FILE") 2>/dev/null; then
        echo "📊 Data Pipeline: ✅ Running (PID: $(cat $PIPELINE_PID_FILE))"
    else
        echo "📊 Data Pipeline: ❌ Stopped"
    fi
    
    # Check API server
    if [ -f "$API_PID_FILE" ] && kill -0 $(cat "$API_PID_FILE") 2>/dev/null; then
        echo "🌐 API Server: ✅ Running (PID: $(cat $API_PID_FILE))"
    else
        echo "🌐 API Server: ❌ Stopped"
    fi
    
    echo
    
    # Check database connections
    echo "🗄️  Database Status"
    echo "==================="
    echo
    
    if pg_isready -h ${DB_HOST:-localhost} -p ${DB_PORT:-5433} -U ${DB_USER:-postgres} &> /dev/null; then
        echo "🐘 PostgreSQL: ✅ Connected"
    else
        echo "🐘 PostgreSQL: ❌ Not reachable"
    fi
    
    if redis-cli -h ${REDIS_HOST:-localhost} -p ${REDIS_PORT:-6379} ping &> /dev/null; then
        echo "🔴 Redis: ✅ Connected"
    else
        echo "🔴 Redis: ❌ Not reachable"
    fi
    
    echo
    
    # API Health Check
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "🌐 API Health: ✅ Healthy"
        
        # Get API metrics
        echo
        echo "📈 API Metrics"
        echo "=============="
        curl -s http://localhost:8080/api/v1/system/metrics | jq '.' 2>/dev/null || echo "Unable to parse metrics"
    else
        echo "🌐 API Health: ❌ Unhealthy"
    fi
}

show_logs() {
    echo "📝 Recent Pipeline Logs"
    echo "======================="
    echo
    
    if [ -f "pipeline.log" ]; then
        tail -n 50 pipeline.log
    else
        echo "No pipeline logs found"
    fi
    
    echo
    echo "📝 Recent API Logs"
    echo "=================="
    echo
    
    if [ -f "api.log" ]; then
        tail -n 50 api.log
    else
        echo "No API logs found"
    fi
}

test_data_ingestion() {
    echo "🧪 Testing Data Ingestion"
    echo "========================="
    echo
    
    # Test Alpha Vantage
    if [ ! -z "$ALPHA_VANTAGE_API_KEY" ]; then
        echo "📈 Testing Alpha Vantage..."
        response=$(curl -s "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=$ALPHA_VANTAGE_API_KEY")
        if echo "$response" | grep -q "Global Quote"; then
            echo "✅ Alpha Vantage: Working"
        else
            echo "❌ Alpha Vantage: Failed"
        fi
    else
        echo "⏭️  Alpha Vantage: No API key"
    fi
    
    # Test NewsAPI
    if [ ! -z "$NEWS_API_KEY" ]; then
        echo "📰 Testing NewsAPI..."
        response=$(curl -s "https://newsapi.org/v2/everything?q=supply%20chain&pageSize=1&apiKey=$NEWS_API_KEY")
        if echo "$response" | grep -q "totalResults"; then
            echo "✅ NewsAPI: Working"
        else
            echo "❌ NewsAPI: Failed"
        fi
    else
        echo "⏭️  NewsAPI: No API key"
    fi
    
    # Test database data
    echo "🗄️  Testing Database Data..."
    if command -v psql &> /dev/null; then
        count=$(PGPASSWORD=${DB_PASSWORD:-password} psql -h ${DB_HOST:-localhost} -p ${DB_PORT:-5433} -U ${DB_USER:-postgres} -d ${DB_NAME:-supply_chain_ml} -t -c "SELECT COUNT(*) FROM companies WHERE active = true;" 2>/dev/null | xargs)
        if [ ! -z "$count" ] && [ "$count" -gt 0 ]; then
            echo "✅ Database: $count active companies"
        else
            echo "❌ Database: No active companies found"
        fi
    else
        echo "⏭️  Database: PostgreSQL client not available"
    fi
    
    echo
}

monitor_pipeline() {
    echo "📊 Real-time Pipeline Monitoring"
    echo "================================="
    echo
    echo "Press Ctrl+C to stop monitoring"
    echo
    
    while true; do
        clear
        echo "📊 Live Data Pipeline Status - $(date)"
        echo "======================================="
        echo
        
        # Show recent pipeline activity
        if [ -f "pipeline.log" ]; then
            echo "🔄 Recent Activity:"
            tail -n 5 pipeline.log | grep -E "(Starting|Processing|Completed|Error)" || echo "No recent activity"
        fi
        
        echo
        
        # Show API metrics
        if curl -s http://localhost:8080/api/v1/system/metrics > /dev/null 2>&1; then
            echo "📈 API Metrics:"
            curl -s http://localhost:8080/api/v1/system/metrics | jq '.data' 2>/dev/null || echo "Unable to parse metrics"
        else
            echo "❌ API not responding"
        fi
        
        echo
        echo "Refreshing in 10 seconds..."
        sleep 10
    done
}

# Main command handling
case "${1:-help}" in
    start)
        check_dependencies
        build_services
        start_pipeline
        start_api
        echo
        echo "🎉 All services started!"
        echo "Monitor at: http://localhost:8080/health"
        echo "Dashboard: http://localhost:3000"
        ;;
    stop)
        stop_pipeline
        stop_api
        echo "🛑 All services stopped"
        ;;
    restart)
        stop_pipeline
        stop_api
        build_services
        start_pipeline
        start_api
        echo "🔄 All services restarted"
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    test)
        test_data_ingestion
        ;;
    monitor)
        monitor_pipeline
        ;;
    help|*)
        show_help
        ;;
esac 