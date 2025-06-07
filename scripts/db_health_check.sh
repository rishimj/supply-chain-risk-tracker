#!/bin/bash

# Supply Chain ML System - Database Health Check Script
# This script monitors database health, performance, and data quality

set -e

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-supply_chain_ml}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-password}"

# Thresholds
MAX_CONNECTIONS_THRESHOLD=150
SLOW_QUERY_THRESHOLD="5 seconds"
DISK_USAGE_THRESHOLD=80
TABLE_BLOAT_THRESHOLD=20

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Function to execute SQL and return result
execute_sql() {
    local query="$1"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "$query" 2>/dev/null | xargs
}

# Function to check database connectivity
check_connectivity() {
    log "Checking database connectivity..."
    
    if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" > /dev/null 2>&1; then
        success "Database is accessible"
        return 0
    else
        error "Database is not accessible"
        return 1
    fi
}

# Function to check connection count
check_connections() {
    log "Checking connection count..."
    
    local current_connections=$(execute_sql "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';")
    local max_connections=$(execute_sql "SHOW max_connections;")
    
    echo "Active connections: $current_connections / $max_connections"
    
    if [[ $current_connections -gt $MAX_CONNECTIONS_THRESHOLD ]]; then
        warning "High number of active connections: $current_connections"
        
        # Show top connection users
        echo "Top connection users:"
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
            SELECT usename, count(*) as connections 
            FROM pg_stat_activity 
            WHERE state = 'active' 
            GROUP BY usename 
            ORDER BY connections DESC 
            LIMIT 5;
        "
    else
        success "Connection count is normal: $current_connections"
    fi
}

# Function to check database size and growth
check_database_size() {
    log "Checking database size..."
    
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            pg_database.datname,
            pg_size_pretty(pg_database_size(pg_database.datname)) AS size
        FROM pg_database 
        WHERE datname = '$DB_NAME';
    "
    
    echo ""
    echo "Largest tables:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
            pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS index_size
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC 
        LIMIT 10;
    "
}

# Function to check for slow queries
check_slow_queries() {
    log "Checking for slow queries (if pg_stat_statements is enabled)..."
    
    # Check if pg_stat_statements is available
    local has_pg_stat_statements=$(execute_sql "SELECT COUNT(*) FROM pg_extension WHERE extname = 'pg_stat_statements';")
    
    if [[ $has_pg_stat_statements -eq 0 ]]; then
        warning "pg_stat_statements extension not installed. Cannot check slow queries."
        return
    fi
    
    echo "Top 10 slowest queries:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            round(mean_exec_time::numeric, 2) AS avg_time_ms,
            calls,
            round(total_exec_time::numeric, 2) AS total_time_ms,
            left(query, 80) AS query_snippet
        FROM pg_stat_statements 
        WHERE calls > 5
        ORDER BY mean_exec_time DESC 
        LIMIT 10;
    " || warning "Could not retrieve slow query statistics"
}

# Function to check index usage
check_index_usage() {
    log "Checking index usage..."
    
    echo "Unused indexes (scan count = 0):"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            schemaname,
            tablename,
            indexname,
            pg_size_pretty(pg_relation_size(indexrelid)) AS size
        FROM pg_stat_user_indexes 
        WHERE idx_scan = 0 
        AND schemaname = 'public'
        ORDER BY pg_relation_size(indexrelid) DESC;
    "
    
    echo ""
    echo "Tables without primary key or unique index:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            t.table_name
        FROM information_schema.tables t
        LEFT JOIN information_schema.table_constraints tc 
            ON t.table_name = tc.table_name 
            AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
        WHERE t.table_schema = 'public' 
            AND t.table_type = 'BASE TABLE'
            AND tc.table_name IS NULL;
    "
}

# Function to check table bloat
check_table_bloat() {
    log "Checking table bloat..."
    
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            schemaname,
            tablename,
            n_dead_tup,
            n_live_tup,
            CASE 
                WHEN n_live_tup > 0 
                THEN round(100.0 * n_dead_tup / n_live_tup, 2) 
                ELSE 0 
            END AS dead_tuple_percent
        FROM pg_stat_user_tables 
        WHERE n_dead_tup > 1000
        ORDER BY dead_tuple_percent DESC;
    "
}

# Function to check replication lag (if applicable)
check_replication() {
    log "Checking replication status..."
    
    local is_replica=$(execute_sql "SELECT pg_is_in_recovery();")
    
    if [[ "$is_replica" == "t" ]]; then
        echo "This is a replica database"
        local lag=$(execute_sql "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()));")
        echo "Replication lag: ${lag} seconds"
        
        if [[ $(echo "$lag > 30" | bc -l) -eq 1 ]]; then
            warning "High replication lag: ${lag} seconds"
        fi
    else
        echo "This is a master database"
        
        # Check if there are any replicas
        local replica_count=$(execute_sql "SELECT count(*) FROM pg_stat_replication;")
        echo "Connected replicas: $replica_count"
    fi
}

# Function to check data quality metrics
check_data_quality() {
    log "Checking data quality metrics..."
    
    # Check for NULL values in critical columns
    echo "Critical columns with NULL values:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 'companies.symbol' as column_name, COUNT(*) as null_count FROM companies WHERE symbol IS NULL
        UNION ALL
        SELECT 'financial_data.company_id', COUNT(*) FROM financial_data WHERE company_id IS NULL
        UNION ALL
        SELECT 'feature_store.company_id', COUNT(*) FROM feature_store WHERE company_id IS NULL
        UNION ALL
        SELECT 'market_data_daily.company_id', COUNT(*) FROM market_data_daily WHERE company_id IS NULL;
    "
    
    echo ""
    echo "Recent data freshness:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            'companies' as table_name,
            MAX(updated_at) as latest_update,
            AGE(CURRENT_TIMESTAMP, MAX(updated_at)) as age
        FROM companies
        UNION ALL
        SELECT 
            'financial_data',
            MAX(updated_at),
            AGE(CURRENT_TIMESTAMP, MAX(updated_at))
        FROM financial_data
        UNION ALL
        SELECT 
            'market_data_daily',
            MAX(created_at),
            AGE(CURRENT_TIMESTAMP, MAX(created_at))
        FROM market_data_daily
        UNION ALL
        SELECT 
            'feature_store',
            MAX(updated_at),
            AGE(CURRENT_TIMESTAMP, MAX(updated_at))
        FROM feature_store;
    "
}

# Function to check system resource usage
check_system_resources() {
    log "Checking system resources..."
    
    # Check cache hit ratio
    echo "Cache hit ratio:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            round(
                sum(blks_hit) * 100.0 / 
                NULLIF(sum(blks_hit) + sum(blks_read), 0), 
                2
            ) AS cache_hit_ratio_percent
        FROM pg_stat_database;
    "
    
    # Check lock information
    echo ""
    echo "Current locks:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            mode,
            count(*) as lock_count
        FROM pg_locks 
        WHERE database = (SELECT oid FROM pg_database WHERE datname = '$DB_NAME')
        GROUP BY mode
        ORDER BY lock_count DESC;
    "
    
    # Check for long-running transactions
    echo ""
    echo "Long-running transactions (>5 minutes):"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            pid,
            usename,
            application_name,
            state,
            AGE(CURRENT_TIMESTAMP, xact_start) as duration,
            left(query, 50) as query_snippet
        FROM pg_stat_activity 
        WHERE xact_start < CURRENT_TIMESTAMP - INTERVAL '5 minutes'
        AND state != 'idle'
        ORDER BY duration DESC;
    "
}

# Function to generate recommendations
generate_recommendations() {
    log "Generating maintenance recommendations..."
    
    # Check if VACUUM is needed
    echo "Tables that may need VACUUM:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            schemaname,
            tablename,
            n_dead_tup,
            last_vacuum,
            last_autovacuum
        FROM pg_stat_user_tables 
        WHERE n_dead_tup > 1000 
        AND (
            last_vacuum IS NULL 
            OR last_vacuum < CURRENT_DATE - INTERVAL '7 days'
        )
        AND (
            last_autovacuum IS NULL 
            OR last_autovacuum < CURRENT_DATE - INTERVAL '7 days'
        )
        ORDER BY n_dead_tup DESC;
    "
    
    echo ""
    echo "Maintenance recommendations:"
    echo "1. Consider running VACUUM on tables with high dead tuple counts"
    echo "2. Monitor slow queries and consider adding indexes"
    echo "3. Check for unused indexes and consider dropping them"
    echo "4. Set up regular automated backups"
    echo "5. Monitor connection counts during peak hours"
}

# Function to run all health checks
run_all_checks() {
    echo "======================================"
    echo "Supply Chain ML Database Health Check"
    echo "======================================"
    echo "Database: $DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
    echo "Timestamp: $(date)"
    echo ""
    
    if ! check_connectivity; then
        error "Cannot connect to database. Exiting."
        exit 1
    fi
    
    check_connections
    echo ""
    
    check_database_size
    echo ""
    
    check_slow_queries
    echo ""
    
    check_index_usage
    echo ""
    
    check_table_bloat
    echo ""
    
    check_replication
    echo ""
    
    check_data_quality
    echo ""
    
    check_system_resources
    echo ""
    
    generate_recommendations
    echo ""
    
    success "Health check completed!"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --connections       Check only connection status"
        echo "  --size              Check only database size"
        echo "  --slow-queries      Check only slow queries"
        echo "  --indexes           Check only index usage"
        echo "  --bloat             Check only table bloat"
        echo "  --replication       Check only replication status"
        echo "  --data-quality      Check only data quality"
        echo "  --resources         Check only system resources"
        echo "  --recommendations   Generate only recommendations"
        echo ""
        echo "Environment Variables:"
        echo "  DB_HOST             Database host (default: localhost)"
        echo "  DB_PORT             Database port (default: 5432)"
        echo "  DB_NAME             Database name (default: supply_chain_ml)"
        echo "  DB_USER             Database user (default: postgres)"
        echo "  DB_PASSWORD         Database password (default: password)"
        exit 0
        ;;
    --connections)
        check_connectivity && check_connections
        ;;
    --size)
        check_connectivity && check_database_size
        ;;
    --slow-queries)
        check_connectivity && check_slow_queries
        ;;
    --indexes)
        check_connectivity && check_index_usage
        ;;
    --bloat)
        check_connectivity && check_table_bloat
        ;;
    --replication)
        check_connectivity && check_replication
        ;;
    --data-quality)
        check_connectivity && check_data_quality
        ;;
    --resources)
        check_connectivity && check_system_resources
        ;;
    --recommendations)
        check_connectivity && generate_recommendations
        ;;
    "")
        run_all_checks
        ;;
    *)
        error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac 