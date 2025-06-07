#!/bin/bash

# Supply Chain ML System - Database Initialization Script
# This script initializes the PostgreSQL database with all required tables and seed data

set -e  # Exit on any error

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5433}"
DB_NAME="${DB_NAME:-supply_chain_ml}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-password}"
SCHEMA_DIR="./data/schemas"
SCRIPT_DIR="./scripts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if PostgreSQL is running
check_postgres() {
    log "Checking PostgreSQL connection..."
    if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" > /dev/null 2>&1; then
        error "PostgreSQL is not running or not accessible"
        error "Please ensure PostgreSQL is running and accessible at $DB_HOST:$DB_PORT"
        exit 1
    fi
    success "PostgreSQL is running and accessible"
}

# Function to check if database exists
check_database() {
    log "Checking if database '$DB_NAME' exists..."
    if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        log "Database '$DB_NAME' does not exist. Creating..."
        PGPASSWORD="$DB_PASSWORD" createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
        success "Database '$DB_NAME' created successfully"
    else
        success "Database '$DB_NAME' already exists"
    fi
}

# Function to execute SQL file
execute_sql_file() {
    local file="$1"
    local description="$2"
    
    if [[ ! -f "$file" ]]; then
        error "SQL file not found: $file"
        return 1
    fi
    
    log "Executing $description..."
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$file" > /dev/null 2>&1; then
        success "$description completed successfully"
        return 0
    else
        error "$description failed"
        return 1
    fi
}

# Function to verify table creation
verify_tables() {
    log "Verifying table creation..."
    
    local expected_tables=(
        "company_info"
        "financial_data"
        "earnings_calls"
        "sec_filings"
        "supplier_relationships"
        "supplier_companies"
        "market_data"
        "news_articles"
        "features"
        "predictions"
        "model_training_logs"
        "companies"
        "feature_store"
        "processed_filings"
        "market_data_daily"
        "realtime_market_data"
        "detected_anomalies"
        "pipeline_jobs"
        "data_quality_metrics"
        "system_alerts"
    )
    
    local missing_tables=()
    
    for table in "${expected_tables[@]}"; do
        if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "\dt $table" > /dev/null 2>&1; then
            missing_tables+=("$table")
        fi
    done
    
    if [[ ${#missing_tables[@]} -eq 0 ]]; then
        success "All ${#expected_tables[@]} tables created successfully"
        return 0
    else
        error "Missing tables: ${missing_tables[*]}"
        return 1
    fi
}

# Function to check data insertion
verify_data() {
    log "Verifying seed data insertion..."
    
    # Check if companies table has data
    local company_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM companies;" | xargs)
    
    if [[ "$company_count" -gt 0 ]]; then
        success "Seed data inserted successfully ($company_count companies)"
        
        # Show some sample data
        log "Sample companies in database:"
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT symbol, name, sector FROM companies LIMIT 5;"
        
        return 0
    else
        error "No seed data found in companies table"
        return 1
    fi
}

# Function to create database backup
create_backup() {
    log "Creating database backup..."
    local backup_file="./backups/supply_chain_ml_backup_$(date +%Y%m%d_%H%M%S).sql"
    
    # Create backups directory if it doesn't exist
    mkdir -p "./backups"
    
    if PGPASSWORD="$DB_PASSWORD" pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" > "$backup_file"; then
        success "Database backup created: $backup_file"
    else
        warning "Failed to create database backup"
    fi
}

# Function to show database statistics
show_statistics() {
    log "Database Statistics:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples
        FROM pg_stat_user_tables 
        WHERE n_live_tup > 0
        ORDER BY n_live_tup DESC
        LIMIT 10;
    "
}

# Function to set up database maintenance
setup_maintenance() {
    log "Setting up database maintenance..."
    
    # Create maintenance SQL
    cat > /tmp/maintenance_setup.sql << 'EOF'
-- Set up automatic vacuuming for better performance
ALTER TABLE companies SET (autovacuum_enabled = true);
ALTER TABLE financial_data SET (autovacuum_enabled = true);
ALTER TABLE feature_store SET (autovacuum_enabled = true);
ALTER TABLE market_data_daily SET (autovacuum_enabled = true);
ALTER TABLE realtime_market_data SET (autovacuum_enabled = true);

-- Create indexes for better query performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feature_store_company_timestamp ON feature_store(company_id, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_daily_symbol_date ON market_data_daily(company_id, date DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_realtime_market_data_expires ON realtime_market_data(expires_at) WHERE expires_at IS NOT NULL;

-- Set up connection limits
ALTER DATABASE supply_chain_ml SET max_connections = 200;
ALTER DATABASE supply_chain_ml SET shared_preload_libraries = 'pg_stat_statements';
EOF
    
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f /tmp/maintenance_setup.sql > /dev/null 2>&1; then
        success "Database maintenance configured"
    else
        warning "Some maintenance configurations may have failed"
    fi
    
    rm -f /tmp/maintenance_setup.sql
}

# Main execution function
main() {
    log "Starting Supply Chain ML Database Initialization"
    log "Target: $DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
    
    # Check prerequisites
    check_postgres
    check_database
    
    # Execute schema files in order
    local schema_files=(
        "$SCHEMA_DIR/001_init.sql"
        "$SCHEMA_DIR/002_pipeline_tables.sql"
        "$SCHEMA_DIR/003_seed_data.sql"
    )
    
    local descriptions=(
        "Base schema creation"
        "Data pipeline tables creation"
        "Seed data insertion"
    )
    
    for i in "${!schema_files[@]}"; do
        if ! execute_sql_file "${schema_files[$i]}" "${descriptions[$i]}"; then
            error "Failed to execute ${schema_files[$i]}"
            exit 1
        fi
    done
    
    # Verify installation
    if ! verify_tables; then
        error "Table verification failed"
        exit 1
    fi
    
    if ! verify_data; then
        error "Data verification failed"
        exit 1
    fi
    
    # Set up maintenance and optimization
    setup_maintenance
    
    # Create backup
    create_backup
    
    # Show statistics
    show_statistics
    
    success "Database initialization completed successfully!"
    log "You can now start the API server and data pipeline services"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --verify-only       Only verify existing database"
        echo "  --backup-only       Only create a backup"
        echo "  --reset             Drop and recreate database (WARNING: destructive)"
        echo ""
        echo "Environment Variables:"
        echo "  DB_HOST             Database host (default: localhost)"
        echo "  DB_PORT             Database port (default: 5432)"
        echo "  DB_NAME             Database name (default: supply_chain_ml)"
        echo "  DB_USER             Database user (default: postgres)"
        echo "  DB_PASSWORD         Database password (default: password)"
        exit 0
        ;;
    --verify-only)
        log "Verification mode"
        check_postgres
        verify_tables
        verify_data
        show_statistics
        exit 0
        ;;
    --backup-only)
        log "Backup mode"
        check_postgres
        create_backup
        exit 0
        ;;
    --reset)
        warning "DESTRUCTIVE OPERATION: This will drop and recreate the database"
        read -p "Are you sure? Type 'yes' to continue: " -r
        if [[ $REPLY == "yes" ]]; then
            log "Dropping database..."
            PGPASSWORD="$DB_PASSWORD" dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" 2>/dev/null || true
            main
        else
            log "Operation cancelled"
            exit 0
        fi
        ;;
    "")
        main
        ;;
    *)
        error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac 