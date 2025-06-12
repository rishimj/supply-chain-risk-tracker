.PHONY: help build run test clean docker-build docker-run setup-dev lint build-data-pipeline run-data-pipeline test-data-pipeline

# Default target
help:
	@echo "Supply Chain Risk Tracker - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup-dev     - Setup development environment"
	@echo "  install       - Install all dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  run-api       - Run API server locally"
	@echo "  run-ml        - Run ML model server locally"
	@echo "  run-web       - Run frontend development server"
	@echo "  run-pipeline  - Run data pipeline service"
	@echo "  run-all       - Run all services locally (requires Docker)"
	@echo ""
	@echo "Live Data Pipeline Commands:"
	@echo "  setup-live-data    - Setup live data sources (API keys, etc.)"
	@echo "  pipeline-start     - Start data pipeline with real data ingestion"
	@echo "  pipeline-stop      - Stop data pipeline"
	@echo "  pipeline-status    - Check pipeline and data source status"
	@echo "  pipeline-monitor   - Monitor real-time data ingestion"
	@echo "  pipeline-test      - Test API connectivity and data sources"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build  - Build all Docker images"
	@echo "  docker-run    - Run all services with Docker Compose"
	@echo "  docker-stop   - Stop all Docker services"
	@echo "  docker-clean  - Clean up Docker resources"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test          - Run all tests"
	@echo "  test-go       - Run Go tests"
	@echo "  test-python   - Run Python tests"
	@echo "  test-api      - Run API integration tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          - Run linters for all languages"
	@echo "  fmt           - Format code"
	@echo "  security      - Run security scans"
	@echo ""
	@echo "Database Commands:"
	@echo "  db-init       - Initialize database with schema and seed data"
	@echo "  db-reset      - Reset database (WARNING: destructive)"
	@echo "  db-health     - Check database health and performance"
	@echo "  db-backup     - Create database backup"
	@echo "  db-verify     - Verify database setup"
	@echo "  db-start      - Start database services only"
	@echo "  db-stop       - Stop database services"
	@echo "  db-logs       - Show database logs"
	@echo ""
	@echo "ML Commands:"
	@echo "  train-model   - Train ML model"
	@echo ""

# Setup development environment
setup-dev:
	@echo "Setting up development environment..."
	@echo "Installing Go dependencies..."
	go mod download
	@echo "Setting up Python virtual environment..."
	python -m venv python/venv
	@echo "Installing Python dependencies..."
	python/venv/bin/pip install -r python/requirements.txt
	@echo "Installing Node.js dependencies..."
	cd web && npm install
	@echo "Creating environment file..."
	cp .env.example .env || echo ".env already exists"
	@echo "Development environment setup complete!"

# Install all dependencies
install:
	go mod download
	cd python && pip install -r requirements.txt
	cd web && npm install

# Run commands
run-api:
	@echo "Starting API server..."
	go run cmd/api-server/main.go

run-ml:
	@echo "Starting ML model server..."
	cd python && uvicorn inference.server:app --host 0.0.0.0 --port 8001 --reload

run-web:
	@echo "Starting frontend development server..."
	cd web && npm run dev

# Data pipeline commands
run-pipeline:
	@echo "Starting data pipeline service..."
	go run cmd/data-pipeline/main.go

setup-live-data:
	@echo "Setting up live data sources..."
	./scripts/setup_live_data.sh

pipeline-start:
	@echo "Starting data pipeline with management script..."
	./scripts/manage_pipeline.sh start

pipeline-stop:
	@echo "Stopping data pipeline..."
	./scripts/manage_pipeline.sh stop

pipeline-status:
	@echo "Checking pipeline status..."
	./scripts/manage_pipeline.sh status

pipeline-monitor:
	@echo "Monitoring data pipeline..."
	./scripts/manage_pipeline.sh monitor

pipeline-test:
	@echo "Testing data sources..."
	./scripts/manage_pipeline.sh test

run-all:
	@echo "Starting all services with Docker Compose..."
	docker-compose up -d

# Docker commands
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-run:
	@echo "Starting all services with Docker Compose..."
	docker-compose up -d
	@echo "Services started! Access points:"
	@echo "  Dashboard: http://localhost:3000"
	@echo "  API: http://localhost:8080"
	@echo "  Model Server: http://localhost:8001"
	@echo "  Grafana: http://localhost:3001"

docker-stop:
	@echo "Stopping Docker services..."
	docker-compose down

docker-clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f

# Testing
test: test-go test-python
	@echo "All tests completed!"

test-go:
	@echo "Running Go tests..."
	go test -v ./...

test-python:
	@echo "Running Python tests..."
	cd python && python -m pytest tests/ -v

test-api:
	@echo "Running API integration tests..."
	# Add API integration test commands here
	curl -f http://localhost:8080/health || echo "API not responding"

# Code quality
lint: lint-go lint-python lint-web
	@echo "Linting completed!"

lint-go:
	@echo "Linting Go code..."
	golangci-lint run ./...

lint-python:
	@echo "Linting Python code..."
	cd python && flake8 . && black --check . && mypy .

lint-web:
	@echo "Linting frontend code..."
	cd web && npm run lint

fmt: fmt-go fmt-python fmt-web
	@echo "Code formatting completed!"

fmt-go:
	@echo "Formatting Go code..."
	go fmt ./...
	goimports -w .

fmt-python:
	@echo "Formatting Python code..."
	cd python && black . && isort .

fmt-web:
	@echo "Formatting frontend code..."
	cd web && npm run lint --fix || echo "No format script available"

security:
	@echo "Running security scans..."
	gosec ./...
	cd python && bandit -r .

# Database operations
db-init:
	@echo "Initializing database..."
	chmod +x scripts/init_database.sh
	./scripts/init_database.sh

db-reset:
	@echo "Resetting database (WARNING: This will delete all data)..."
	chmod +x scripts/init_database.sh
	./scripts/init_database.sh --reset

db-health:
	@echo "Checking database health..."
	chmod +x scripts/db_health_check.sh
	./scripts/db_health_check.sh

db-backup:
	@echo "Creating database backup..."
	chmod +x scripts/init_database.sh
	./scripts/init_database.sh --backup-only

db-verify:
	@echo "Verifying database setup..."
	chmod +x scripts/init_database.sh
	./scripts/init_database.sh --verify-only

db-start:
	@echo "Starting database services..."
	docker-compose up -d postgres redis neo4j

db-stop:
	@echo "Stopping database services..."
	docker-compose stop postgres redis neo4j

db-logs:
	@echo "Showing database logs..."
	docker-compose logs -f postgres

# Legacy commands (deprecated)
migrate: db-init
	@echo "Note: 'migrate' is deprecated. Use 'db-init' instead."

seed-data: db-init
	@echo "Note: 'seed-data' is deprecated. Use 'db-init' instead."

# ML operations
train-model:
	@echo "Training ML model..."
	cd python && python train_model.py --config config/config.yaml

train-model-quick:
	@echo "Quick training ML model (for testing)..."
	cd python && python train_model.py --quick --mock-data

train-model-retrain:
	@echo "Retraining existing ML model..."
	cd python && python train_model.py --retrain models/artifacts/latest

mlflow-ui:
	@echo "Starting MLflow UI..."
	cd python && mlflow ui --host 0.0.0.0 --port 5000

model-serve:
	@echo "Starting model inference server..."
	cd python && python inference/server.py

test-ml:
	@echo "Testing ML pipeline..."
	chmod +x scripts/test_ml_pipeline.sh
	./scripts/test_ml_pipeline.sh

# Development helpers
logs:
	@echo "Showing service logs..."
	docker-compose logs -f

status:
	@echo "Service status:"
	docker-compose ps

shell-api:
	@echo "Opening shell in API container..."
	docker-compose exec api-server /bin/sh

shell-ml:
	@echo "Opening shell in ML container..."
	docker-compose exec model-server /bin/bash

shell-db:
	@echo "Opening PostgreSQL shell..."
	docker-compose exec postgres psql -U postgres -d supply_chain_ml

# Environment management
env-create:
	cp .env.example .env
	@echo "Created .env file. Please edit with your configuration."

env-validate:
	@echo "Validating environment configuration..."
	# Add validation commands here

# Build artifacts
build: build-go build-python build-web
	@echo "Build completed!"

build-go:
	@echo "Building Go binaries..."
	mkdir -p bin
	go build -o bin/api-server cmd/api-server/main.go
	go build -o bin/stream-processor cmd/stream-processor/main.go
	go build -o bin/batch-processor cmd/batch-processor/main.go

build-python:
	@echo "Building Python packages..."
	cd python && python setup.py build

build-web:
	@echo "Building frontend..."
	cd web && npm run build

# Cleanup
clean: clean-go clean-python clean-web
	@echo "Cleanup completed!"

clean-go:
	@echo "Cleaning Go artifacts..."
	go clean
	rm -rf bin/

clean-python:
	@echo "Cleaning Python artifacts..."
	cd python && find . -type f -name "*.pyc" -delete
	cd python && find . -type d -name "__pycache__" -delete
	cd python && rm -rf build/ dist/ *.egg-info/

clean-web:
	@echo "Cleaning frontend artifacts..."
	cd web && rm -rf build/ dist/

# Release
release-check:
	@echo "Running pre-release checks..."
	make test
	make lint
	make security
	make build

# Documentation
docs-serve:
	@echo "Serving documentation..."
	# Add documentation server command

docs-build:
	@echo "Building documentation..."
	# Add documentation build command

# Quick start command
quickstart:
	@echo "Starting Supply Chain Risk Tracker..."
	@echo "This will set up and run the entire system."
	make setup-dev
	make docker-run
	@echo ""
	@echo "🚀 System is starting up!"
	@echo "📊 Dashboard: http://localhost:3000"
	@echo "🔗 API: http://localhost:8080"
	@echo "🤖 ML Server: http://localhost:8001"
	@echo "📈 Grafana: http://localhost:3001"
	@echo ""
	@echo "Run 'make logs' to see service logs"
	@echo "Run 'make status' to check service status"

# Data Pipeline Commands
build-data-pipeline:
	@echo "Building data pipeline service..."
	go build -o data-pipeline cmd/data-pipeline/main.go

run-data-pipeline: build-data-pipeline
	@echo "Starting data pipeline service..."
	./data-pipeline

test-data-pipeline:
	@echo "Testing data pipeline components..."
	go test ./pkg/pipeline/... -v

# Build all services
.PHONY: build-all
build-all: build-api build-data-pipeline
	@echo "All services built successfully"

test-ensemble: ## Test the ensemble model with mock data
	@echo "Testing ensemble model with mock data..."
	cd python && python test_ensemble.py

test-ensemble-real: ## Test the ensemble model with real data
	@echo "Testing ensemble model with real data..."
	cd python && python test_ensemble.py --use-real-data

# Build all services
.PHONY: build-all
build-all: build-api build-data-pipeline
	@echo "All services built successfully" 