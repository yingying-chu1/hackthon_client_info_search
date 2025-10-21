# Makefile for Client Info Search API

.PHONY: help install dev test lint format clean docker-build docker-run docker-stop setup

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  dev          - Run development server"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean up generated files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  docker-stop  - Stop Docker container"
	@echo "  setup        - Initial setup"

# Install dependencies
install:
	pip install -r requirements.txt

# Run development server
dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

# Run linting
lint:
	flake8 .
	mypy .

# Format code
format:
	black .
	isort .

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf chroma_db
	rm -f client_info.db

# Docker commands
docker-build:
	docker build -t client-info-search .

docker-run:
	docker run -d --name client-info-search -p 8000:8000 \
		-e OPENAI_API_KEY=$$OPENAI_API_KEY \
		-v $$(pwd)/chroma_db:/app/chroma_db \
		-v $$(pwd)/data:/app/data \
		client-info-search

docker-stop:
	docker stop client-info-search
	docker rm client-info-search

# Docker Compose commands
compose-up:
	docker-compose up -d

compose-down:
	docker-compose down

compose-logs:
	docker-compose logs -f

# Initial setup
setup: install
	@echo "Setting up environment..."
	@if [ ! -f .env ]; then \
		cp env.example .env; \
		echo "Created .env file from env.example"; \
		echo "Please edit .env file with your OpenAI API key"; \
	fi
	@mkdir -p chroma_db data
	@echo "Setup complete!"

# Database commands
db-init:
	python -c "from database import init_db; init_db()"

# Production commands
prod-install:
	pip install -r requirements.txt --no-dev

prod-run:
	gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Health check
health:
	curl -f http://localhost:8000/health || echo "Service not running"
