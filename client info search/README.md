# Client Info Search API

A comprehensive FastAPI application with SQLite database, Chroma RAG (Retrieval-Augmented Generation), and OpenAI function-calling tools for client information management and document search.

## Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **SQLite Database**: Lightweight database for storing client and document information
- **Chroma RAG**: Vector database for semantic search and document retrieval
- **OpenAI Integration**: Function calling capabilities and AI-powered features
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **DevContainer**: VS Code development container configuration
- **Comprehensive Testing**: Full test suite with pytest
- **Development Tools**: Makefile with common development commands

## Project Structure

```
client-info-search/
├── main.py                 # FastAPI application entry point
├── database.py            # SQLite database models and configuration
├── models.py              # Database model imports
├── schemas.py             # Pydantic schemas for request/response models
├── rag_service.py         # Chroma RAG service implementation
├── openai_service.py      # OpenAI service with function calling
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker container configuration
├── docker-compose.yml     # Docker Compose configuration
├── Makefile              # Development commands
├── pyproject.toml        # Project configuration
├── env.example           # Environment variables template
├── .devcontainer/        # VS Code devcontainer configuration
│   └── devcontainer.json
└── tests/                # Test suite
    ├── conftest.py       # Test configuration and fixtures
    ├── test_main.py      # Main application tests
    ├── test_database.py  # Database model tests
    ├── test_rag_service.py # RAG service tests
    ├── test_openai_service.py # OpenAI service tests
    └── test_api.py       # API endpoint tests
```

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional)
- OpenAI API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd client-info-search
   ```

2. **Set up environment**:
   ```bash
   make setup
   ```

3. **Configure environment variables**:
   ```bash
   cp env.example .env
   # Edit .env file with your OpenAI API key
   ```

4. **Install dependencies**:
   ```bash
   make install
   ```

5. **Run the application**:
   ```bash
   make dev
   ```

The API will be available at `http://localhost:8000`

### Using Docker

1. **Build and run with Docker Compose**:
   ```bash
   make compose-up
   ```

2. **Or build and run manually**:
   ```bash
   make docker-build
   make docker-run
   ```

## API Documentation

Once the application is running, you can access:

- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`
- **OpenAPI schema**: `http://localhost:8000/openapi.json`

## API Endpoints

### Core Endpoints
- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint

### Document Management
- `POST /documents/` - Create a new document
- `GET /documents/` - Get all documents (with pagination)
- `GET /documents/{document_id}` - Get specific document

### Client Management
- `POST /clients/` - Create a new client
- `GET /clients/` - Get all clients (with pagination)
- `GET /clients/{client_id}` - Get specific client

### RAG Search
- `POST /search/` - Search documents using RAG

### OpenAI Function Calling
- `POST /function-call/` - Execute OpenAI function calling
- `GET /functions/` - List available functions

## Development

### Available Commands

```bash
make help          # Show all available commands
make install       # Install dependencies
make dev           # Run development server
make test          # Run tests
make lint          # Run linting
make format        # Format code
make clean         # Clean up generated files
make docker-build  # Build Docker image
make docker-run    # Run Docker container
make docker-stop   # Stop Docker container
make setup         # Initial setup
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
mypy .
```

## Environment Variables

Create a `.env` file based on `env.example`:

```env
DATABASE_URL=sqlite:///./client_info.db
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
DEBUG=True
LOG_LEVEL=INFO
```

## Database Schema

The database is designed to handle both **structured** and **unstructured** client information with maximum flexibility.

### Clients Table
**Structured Fields:**
- `id`: Primary key
- `name`: Client name (required)
- `email`: Client email (unique, required)
- `phone`: Phone number
- `company`: Company name
- `job_title`: Job title
- `industry`: Industry classification
- `location`: Geographic location
- `website`: Company website
- `status`: Client status (active, inactive, prospect, lead)
- `priority`: Priority level (low, medium, high, urgent)
- `source`: Lead source (referral, website, cold_call, etc.)
- `budget_range`: Budget range
- `annual_revenue`: Annual revenue
- `preferred_contact_method`: Communication preference
- `timezone`: Client timezone
- `language`: Preferred language
- `notes`: Free-form notes
- `last_contact_date`: Last contact timestamp
- `next_follow_up`: Next follow-up date

**Unstructured Data Storage:**
- `raw_data`: JSON field for any unstructured data
- `custom_fields`: JSON field for custom business fields
- `tags`: JSON array for flexible tagging
- `metadata`: JSON field for additional metadata

### Client Interactions Table
- `id`: Primary key
- `client_id`: Foreign key to clients
- `interaction_type`: Type of interaction (call, email, meeting, note)
- `subject`: Interaction subject
- `content`: Interaction content
- `outcome`: Interaction outcome
- `participants`: List of participants
- `location`: Meeting location or call type
- `duration_minutes`: Interaction duration
- `follow_up_required`: Boolean flag
- `follow_up_date`: Follow-up date
- `follow_up_notes`: Follow-up notes
- `raw_data`: JSON field for unstructured interaction data
- `attachments`: List of file attachments
- `tags`: JSON array for tagging
- `metadata`: JSON field for additional metadata

### Client Documents Table
- `id`: Primary key
- `client_id`: Foreign key to clients
- `title`: Document title
- `content`: Document content
- `document_type`: Type of document (contract, proposal, invoice, etc.)
- `file_path`: Path to actual file
- `file_size`: File size in bytes
- `mime_type`: MIME type
- `summary`: AI-generated summary
- `keywords`: Extracted keywords
- `entities`: Named entities
- `sentiment`: Sentiment analysis result
- `category`: Document category
- `subcategory`: Document subcategory
- `priority`: Document priority
- `confidential`: Confidentiality flag
- `raw_data`: JSON field for unstructured document data
- `custom_fields`: JSON field for custom document fields
- `tags`: JSON array for tagging
- `metadata`: JSON field for additional metadata

### Client Info Searches Table
- `id`: Primary key
- `query`: Search query
- `search_type`: Type of search performed
- `filters`: Applied filters
- `results_count`: Number of results
- `execution_time_ms`: Query execution time
- `user_id`: User who performed search
- `session_id`: Session identifier
- `results_summary`: Summary of results
- `created_at`: Search timestamp

## Enhanced Data Handling

The database is designed to handle both structured and unstructured client information:

### Structured Data
- **Standard Fields**: Name, email, phone, company, etc.
- **Business Fields**: Status, priority, source, budget, etc.
- **Communication**: Contact preferences, timezone, language
- **Tracking**: Last contact, follow-up dates, timestamps

### Unstructured Data Storage
- **raw_data**: JSON field for any unstructured data
  ```json
  {
    "social_media": {"linkedin": "url", "twitter": "handle"},
    "preferences": {"communication_style": "casual"},
    "internal_notes": "Additional context..."
  }
  ```
- **custom_fields**: Business-specific fields
  ```json
  {
    "custom_sales_stage": "qualification",
    "custom_lead_score": 85,
    "custom_referral_source": "conference_2024"
  }
  ```
- **tags**: Flexible tagging system
  ```json
  ["enterprise", "ai-interested", "high-value"]
  ```
- **metadata**: Additional metadata
  ```json
  {
    "lead_source_detail": "TechCrunch Disrupt 2024",
    "assigned_sales_rep": "Mike Chen",
    "territory": "West Coast"
  }
  ```

### Data Handling Utilities
- **ClientDataHandler**: Utility class for creating/updating clients
- **DocumentDataHandler**: Utility class for document management
- **Search Functions**: Search across structured and unstructured data
- **Analytics**: Comprehensive client analytics and reporting

## RAG Service

The RAG service uses Chroma for vector storage and retrieval:

- **Document Storage**: Documents are stored in both SQLite and Chroma
- **Vector Search**: Semantic search using embeddings
- **Metadata Filtering**: Filter results by metadata
- **Persistence**: Chroma database persists between sessions

## OpenAI Integration

The OpenAI service provides:

- **Function Calling**: Execute predefined functions
- **Text Analysis**: Summarization, sentiment analysis, etc.
- **Embeddings**: Generate text embeddings
- **Moderation**: Content moderation

### Available Functions

- `search_documents`: Search for documents using RAG
- `get_client_info`: Get information about a specific client
- `create_client`: Create a new client
- `analyze_text`: Analyze and summarize text content

## Docker Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
# Build production image
docker build -t client-info-search .

# Run with environment variables
docker run -d \
  --name client-info-search \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/data:/app/data \
  client-info-search
```

## VS Code DevContainer

The project includes a VS Code devcontainer configuration:

1. Open the project in VS Code
2. Install the "Dev Containers" extension
3. Press `Ctrl+Shift+P` and select "Dev Containers: Reopen in Container"
4. The development environment will be set up automatically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions or issues, please open an issue on the GitHub repository.
