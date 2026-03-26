# Instagram Saved Content Analyzer

An AI-powered local web app for turning Instagram saved posts into a searchable knowledge base, action list, and recall assistant.

## 🚀 Major Changes & Improvements

### Security Fixes
- ✅ **Removed hardcoded credentials** - All credentials now loaded from `.env` file
- ✅ **Added `.gitignore`** - Prevents accidental commits of sensitive files
- ✅ **Rate limiting** - Flask-Limiter protects API endpoints from abuse
- ✅ **Security headers** - Flask-Talisman adds security headers
- ✅ **Input validation** - Pydantic schemas validate all API inputs

### Architecture Improvements
- ✅ **Centralized configuration** - All config in `config.py` with proper validation
- ✅ **Thread-safe status management** - No more race conditions in global state
- ✅ **Proper service separation**:
  - `services/instagram_client.py` - Instagram API interactions
  - `services/ai_analyzer.py` - AI analysis with Azure OpenAI
  - `services/rag_service.py` - Semantic search with FAISS
- ✅ **Database migrations** - Proper schema versioning with `migrations.py`
- ✅ **Custom JSON type** - Eliminates double-encoding bugs

### Code Quality
- ✅ **Type hints** - Comprehensive typing throughout
- ✅ **Pydantic schemas** - Request/response validation
- ✅ **Error handling** - Proper exception hierarchy and handling
- ✅ **Logging** - Structured logging with proper levels
- ✅ **Tests** - Comprehensive test suite with pytest

### Performance
- ✅ **Pagination** - All list endpoints support pagination
- ✅ **Rate limiting** - Built-in request throttling
- ✅ **N+1 query fixes** - Proper eager loading with joins
- ✅ **Connection pooling** - Database connection management

## What Works Today

- Sync saved Instagram posts into a local database
- Analyze posts with Azure OpenAI
- Browse posts by category, sentiment, and collection
- Generate action tasks from saved content
- Chat with your saved library using RAG search
- Export reports as Markdown, JSON, and CSV

## Startup

Use `run.py` to start the app.

`app.py` defines the Flask application and routes, but `run.py` is the correct entrypoint because it:

- ensures required directories exist
- initializes database tables
- runs migrations
- logs configuration status before serving the app

Start the app with:

```bash
python run.py
```

Then open:

```bash
http://127.0.0.1:5001
```

Health check:

```bash
http://127.0.0.1:5001/api/health
```

## Project Structure

```
instagram-saved-content/
├── app.py                          # Flask app and API routes
├── config.py                       # Centralized configuration
├── database.py                     # Database models with JSON type
├── schemas.py                      # Pydantic validation schemas
├── status_manager.py               # Thread-safe status management
├── migrations.py                   # Database migrations
├── run.py                          # Correct startup entrypoint
├── requirements.txt                # Pinned dependencies
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore patterns
├── services/
│   ├── __init__.py
│   ├── instagram_client.py         # Instagram API client
│   ├── ai_analyzer.py              # AI analysis service
│   └── rag_service.py              # RAG/semantic search
└── tests/
    ├── conftest.py                 # Pytest fixtures
    └── test_models.py              # Model tests
```

## Setup

1. **Create or reuse a conda environment:**
```bash
cd /path/to/instagram-saved-content
conda activate env1
```

If `env1` is already working, keep using it.

If you want a clean environment instead:
```bash
conda create -n instagram-saved-content python=3.11 -y
conda activate instagram-saved-content
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. **Run migrations:**
```bash
python -c "from app import app; from migrations import run_migrations; app.app_context().push(); run_migrations()"
```

5. **Start the server:**
```bash
python run.py
```

## Environment Notes

- Prefer `conda` for the Python environment and `pip` inside that environment for project packages.
- The project currently works with `openai==1.54.0` and `httpx==0.27.2`.
- If you see `Client.__init__() got an unexpected keyword argument 'proxies'`, your environment likely has an older `openai` package than the repo expects.

## Current UX State

The app is functional and the backend is tested, but the frontend is still evolving. If the UI feels visually underwhelming, that is a product polish issue rather than a startup or backend issue.

## 🔒 Security Checklist

- [x] No hardcoded credentials in source code
- [x] `.env` file in `.gitignore`
- [x] Rate limiting on all endpoints
- [x] Security headers enabled
- [x] Input validation on all API endpoints
- [x] SQL injection protection via SQLAlchemy ORM
- [x] XSS protection via proper escaping

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=.
```

Run with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

## 📊 API Endpoints

### Authentication
- `POST /api/auth/browser-cookies` - Extract cookies from browser
- `POST /api/test-connection` - Test Instagram connection

### Analysis
- `POST /api/analyze` - Start analysis job (rate limited)
- `GET /api/status` - Get analysis status
- `GET /api/stats` - Get statistics

### Posts
- `GET /api/posts` - List posts with filtering/pagination
- `POST /api/rebuild-rag` - Rebuild search index

### Tasks
- `GET /api/tasks` - List tasks
- `POST /api/tasks` - Create task
- `PATCH /api/tasks/<id>` - Update task
- `POST /api/tasks/bootstrap` - Generate tasks from analysis

### Chat
- `POST /api/chat` - Chat with content (RAG)

## 🐛 Fixed Issues

1. **Security vulnerabilities**
   - Hardcoded Azure API key removed
   - `.env` added to `.gitignore`

2. **Race conditions**
   - Global state now thread-safe with `ThreadSafeStatusManager`

3. **Double JSON encoding**
   - Custom `JSONList` type handles serialization properly

4. **No input validation**
   - Pydantic schemas validate all inputs

5. **No rate limiting**
   - Flask-Limiter added with sensible defaults

6. **N+1 queries**
   - Proper eager loading with `joinedload`

7. **No pagination**
   - All list endpoints support pagination

8. **Monolithic code**
   - Split into logical services

9. **No tests**
   - Comprehensive test suite added

10. **No error handling**
    - Proper exception hierarchy and handling

## 📝 Environment Variables

See `.env.example` for all available options:

```env
# Instagram
INSTAGRAM_SESSIONID=your_session_id
RAW_COOKIE=your_cookie_string
USER_AGENT=your_user_agent

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview
MODEL=DeepSeek-V3.2

# Embedding
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072

# App
FLASK_DEBUG=0
FLASK_PORT=5001
FLASK_HOST=127.0.0.1
SECRET_KEY=your_secret_key
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

MIT License
