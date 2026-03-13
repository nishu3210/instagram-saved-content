# Database & Storage Documentation

## Active Database
**Location**: `output/instagram_analyzer.db`  
**Purpose**: Main SQLite database storing all Instagram posts and AI analysis

**Why this location?**
- Configured in `database.py` via `get_db_uri()` function
- All application data (posts, analysis, conversations) lives here
- **Current data**: 886 posts with full analysis

## RAG Index Files (After Manual Build)
**Location**: `output/`
- `rag_index.bin` - FAISS vector index
- `indexed_posts.json` - Metadata mapping post IDs to vectors

**Purpose**: Enable semantic search in chat feature
**How to build**: Click "🧠 Rebuild RAG Index" button in Dashboard

## Old Files (Removed)
- `instance/instagram_analyzer.db` - Old empty DB from Flask SQLAlchemy default, now deleted
- Created automatically but never used due to custom `get_db_uri()` override

## Directory Structure
```
/instagram-saved-content/
├── output/                          # Active data directory
│   ├── instagram_analyzer.db       ✅ Main database (3.4MB)
│   ├── rag_index.bin               🔄 Built on-demand
│   ├── indexed_posts.json          🔄 Built on-demand
│   └── analyzed_results.json       📊 Export cache
├── instance/                        # Flask default (now empty)
├── archive/old-ui/                  # Archived UI files
└── *.py                            # Application code
```

## Important Notes
1. **Single source of truth**: `output/instagram_analyzer.db`
2. **Backups**: Recommended to backup `output/` directory periodically
3. **RAG persistence**: Index rebuilds are expensive (API calls), so they persist to disk
