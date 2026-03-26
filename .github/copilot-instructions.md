# Instagram Saved Posts AI Analyzer - AI Agent Instructions

## Architecture Overview

This is a Flask-based web application that analyzes Instagram saved posts using Azure OpenAI. The system consists of:

- **Flask Backend** (`flask-backend.py`): REST API server with endpoints for analysis control and results
- **Web UI** (`templates/index.html`): Single-page dashboard with credential input, progress tracking, and results visualization
- **Analysis Script** (`instagram_analyzer.py`): External Python script that performs the actual Instagram scraping and AI analysis
- **Output Directory** (`output/`): Stores analysis results as JSON files

## Key Design Patterns

### Credential Management
- Credentials are passed via API and stored temporarily in `.env.temp` files during analysis
- Never persist credentials - always clean up temporary files after use
- Example: Backend creates `.env.temp` with `INSTAGRAM_SESSIONID`, `AZURE_ENDPOINT`, `AZURE_KEY` before calling analysis script

### Background Processing
- Long-running analysis runs in background threads to avoid blocking the UI
- Frontend polls `/api/status` endpoint every 1 second for progress updates
- Status object tracks: `running` (bool), `progress` (0-100), `message` (string), `results` (dict), `error` (string)

### API Endpoints Structure
```
POST /api/analyze       # Start analysis with credentials
GET  /api/status        # Poll analysis progress
GET  /api/results       # Get full analysis results
GET  /api/results/download  # Download JSON results
GET  /api/results/csv   # Export as CSV
POST /api/test-credentials # Validate Instagram/Azure connections
GET  /api/summary       # Get summary stats only
```

### Data Flow
1. User inputs credentials in web UI
2. Frontend calls `/api/analyze` with sessionid, azure_endpoint, azure_key, model, max_posts
3. Backend creates `.env.temp` file and starts background thread
4. Thread calls `python instagram_analyzer.py` as subprocess
5. Analysis script reads `.env.temp`, fetches Instagram saved posts, analyzes with Azure OpenAI
6. Results written to `output/analyzed_results.json`
7. Frontend polls status until `running: false`, then displays results

## Critical Workflows

### Starting Analysis
```python
# In flask-backend.py run_analysis() function
env_content = f"""
INSTAGRAM_SESSIONID={sessionid}
AZURE_ENDPOINT={azure_endpoint}
AZURE_KEY={azure_key}
MODEL={model}
MAX_POSTS={max_posts}
"""
with open('.env.temp', 'w') as f:
    f.write(env_content)

result = subprocess.run(['python', 'instagram_analyzer.py'], ...)
```

### Progress Tracking
- Update `analysis_status` dict in global scope from background thread
- Frontend JavaScript polls with `fetch('/api/status')` every 1000ms
- Progress messages: "Initializing...", "Testing Instagram connection...", "Fetching saved posts...", "Analyzing with Azure AI..."

### Results Processing
- Analysis script outputs structured JSON with `analyzed_posts` array and `summary` object
- Each post contains: `post` (original data) and `analysis` (AI insights)
- Summary includes: total_posts_analyzed, average_sentiment_score, categories_breakdown, sentiment_distribution, top_topics, total_tokens, total_cost

## Dependencies & Environment

### Required Packages
```
flask flask-cors requests openai pandas
```

### External Services
- **Instagram API**: Uses session cookie authentication, no official API
- **Azure OpenAI**: Requires deployed model (gpt-4o-mini recommended), endpoint URL, and API key
- **Chart.js**: CDN-loaded for data visualization in frontend

## Project-Specific Conventions

### Error Handling
- Use try/catch in background threads, set `analysis_status['error']` on failure
- Frontend checks `status.error` and shows alert dialogs
- Log errors with `logger.error()` but don't expose internal details to UI

### File Organization
- Flask templates in `templates/` directory (even though UI can be standalone)
- Analysis outputs in `output/` directory, created automatically if missing
- No `requirements.txt` - dependencies installed manually via pip

### UI Patterns
- Single HTML file with embedded CSS/JS, no build process
- Tab-based navigation with `tab-content` divs and `active` class switching
- Gradient backgrounds and glass-morphism styling with Tailwind CSS
- Real-time progress bars and spinners during analysis

### Security Considerations
- Instagram session cookies are sensitive - handle as temporary credentials only
- Azure API keys passed through API but not stored permanently
- CORS enabled for local development (`flask-cors`)

## Development Commands

### Running the Application
```bash
# Terminal 1: Start Flask backend
python flask-backend.py

# Terminal 2: Open browser to http://localhost:5000
```

### Testing Credentials
- Use `/api/test-credentials` endpoint to validate Instagram and Azure connections
- Instagram test: GET request to Instagram API with session cookie
- Azure test: Simple chat completion call to verify endpoint/key

### Debugging Analysis
- Check `output/analyzed_results.json` for results after completion
- Monitor Flask logs for subprocess errors
- Use browser DevTools Network tab to inspect API calls

## Common Issues & Solutions

### Analysis Fails
- Check if `instagram_analyzer.py` exists and is executable
- Verify credentials are valid (use "Test Credentials" button)
- Check Flask logs for subprocess error output
- Ensure `output/` directory is writable

### UI Not Loading
- Confirm Flask is running on port 5000
- Check browser console for CORS errors
- Verify Chart.js CDN is accessible

### No Results Displayed
- Ensure analysis completed successfully (check status endpoint)
- Verify `analyzed_results.json` contains expected structure
- Check browser console for JavaScript errors in display functions