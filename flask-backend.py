"""
Flask Backend API for Instagram AI Analyzer
Provides REST endpoints to trigger analysis and retrieve results
"""

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import json
import os
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging
import browser_cookie3
from dotenv import load_dotenv
from openai import AzureOpenAI
from services.rag_service import RAGService
from instagram_analyzer import InstagramAnalyzer
import hashlib
import requests
import re
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

import uuid
from database import db, Post, Analysis, ActionTask, Conversation, Message, get_db_uri

app = Flask(__name__)
# Use centralized database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = get_db_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost:5001", "http://127.0.0.1:5001"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
analysis_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'error': None,
    'results': None
}

task_job_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'error': None,
    'results': None
}

last_instagram_auth = {
    'cookie': '',
    'user_agent': 'Mozilla/5.0'
}

# Initialize RAG Service
rag_service = RAGService()

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

TASK_STATUSES = {'pending', 'in_progress', 'done', 'archived'}


def get_azure_endpoint():
    return os.getenv('AZURE_OPENAI_API_BASE') or os.getenv('AZURE_ENDPOINT') or ''


def get_azure_key():
    return os.getenv('AZURE_OPENAI_API_KEY') or os.getenv('AZURE_KEY') or ''


def parse_iso_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime):
        if value.tzinfo:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    try:
        text = str(value).strip()
        if text.endswith('Z'):
            text = text[:-1] + '+00:00'
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except (TypeError, ValueError):
        return None


def normalize_status(value):
    status = str(value or 'pending').strip().lower()
    return status if status in TASK_STATUSES else 'pending'


def normalize_priority(value):
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return 2
    return max(1, min(3, priority))


def task_source_key(post_id, title):
    normalized = f"{post_id or 'none'}::{(title or '').strip().lower()}"
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def remember_instagram_auth(data):
    if not isinstance(data, dict):
        return

    sessionid = (data.get('sessionid') or '').strip()
    raw_cookie = (data.get('raw_cookie') or '').strip()
    user_agent = (data.get('user_agent') or '').strip()

    if raw_cookie:
        last_instagram_auth['cookie'] = raw_cookie
    elif sessionid:
        last_instagram_auth['cookie'] = f"sessionid={sessionid}"

    if user_agent:
        last_instagram_auth['user_agent'] = user_agent


def build_instagram_headers():
    headers = {
        'User-Agent': last_instagram_auth.get('user_agent') or 'Mozilla/5.0',
        'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
        'Referer': 'https://www.instagram.com/',
    }
    cookie = (last_instagram_auth.get('cookie') or '').strip()
    if cookie:
        headers['Cookie'] = cookie
    return headers


def is_allowed_media_host(hostname):
    if not hostname:
        return False
    host = hostname.lower()
    allowed_suffixes = (
        'cdninstagram.com',
        'fbcdn.net',
        'instagram.com'
    )
    return any(host == suffix or host.endswith(f'.{suffix}') for suffix in allowed_suffixes)


def extract_og_image_url(html):
    if not html:
        return None
    match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if not match:
        return None
    return match.group(1)

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get Azure config from environment"""
    return jsonify({
        'azure_endpoint': get_azure_endpoint(),
        'has_azure_key': bool(get_azure_key())
    })

@app.route('/api/status', methods=['GET'])
def status():
    """Get current analysis status"""
    return jsonify(analysis_status)

@app.route('/api/unanalyzed-count', methods=['GET'])
def get_unanalyzed_count():
    """Get count of unanalyzed posts"""
    with app.app_context():
        # Ensure tables exist
        db.create_all()
        count = Post.query.filter(Post.analysis == None).count()
        return jsonify({'count': count})

@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    """Start Instagram analysis in background"""
    
    if analysis_status['running']:
        return jsonify({'error': 'Analysis already running'}), 400
    
    data = request.json or {}
    remember_instagram_auth(data)
    sessionid = data.get('sessionid')
    raw_cookie = data.get('raw_cookie')
    user_agent = data.get('user_agent')
    azure_endpoint = data.get('azure_endpoint') or get_azure_endpoint()
    azure_key = data.get('azure_key') or get_azure_key()
    model = data.get('model', os.getenv('MODEL', 'DeepSeek-V3.2'))
    max_posts = int(data.get('max_posts', 20))
    
    # Validate inputs
    if not all([sessionid, azure_endpoint, azure_key]):
        return jsonify({'error': 'Missing required credentials'}), 400
    
    # Reset status
    analysis_status['running'] = True
    analysis_status['progress'] = 0
    analysis_status['message'] = 'Initializing...'
    analysis_status['error'] = None
    analysis_status['results'] = None
    
    # Run analysis in background thread
    thread = threading.Thread(
        target=run_analysis,
        args=(sessionid, raw_cookie, user_agent, azure_endpoint, azure_key, model, max_posts)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Analysis started in background'})

@app.route('/api/fetch-posts', methods=['POST'])
def fetch_posts():
    """Fetch all posts and sync to DB"""
    if analysis_status['running']:
        return jsonify({'error': 'Task already running'}), 400
        
    data = request.json or {}
    remember_instagram_auth(data)
    sessionid = data.get('sessionid')
    
    if not sessionid:
        return jsonify({'error': 'Session ID required'}), 400
        
    # Reset status
    analysis_status['running'] = True
    analysis_status['progress'] = 0
    analysis_status['message'] = 'Initializing sync...'
    analysis_status['error'] = None
    
    # Run in background
    thread = threading.Thread(
        target=run_sync,
        args=(data,) # data already contains 'browser' from request
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Sync started'})

@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch():
    """Analyze unanalyzed posts"""
    if analysis_status['running']:
        return jsonify({'error': 'Task already running'}), 400
        
    data = request.json or {}
    batch_size = data.get('batch_size')
    
    # Reset status
    analysis_status['running'] = True
    analysis_status['progress'] = 0
    analysis_status['message'] = 'Initializing analysis...'
    analysis_status['error'] = None
    
    # Run in background
    thread = threading.Thread(
        target=run_batch_analysis,
        args=(data,)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Batch analysis started'})

@app.route('/api/test-connection', methods=['POST'])
def test_connection():
    """Test Instagram connection"""
    try:
        data = request.json or {}
        remember_instagram_auth(data)
        config = {
            'sessionid': data.get('sessionid'),
            'raw_cookie': data.get('raw_cookie'),
            'browser': data.get('browser'), # Pass browser selection
            'user_agent': data.get('user_agent'),
            'azure_endpoint': '',
            'azure_key': '',
            'model': os.getenv('MODEL', 'DeepSeek-V3.2'),
            'max_posts': 1
        }
        
        analyzer = InstagramAnalyzer(config)
        # Use validate_session for accurate check
        is_valid, message = analyzer.validate_session()
        
        if is_valid:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'error': message})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def run_sync(data):
    """Background sync task"""
    try:
        analyzer = InstagramAnalyzer(data)
        
        def update_progress(p, msg):
            analysis_status['progress'] = p
            analysis_status['message'] = msg
            
        result = analyzer.fetch_and_sync_posts(progress_callback=update_progress)
        
        analysis_status['message'] = f"Sync complete! Found {result['count']} posts ({result['new']} new)."
        analysis_status['progress'] = 100
        
    except Exception as e:
        analysis_status['error'] = str(e)
        logger.error(f"Sync error: {e}")
    finally:
        analysis_status['running'] = False

def run_batch_analysis(data):
    """Background batch analysis task"""
    try:
        analyzer = InstagramAnalyzer(data)
        
        def update_progress(p, msg):
            analysis_status['progress'] = p
            analysis_status['message'] = msg
            
        result = analyzer.analyze_batch(
            batch_size=data.get('batch_size'),
            progress_callback=update_progress
        )
        
        # Re-index RAG
        with app.app_context():
            posts = Post.query.filter(Post.analysis != None).all()
            rag_service.index_posts(posts)
            
        analysis_status['message'] = f"Analysis complete! Processed {result['analyzed_count']} posts."
        analysis_status['results'] = result
        analysis_status['progress'] = 100
        
    except Exception as e:
        analysis_status['error'] = str(e)
        logger.error(f"Batch analysis error: {e}")
    finally:
        analysis_status['running'] = False

def run_analysis(sessionid, raw_cookie, user_agent, azure_endpoint, azure_key, model, max_posts):
    """Run the actual analysis (called in background thread)"""
    try:
        def update_progress(percent, message):
            analysis_status['progress'] = percent
            analysis_status['message'] = message
            
        # Initialize Analyzer
        config = {
            'sessionid': sessionid,
            'raw_cookie': raw_cookie,
            'user_agent': user_agent,
            'azure_endpoint': azure_endpoint,
            'azure_key': azure_key,
            'model': model,
            'max_posts': max_posts
        }
        
        analyzer = InstagramAnalyzer(config)
        
        # Run Analysis
        analyzer.run_analysis(progress_callback=update_progress)
        
        # Load results (analyzer saves to file)
        results_file = OUTPUT_DIR / 'analyzed_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                analysis_status['results'] = json.load(f)
            
            analysis_status['message'] = 'Indexing for RAG...'
            analysis_status['progress'] = 90
            
            # Re-index RAG
            with app.app_context():
                posts = Post.query.all()
                rag_service.index_posts(posts)

            analysis_status['message'] = 'Analysis complete!'
            analysis_status['progress'] = 100
        else:
            analysis_status['error'] = 'Results file not found'
    
    except Exception as e:
        analysis_status['error'] = f"Error: {str(e)}"
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
    finally:
        analysis_status['running'] = False
    



@app.route('/api/results', methods=['GET'])
def get_results():
    """Get analysis results"""
    if not analysis_status['results']:
        return jsonify({'error': 'No results available'}), 404
    
    return jsonify(analysis_status['results'])

@app.route('/api/results/download', methods=['GET'])
def download_results():
    """Download results as JSON"""
    results_file = OUTPUT_DIR / 'analyzed_results.json'
    if not results_file.exists():
        return jsonify({'error': 'No results available'}), 404
    
    return send_file(
        results_file,
        mimetype='application/json',
        as_attachment=True,
        download_name=f"instagram_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

@app.route('/api/results/csv', methods=['GET'])
def export_csv():
    """Export results as CSV"""
    try:
        import pandas as pd
        
        if not analysis_status['results']:
            return jsonify({'error': 'No results available'}), 404
        
        posts = analysis_status['results'].get('analyzed_posts', [])
        
        # Flatten data for CSV
        csv_data = []
        for item in posts:
            post = item.get('post', {})
            analysis = item.get('analysis', {})
            
            csv_data.append({
                'username': post.get('username'),
                'caption': post.get('caption', '')[:100],  # Truncate caption
                'category': analysis.get('category'),
                'sentiment_score': analysis.get('sentiment', {}).get('score'),
                'sentiment_label': analysis.get('sentiment', {}).get('label'),
                'credibility_score': analysis.get('credibility_score'),
                'topics': ', '.join(analysis.get('topics', [])),
                'url': post.get('url'),
                'likes': post.get('likes'),
                'comments': post.get('comments'),
                'timestamp': post.get('timestamp')
            })
        
        df = pd.DataFrame(csv_data)
        
        # Save to file
        csv_file = OUTPUT_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        
        return send_file(
            csv_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=csv_file.name
        )
    
    except ImportError:
        return jsonify({'error': 'pandas not installed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/browser-cookies', methods=['POST'])
def get_browser_cookies():
    """Extract cookies from browser"""
    try:
        data = request.json
        browser_name = data.get('browser', 'all')
        
        # List of browsers to try
        browsers = []
        if browser_name == 'all':
            browsers = [
                ('Chrome', browser_cookie3.chrome),
                ('Firefox', browser_cookie3.firefox),
                ('Safari', browser_cookie3.safari),
                ('Edge', browser_cookie3.edge),
                ('Chromium', browser_cookie3.chromium),
                ('Brave', browser_cookie3.brave),
                ('Opera', browser_cookie3.opera)
            ]
        elif hasattr(browser_cookie3, browser_name.lower()):
            browsers = [(browser_name, getattr(browser_cookie3, browser_name.lower()))]
            
        found_session = None
        found_browser = None
        
        for name, func in browsers:
            try:
                print(f"Attempting to load cookies from {name}...")
                cj = func(domain_name='.instagram.com')
                
                for cookie in cj:
                    if cookie.name == 'sessionid':
                        found_session = cookie.value
                        found_browser = name
                        break
                
                if found_session:
                    break
            except Exception as e:
                print(f"Could not load from {name}: {e}")
                continue
                
        if found_session:
            payload = {
                'sessionid': found_session,
                'raw_cookie': '', # Clear raw cookie to avoid conflicts
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0',
                'browser': found_browser
            }
            remember_instagram_auth(payload)
            return jsonify({
                'sessionid': found_session,
                'raw_cookie': '', # Clear raw cookie to avoid conflicts
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0',
                'browser': found_browser
            })
        else:
            return jsonify({'error': 'Could not find Instagram session in any browser. Please make sure you are logged in.'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/test-credentials', methods=['POST'])
def test_credentials():
    """Test if Instagram and Azure credentials work"""
    data = request.json or {}
    remember_instagram_auth(data)
    sessionid = data.get('sessionid')
    raw_cookie = data.get('raw_cookie')
    user_agent = data.get('user_agent')
    azure_endpoint = data.get('azure_endpoint') or get_azure_endpoint()
    azure_key = data.get('azure_key') or get_azure_key()
    model = data.get('model', os.getenv('MODEL', 'DeepSeek-V3.2'))
    
    results = {
        'instagram': False,
        'azure': False,
        'messages': []
    }
    
    # Test Instagram
    try:
        import requests
        
        # Build headers exactly like the working debug script
        headers = {
            'User-Agent': user_agent or 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0',
            'Accept': 'application/json, text/plain, */*',
            'X-IG-App-ID': '936619743392459',
            'X-Requested-With': 'XMLHttpRequest',
        }
        
        # Add the raw cookie directly to headers (like debug_ua.py)
        if raw_cookie:
            headers['Cookie'] = raw_cookie
        elif sessionid:
            headers['Cookie'] = f'sessionid={sessionid}'
        
        # Test with the saved posts endpoint using simple requests.get (not session)
        response = requests.get(
            'https://www.instagram.com/api/v1/feed/saved/posts/',
            headers=headers,
            timeout=10
        )
        logger.info(f"Instagram test response status: {response.status_code}")
        logger.info(f"Instagram test response headers: {dict(response.headers)}")
        try:
            response_text = response.text[:500]  # First 500 chars
            logger.info(f"Instagram test response body: {response_text}")
        except:
            logger.info("Could not read response body")
        
        if response.status_code == 200:
            results['instagram'] = True
            results['messages'].append('✓ Instagram connection successful')
        else:
            results['messages'].append(f'✗ Instagram error: {response.status_code} - Check if session ID is valid and fresh')
    except Exception as e:
        results['messages'].append(f'✗ Instagram error: {str(e)}')
    
    # Test Azure
    try:
        if not azure_endpoint or not azure_key:
            raise ValueError("Azure endpoint/key missing")
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version="2024-08-01-preview"
        )
        # Try a simple API call
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10
        )
        results['azure'] = True
        results['messages'].append('✓ Azure OpenAI connection successful')
    except Exception as e:
        results['messages'].append(f'✗ Azure error: {str(e)}')
    
    return jsonify(results)

@app.route('/', methods=['GET'])
def index():
    """Serve the main UI"""
    return send_file('index-ui.html')


@app.route('/api/media-proxy', methods=['GET'])
def media_proxy():
    """Proxy Instagram media so browser can render authenticated or stale URLs more reliably."""
    try:
        raw_url = (request.args.get('url') or '').strip()
        shortcode = (request.args.get('shortcode') or '').strip()

        candidates = []
        if raw_url:
            parsed = urlparse(raw_url)
            if parsed.scheme != 'https' or not is_allowed_media_host(parsed.hostname):
                return jsonify({'error': 'Invalid media URL host'}), 400
            candidates.append(raw_url)
        if shortcode:
            candidates.append(f'https://www.instagram.com/p/{shortcode}/')

        if not candidates:
            return jsonify({'error': 'url or shortcode is required'}), 400

        headers = build_instagram_headers()
        seen = set()
        while candidates:
            candidate = candidates.pop(0)
            if candidate in seen:
                continue
            seen.add(candidate)

            try:
                resp = requests.get(candidate, headers=headers, timeout=20, allow_redirects=True)
            except Exception:
                continue

            if resp.status_code != 200:
                continue

            content_type = (resp.headers.get('Content-Type') or '').lower()
            if content_type.startswith('image/'):
                proxy_resp = Response(resp.content, status=200, content_type=resp.headers.get('Content-Type', 'image/jpeg'))
                cache_control = resp.headers.get('Cache-Control', 'public, max-age=3600')
                proxy_resp.headers['Cache-Control'] = cache_control
                return proxy_resp

            # If shortcode page HTML was returned, try og:image URL next.
            if 'text/html' in content_type:
                og_image_url = extract_og_image_url(resp.text)
                if og_image_url:
                    parsed = urlparse(og_image_url)
                    if parsed.scheme == 'https' and is_allowed_media_host(parsed.hostname):
                        candidates.append(og_image_url)

        return jsonify({'error': 'Media not available'}), 404
    except Exception as e:
        logger.error(f"Media proxy error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get analysis summary without full posts"""
    if not analysis_status['results']:
        return jsonify({'error': 'No results available'}), 404
    
    return jsonify(analysis_status['results'].get('summary', {}))

@app.route('/api/posts', methods=['GET'])
def get_posts():
    """Get posts from database with pagination, filtering, and sorting"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        sort_by = request.args.get('sort', 'newest')
        category = request.args.get('category', 'all')
        collection = request.args.get('collection', 'all')
        sentiment = request.args.get('sentiment', 'all')
        
        query = Post.query
        
        # Join with Analysis if filtering/sorting by analysis fields
        if category != 'all' or sentiment != 'all' or 'sentiment' in sort_by:
            query = query.join(Analysis)
            
        # Apply Filters
        if collection != 'all':
            # naive string search in JSON column
            query = query.filter(Post.collections.contains(collection))

        if category != 'all':
            query = query.filter(Analysis.category == category)
            
        if sentiment != 'all':
            query = query.filter(Analysis.sentiment_label == sentiment)
            
        # Apply Sorting
        if sort_by == 'newest':
            query = query.order_by(Post.timestamp.desc())
        elif sort_by == 'oldest':
            query = query.order_by(Post.timestamp.asc())
        elif sort_by == 'sentiment_desc': # Positive first
            query = query.order_by(Analysis.sentiment_score.desc())
        elif sort_by == 'sentiment_asc': # Negative first
            query = query.order_by(Analysis.sentiment_score.asc())
        
        # Pagination
        pagination = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'posts': [post.to_dict() for post in pagination.items],
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        })
    except Exception as e:
        logger.error(f"Error fetching posts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get analysis statistics"""
    try:
        total_posts = Post.query.count()
        analyzed_posts = db.session.query(db.func.count(db.distinct(Analysis.post_id))).scalar() or 0
        
        # Calculate average sentiment
        avg_sentiment = db.session.query(db.func.avg(Analysis.sentiment_score)).scalar() or 0.0
        
        # Category breakdown (Normalized)
        raw_categories = db.session.query(Analysis.category, db.func.count(Analysis.category))\
            .group_by(Analysis.category).all()
            
        categories = {}
        for cat, count in raw_categories:
            # Normalize to Title Case to prevent duplicates like "Tech" vs "tech"
            norm_cat = cat.title() if cat else "Uncategorized"
            categories[norm_cat] = categories.get(norm_cat, 0) + count
        
        # Sentiment distribution
        sentiments = db.session.query(Analysis.sentiment_label, db.func.count(Analysis.sentiment_label))\
            .group_by(Analysis.sentiment_label).all()
        normalized_sentiments = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        for label, count in sentiments:
            key = str(label or '').strip().lower()
            if key == 'positive':
                normalized_sentiments['Positive'] += count
            elif key == 'negative':
                normalized_sentiments['Negative'] += count
            else:
                normalized_sentiments['Neutral'] += count
            
        return jsonify({
            'total_posts': total_posts,
            'analyzed_count': analyzed_posts, # Fixed key name
            'embedded_count': rag_service.index.ntotal if rag_service and rag_service.index else 0,
            'avg_sentiment': round(avg_sentiment, 2),
            'categories': categories,
            'sentiments': normalized_sentiments
        })
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """List tasks with pagination and filters"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 25, type=int), 100)
        status = request.args.get('status', 'all')
        due = request.args.get('due', 'all')
        now = datetime.now(timezone.utc)

        query = ActionTask.query
        if status != 'all':
            query = query.filter(ActionTask.status == normalize_status(status))

        if due == 'today':
            start = datetime(now.year, now.month, now.day)
            end = start + timedelta(days=1)
            query = query.filter(ActionTask.due_date >= start, ActionTask.due_date < end)
        elif due == 'overdue':
            query = query.filter(
                ActionTask.due_date != None,
                ActionTask.due_date < now,
                ActionTask.status.notin_(['done', 'archived'])
            )
        elif due == 'week':
            end = now + timedelta(days=7)
            query = query.filter(
                ActionTask.due_date != None,
                ActionTask.due_date >= now,
                ActionTask.due_date <= end
            )

        query = query.order_by(
            db.case(
                (ActionTask.status == 'in_progress', 0),
                (ActionTask.status == 'pending', 1),
                (ActionTask.status == 'done', 2),
                else_=3
            ),
            ActionTask.due_date.is_(None),
            ActionTask.due_date.asc(),
            ActionTask.priority.desc(),
            ActionTask.created_at.desc()
        )

        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        return jsonify({
            'tasks': [task.to_dict() for task in pagination.items],
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        })
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/tasks', methods=['POST'])
def create_task():
    """Create a new action task"""
    try:
        data = request.json or {}
        title = (data.get('title') or '').strip()
        if not title:
            return jsonify({'error': 'title is required'}), 400

        post_id = data.get('post_id')
        source = (data.get('source') or 'manual')[:50]
        source_key = data.get('source_key')
        if not source_key and title and (source != 'manual' or post_id):
            source_key = task_source_key(post_id, title)

        existing_task = ActionTask.query.filter_by(source_key=source_key).first() if source_key else None
        if existing_task:
            return jsonify({'task': existing_task.to_dict(), 'created': False})

        task = ActionTask(
            post_id=post_id,
            title=title,
            notes=data.get('notes'),
            status=normalize_status(data.get('status')),
            priority=normalize_priority(data.get('priority')),
            due_date=parse_iso_datetime(data.get('due_date')),
            scheduled_for=parse_iso_datetime(data.get('scheduled_for')),
            source=source,
            source_key=source_key,
            evidence_text=data.get('evidence_text')
        )
        if task.status == 'done':
            task.completed_at = datetime.now(timezone.utc)

        db.session.add(task)
        db.session.commit()
        return jsonify({'task': task.to_dict(), 'created': True}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating task: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/tasks/<int:task_id>', methods=['PATCH'])
def update_task(task_id):
    """Update task status/details"""
    try:
        task = ActionTask.query.get(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404

        data = request.json or {}

        if 'title' in data:
            title = (data.get('title') or '').strip()
            if not title:
                return jsonify({'error': 'title cannot be empty'}), 400
            task.title = title
        if 'notes' in data:
            task.notes = data.get('notes')
        if 'priority' in data:
            task.priority = normalize_priority(data.get('priority'))
        if 'due_date' in data:
            task.due_date = parse_iso_datetime(data.get('due_date'))
        if 'scheduled_for' in data:
            task.scheduled_for = parse_iso_datetime(data.get('scheduled_for'))
        if 'evidence_text' in data:
            task.evidence_text = data.get('evidence_text')
        if 'source' in data and data.get('source'):
            task.source = str(data.get('source'))[:50]

        if 'status' in data:
            status = normalize_status(data.get('status'))
            if status == 'done' and task.status != 'done':
                task.completed_at = datetime.now(timezone.utc)
            if status != 'done' and task.status == 'done':
                task.completed_at = None
            task.status = status

        if task.source != 'manual' or task.post_id:
            task.source_key = task_source_key(task.post_id, task.title)
        db.session.commit()
        return jsonify({'task': task.to_dict()})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating task: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/tasks/stats', methods=['GET'])
def task_stats():
    """Task compliance and follow-through metrics"""
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today_start = datetime(now.year, now.month, now.day)
        tomorrow = today_start + timedelta(days=1)

        total = ActionTask.query.count()
        open_count = ActionTask.query.filter(ActionTask.status.in_(['pending', 'in_progress'])).count()
        done_count = ActionTask.query.filter(ActionTask.status == 'done').count()
        overdue_count = ActionTask.query.filter(
            ActionTask.due_date != None,
            ActionTask.due_date < now,
            ActionTask.status.in_(['pending', 'in_progress'])
        ).count()
        due_today = ActionTask.query.filter(
            ActionTask.due_date != None,
            ActionTask.due_date >= today_start,
            ActionTask.due_date < tomorrow,
            ActionTask.status.in_(['pending', 'in_progress'])
        ).count()

        completed_7d = ActionTask.query.filter(
            ActionTask.completed_at != None,
            ActionTask.completed_at >= now - timedelta(days=7)
        ).count()
        created_7d = ActionTask.query.filter(
            ActionTask.created_at >= now - timedelta(days=7)
        ).count()

        oldest_open = db.session.query(db.func.min(ActionTask.created_at)).filter(
            ActionTask.status.in_(['pending', 'in_progress'])
        ).scalar()
        oldest_open_age_days = (now - oldest_open).days if oldest_open else 0

        completion_rate_7d = round((completed_7d / created_7d) * 100, 1) if created_7d else 0.0

        return jsonify({
            'total': total,
            'open': open_count,
            'done': done_count,
            'overdue': overdue_count,
            'due_today': due_today,
            'completed_7d': completed_7d,
            'created_7d': created_7d,
            'completion_rate_7d': completion_rate_7d,
            'oldest_open_age_days': oldest_open_age_days
        })
    except Exception as e:
        logger.error(f"Error loading task stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/tasks/today-plan', methods=['GET'])
def get_today_plan():
    """Return top prioritized tasks for today's execution."""
    try:
        max_items = min(request.args.get('max_items', 3, type=int), 10)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today_start = datetime(now.year, now.month, now.day)
        tomorrow = today_start + timedelta(days=1)

        candidates = ActionTask.query.filter(
            ActionTask.status.in_(['pending', 'in_progress'])
        ).all()

        ranked = []
        for task in candidates:
            score = 0
            if task.status == 'in_progress':
                score += 30
            score += task.priority * 10

            if task.due_date:
                if task.due_date < now:
                    score += 50
                elif today_start <= task.due_date < tomorrow:
                    score += 25
                elif task.due_date <= now + timedelta(days=3):
                    score += 10

            age_days = (now - task.created_at).days if task.created_at else 0
            score += min(age_days, 30)
            ranked.append((score, task))

        ranked.sort(key=lambda item: item[0], reverse=True)
        top_tasks = [task.to_dict() for _, task in ranked[:max_items]]

        return jsonify({
            'plan': top_tasks,
            'count': len(top_tasks),
            'generated_at': now.isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating today plan: {e}")
        return jsonify({'error': str(e)}), 500


def run_task_bootstrap_job(limit, due_days):
    """Background task generation job with parallel candidate extraction."""
    try:
        task_job_status['progress'] = 5
        task_job_status['message'] = 'Loading analyzed posts...'
        task_job_status['error'] = None
        task_job_status['results'] = None

        with app.app_context():
            posts = Post.query.join(Analysis).order_by(Post.timestamp.desc()).all()
            snapshots = []
            for post in posts:
                if not post.analysis:
                    continue
                snapshots.append({
                    'post_id': post.id,
                    'username': post.username or 'unknown',
                    'shortcode': post.shortcode or '',
                    'action_items': post.analysis.action_items or []
                })

            existing_keys = {
                key for (key,) in db.session.query(ActionTask.source_key)
                .filter(ActionTask.source_key != None)
                .all()
                if key
            }

        if not snapshots:
            task_job_status['progress'] = 100
            task_job_status['message'] = 'No analyzed posts available for task generation.'
            task_job_status['results'] = {'created_count': 0, 'total_candidates': 0}
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def extract_candidates(snapshot):
            items = []
            for raw_action in snapshot['action_items']:
                title = str(raw_action).strip()
                if not title:
                    continue
                source_key = task_source_key(snapshot['post_id'], title)
                source_url = f"https://instagram.com/p/{snapshot['shortcode']}" if snapshot['shortcode'] else ""
                notes = f"Source: @{snapshot['username']} {source_url}".strip()
                items.append({
                    'post_id': snapshot['post_id'],
                    'title': title,
                    'notes': notes,
                    'source_key': source_key
                })
            return items

        task_job_status['progress'] = 15
        task_job_status['message'] = 'Extracting task candidates in parallel...'

        candidates = []
        total = len(snapshots)
        max_workers = min(16, max(4, os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(extract_candidates, snap) for snap in snapshots]
            completed = 0
            for future in as_completed(futures):
                completed += 1
                task_job_status['progress'] = 15 + int((completed / total) * 50)
                task_job_status['message'] = f'Processing posts {completed}/{total}...'
                try:
                    candidates.extend(future.result())
                except Exception as e:
                    logger.error(f"Task candidate extraction failed: {e}")

        deduped = []
        seen_keys = set(existing_keys)
        for candidate in candidates:
            key = candidate['source_key']
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(candidate)
            if len(deduped) >= limit:
                break

        task_job_status['progress'] = 75
        task_job_status['message'] = 'Saving tasks...'
        due_date = (datetime.now(timezone.utc) + timedelta(days=due_days)).replace(tzinfo=None)

        created_tasks = []
        with app.app_context():
            for idx, candidate in enumerate(deduped, start=1):
                task = ActionTask(
                    post_id=candidate['post_id'],
                    title=candidate['title'],
                    notes=candidate['notes'],
                    status='pending',
                    priority=2,
                    due_date=due_date,
                    source='ai',
                    source_key=candidate['source_key']
                )
                db.session.add(task)
                created_tasks.append(task)
                if idx % 100 == 0:
                    db.session.flush()

            db.session.commit()

            task_job_status['progress'] = 100
            task_job_status['message'] = f"Task generation complete! Created {len(created_tasks)} tasks."
            task_job_status['results'] = {
                'created_count': len(created_tasks),
                'total_candidates': len(candidates),
                'deduped_candidates': len(deduped),
                'tasks': [task.to_dict() for task in created_tasks[:20]]
            }
    except Exception as e:
        logger.error(f"Error bootstrapping tasks: {e}")
        task_job_status['error'] = str(e)
    finally:
        task_job_status['running'] = False


@app.route('/api/tasks/status', methods=['GET'])
def get_task_job_status():
    """Get background task generation status."""
    return jsonify(task_job_status)


@app.route('/api/tasks/bootstrap', methods=['POST'])
def bootstrap_tasks_from_analysis():
    """Start background task generation from analyzed action_items."""
    if task_job_status['running']:
        return jsonify({'error': 'Task generation already running'}), 400

    try:
        data = request.json or {}
        limit = max(1, min(int(data.get('limit', 50)), 1000))
        due_days = max(1, min(int(data.get('due_days', 7)), 90))

        task_job_status['running'] = True
        task_job_status['progress'] = 0
        task_job_status['message'] = 'Initializing task generation...'
        task_job_status['error'] = None
        task_job_status['results'] = None

        thread = threading.Thread(target=run_task_bootstrap_job, args=(limit, due_days))
        thread.daemon = True
        thread.start()

        return jsonify({'status': 'started', 'message': 'Task generation started in background'})
    except Exception as e:
        task_job_status['running'] = False
        task_job_status['error'] = str(e)
        logger.error(f"Error starting task bootstrap: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/normalize-analysis', methods=['POST'])
def normalize_analysis_rows():
    """Normalize legacy double-encoded analysis list fields."""
    try:
        updated = 0
        rows = Analysis.query.all()
        for row in rows:
            before_topics = row._topics
            before_learning = row._learning_points
            before_actions = row._action_items

            # Re-assign through model properties to normalize storage
            row.topics = row.topics
            row.learning_points = row.learning_points
            row.action_items = row.action_items

            if row.credibility_score is None and row.raw_analysis:
                try:
                    raw = json.loads(row.raw_analysis)
                    cred = raw.get('credibility_score')
                    if cred is not None:
                        row.credibility_score = max(0, min(100, int(cred)))
                except Exception:
                    pass

            if (
                before_topics != row._topics
                or before_learning != row._learning_points
                or before_actions != row._action_items
            ):
                updated += 1

        db.session.commit()
        return jsonify({'updated_rows': updated, 'total_rows': len(rows)})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error normalizing analysis rows: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile', methods=['GET'])
def get_psychometric_profile():
    """Generate and return psychometric profile"""
    try:
        # Check if we have analyzed posts
        analyzed_posts = Post.query.join(Analysis).all()
        if not analyzed_posts:
            return jsonify({'error': 'No analyzed posts found'}), 404

        # Prepare simple list of dicts for the analyzer
        posts_data = []
        for p in analyzed_posts:
            # Reconstruct basic dict needed for profile generation
            data = p.analysis.to_dict()
            data['id'] = p.id
            posts_data.append(data)

        # Initialize Analyzer just for this method (using env vars)
        # Note: In production, we'd want a cleaner dependency injection or singleton
        config = {
            'sessionid': os.getenv('INSTAGRAM_SESSIONID'),
            'raw_cookie': os.getenv('RAW_COOKIE'),
            'user_agent': os.getenv('USER_AGENT', 'Mozilla/5.0'),
            'azure_endpoint': get_azure_endpoint(),
            'azure_key': get_azure_key(),
            'model': os.getenv('MODEL', 'DeepSeek-V3.2'),
            'max_posts': 10 # Irrelevant here
        }
        
        # Quick validation
        if not config['azure_endpoint'] or not config['azure_key']:
             return jsonify({'error': 'Azure credentials not set in environment'}), 500

        analyzer = InstagramAnalyzer(config)
        profile = analyzer.generate_psychometric_profile(posts_data)
        
        if not profile:
            return jsonify({'error': 'Failed to generate profile'}), 500
            
        return jsonify(profile)

    except Exception as e:
        logger.error(f"Error generating profile: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rebuild-rag', methods=['POST'])
def rebuild_rag():
    """Manually rebuild RAG index from all analyzed posts"""
    try:
        posts = Post.query.join(Analysis).all()
        if not posts:
            return jsonify({'error': 'No analyzed posts found'}), 404
        
        logger.info(f"Manually rebuilding RAG index for {len(posts)} posts...")
        rag_service.index_posts(posts)
        
        return jsonify({
            'success': True,
            'indexed_count': rag_service.index.ntotal if rag_service.index else 0,
            'message': f'Successfully indexed {rag_service.index.ntotal if rag_service.index else 0} posts'
        })
    except Exception as e:
        logger.error(f"Error rebuilding RAG: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_content():
    """Chat with saved content using RAG"""
    try:
        data = request.json or {}
        query = data.get('query')
        conversation_id = data.get('conversation_id')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Handle conversation
        if conversation_id:
            conversation = Conversation.query.get(conversation_id)
            if not conversation:
                # If ID provided but not found, create new
                conversation = Conversation(id=conversation_id)
                db.session.add(conversation)
                db.session.commit()
        else:
            conversation_id = str(uuid.uuid4())
            conversation = Conversation(id=conversation_id)
            db.session.add(conversation)
            db.session.commit()

        # Save user message
        user_msg = Message(conversation_id=conversation_id, role='user', content=query)
        db.session.add(user_msg)
        db.session.commit()

        # Get history
        history = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.created_at).all()

        # Use Vector Search to find relevant posts
        relevant_posts = rag_service.search(query, k=5)
        
        # Fallback if index is empty (e.g. first run)
        if not relevant_posts:
            relevant_posts = Post.query.limit(10).all()
        
        # Create context
        context = []
        for post in relevant_posts:
            analysis = post.analysis
            context.append(f"""
Post by {post.username} ({post.timestamp}):
Caption: {post.caption[:300]}...
Category: {analysis.category if analysis else 'Unknown'}
Topics: {', '.join(analysis.topics) if analysis else 'None'}
Action Items: {', '.join(analysis.action_items) if analysis and analysis.action_items else 'None'}
Transcript: {analysis.video_transcript[:1000] if analysis and analysis.video_transcript else 'N/A'}
OCR Text: {analysis.ocr_text[:500] if analysis and analysis.ocr_text else 'N/A'}
Visuals: {analysis.visual_description if analysis and analysis.visual_description else 'N/A'}
""")
        
        context_str = "\n".join(context)
        
        # Call Azure OpenAI
        azure_endpoint = get_azure_endpoint()
        azure_key = get_azure_key()
        if not azure_endpoint or not azure_key:
            return jsonify({'error': 'Azure credentials are not configured'}), 500

        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version="2024-08-01-preview"
        )
        
        # Build messages with history
        messages = [
            {"role": "system", "content": f"""You are Oracle, an AI assistant helping the user explore their saved Instagram content.

FORMATTING RULES:
- Use numbered lists (1., 2., 3.) for multiple items
- Use **bold** for usernames and categories
- Use bullet points (-) for sub-items like action items
- Keep responses concise and well-structured

Answer based on the provided context. If the context doesn't have the answer, say so politely.

Context:
{context_str}"""}
        ]
        
        # Add history (excluding the current user message which we just added to DB but want to send as part of flow)
        # Actually, we should just send all messages including the latest one
        # But we need to be careful about the system message being first.
        
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
            
        response = client.chat.completions.create(
            model=str(os.getenv('MODEL', 'DeepSeek-V3.2')),
            messages=messages,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        # Save assistant message
        ai_msg = Message(conversation_id=conversation_id, role='assistant', content=answer)
        db.session.add(ai_msg)
        db.session.commit()
        
        return jsonify({
            'answer': answer,
            'conversation_id': conversation_id
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['GET'])
def export_analysis():
    """Export analysis to Markdown"""
    try:
        posts = Post.query.all()
        
        # Calculate stats
        total_posts = len(posts)
        categories = {}
        for post in posts:
            if post.analysis:
                cat = post.analysis.category
                categories[cat] = categories.get(cat, 0) + 1
        
        # Generate Markdown
        md = f"# Instagram Saved Content Analysis\n\n"
        md += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        md += "## 📊 Overview\n"
        md += f"- **Total Posts**: {total_posts}\n"
        md += f"- **Top Category**: {max(categories, key=lambda k: categories[k]) if categories else 'N/A'}\n\n"
        
        md += "## 🧠 Psychoanalysis Profile\n"
        # Fetch profile from latest analysis result file if available, or skip
        results_file = OUTPUT_DIR / 'analyzed_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                profile = data.get('summary', {}).get('user_profile', {})
                if profile:
                    md += "### Personality Traits\n"
                    for trait in profile.get('personality_traits', []):
                        md += f"- {trait}\n"
                    md += "\n### Interests\n"
                    for interest in profile.get('interests', []):
                        md += f"- {interest}\n"
                    md += f"\n**Summary**: {profile.get('psychological_summary', '')}\n\n"

        md += "## 📝 Saved Posts\n"
        for post in posts:
            analysis = post.analysis
            if not analysis: continue
            
            md += f"### {analysis.category}: {post.username}\n"
            md += f"**Date**: {post.timestamp}\n\n"
            md += f"![Thumbnail]({post.thumbnail_url})\n\n"
            md += f"**Caption**: {post.caption}\n\n"
            md += f"**Analysis**:\n"
            md += f"- Sentiment: {analysis.sentiment_label} ({analysis.sentiment_score})\n"
            md += f"- Topics: {', '.join(analysis.topics)}\n"
            
            if analysis.action_items:
                md += "\n**Action Items**:\n"
                for item in analysis.action_items:
                    md += f"- [ ] {item}\n"
            
            md += "\n---\n\n"
            
        # Save to file
        export_file = OUTPUT_DIR / f"instagram_export_{datetime.now().strftime('%Y%m%d')}.md"
        with open(export_file, 'w') as f:
            f.write(md)
            
        return send_file(
            export_file,
            mimetype='text/markdown',
            as_attachment=True,
            download_name=export_file.name
        )

    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Instagram AI Analyzer API Server")
    print("📍 Open UI at http://localhost:5001")
    print("🔗 API docs at http://localhost:5001/api/health")
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, port=5001, host='127.0.0.1')
