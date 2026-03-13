# Instagram AI Analyzer - Complete UI Setup Guide

## 🎯 What You Get Now

**Full-stack solution with beautiful UI:**
- ✅ Backend API (Flask) running on `localhost:5001`
- ✅ Beautiful Web UI on `localhost:5001`

- ✅ Real Instagram data pulling
- ✅ Real Azure AI analysis
- ✅ Live progress tracking
- ✅ Charts & analytics dashboard
- ✅ Export to JSON/CSV

---

## 📦 Installation (2 Minutes)

### Step 1: Install Dependencies

```bash
pip install flask flask-cors requests openai pandas browser-cookie3
```

### Step 2: Organize Files

Create this folder structure:
```
instagram-analyzer/
├── flask-backend.py         # Backend API
├── instagram_analyzer.py    # Analysis script
├── requirements.txt         # Dependencies list
└── templates/
    └── index.html           # UI (paste the HTML file content)

```

### Step 3: Place UI File

Save the `index-ui.html` content as `templates/index.html`

```bash
mkdir templates
# Paste index-ui.html content into templates/index.html
```

---

## 🚀 Running the Application

### Terminal 1 - Start Backend API

```bash
python flask-backend.py
```

You'll see:
```
🚀 Instagram AI Analyzer API Server
📍 Open UI at http://localhost:5001
🔗 API docs at http://localhost:5001/api/health
```

### Terminal 2 - Open in Browser

Go to: **http://localhost:5001**


You'll see the beautiful UI!

---

## 🎨 How to Use the UI

### Tab 1: 🔑 Credentials
1. **Get Instagram Session ID:**
   - Open Instagram, login
   - Press F12 → Application → Cookies → instagram.com
   - Find `sessionid`, copy VALUE
   - Paste in the textarea

2. **Get Azure Credentials:**
   - Go to Azure Portal
   - Find your OpenAI resource
   - Copy Endpoint & API Key
   - Paste in the form

3. **Test Credentials:**
   - Click "🧪 Test Credentials"
   - Wait for ✓ marks next to both services

### Tab 2: 🚀 Analyze
1. Click "🚀 Start Analysis"
2. Watch the progress bar
3. Analysis runs in the background
4. When complete, results auto-load

### Tab 3: 📊 Results
- See all the analysis data
- View charts and statistics
- See individual post analysis
- Download as JSON or CSV

### Tab 4: 💡 Insights
- Top topics extracted
- Learning points compiled
- Category breakdown
- Growth recommendations

---

## 📊 Example Workflow

```
1. Paste Instagram sessionid + Azure keys
2. Click "Test Credentials" ✓
3. Set posts to analyze (e.g., 20)
4. Click "Start Analysis"
5. Watch progress (10% → 100%)
6. View results in Charts tab
7. Download data as JSON/CSV
```

---

## 🔍 What Happens Behind the Scenes

1. **Backend receives credentials** via API
2. **instagram_analyzer.py script runs** in background
3. **Fetches saved posts** from Instagram
4. **Sends each to Azure OpenAI** for analysis
5. **Saves results** to `output/analyzed_results.json`
6. **UI polls status** and displays live progress
7. **Charts render** from final data
8. **User can export** results


---

## 📂 File Structure After First Run

```
instagram-analyzer/
├── flask-backend.py
├── instagram_analyzer.py
├── requirements.txt
├── templates/
│   └── index.html
└── output/
    ├── raw_posts.json          # Created by analyzer
    └── analyzed_results.json   # Created by analyzer

```

---

## 💻 API Endpoints (Advanced)

If you want to use the API directly:

```bash
# Health check
curl http://localhost:5001/api/health

# Start analysis
curl -X POST http://localhost:5001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "sessionid": "your_sessionid",
    "azure_endpoint": "https://...",
    "azure_key": "your_key",
    "model": "gpt-4o-mini",
    "max_posts": 20
  }'

# Check status
curl http://localhost:5001/api/status

# Get results
curl http://localhost:5001/api/results

# Download JSON
curl http://localhost:5001/api/results/download > results.json

# Download CSV
curl http://localhost:5001/api/results/csv > results.csv

```

---

## 🛠️ Troubleshooting

### "Connection refused on localhost:5001"
```bash
# Make sure flask is running in another terminal
python flask-backend.py

# If port is busy:
# Edit flask-backend.py, change port=5001 to port=8000

```

### "Cannot fetch from Instagram"
- Check sessionid is correct (not expired)
- Try logging out and back into Instagram
- Get fresh sessionid from DevTools

### "Azure API error"
- Verify endpoint URL is correct
- Check API key is not expired
- Ensure model is deployed in Azure
- Check you have sufficient quota

### "Analysis seems stuck"
- Check Terminal 1 for errors
- Analysis takes time (5-10 min for 20 posts)
- Don't close Terminal 1 (backend)
- Refresh browser if UI freezes

---

## ⚡ Performance Tips

1. **Start small:** Test with 5 posts first
2. **Don't close Terminal 1:** That's your backend!
3. **Use gpt-4o-mini:** Cheapest & fastest for this
4. **Batch size:** 20-50 posts at a time works best
5. **Run overnight:** Analyze 100+ posts when you sleep

---

## 📱 Mobile Access (Advanced)

To access from phone/other computers:

1. Find your computer's IP:
   ```bash
   # Mac/Linux
   ifconfig | grep "inet "
   
   # Windows
   ipconfig
   ```

2. Edit `flask-backend.py`, change last line to:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

3. Access from phone: `http://[YOUR_IP]:5000`

---

## 🔒 Security Notes

- Keep sessionid & API key **private**
- Don't commit `.env` or `flask-backend.py` to GitHub
- The UI runs **locally only** (not on the internet)
- All data stays on your computer
- Delete `output/` folder to clear all data

---

## 📊 Export Formats

### JSON Format
```json
{
  "summary": {
    "total_posts_analyzed": 20,
    "categories_breakdown": {...},
    "average_sentiment_score": 0.65,
    "top_topics": [...]
  },
  "analyzed_posts": [
    {
      "post": {...},
      "analysis": {...}
    }
  ]
}
```

### CSV Format
Columns: username, caption, category, sentiment_score, sentiment_label, credibility_score, topics, url, likes, comments, timestamp

---

## 🎯 Next Steps

1. **Install & setup** (5 min)
2. **Run backend** in Terminal 1
3. **Open UI** in browser
4. **Add credentials**
5. **Start first analysis** (test with 5 posts)
6. **Check results** in dashboard
7. **Export data** for further analysis

---

## 💡 Use Cases After Analysis

- **Track learning:** See how your interests evolve
- **Categorize content:** Understand what you save
- **Find patterns:** Identify content gaps
- **Build reports:** Export insights as PDF
- **Share learnings:** Show what you've learned
- **Optimize time:** Focus on high-value content
- **Create content:** Ideas based on what you save

---

## 🚀 You're All Set!

**Run this now:**

Terminal 1:
```bash
python flask-backend.py
```

Then open in browser:
```
http://localhost:5001
```

Enjoy! 📊✨

---

## ❓ Quick FAQ

**Q: Do I need the UI?**
A: No, you can use `python instagram_analyzer.py` directly. UI is optional but nicer.

**Q: Can I close my laptop while analyzing?**
A: No, the analysis runs locally. Keep it on.

**Q: How long does analysis take?**
A: ~5-10 seconds per post. So 20 posts = 2-3 minutes.

**Q: Can I use my phone as the backend?**
A: Not easily. Backend needs Python + Azure credentials.

**Q: Is my data stored anywhere?**
A: Only on your computer in `output/` folder.

**Q: Can I pause analysis?**
A: Close the browser or Terminal 1. It stops gracefully.

**Q: What if power goes out?**
A: Analysis stops. Run again to restart from beginning.
