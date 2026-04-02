import {
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useState,
} from "react";

const API_BASE = "/api";
const LOCAL_SETUP_KEY = "saved-intelligence-setup-v2";
const DEFAULT_SETUP = {
  sessionid: "",
  raw_cookie: "",
  user_agent: "",
  browser: "chrome",
  azure_endpoint: "",
  azure_key: "",
  model: "DeepSeek-V3.2",
  verification_provider: "tavily_gemini",
  verification_model: "gemini-3.1-flash-lite-preview",
  verification_api_key: "",
  tavily_api_key: "",
  verification_max_claims: 5,
  verification_max_sources: 5,
};

function getStoredSetup() {
  try {
    return {
      ...DEFAULT_SETUP,
      ...JSON.parse(window.localStorage.getItem(LOCAL_SETUP_KEY) || "{}"),
    };
  } catch {
    return DEFAULT_SETUP;
  }
}

async function fetchJson(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.error) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function toMultiline(value) {
  return Array.isArray(value) ? value.join("\n") : "";
}

function fromMultiline(value) {
  return value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatDate(value) {
  if (!value) return "No due date";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "No due date";
  return date.toLocaleDateString();
}

function classNames(...values) {
  return values.filter(Boolean).join(" ");
}

function App() {
  const [view, setView] = useState("dashboard");
  const [setup, setSetup] = useState(getStoredSetup);
  const [config, setConfig] = useState(null);
  const [stats, setStats] = useState(null);
  const [taskStats, setTaskStats] = useState(null);
  const [profile, setProfile] = useState(null);
  const [profileForm, setProfileForm] = useState({
    manual_goals: "",
    priorities: "",
    constraints: "",
    focus_areas: "",
  });
  const [posts, setPosts] = useState([]);
  const [postsPagination, setPostsPagination] = useState({
    current_page: 1,
    pages: 1,
    total: 0,
  });
  const [postFilters, setPostFilters] = useState({
    sort: "newest",
    category: "all",
    collection: "all",
    sentiment: "all",
    verification: "all",
  });
  const [tasks, setTasks] = useState([]);
  const [taskPagination, setTaskPagination] = useState({
    current_page: 1,
    pages: 1,
  });
  const [taskFilters, setTaskFilters] = useState({
    status: "all",
    due: "all",
  });
  const [todayPlan, setTodayPlan] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState([
    { role: "assistant", content: "I am Oracle. Ask about your saved content." },
  ]);
  const [conversationId, setConversationId] = useState(null);
  const [verificationReport, setVerificationReport] = useState(null);
  const [verificationOpen, setVerificationOpen] = useState(false);
  const [toast, setToast] = useState(null);
  const [busyLabel, setBusyLabel] = useState("");
  const [statusLabel, setStatusLabel] = useState("Checking...");
  const [manualTask, setManualTask] = useState({ title: "", due_date: "" });

  const deferredPosts = useDeferredValue(posts);

  const categoryOptions = useMemo(
    () => ["all", ...Object.keys(stats?.categories || {})],
    [stats]
  );

  const collectionOptions = useMemo(() => {
    const items = new Set();
    posts.forEach((post) => (post.collections || []).forEach((item) => items.add(item)));
    return ["all", ...Array.from(items).sort()];
  }, [posts]);

  const groupedTasks = useMemo(() => {
    const buckets = { today: [], this_week: [], later: [] };
    tasks.forEach((task) => {
      const key = task.horizon || "later";
      if (!buckets[key]) {
        buckets.later.push(task);
      } else {
        buckets[key].push(task);
      }
    });
    return buckets;
  }, [tasks]);

  useEffect(() => {
    window.localStorage.setItem(LOCAL_SETUP_KEY, JSON.stringify(setup));
  }, [setup]);

  useEffect(() => {
    loadBootData();
    const interval = window.setInterval(loadStatusSummary, 15000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (view === "gallery") {
      loadPosts(1);
    }
    if (view === "execution") {
      loadTasks(1);
      loadTodayPlan();
      loadTaskStats();
    }
  }, [view, postFilters, taskFilters]);

  function showToast(message, type = "info") {
    setToast({ message, type });
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => setToast(null), 3000);
  }

  async function loadBootData() {
    await Promise.all([
      loadConfig(),
      loadStats(),
      loadTaskStats(),
      loadTodayPlan(),
      loadTasks(1),
      loadProfileSettings(),
      loadStatusSummary(),
    ]);
  }

  async function loadConfig() {
    const data = await fetchJson("/config");
    setConfig(data);
    setSetup((current) => ({
      ...current,
      azure_endpoint: current.azure_endpoint || data.azure_endpoint || "",
      model: current.model || data.model || DEFAULT_SETUP.model,
      verification_provider:
        current.verification_provider || data.verification_provider || DEFAULT_SETUP.verification_provider,
      verification_model:
        current.verification_model || data.verification_model || DEFAULT_SETUP.verification_model,
      verification_max_claims:
        current.verification_max_claims || data.verification_max_claims || DEFAULT_SETUP.verification_max_claims,
      verification_max_sources:
        current.verification_max_sources || data.verification_max_sources || DEFAULT_SETUP.verification_max_sources,
    }));
  }

  async function loadStats() {
    const data = await fetchJson("/stats");
    setStats(data);
  }

  async function loadTaskStats() {
    const data = await fetchJson("/tasks/stats");
    setTaskStats(data);
  }

  async function loadTodayPlan() {
    const data = await fetchJson("/tasks/today-plan?max_items=3");
    setTodayPlan(data.plan || []);
  }

  async function loadTasks(page = 1) {
    const params = new URLSearchParams({
      page: String(page),
      per_page: "20",
      status: taskFilters.status,
      due: taskFilters.due,
    });
    const data = await fetchJson(`/tasks?${params.toString()}`);
    setTasks(data.tasks || []);
    setTaskPagination({
      current_page: data.current_page || 1,
      pages: data.pages || 1,
    });
  }

  async function loadPosts(page = 1) {
    const params = new URLSearchParams({
      page: String(page),
      per_page: "50",
      sort: postFilters.sort,
      category: postFilters.category,
      collection: postFilters.collection,
      sentiment: postFilters.sentiment,
      verification: postFilters.verification,
    });
    const data = await fetchJson(`/posts?${params.toString()}`);
    setPosts(data.posts || []);
    setPostsPagination({
      current_page: data.current_page || 1,
      pages: data.pages || 1,
      total: data.total || 0,
    });
  }

  async function loadProfileSettings() {
    const data = await fetchJson("/profile-settings");
    setProfile(data);
    setProfileForm({
      manual_goals: toMultiline(data.manual_goals),
      priorities: toMultiline(data.priorities),
      constraints: toMultiline(data.constraints),
      focus_areas: toMultiline(data.focus_areas),
    });
  }

  async function loadStatusSummary() {
    try {
      const status = await fetchJson("/status");
      setStatusLabel(status.running ? `Working... ${status.progress || 0}%` : "Ready");
    } catch {
      setStatusLabel("Unavailable");
    }
  }

  async function runAction(label, action) {
    try {
      setBusyLabel(label);
      await action();
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      setBusyLabel("");
    }
  }

  async function saveProfileSettings() {
    const payload = {
      manual_goals: fromMultiline(profileForm.manual_goals),
      priorities: fromMultiline(profileForm.priorities),
      constraints: fromMultiline(profileForm.constraints),
      focus_areas: fromMultiline(profileForm.focus_areas),
    };
    await fetchJson("/profile-settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadProfileSettings();
    showToast("Goals saved", "success");
  }

  async function refreshProfile() {
    await fetchJson("/profile/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        azure_endpoint: setup.azure_endpoint,
        azure_key: setup.azure_key,
        model: setup.model,
      }),
    });
    await loadProfileSettings();
    showToast("Psychometric snapshot refreshed", "success");
  }

  async function syncLibrary() {
    await fetchJson("/fetch-posts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(setup),
    });
    await Promise.all([loadStats(), loadPosts(1)]);
    showToast("Library synced", "success");
  }

  async function analyzeBatch() {
    await fetchJson("/analyze-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        batch_size: 1000,
        azure_endpoint: setup.azure_endpoint,
        azure_key: setup.azure_key,
        model: setup.model,
      }),
    });
    await Promise.all([loadStats(), loadPosts(1), loadProfileSettings()]);
    showToast("Analysis complete", "success");
  }

  async function refreshCategories() {
    await fetchJson("/admin/refresh-categories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        azure_endpoint: setup.azure_endpoint,
        azure_key: setup.azure_key,
        model: setup.model,
      }),
    });
    await Promise.all([loadStats(), loadPosts(postsPagination.current_page || 1)]);
    showToast("Categories refreshed", "success");
  }

  async function rebuildRag() {
    await fetchJson("/rebuild-rag", { method: "POST" });
    await loadStats();
    showToast("RAG index rebuilt", "success");
  }

  async function verifyPost(postId) {
    await fetchJson(`/posts/${postId}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: setup.verification_provider,
        verification_model: setup.verification_model,
        verification_api_key: setup.verification_api_key,
        tavily_api_key: setup.tavily_api_key,
        max_claims: Number(setup.verification_max_claims || 5),
        max_sources: Number(setup.verification_max_sources || 5),
        azure_endpoint: setup.azure_endpoint,
        azure_key: setup.azure_key,
        model: setup.model,
      }),
    });
    await Promise.all([loadStats(), loadPosts(postsPagination.current_page || 1)]);
    await openVerification(postId);
    showToast("Verification complete", "success");
  }

  async function openVerification(postId) {
    const data = await fetchJson(`/posts/${postId}/verification`);
    setVerificationReport(data);
    setVerificationOpen(true);
  }

  async function bootstrapTasks() {
    await fetchJson("/tasks/bootstrap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit: 120, due_days: 7 }),
    });
    await Promise.all([loadTaskStats(), loadTodayPlan(), loadTasks(1)]);
    showToast("Tasks generated", "success");
  }

  async function updateTaskStatus(id, status) {
    await fetchJson(`/tasks/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
    await Promise.all([
      loadTaskStats(),
      loadTodayPlan(),
      loadTasks(taskPagination.current_page || 1),
    ]);
  }

  async function addManualTask(event) {
    event.preventDefault();
    if (!manualTask.title.trim()) return;
    await fetchJson("/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: manualTask.title.trim(),
        due_date: manualTask.due_date
          ? new Date(`${manualTask.due_date}T18:00:00`).toISOString()
          : null,
        source: "manual",
        priority: 2,
        horizon: "this_week",
        effort: "medium",
        impact: "medium",
      }),
    });
    setManualTask({ title: "", due_date: "" });
    await Promise.all([loadTaskStats(), loadTodayPlan(), loadTasks(1)]);
    showToast("Task added", "success");
  }

  async function sendChat(event) {
    event.preventDefault();
    if (!chatInput.trim()) return;
    const userMessage = { role: "user", content: chatInput.trim() };
    setChatMessages((items) => [...items, userMessage]);
    const question = chatInput.trim();
    setChatInput("");
    try {
      const data = await fetchJson("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question, conversation_id: conversationId }),
      });
      setConversationId(data.conversation_id || null);
      setChatMessages((items) => [
        ...items,
        { role: "assistant", content: data.answer || "No answer returned." },
      ]);
    } catch (error) {
      setChatMessages((items) => [
        ...items,
        { role: "assistant", content: `Error: ${error.message}` },
      ]);
    }
  }

  const hero = {
    totalPosts: stats?.total_posts || 0,
    analyzed: stats?.analysis_count || 0,
    verified: stats?.verification_count || 0,
    openTasks: taskStats?.open || 0,
    analysisCoverage:
      stats?.total_posts > 0
        ? Math.round((stats.analysis_count / stats.total_posts) * 100)
        : 0,
    verificationCoverage:
      stats?.analysis_count > 0
        ? Math.round((stats.verification_count / stats.analysis_count) * 100)
        : 0,
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">SI</span>
          <div>
            <p className="brand-title">Saved Intelligence</p>
            <p className="brand-subtitle">Vite + React frontend on Flask APIs</p>
          </div>
        </div>
        <nav className="nav-tabs">
          {["dashboard", "gallery", "execution", "oracle", "settings"].map((item) => (
            <button
              key={item}
              type="button"
              className={classNames("nav-tab", view === item && "active")}
              onClick={() => startTransition(() => setView(item))}
            >
              {item === "oracle" ? "Oracle" : item[0].toUpperCase() + item.slice(1)}
            </button>
          ))}
        </nav>
        <div className="status-chip">{busyLabel || statusLabel}</div>
      </header>

      <main className="page-shell">
        {view === "dashboard" && (
          <section className="stack-xl">
            <section className="hero">
              <div className="hero-copy panel">
                <div className="eyebrow-row">
                  <span className="eyebrow">
                    {stats?.index_integrity_status === "warning" ? "Needs attention" : "System ready"}
                  </span>
                  <span className="eyebrow">React frontend migration</span>
                  <span className="eyebrow">Flask backend retained</span>
                </div>
                <h1>Turn saved reels into a living operating system.</h1>
                <p>
                  Keep the Python backend, modernize the UI, and make the library,
                  search, verification, and execution flows easier to operate.
                </p>
                <div className="hero-actions">
                  <button onClick={() => runAction("Syncing library...", syncLibrary)}>
                    Sync Library
                  </button>
                  <button className="ghost" onClick={() => runAction("Analyzing posts...", analyzeBatch)}>
                    Analyze Batch
                  </button>
                  <button className="ghost" onClick={() => setView("settings")}>
                    Review Setup
                  </button>
                </div>
              </div>
              <aside className="panel">
                <p className="section-label">Current signal</p>
                <div className="metric-grid">
                  <MetricCard label="Saved Posts" value={hero.totalPosts} detail="Rows in SQL library" />
                  <MetricCard label="Analyzed" value={hero.analyzed} detail={`${hero.analysisCoverage}% of library`} />
                  <MetricCard label="Verified" value={hero.verified} detail={`${hero.verificationCoverage}% of analyses`} />
                  <MetricCard label="Open Tasks" value={hero.openTasks} detail="Current execution load" />
                </div>
              </aside>
            </section>

            <section className="grid-3">
              <div className="panel">
                <p className="section-label">Integrity</p>
                <h2>Truthful counts</h2>
                <ul className="metric-list">
                  <li><span>DB analyses</span><strong>{stats?.analysis_count || 0}</strong></li>
                  <li><span>Completed verifications</span><strong>{stats?.verification_count || 0}</strong></li>
                  <li><span>Index vectors</span><strong>{stats?.index_vector_count || 0}</strong></li>
                  <li><span>Index metadata</span><strong>{stats?.index_metadata_count || 0}</strong></li>
                </ul>
                {stats?.index_integrity_warning && (
                  <div className="warning-box">{stats.index_integrity_warning}</div>
                )}
                <button className="ghost" onClick={() => runAction("Rebuilding index...", rebuildRag)}>
                  Rebuild Search Index
                </button>
              </div>
              <div className="panel">
                <p className="section-label">Category spread</p>
                <h2>Primary categories</h2>
                <div className="tag-flow">
                  {Object.entries(stats?.categories || {}).map(([category, count]) => (
                    <span key={category} className="tag-chip">
                      {category} · {count}
                    </span>
                  ))}
                </div>
              </div>
              <div className="panel">
                <p className="section-label">Profile</p>
                <h2>{profile?.psychometric_profile?.archetype || "No saved snapshot yet"}</h2>
                <p className="muted-copy">
                  {profile?.psychometric_profile?.one_liner ||
                    "Refresh the psychometric profile after analysis to persist your content snapshot."}
                </p>
                <div className="tag-flow">
                  {(profile?.psychometric_profile?.traits || []).map((trait) => (
                    <span key={trait} className="tag-chip">
                      {trait}
                    </span>
                  ))}
                </div>
              </div>
            </section>
          </section>
        )}

        {view === "gallery" && (
          <section className="stack-xl">
            <section className="panel toolbar">
              <div className="toolbar-group">
                <SelectField
                  label="Sort"
                  value={postFilters.sort}
                  onChange={(value) => setPostFilters((items) => ({ ...items, sort: value }))}
                  options={[
                    { value: "newest", label: "Newest" },
                    { value: "oldest", label: "Oldest" },
                  ]}
                />
                <SelectField
                  label="Category"
                  value={postFilters.category}
                  onChange={(value) => setPostFilters((items) => ({ ...items, category: value }))}
                  options={categoryOptions.map((value) => ({ value, label: value === "all" ? "All Categories" : value }))}
                />
                <SelectField
                  label="Collection"
                  value={postFilters.collection}
                  onChange={(value) => setPostFilters((items) => ({ ...items, collection: value }))}
                  options={collectionOptions.map((value) => ({ value, label: value === "all" ? "All Collections" : value }))}
                />
                <SelectField
                  label="Sentiment"
                  value={postFilters.sentiment}
                  onChange={(value) => setPostFilters((items) => ({ ...items, sentiment: value }))}
                  options={[
                    { value: "all", label: "All Sentiments" },
                    { value: "Positive", label: "Positive" },
                    { value: "Neutral", label: "Neutral" },
                    { value: "Negative", label: "Negative" },
                  ]}
                />
                <SelectField
                  label="Verification"
                  value={postFilters.verification}
                  onChange={(value) => setPostFilters((items) => ({ ...items, verification: value }))}
                  options={[
                    { value: "all", label: "All Verification" },
                    { value: "verified", label: "Verified" },
                    { value: "unverified", label: "Not Verified" },
                    { value: "failed", label: "Verification Failed" },
                  ]}
                />
              </div>
              <div className="toolbar-actions">
                <button className="ghost" onClick={() => runAction("Refreshing categories...", refreshCategories)}>
                  Refresh Categories
                </button>
                <button onClick={() => runAction("Syncing library...", syncLibrary)}>Sync</button>
                <button onClick={() => runAction("Analyzing posts...", analyzeBatch)}>Analyze</button>
              </div>
            </section>
            {deferredPosts.length === 0 ? (
              <EmptyState
                title="No posts match this view."
                body="Loosen the filters or sync and analyze more saved content."
              />
            ) : (
              <section className="post-grid">
                {deferredPosts.map((post) => (
                  <article key={post.id} className="post-card">
                    <button
                      type="button"
                      className="post-link"
                      onClick={() =>
                        window.open(`https://www.instagram.com/p/${post.shortcode || ""}`, "_blank")
                      }
                    >
                      <div className="post-media">
                        {post.thumbnail_url ? (
                          <img
                            src={`${API_BASE}/media-proxy?shortcode=${encodeURIComponent(
                              post.shortcode || ""
                            )}&url=${encodeURIComponent(post.thumbnail_url)}`}
                            alt={post.caption || post.username || "Saved post"}
                          />
                        ) : (
                          <div className="media-fallback">No preview</div>
                        )}
                        <span className="category-pill">
                          {post.analysis?.category || "Uncategorized"}
                        </span>
                      </div>
                    </button>
                    <div className="post-body">
                      <div className="post-topline">
                        <span>@{post.username || "unknown"}</span>
                        <span>{post.analysis?.sentiment?.label || "Neutral"}</span>
                      </div>
                      <p className="post-summary">
                        {post.analysis?.video_summary ||
                          post.analysis?.visual_description ||
                          post.caption ||
                          "No summary available."}
                      </p>
                      <p className="post-caption">{post.caption || "No caption available."}</p>
                      <div className="tag-flow">
                        {(post.analysis?.topics || []).slice(0, 4).map((topic) => (
                          <span key={topic} className="tag-chip">
                            {topic}
                          </span>
                        ))}
                      </div>
                      <div className="post-actions">
                        <button
                          className="ghost small"
                          onClick={() => runAction("Running verification...", () => verifyPost(post.id))}
                        >
                          {post.verification ? "Re-verify" : "Verify"}
                        </button>
                        <button
                          className="ghost small"
                          onClick={() => openVerification(post.id).catch((error) => showToast(error.message, "error"))}
                        >
                          {post.verification?.status === "completed"
                            ? "View Verification"
                            : post.verification?.status === "failed"
                              ? "View Failure"
                              : "Details"}
                        </button>
                      </div>
                    </div>
                  </article>
                ))}
              </section>
            )}
          </section>
        )}

        {view === "execution" && (
          <section className="stack-xl">
            <section className="grid-4">
              <MetricCard label="Open" value={taskStats?.open || 0} detail="Pending and in progress" />
              <MetricCard label="Due Today" value={taskStats?.due_today || 0} detail="Immediate follow-through" />
              <MetricCard label="Overdue" value={taskStats?.overdue || 0} detail="Needs cleanup" />
              <MetricCard
                label="7d Completion"
                value={`${taskStats?.completion_rate_7d || 0}%`}
                detail="Recent execution rate"
              />
            </section>

            <section className="panel">
              <div className="section-head">
                <div>
                  <p className="section-label">Planner</p>
                  <h2>Today’s execution plan</h2>
                </div>
                <button onClick={() => runAction("Generating tasks...", bootstrapTasks)}>Generate Tasks</button>
              </div>
              {todayPlan.length === 0 ? (
                <EmptyState
                  title="No open tasks right now."
                  body="Generate goal-aware tasks from recent content or add one manually."
                />
              ) : (
                <div className="plan-grid">
                  {todayPlan.map((task) => (
                    <TaskCard key={task.id} task={task} onStatusChange={updateTaskStatus} />
                  ))}
                </div>
              )}
            </section>

            <section className="panel">
              <div className="section-head">
                <div>
                  <p className="section-label">Manual capture</p>
                  <h2>Add a task</h2>
                </div>
              </div>
              <form className="manual-task-form" onSubmit={addManualTask}>
                <input
                  value={manualTask.title}
                  onChange={(event) =>
                    setManualTask((current) => ({ ...current, title: event.target.value }))
                  }
                  placeholder="Define the next concrete action..."
                />
                <input
                  type="date"
                  value={manualTask.due_date}
                  onChange={(event) =>
                    setManualTask((current) => ({ ...current, due_date: event.target.value }))
                  }
                />
                <button type="submit">Add</button>
              </form>
            </section>

            <section className="panel">
              <div className="section-head">
                <div>
                  <p className="section-label">Inbox</p>
                  <h2>Task workflow</h2>
                </div>
                <div className="toolbar-group">
                  <SelectField
                    label="Status"
                    value={taskFilters.status}
                    onChange={(value) => setTaskFilters((items) => ({ ...items, status: value }))}
                    options={[
                      { value: "all", label: "All Status" },
                      { value: "pending", label: "Pending" },
                      { value: "in_progress", label: "In Progress" },
                      { value: "done", label: "Done" },
                    ]}
                  />
                  <SelectField
                    label="Due"
                    value={taskFilters.due}
                    onChange={(value) => setTaskFilters((items) => ({ ...items, due: value }))}
                    options={[
                      { value: "all", label: "Any Due Date" },
                      { value: "today", label: "Due Today" },
                      { value: "overdue", label: "Overdue" },
                      { value: "week", label: "Due This Week" },
                    ]}
                  />
                </div>
              </div>

              <div className="lane-grid">
                {[
                  ["today", "Today"],
                  ["this_week", "This Week"],
                  ["later", "Later"],
                ].map(([key, label]) => (
                  <div key={key} className="task-lane">
                    <div className="lane-head">
                      <h3>{label}</h3>
                      <span>{groupedTasks[key].length}</span>
                    </div>
                    <div className="stack-md">
                      {groupedTasks[key].length === 0 ? (
                        <p className="empty-copy">No tasks here.</p>
                      ) : (
                        groupedTasks[key].map((task) => (
                          <TaskCard key={task.id} task={task} onStatusChange={updateTaskStatus} />
                        ))
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </section>
        )}

        {view === "oracle" && (
          <section className="panel oracle-shell">
            <div className="chat-thread">
              {chatMessages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={classNames("chat-bubble", message.role === "user" && "user")}
                >
                  {message.content}
                </div>
              ))}
            </div>
            <form className="chat-form" onSubmit={sendChat}>
              <input
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                placeholder="Ask what you saved and why it matters."
              />
              <button type="submit">Send</button>
            </form>
          </section>
        )}

        {view === "settings" && (
          <section className="stack-xl">
            <section className="grid-2">
              <div className="panel">
                <div className="section-head">
                  <div>
                    <p className="section-label">Instagram</p>
                    <h2>Connection</h2>
                  </div>
                  <button className="ghost" onClick={() => window.open("/legacy", "_blank")}>
                    Open Legacy UI
                  </button>
                </div>
                <div className="stack-md">
                  <TextField label="Session ID" value={setup.sessionid} onChange={(value) => setSetup((items) => ({ ...items, sessionid: value }))} />
                  <TextField label="Raw Cookie" value={setup.raw_cookie} onChange={(value) => setSetup((items) => ({ ...items, raw_cookie: value }))} />
                  <TextField label="User Agent" value={setup.user_agent} onChange={(value) => setSetup((items) => ({ ...items, user_agent: value }))} />
                </div>
              </div>
              <div className="panel">
                <div className="section-head">
                  <div>
                    <p className="section-label">Models</p>
                    <h2>Analysis and verification</h2>
                  </div>
                </div>
                <div className="stack-md">
                  <TextField label="Azure Endpoint" value={setup.azure_endpoint} onChange={(value) => setSetup((items) => ({ ...items, azure_endpoint: value }))} />
                  <TextField label="Azure Key" type="password" value={setup.azure_key} onChange={(value) => setSetup((items) => ({ ...items, azure_key: value }))} />
                  <TextField label="Azure Model" value={setup.model} onChange={(value) => setSetup((items) => ({ ...items, model: value }))} />
                  <SelectField
                    label="Verification Provider"
                    value={setup.verification_provider}
                    onChange={(value) => setSetup((items) => ({ ...items, verification_provider: value }))}
                    options={(config?.verification_providers || ["tavily_gemini"]).map((value) => ({
                      value,
                      label: value,
                    }))}
                  />
                  <TextField label="Verification Model" value={setup.verification_model} onChange={(value) => setSetup((items) => ({ ...items, verification_model: value }))} />
                  <TextField label="LLM API Key" type="password" value={setup.verification_api_key} onChange={(value) => setSetup((items) => ({ ...items, verification_api_key: value }))} />
                  <TextField label="Tavily API Key" type="password" value={setup.tavily_api_key} onChange={(value) => setSetup((items) => ({ ...items, tavily_api_key: value }))} />
                </div>
              </div>
            </section>

            <section className="panel">
              <div className="section-head">
                <div>
                  <p className="section-label">Workspace profile</p>
                  <h2>Goals and priorities</h2>
                </div>
                <div className="toolbar-actions">
                  <button className="ghost" onClick={() => runAction("Saving goals...", saveProfileSettings)}>
                    Save Goals
                  </button>
                  <button onClick={() => runAction("Refreshing profile...", refreshProfile)}>
                    Refresh Snapshot
                  </button>
                </div>
              </div>
              <div className="grid-2">
                <TextAreaField label="Manual Goals" value={profileForm.manual_goals} onChange={(value) => setProfileForm((items) => ({ ...items, manual_goals: value }))} />
                <TextAreaField label="Priorities" value={profileForm.priorities} onChange={(value) => setProfileForm((items) => ({ ...items, priorities: value }))} />
                <TextAreaField label="Constraints" value={profileForm.constraints} onChange={(value) => setProfileForm((items) => ({ ...items, constraints: value }))} />
                <TextAreaField label="Focus Areas" value={profileForm.focus_areas} onChange={(value) => setProfileForm((items) => ({ ...items, focus_areas: value }))} />
              </div>
            </section>
          </section>
        )}
      </main>

      {verificationOpen && verificationReport && (
        <div className="modal-backdrop" onClick={() => setVerificationOpen(false)}>
          <div className="modal-card" onClick={(event) => event.stopPropagation()}>
            <div className="section-head">
              <div>
                <p className="section-label">Grounded verification</p>
                <h2>{verificationReport.verdict || verificationReport.status}</h2>
              </div>
              <button className="ghost" onClick={() => setVerificationOpen(false)}>
                Close
              </button>
            </div>
            <div className="grid-3">
              <MetricCard label="Status" value={verificationReport.status} detail={verificationReport.provider} />
              <MetricCard label="Confidence" value={`${Math.round((verificationReport.confidence || 0) * 100)}%`} detail={verificationReport.model} />
              <MetricCard label="Sources" value={(verificationReport.source_links || []).length} detail="Linked references" />
            </div>
            <div className="panel soft">
              <p className="section-label">Evidence summary</p>
              <p>{verificationReport.evidence_summary || "No summary available."}</p>
            </div>
            <div className="stack-md">
              {(verificationReport.claims || []).map((claim, index) => (
                <div key={`${claim.claim}-${index}`} className="panel soft">
                  <div className="task-meta">
                    <span className="tag-chip">{claim.verdict || "unknown"}</span>
                    <span className="muted-copy">{Math.round((claim.confidence || 0) * 100)}% confidence</span>
                  </div>
                  <p className="claim-title">{claim.claim}</p>
                  <p className="muted-copy">{claim.rationale}</p>
                </div>
              ))}
            </div>
            <div className="stack-md">
              {(verificationReport.source_links || []).map((source, index) => (
                <a key={`${source.url}-${index}`} className="source-link" href={source.url} target="_blank" rel="noreferrer">
                  <strong>{source.title || source.url}</strong>
                  <span>{source.publisher || source.url}</span>
                </a>
              ))}
            </div>
          </div>
        </div>
      )}

      {toast && <div className={classNames("toast", toast.type)}>{toast.message}</div>}
    </div>
  );
}

function MetricCard({ label, value, detail }) {
  return (
    <div className="metric-card">
      <p className="metric-label">{label}</p>
      <p className="metric-value">{value}</p>
      <p className="metric-detail">{detail}</p>
    </div>
  );
}

function SelectField({ label, value, onChange, options }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function TextField({ label, value, onChange, type = "text" }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type={type} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function TextAreaField({ label, value, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <textarea rows={5} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function EmptyState({ title, body }) {
  return (
    <div className="panel empty-panel">
      <h3>{title}</h3>
      <p>{body}</p>
    </div>
  );
}

function TaskCard({ task, onStatusChange }) {
  return (
    <article className="task-card">
      <div className="task-meta">
        <span className={classNames("tag-chip", task.source === "goal_planner" ? "accent" : "warm")}>
          {task.source === "goal_planner" ? "AI Suggested" : "Manual"}
        </span>
        <span className="tag-chip">{task.effort || "medium"} effort</span>
        <span className="tag-chip">{task.impact || "medium"} impact</span>
      </div>
      <h3>{task.title}</h3>
      <p className="task-next-step">{task.next_step || "No next step captured yet."}</p>
      {task.evidence_text && <p className="muted-copy">{task.evidence_text}</p>}
      {task.notes && <p className="muted-copy">{task.notes}</p>}
      <div className="task-actions">
        <span className="muted-copy">Due {formatDate(task.due_date)}</span>
        <select
          value={task.status}
          onChange={(event) => onStatusChange(task.id, event.target.value)}
        >
          <option value="pending">Pending</option>
          <option value="in_progress">In Progress</option>
          <option value="done">Done</option>
        </select>
      </div>
    </article>
  );
}

export default App;
