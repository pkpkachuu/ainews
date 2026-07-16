import { useState, useEffect } from 'react';
import {
  Newspaper,
  Search,
  TrendingUp,
  Brain,
  RefreshCw,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  Sparkles,
} from 'lucide-react';

// Types
interface FeedItem {
  feed_item_id: string;
  type: string;
  topic_label: string;
  trigger_reason: string;
  summary: string;
  key_entities: string[];
  supporting_article_ids: string[];
  article_count: number;
  combined_score: number;
  generated_at: string;
}

interface Article {
  article_id: string;
  title: string;
  body?: string;
  source: string;
  published_at: string;
  sentiment_score?: number;
  category?: string;
  url?: string;
  _score?: number;
}

interface TrendingEntity {
  entity: string;
  total_articles: number;
  recent_articles: number;
  velocity: number;
  avg_sentiment: number;
}

interface TimelineEvent {
  date: string;
  event_count: number;
  headline_titles: string[];
  sources: string[];
  sentiment: number;
}

// API Configuration
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Utility functions
const formatDate = (dateStr: string): string => {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const getSentimentColor = (score: number): string => {
  if (score > 0.3) return 'text-emerald-600';
  if (score < -0.3) return 'text-red-600';
  return 'text-slate-500';
};

const getSentimentBg = (score: number): string => {
  if (score > 0.3) return 'bg-emerald-50 border-emerald-200';
  if (score < -0.3) return 'bg-red-50 border-red-200';
  return 'bg-slate-50 border-slate-200';
};

function Navigation({
  activeTab,
  setActiveTab,
}: {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}) {
  const tabs = [
    { id: 'feed', label: 'Intelligence Feed', icon: Sparkles },
    { id: 'search', label: 'Search', icon: Search },
    { id: 'trending', label: 'Trending', icon: TrendingUp },
    { id: 'agent', label: 'Analyst', icon: Brain },
  ];

  return (
    <nav className="bg-white border-b border-slate-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-2">
            <Newspaper className="w-8 h-8 text-sky-600" />
            <span className="text-xl font-bold text-slate-800">News Analytics</span>
          </div>
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  activeTab === tab.id
                    ? 'bg-sky-100 text-sky-700 font-medium'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}

function IntelligenceFeedTab() {
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [generatedAt, setGeneratedAt] = useState<string | null>(null);

  const fetchFeed = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/intelligence/feed`);
      const data = await response.json();
      setFeed(data.feed || []);
      setGeneratedAt(data.generated_at);
    } catch (err) {
      setError('Failed to load intelligence feed. Make sure the backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const generateFeed = async () => {
    setLoading(true);
    try {
      await fetch(`${API_BASE}/intelligence/feed/generate`, { method: 'POST' });
      await fetchFeed();
    } catch (err) {
      setError('Failed to generate feed');
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFeed();
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Intelligence Feed</h2>
          <p className="text-slate-500">
            {generatedAt ? `Generated: ${formatDate(generatedAt)}` : 'No feed generated yet'}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={fetchFeed}
            className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
          <button
            onClick={generateFeed}
            className="flex items-center gap-2 px-4 py-2 bg-sky-600 text-white rounded-lg hover:bg-sky-700 transition"
          >
            <Sparkles className="w-4 h-4" />
            Generate Feed
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <RefreshCw className="w-8 h-8 animate-spin text-sky-600" />
        </div>
      ) : error ? (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      ) : feed.length === 0 ? (
        <div className="text-center py-20 bg-slate-50 rounded-xl border border-slate-200">
          <Sparkles className="w-12 h-12 text-slate-400 mx-auto mb-4" />
          <p className="text-slate-600">No intelligence items available.</p>
          <p className="text-slate-500 text-sm">Click "Generate Feed" to create the intelligence feed.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {feed.map((item) => (
            <div
              key={item.feed_item_id}
              className={`bg-white rounded-xl border p-6 shadow-sm hover:shadow-md transition ${
                item.type === 'EMERGING_TOPIC'
                  ? 'border-amber-200 bg-gradient-to-r from-amber-50 to-white'
                  : 'border-slate-200'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    {item.type === 'EMERGING_TOPIC' ? (
                      <span className="px-2 py-1 bg-amber-100 text-amber-700 text-xs font-medium rounded-full">
                        Emerging Topic
                      </span>
                    ) : (
                      <span className="px-2 py-1 bg-sky-100 text-sky-700 text-xs font-medium rounded-full">
                        Narrative Shift
                      </span>
                    )}
                    <span className="text-sm text-slate-500">{item.article_count} articles</span>
                    <span className="text-sm text-slate-400">Score: {item.combined_score}</span>
                  </div>
                  <h3 className="text-lg font-semibold text-slate-800 mb-2">{item.topic_label}</h3>
                  <p className="text-slate-600 mb-3">{item.summary}</p>
                  <div className="flex flex-wrap gap-2">
                    {item.key_entities.slice(0, 5).map((entity) => (
                      <span
                        key={entity}
                        className="px-2 py-1 bg-slate-100 text-slate-600 text-sm rounded"
                      >
                        {entity}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="text-right ml-4">
                  <p className="text-sm text-slate-500">{item.trigger_reason}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SearchTab() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Article[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [filters, setFilters] = useState({
    category: '',
    days: '',
  });

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setSearched(true);
    try {
      const searchFilters: Record<string, any> = {};
      if (filters.category) searchFilters.category = filters.category;
      if (filters.days) searchFilters.days = filters.days;

      const response = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          filters: Object.keys(searchFilters).length ? searchFilters : null,
          size: 20,
          use_reranking: true
        })
      });
      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      console.error(err);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Search Articles</h2>
        <p className="text-slate-500">Hybrid search combining keyword and semantic retrieval</p>
      </div>

      <form onSubmit={handleSearch} className="space-y-4">
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for news topics, entities, or events..."
              className="w-full pl-12 pr-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-sky-500 focus:border-sky-500 transition"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-3 bg-sky-600 text-white rounded-xl hover:bg-sky-700 disabled:opacity-50 transition font-medium"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        <div className="flex gap-4">
          <select
            value={filters.category}
            onChange={(e) => setFilters({ ...filters, category: e.target.value })}
            className="px-4 py-2 border border-slate-300 rounded-lg bg-white"
          >
            <option value="">All Categories</option>
            <option value="technology">Technology</option>
            <option value="politics">Politics</option>
            <option value="economy">Economy</option>
            <option value="health">Health</option>
            <option value="world">World</option>
          </select>
          <select
            value={filters.days}
            onChange={(e) => setFilters({ ...filters, days: e.target.value })}
            className="px-4 py-2 border border-slate-300 rounded-lg bg-white"
          >
            <option value="">All Time</option>
            <option value="1">Last 24 Hours</option>
            <option value="7">Last 7 Days</option>
            <option value="30">Last 30 Days</option>
          </select>
        </div>
      </form>

      {searched && results.length === 0 && !loading && (
        <div className="text-center py-10 bg-slate-50 rounded-xl">
          <Search className="w-10 h-10 text-slate-400 mx-auto mb-3" />
          <p className="text-slate-600">No results found for "{query}"</p>
        </div>
      )}

      {results.length > 0 && (
        <div className="space-y-3">
          <p className="text-sm text-slate-500">{results.length} results found</p>
          {results.map((article) => (
            <div
              key={article.article_id}
              className="bg-white rounded-lg border border-slate-200 p-5 hover:shadow-md transition"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm text-slate-500">{article.source}</span>
                    <span className="text-slate-300">|</span>
                    <span className="text-sm text-slate-500">
                      {formatDate(article.published_at)}
                    </span>
                    {article.category && (
                      <>
                        <span className="text-slate-300">|</span>
                        <span className="px-2 py-0.5 bg-slate-100 text-slate-600 text-xs rounded">
                          {article.category}
                        </span>
                      </>
                    )}
                  </div>
                  <h3 className="text-lg font-semibold text-slate-800 mb-2">{article.title}</h3>
                  {article.body && (
                    <p className="text-slate-600 text-sm line-clamp-2">
                      {article.body.slice(0, 200)}...
                    </p>
                  )}
                </div>
                {article.sentiment_score !== undefined && (
                  <div
                    className={`ml-4 px-3 py-2 rounded-lg border text-sm ${getSentimentBg(
                      article.sentiment_score
                    )}`}
                  >
                    <span className={getSentimentColor(article.sentiment_score)}>
                      {article.sentiment_score > 0 ? '+' : ''}
                      {article.sentiment_score.toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TrendingTab() {
  const [trending, setTrending] = useState<TrendingEntity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTrending = async () => {
      try {
        const response = await fetch(`${API_BASE}/intelligence/trending`);
        const data = await response.json();
        setTrending(data.trending || []);
      } catch (err) {
        setError('Failed to load trending entities');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchTrending();
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Trending Entities</h2>
          <p className="text-slate-500">Entities ranked by velocity of recent coverage</p>
        </div>
        <button
          onClick={() => window.location.reload()}
          className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <RefreshCw className="w-8 h-8 animate-spin text-sky-600" />
        </div>
      ) : error ? (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      ) : trending.length === 0 ? (
        <div className="text-center py-20 bg-slate-50 rounded-xl border border-slate-200">
          <TrendingUp className="w-12 h-12 text-slate-400 mx-auto mb-4" />
          <p className="text-slate-600">No trending entities found.</p>
          <p className="text-slate-500 text-sm">Ingest some articles first to see trending data.</p>
        </div>
      ) : (
        <div className="grid gap-4">
          {trending.map((entity, index) => (
            <div
              key={entity.entity}
              className="bg-white rounded-lg border border-slate-200 p-5 hover:shadow-md transition flex items-center justify-between"
            >
              <div className="flex items-center gap-4">
                <span className="text-2xl font-bold text-slate-400 w-8">{index + 1}</span>
                <div>
                  <h3 className="text-lg font-semibold text-slate-800">{entity.entity}</h3>
                  <p className="text-sm text-slate-500">
                    {entity.total_articles} total articles, {entity.recent_articles} in last 24h
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-6">
                <div className="text-center">
                  <p className="text-2xl font-bold text-sky-600">{entity.velocity}%</p>
                  <p className="text-xs text-slate-500">Velocity</p>
                </div>
                <div className="text-center">
                  {entity.avg_sentiment > 0 ? (
                    <ArrowUpRight className="w-6 h-6 text-emerald-500" />
                  ) : entity.avg_sentiment < 0 ? (
                    <ArrowDownRight className="w-6 h-6 text-red-500" />
                  ) : (
                    <div className="w-6 h-6 rounded-full bg-slate-200" />
                  )}
                  <p className="text-xs text-slate-500">
                    {entity.avg_sentiment > 0 ? 'Positive' : entity.avg_sentiment < 0 ? 'Negative' : 'Neutral'}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AgentTab() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState<{
    type: string;
    entity?: string;
    topic?: string;
    narrative?: string;
    explanation?: string;
    timeline?: TimelineEvent[];
    trend_data?: Record<string, unknown>;
    article_count?: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<'timeline' | 'explain'>('timeline');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResponse(null);

    try {
      const endpoint = mode === 'timeline' ? '/agent/timeline' : '/agent/explain-trend';
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setResponse({ ...data, type: mode });
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-800">AI Analyst</h2>
        <p className="text-slate-500">Ask questions about news trends and events</p>
      </div>

      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setMode('timeline')}
          className={`px-4 py-2 rounded-lg transition ${
            mode === 'timeline'
              ? 'bg-sky-100 text-sky-700 font-medium'
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
          }`}
        >
          <Clock className="w-4 h-4 inline mr-2" />
          Timeline
        </button>
        <button
          onClick={() => setMode('explain')}
          className={`px-4 py-2 rounded-lg transition ${
            mode === 'explain'
              ? 'bg-sky-100 text-sky-700 font-medium'
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
          }`}
        >
          <TrendingUp className="w-4 h-4 inline mr-2" />
          Trend Explanation
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="relative">
          <Brain className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              mode === 'timeline'
                ? 'e.g., "What has been happening with AI regulation?"'
                : 'e.g., "Why is OpenAI trending?"'
            }
            className="w-full pl-12 pr-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-sky-500 focus:border-sky-500 transition"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="w-full px-6 py-3 bg-sky-600 text-white rounded-xl hover:bg-sky-700 disabled:opacity-50 transition font-medium"
        >
          {loading ? 'Analyzing...' : 'Ask Analyst'}
        </button>
      </form>

      {response && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-slate-800">
              {mode === 'timeline' ? 'Timeline' : 'Explanation'}
            </h3>
            <span className="text-sm text-slate-500">
              {response.article_count} articles analyzed
            </span>
          </div>

          {(response.narrative || response.explanation) && (
            <div className="bg-sky-50 border border-sky-200 rounded-lg p-4">
              <p className="text-slate-700">{response.narrative || response.explanation}</p>
            </div>
          )}

          {response.timeline && response.timeline.length > 0 && (
            <div className="space-y-3 mt-4">
              <h4 className="font-medium text-slate-700">Timeline:</h4>
              <div className="relative">
                <div className="absolute left-4 top-0 bottom-0 w-px bg-slate-200" />
                {response.timeline.map((event, index) => (
                  <div key={index} className="relative pl-10 pb-4">
                    <div className="absolute left-2 top-1 w-4 h-4 rounded-full bg-sky-500 border-2 border-white" />
                    <div className="bg-slate-50 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-slate-800">{event.date}</span>
                        <span className="text-sm text-slate-500">{event.event_count} articles</span>
                      </div>
                      <ul className="text-sm text-slate-600">
                        {event.headline_titles.slice(0, 2).map((title, i) => (
                          <li key={i} className="truncate">
                            {title}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('feed');

  return (
    <div className="min-h-screen bg-slate-100">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'feed' && <IntelligenceFeedTab />}
        {activeTab === 'search' && <SearchTab />}
        {activeTab === 'trending' && <TrendingTab />}
        {activeTab === 'agent' && <AgentTab />}
      </main>

      <footer className="border-t border-slate-200 bg-white py-6 mt-auto">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-slate-500">
          AI-Powered Intelligent News Analytics Platform | MVP Version
        </div>
      </footer>
    </div>
  );
}

export default App;
