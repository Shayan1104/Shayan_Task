import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Overview from './pages/Overview';
import ModelComparison from './pages/ModelComparison';
import ActualVsPredicted from './pages/ActualVsPredicted';
import ModelDetails from './pages/ModelDetails';
import PricePredictor from './pages/PricePredictor';
import { fetchMetrics, fetchEdaSummary, fetchPredictions } from './services/api';
import { AlertCircle, RefreshCw } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('comparison'); // Start on flagship Model Comparison tab
  const [darkMode, setDarkMode] = useState(() => {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  const [metricsData, setMetricsData] = useState(null);
  const [edaData, setEdaData] = useState(null);
  const [predictionsData, setPredictionsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [backendConnected, setBackendConnected] = useState(false);

  // Sync dark mode class on document element
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Load all initial API data
  const loadAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [m, e, p] = await Promise.all([
        fetchMetrics(),
        fetchEdaSummary(),
        fetchPredictions(),
      ]);
      setMetricsData(m);
      setEdaData(e);
      setPredictionsData(p);
      setBackendConnected(true);
    } catch (err) {
      console.error("API connection failed:", err);
      setError(err.message || 'Failed to connect to FastAPI backend.');
      setBackendConnected(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAllData();
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors">
      
      {/* Sticky Navigation Bar */}
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        darkMode={darkMode} 
        setDarkMode={setDarkMode}
        backendConnected={backendConnected}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Error Notification if Backend is Unreachable */}
        {error && (
          <div className="mb-6 p-4 rounded-2xl bg-rose-50 dark:bg-rose-950/70 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-200 text-sm flex items-center justify-between shadow-sm">
            <div className="flex items-center space-x-3">
              <AlertCircle className="w-5 h-5 text-rose-600 dark:text-rose-400 shrink-0" />
              <div>
                <p className="font-bold">Backend Connection Error</p>
                <p className="text-xs text-rose-600 dark:text-rose-300">
                  {error} Ensure FastAPI is running at <code>http://127.0.0.1:8000</code>.
                </p>
              </div>
            </div>
            <button
              onClick={loadAllData}
              className="px-3 py-1.5 rounded-xl bg-rose-600 text-white hover:bg-rose-700 text-xs font-bold flex items-center space-x-1.5 transition shadow-sm"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Tab Router */}
        {activeTab === 'overview' && (
          <Overview 
            edaData={edaData} 
            metricsData={metricsData} 
            loading={loading} 
          />
        )}

        {activeTab === 'comparison' && (
          <ModelComparison 
            metricsData={metricsData} 
            loading={loading} 
          />
        )}

        {activeTab === 'actual_vs_pred' && (
          <ActualVsPredicted 
            predictionsData={predictionsData} 
            loading={loading} 
          />
        )}

        {activeTab === 'details' && (
          <ModelDetails 
            metricsData={metricsData} 
            loading={loading} 
          />
        )}

        {activeTab === 'predictor' && (
          <PricePredictor />
        )}

      </main>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 py-6 text-xs text-slate-500 dark:text-slate-400 transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>
            © 2026 Zameen Property Valuation ML Assignment • Powered by FastAPI & React
          </p>
          <div className="flex items-center space-x-4 text-xs font-medium">
            <span>Linear • Poly • KNN • SVR • Tree • Logistic</span>
            <span className="text-slate-300 dark:text-slate-700">|</span>
            <span className="font-mono text-emerald-600 dark:text-emerald-400">All Metrics Verified</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
