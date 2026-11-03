import React from 'react';
import { 
  Cpu, 
  Settings2, 
  ThumbsUp, 
  ThumbsDown, 
  Sparkles, 
  Grid, 
  CheckCircle, 
  AlertCircle,
  HelpCircle,
  ArrowRight
} from 'lucide-react';
import { MODEL_COLORS, MODEL_DESCRIPTIONS, formatPKR } from '../constants/models';

export default function ModelDetails({ metricsData, loading }) {
  if (loading || !metricsData || !metricsData.metrics) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-slate-500 dark:text-slate-400 font-medium">Loading algorithm architectures...</p>
      </div>
    );
  }

  const rawMetrics = metricsData.metrics;
  const logisticData = rawMetrics['Logistic Regression'] || {};
  const logMetrics = logisticData.metrics || {};
  const cm = logMetrics.confusion_matrix || [[153, 30, 0], [34, 165, 25], [1, 26, 163]];
  const labels = logMetrics.labels || ['Low', 'Medium', 'High'];

  // All 6 models
  const modelKeys = [
    'Linear Regression',
    'Polynomial Regression',
    'KNN',
    'SVM',
    'Decision Tree',
    'Logistic Regression'
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      
      {/* Header Banner */}
      <div>
        <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-950/70 text-blue-700 dark:text-blue-300 text-xs font-semibold mb-2">
          <Cpu className="w-3.5 h-3.5" />
          <span>Architectural Breakdown</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
          Algorithm Specifications & Deep Dives
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Detailed breakdown of all 6 algorithms: mathematical behavior, optimal hyperparameters tuned via GridSearchCV, strengths, and limitations.
        </p>
      </div>

      {/* Grid of 6 Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {modelKeys.map(name => {
          const info = rawMetrics[name] || {};
          const desc = MODEL_DESCRIPTIONS[name] || {};
          const isClassifier = info.type === 'classification';
          const params = info.hyperparameters || {};
          const color = MODEL_COLORS[name] || '#3b82f6';

          return (
            <div 
              key={name}
              className="p-6 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm flex flex-col justify-between hover:shadow-md transition"
            >
              <div className="space-y-4">
                
                {/* Header */}
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-2.5">
                    <span className="w-3.5 h-3.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                    <h3 className="font-extrabold text-slate-900 dark:text-white text-base">
                      {name}
                    </h3>
                  </div>
                  <span className={`text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full border ${
                    isClassifier 
                      ? 'bg-pink-50 text-pink-700 dark:bg-pink-950 dark:text-pink-300 border-pink-200 dark:border-pink-800'
                      : 'bg-blue-50 text-blue-700 dark:bg-blue-950 dark:text-blue-300 border-blue-200 dark:border-blue-800'
                  }`}>
                    {isClassifier ? 'Classifier' : 'Regressor'}
                  </span>
                </div>

                {/* 2-Sentence Summary */}
                <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                  {desc.summary}
                </p>

                {/* Performance Highlight Pill */}
                <div className="p-3 rounded-2xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 flex items-center justify-between text-xs">
                  <span className="text-slate-500 font-medium">
                    {isClassifier ? 'Test Accuracy' : 'Test R² Score'}
                  </span>
                  <span className="font-mono font-bold text-slate-900 dark:text-white text-sm">
                    {isClassifier 
                      ? `${((info.metrics?.accuracy_test || 0) * 100).toFixed(1)}%`
                      : `${((info.metrics?.r2_test || 0) * 100).toFixed(1)}%`
                    }
                  </span>
                </div>

                {/* Best Hyperparameters */}
                <div className="space-y-1.5">
                  <span className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider flex items-center space-x-1">
                    <Settings2 className="w-3 h-3 text-blue-500" />
                    <span>Best Hyperparameters:</span>
                  </span>
                  <div className="p-2.5 rounded-xl bg-slate-900 text-slate-200 font-mono text-[11px] overflow-x-auto">
                    {Object.entries(params).map(([k, v]) => (
                      <div key={k} className="flex justify-between py-0.5">
                        <span className="text-slate-400">{k}:</span>
                        <span className="text-emerald-400 font-bold">{String(v)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Strengths & Weaknesses */}
                <div className="space-y-2 pt-2 border-t border-slate-100 dark:border-slate-800 text-xs">
                  <div className="flex items-start space-x-2">
                    <ThumbsUp className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                    <span className="text-slate-600 dark:text-slate-300">
                      <strong>Strength:</strong> {desc.strengths}
                    </span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <ThumbsDown className="w-3.5 h-3.5 text-rose-500 shrink-0 mt-0.5" />
                    <span className="text-slate-600 dark:text-slate-300">
                      <strong>Limitation:</strong> {desc.weaknesses}
                    </span>
                  </div>
                </div>

              </div>
            </div>
          );
        })}
      </div>

      {/* Dedicated Deep Dive: Logistic Regression & Confusion Matrix */}
      <div className="p-8 rounded-3xl border border-pink-200 dark:border-pink-900/60 bg-gradient-to-br from-white via-pink-50/20 to-white dark:from-slate-900 dark:via-pink-950/10 dark:to-slate-900 shadow-sm space-y-6">
        
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-pink-100 dark:bg-pink-950/70 text-pink-700 dark:text-pink-300 text-xs font-semibold mb-2">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Special Classification Focus</span>
            </div>
            <h3 className="text-xl sm:text-2xl font-black text-slate-900 dark:text-white">
              Logistic Regression: Market Bracket Classification & Confusion Matrix
            </h3>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
              Why use a classification algorithm in a price prediction problem? Logistic regression estimates class probabilities via softmax. By binning plot prices into 3 equal-sized tiers, we evaluate its ability to identify the buyer's budget segment.
            </p>
          </div>

          <div className="flex items-center space-x-4">
            <div className="text-center p-3 rounded-2xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-sm">
              <span className="text-[10px] uppercase font-bold text-slate-500">Test Accuracy</span>
              <p className="text-xl font-extrabold text-pink-600 dark:text-pink-400">
                {((logMetrics.accuracy_test || 0.8057) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-center p-3 rounded-2xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-sm">
              <span className="text-[10px] uppercase font-bold text-slate-500">Macro F1-Score</span>
              <p className="text-xl font-extrabold text-indigo-600 dark:text-indigo-400">
                {(logMetrics.macro_f1 || 0.8096).toFixed(3)}
              </p>
            </div>
          </div>
        </div>

        {/* Confusion Matrix Heatmap */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center pt-4">
          
          <div>
            <h4 className="text-sm font-bold text-slate-900 dark:text-white mb-2 flex items-center space-x-2">
              <Grid className="w-4 h-4 text-pink-600" />
              <span>Test Set Confusion Matrix Heatmap</span>
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
              Diagonal cells (green) indicate correct classifications. Off-diagonal cells represent misclassifications.
            </p>

            <div className="inline-block p-4 rounded-2xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-sm">
              
              {/* Header: Predicted Labels */}
              <div className="text-center text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
                Predicted Market Bracket
              </div>

              <div className="flex">
                {/* Vertical Label: True Labels */}
                <div className="flex items-center justify-center mr-3">
                  <span className="text-xs font-bold text-slate-500 uppercase tracking-wider -rotate-90 whitespace-nowrap">
                    Actual Bracket
                  </span>
                </div>

                <div>
                  {/* Column Headers */}
                  <div className="grid grid-cols-3 gap-2 mb-2 text-center text-xs font-bold text-slate-600 dark:text-slate-300">
                    <div>Low</div>
                    <div>Medium</div>
                    <div>High</div>
                  </div>

                  {/* Matrix Cells */}
                  <div className="space-y-2">
                    {cm.map((row, rIdx) => (
                      <div key={rIdx} className="grid grid-cols-3 gap-2">
                        {row.map((val, cIdx) => {
                          const isDiagonal = rIdx === cIdx;
                          const intensity = Math.min(1, val / 170);
                          return (
                            <div 
                              key={cIdx} 
                              className={`w-20 h-16 rounded-xl flex flex-col items-center justify-center font-mono border transition ${
                                isDiagonal 
                                  ? 'bg-emerald-500 text-white border-emerald-600 shadow-sm'
                                  : val > 25
                                    ? 'bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200 border-amber-300'
                                    : 'bg-slate-50 text-slate-600 dark:bg-slate-800 dark:text-slate-400 border-slate-200 dark:border-slate-700'
                              }`}
                            >
                              <span className="text-base font-extrabold">{val}</span>
                              <span className="text-[10px] opacity-80">
                                {isDiagonal ? 'Correct' : 'Misclassified'}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    ))}
                  </div>

                </div>
              </div>

            </div>
          </div>

          {/* Educational Insights Box */}
          <div className="space-y-4">
            <div className="p-5 rounded-2xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 space-y-3 text-xs text-slate-600 dark:text-slate-300">
              <h5 className="font-bold text-slate-900 dark:text-white text-sm flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                <span>Key Classification Findings:</span>
              </h5>
              <p>
                1. <strong>Heavy Diagonal Concentration:</strong> Out of 597 test listings, 481 plots (80.6%) were assigned to their exact true price band.
              </p>
              <p>
                2. <strong>Zero Extreme Errors:</strong> Only 1 "High" plot was mistakenly labeled "Low", and 0 "Low" plots were labeled "High". Almost all misclassifications occurred at the boundary between Low-Medium and Medium-High.
              </p>
              <p>
                3. <strong>Price Band Thresholds:</strong>
                <br />• <strong>Low:</strong> ≤ PKR 70 Lakh (7M PKR)
                <br />• <strong>Medium:</strong> PKR 70 Lakh to PKR 1.70 Crore
                <br />• <strong>High:</strong> &gt; PKR 1.70 Crore
              </p>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
