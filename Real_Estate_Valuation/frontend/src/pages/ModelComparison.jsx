import React, { useState, useMemo } from 'react';
import { 
  Trophy, 
  ArrowUpDown, 
  ArrowUp, 
  ArrowDown, 
  ShieldCheck, 
  AlertTriangle, 
  HelpCircle, 
  CheckCircle, 
  BarChart2, 
  Compass, 
  Sparkles,
  BookOpen
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar 
} from 'recharts';
import { MODEL_COLORS, formatPKR } from '../constants/models';

export default function ModelComparison({ metricsData, loading }) {
  const [sortField, setSortField] = useState('r2_test');
  const [sortAsc, setSortAsc] = useState(false); // default descending (best first)

  const rawMetrics = metricsData?.metrics || {};

  // Extract only regression models for tabular comparison
  const regressionModels = useMemo(() => {
    if (!rawMetrics || Object.keys(rawMetrics).length === 0) return [];
    return Object.entries(rawMetrics)
      .filter(([_, data]) => data.type === 'regression')
      .map(([name, data]) => {
        const m = data.metrics;
        const gap = m.overfitting_gap !== undefined ? m.overfitting_gap : (m.r2_train - m.r2_test);
        
        let fitStatus = 'Good fit';
        let fitColor = 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 border-emerald-300 dark:border-emerald-800';
        if (gap > 0.35) {
          fitStatus = 'Severe Overfit';
          fitColor = 'bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300 border-rose-300 dark:border-rose-800';
        } else if (gap > 0.15) {
          fitStatus = 'Moderate Overfit';
          fitColor = 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300 border-amber-300 dark:border-amber-800';
        } else if (gap < -0.05) {
          fitStatus = 'Conservative';
          fitColor = 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 border-blue-300 dark:border-blue-800';
        }

        return {
          name,
          r2_test: m.r2_test,
          r2_train: m.r2_train,
          mae: m.mae,
          mae_millions: m.mae / 1e6,
          rmse: m.rmse,
          rmse_millions: m.rmse / 1e6,
          mape: m.mape,
          overfitting_gap: gap,
          fitStatus,
          fitColor,
          color: MODEL_COLORS[name] || '#3b82f6'
        };
      });
  }, [rawMetrics]);

  // Identify best values across columns
  const bestValues = useMemo(() => {
    if (regressionModels.length === 0) return {};
    return {
      r2_test: Math.max(...regressionModels.map(m => m.r2_test)),
      r2_train: Math.max(...regressionModels.map(m => m.r2_train)),
      mae: Math.min(...regressionModels.map(m => m.mae)),
      rmse: Math.min(...regressionModels.map(m => m.rmse)),
      mape: Math.min(...regressionModels.map(m => m.mape)),
      overfitting_gap: Math.min(...regressionModels.map(m => Math.abs(m.overfitting_gap)))
    };
  }, [regressionModels]);

  // Handle column sorting
  const sortedModels = useMemo(() => {
    return [...regressionModels].sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];
      if (sortField === 'overfitting_gap') {
        valA = Math.abs(valA);
        valB = Math.abs(valB);
      }
      if (valA < valB) return sortAsc ? -1 : 1;
      if (valA > valB) return sortAsc ? 1 : -1;
      return 0;
    });
  }, [regressionModels, sortField, sortAsc]);

  const toggleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      // For error metrics, lower is usually better, so default asc
      if (['mae', 'rmse', 'mape', 'overfitting_gap'].includes(field)) {
        setSortAsc(true);
      } else {
        setSortAsc(false);
      }
    }
  };

  // Prepare Grouped Bar Chart Data
  const r2BarData = useMemo(() => {
    return regressionModels.map(m => ({
      name: m.name.replace(' Regression', ''),
      'Test R²': Number((m.r2_test * 100).toFixed(1)),
      'Train R²': Number((m.r2_train * 100).toFixed(1)),
    }));
  }, [regressionModels]);

  const errorBarData = useMemo(() => {
    return regressionModels.map(m => ({
      name: m.name.replace(' Regression', ''),
      'MAE (M PKR)': Number(m.mae_millions.toFixed(2)),
      'RMSE (M PKR)': Number(m.rmse_millions.toFixed(2)),
    }));
  }, [regressionModels]);

  // Prepare Radar Chart Data (Normalized metrics from 20 to 100)
  const radarData = useMemo(() => {
    if (regressionModels.length === 0) return [];
    const maxR2 = Math.max(...regressionModels.map(m => m.r2_test));
    const minMae = Math.min(...regressionModels.map(m => m.mae));
    const minRmse = Math.min(...regressionModels.map(m => m.rmse));
    const minMape = Math.min(...regressionModels.map(m => m.mape));

    return [
      {
        metric: 'Generalization (Test R²)',
        ...Object.fromEntries(regressionModels.map(m => [m.name, Math.round((m.r2_test / maxR2) * 100)]))
      },
      {
        metric: 'Low Absolute Error (1/MAE)',
        ...Object.fromEntries(regressionModels.map(m => [m.name, Math.round((minMae / m.mae) * 100)]))
      },
      {
        metric: 'Large Error Resistance (1/RMSE)',
        ...Object.fromEntries(regressionModels.map(m => [m.name, Math.round((minRmse / m.rmse) * 100)]))
      },
      {
        metric: 'Relative Accuracy (1/MAPE)',
        ...Object.fromEntries(regressionModels.map(m => [m.name, Math.round((minMape / m.mape) * 100)]))
      },
      {
        metric: 'Stability (No Overfit)',
        ...Object.fromEntries(regressionModels.map(m => [
          m.name, 
          Math.max(20, Math.round((1 - Math.min(1, Math.abs(m.overfitting_gap) / 0.5)) * 100))
        ]))
      },
    ];
  }, [regressionModels]);

  if (loading || !metricsData || !metricsData.metrics) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-slate-500 dark:text-slate-400 font-medium">Loading model benchmarks...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-950/70 text-blue-700 dark:text-blue-300 text-xs font-semibold mb-2">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Algorithm Performance Matrix</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
            Model Leaderboard & Comparative Evaluation
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Examine generalization capacity, overfitting gaps, and residual errors across all 5 regression models on 597 unseen test plots.
          </p>
        </div>
      </div>

      {/* "Best Model" Callout Box */}
      <div className="p-6 rounded-3xl bg-gradient-to-r from-emerald-600 via-teal-600 to-blue-700 text-white shadow-xl shadow-emerald-500/10">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="space-y-2 max-w-3xl">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-white/20 backdrop-blur-md text-xs font-bold uppercase tracking-wider text-emerald-100">
              <Trophy className="w-3.5 h-3.5 text-amber-300" />
              <span>Overall Champion: Support Vector Regression (SVR / SVM)</span>
            </div>
            <h3 className="text-xl sm:text-2xl font-black">
              Why SVM (SVR) Outperformed All Other Algorithms
            </h3>
            <p className="text-sm text-emerald-50/90 leading-relaxed">
              <strong>Support Vector Regression (SVR)</strong> achieved the top test score with <strong>R² = 59.39%</strong> and the lowest test RMSE of <strong>PKR 4.32 Crore</strong>.
              Unlike standard Linear Regression which is violently pulled by extreme multi-crore plot outliers, SVR's <em>epsilon-insensitive margin</em> disregards small errors and employs soft-margin linear regularization (C = 2,000,000), making it exceptionally robust on skewed real estate distributions.
            </p>
          </div>

          <div className="flex flex-row md:flex-col items-center justify-center p-4 rounded-2xl bg-white/10 backdrop-blur-md border border-white/20 text-center shrink-0 w-full md:w-auto">
            <span className="text-xs uppercase font-bold text-emerald-200">Test R² Score</span>
            <span className="text-4xl font-extrabold text-white my-1">59.4%</span>
            <span className="text-[11px] text-emerald-100">Rank #1 Generalization</span>
          </div>
        </div>
      </div>

      {/* 1. Sortable Leaderboard Table */}
      <div className="p-6 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm overflow-hidden">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">
              Official Model Benchmark Leaderboard
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Click any column header to sort. The best result in each category is highlighted with a green badge.
            </p>
          </div>
          <span className="text-xs text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700">
            Sorted by: <strong>{sortField.toUpperCase()}</strong> ({sortAsc ? 'Ascending' : 'Descending'})
          </span>
        </div>

        <div className="overflow-x-auto -mx-6 px-6">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                <th className="py-3 px-3">Rank</th>
                <th className="py-3 px-3 cursor-pointer hover:text-blue-600" onClick={() => toggleSort('name')}>
                  <div className="flex items-center space-x-1">
                    <span>Algorithm</span>
                    <ArrowUpDown className="w-3.5 h-3.5" />
                  </div>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-blue-600" onClick={() => toggleSort('r2_test')}>
                  <div className="flex items-center space-x-1">
                    <span>R² Test</span>
                    {sortField === 'r2_test' ? (sortAsc ? <ArrowUp className="w-3.5 h-3.5 text-blue-600" /> : <ArrowDown className="w-3.5 h-3.5 text-blue-600" />) : <ArrowUpDown className="w-3.5 h-3.5" />}
                  </div>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-blue-600" onClick={() => toggleSort('r2_train')}>
                  <div className="flex items-center space-x-1">
                    <span>R² Train</span>
                    {sortField === 'r2_train' ? (sortAsc ? <ArrowUp className="w-3.5 h-3.5 text-blue-600" /> : <ArrowDown className="w-3.5 h-3.5 text-blue-600" />) : <ArrowUpDown className="w-3.5 h-3.5" />}
                  </div>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-blue-600" onClick={() => toggleSort('mae')}>
                  <div className="flex items-center space-x-1">
                    <span>MAE (PKR)</span>
                    {sortField === 'mae' ? (sortAsc ? <ArrowUp className="w-3.5 h-3.5 text-blue-600" /> : <ArrowDown className="w-3.5 h-3.5 text-blue-600" />) : <ArrowUpDown className="w-3.5 h-3.5" />}
                  </div>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-blue-600" onClick={() => toggleSort('rmse')}>
                  <div className="flex items-center space-x-1">
                    <span>RMSE (PKR)</span>
                    {sortField === 'rmse' ? (sortAsc ? <ArrowUp className="w-3.5 h-3.5 text-blue-600" /> : <ArrowDown className="w-3.5 h-3.5 text-blue-600" />) : <ArrowUpDown className="w-3.5 h-3.5" />}
                  </div>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-blue-600" onClick={() => toggleSort('mape')}>
                  <div className="flex items-center space-x-1">
                    <span>Avg % Error</span>
                    {sortField === 'mape' ? (sortAsc ? <ArrowUp className="w-3.5 h-3.5 text-blue-600" /> : <ArrowDown className="w-3.5 h-3.5 text-blue-600" />) : <ArrowUpDown className="w-3.5 h-3.5" />}
                  </div>
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-blue-600" onClick={() => toggleSort('overfitting_gap')}>
                  <div className="flex items-center space-x-1">
                    <span>Overfit Gap</span>
                    {sortField === 'overfitting_gap' ? (sortAsc ? <ArrowUp className="w-3.5 h-3.5 text-blue-600" /> : <ArrowDown className="w-3.5 h-3.5 text-blue-600" />) : <ArrowUpDown className="w-3.5 h-3.5" />}
                  </div>
                </th>
                <th className="py-3 px-3">Fit Diagnosis</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800 text-sm">
              {sortedModels.map((m, idx) => {
                const isBestR2Test = m.r2_test === bestValues.r2_test;
                const isBestR2Train = m.r2_train === bestValues.r2_train;
                const isBestMae = m.mae === bestValues.mae;
                const isBestRmse = m.rmse === bestValues.rmse;
                const isBestMape = m.mape === bestValues.mape;

                return (
                  <tr key={m.name} className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition">
                    <td className="py-3.5 px-3">
                      <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                        idx === 0 
                          ? 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300 border border-amber-300' 
                          : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                      }`}>
                        {idx + 1}
                      </span>
                    </td>
                    <td className="py-3.5 px-3 font-semibold text-slate-900 dark:text-white flex items-center space-x-2">
                      <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: m.color }} />
                      <span>{m.name}</span>
                    </td>
                    <td className="py-3.5 px-3 font-mono">
                      <span className={`px-2 py-0.5 rounded font-bold ${
                        isBestR2Test 
                          ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800' 
                          : 'text-slate-700 dark:text-slate-300'
                      }`}>
                        {(m.r2_test * 100).toFixed(2)}%
                      </span>
                    </td>
                    <td className="py-3.5 px-3 font-mono">
                      <span className={`px-2 py-0.5 rounded ${
                        isBestR2Train 
                          ? 'bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-300 font-bold' 
                          : 'text-slate-600 dark:text-slate-400'
                      }`}>
                        {(m.r2_train * 100).toFixed(2)}%
                      </span>
                    </td>
                    <td className="py-3.5 px-3 font-mono">
                      <span className={`px-2 py-0.5 rounded ${
                        isBestMae 
                          ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 font-bold border border-emerald-300 dark:border-emerald-800' 
                          : 'text-slate-700 dark:text-slate-300'
                      }`}>
                        {formatPKR(m.mae)}
                      </span>
                    </td>
                    <td className="py-3.5 px-3 font-mono">
                      <span className={`px-2 py-0.5 rounded ${
                        isBestRmse 
                          ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 font-bold border border-emerald-300 dark:border-emerald-800' 
                          : 'text-slate-700 dark:text-slate-300'
                      }`}>
                        {formatPKR(m.rmse)}
                      </span>
                    </td>
                    <td className="py-3.5 px-3 font-mono">
                      <span className={`px-2 py-0.5 rounded ${
                        isBestMape 
                          ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 font-bold border border-emerald-300 dark:border-emerald-800' 
                          : 'text-slate-700 dark:text-slate-300'
                      }`}>
                        {m.mape.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3.5 px-3 font-mono">
                      <span className={`text-xs ${m.overfitting_gap > 0.2 ? 'text-rose-600 dark:text-rose-400 font-bold' : 'text-slate-600 dark:text-slate-400'}`}>
                        {m.overfitting_gap > 0 ? `+${(m.overfitting_gap * 100).toFixed(1)}%` : `${(m.overfitting_gap * 100).toFixed(1)}%`}
                      </span>
                    </td>
                    <td className="py-3.5 px-3">
                      <span className={`text-[11px] px-2.5 py-1 rounded-full font-semibold border ${m.fitColor}`}>
                        {m.fitStatus}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. Visual Comparison Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Chart 1: R2 Train vs Test Grouped Bar Chart */}
        <div className="p-6 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
          <div className="mb-4">
            <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
              <BarChart2 className="w-4 h-4 text-blue-600" />
              <span>Generalization Gap: Test R² vs Train R² (%)</span>
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              A high train R² accompanied by a lower test R² signals overfitting (e.g. KNN & Decision Tree).
            </p>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={r2BarData} margin={{ top: 15, right: 15, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 11 }} />
                <Tooltip 
                  formatter={(val) => [`${val}%`, 'Variance Explained']}
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                <Bar dataKey="Test R²" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Train R²" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: MAE & RMSE Error Bars */}
        <div className="p-6 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
          <div className="mb-4">
            <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
              <BarChart2 className="w-4 h-4 text-emerald-600" />
              <span>Model Error Magnitude: MAE vs RMSE (Millions PKR)</span>
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Lower is better. RMSE heavily penalizes large errors on luxury plots.
            </p>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={errorBarData} margin={{ top: 15, right: 15, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis unit="M" tick={{ fontSize: 11 }} />
                <Tooltip 
                  formatter={(val) => [`PKR ${val} Million`, '']}
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                <Bar dataKey="MAE (M PKR)" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="RMSE (M PKR)" fill="#f43f5e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* 3. Radar Chart (Holistic Multi-Metric Evaluation) */}
      <div className="p-6 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
              <Compass className="w-4 h-4 text-indigo-600" />
              <span>Normalized Multi-Dimensional Radar Comparison</span>
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              All metrics scaled 0-100 (where larger area indicates superior performance across accuracy, stability, and error resistance).
            </p>
          </div>
          <span className="text-xs text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700">
            5 Regressors Benchmarked
          </span>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
              <PolarGrid stroke="#94a3b8" strokeDasharray="3 3" opacity={0.3} />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#64748b' }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
              
              <Radar name="SVM" dataKey="SVM" stroke="#10b981" fill="#10b981" fillOpacity={0.25} />
              <Radar name="Linear Reg" dataKey="Linear Regression" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} />
              <Radar name="Decision Tree" dataKey="Decision Tree" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} />
              <Radar name="KNN" dataKey="KNN" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.15} />
              <Radar name="Polynomial" dataKey="Polynomial Regression" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.15} />
              
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px', fontSize: '12px' }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 4. Educational "How to Read This" Section */}
      <div className="p-6 rounded-3xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60 shadow-sm space-y-4">
        <div className="flex items-center space-x-2 text-slate-800 dark:text-slate-100 font-bold">
          <BookOpen className="w-5 h-5 text-blue-600" />
          <span>Student Learning Guide: Understanding Evaluation Metrics in Real Estate ML</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-slate-600 dark:text-slate-400">
          
          <div className="p-4 rounded-2xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/60 space-y-2">
            <h4 className="font-bold text-slate-900 dark:text-white text-sm flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span>R² Score (Variance Explained)</span>
            </h4>
            <p>
              Calculated as <code>1 - (SS_res / SS_tot)</code>. It quantifies what percentage of property price variation is explained by land area and locality. An R² of 0.59 means 59% of price dispersion is accounted for by the model; remaining variance stems from micro-factors (corner plot, park facing, road width).
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/60 space-y-2">
            <h4 className="font-bold text-slate-900 dark:text-white text-sm flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-500" />
              <span>MAE vs RMSE (Error Penalization)</span>
            </h4>
            <p>
              <strong>MAE (Mean Absolute Error)</strong> measures the typical linear miss in PKR (e.g. off by ~PKR 9.3M). 
              <strong> RMSE (Root Mean Squared Error)</strong> squares errors before averaging, making it disproportionately sensitive to huge mistakes on expensive 10 Crore luxury plots.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/60 space-y-2">
            <h4 className="font-bold text-slate-900 dark:text-white text-sm flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              <span>The Overfitting Gap (Train vs Test)</span>
            </h4>
            <p>
              The difference <code>R²_train - R²_test</code>. Notice KNN scored 87.6% on training data but plunged to 42.8% on test data (gap of 44.8%). This proves KNN merely memorized nearest training instances, failing to generalize to new listings as effectively as SVM.
            </p>
          </div>

        </div>
      </div>

    </div>
  );
}
