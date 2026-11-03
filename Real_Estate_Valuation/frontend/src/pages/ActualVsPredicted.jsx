import React, { useState, useMemo } from 'react';
import { 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  ReferenceLine, 
  BarChart, 
  Bar 
} from 'recharts';
import { 
  CheckSquare, 
  Square, 
  Layers, 
  BarChart3, 
  Info, 
  AlertCircle 
} from 'lucide-react';
import { MODEL_COLORS, formatPKR } from '../constants/models';

export default function ActualVsPredicted({ predictionsData, loading }) {
  const [selectedModels, setSelectedModels] = useState({
    'SVM': true,
    'Linear Regression': true,
    'KNN': false,
    'Decision Tree': false,
    'Polynomial Regression': false,
  });

  const [activeResidualModel, setActiveResidualModel] = useState('SVM');

  const modelsList = [
    { key: 'svm_pred', name: 'SVM', color: MODEL_COLORS['SVM'] },
    { key: 'linear_regression_pred', name: 'Linear Regression', color: MODEL_COLORS['Linear Regression'] },
    { key: 'polynomial_regression_pred', name: 'Polynomial Regression', color: MODEL_COLORS['Polynomial Regression'] },
    { key: 'knn_pred', name: 'KNN', color: MODEL_COLORS['KNN'] },
    { key: 'decision_tree_pred', name: 'Decision Tree', color: MODEL_COLORS['Decision Tree'] },
  ];

  const toggleModel = (name) => {
    setSelectedModels(prev => ({ ...prev, [name]: !prev[name] }));
  };

  const rawPoints = predictionsData?.predictions || [];

  // Transform raw data into Crores PKR for clean chart readability
  const formattedData = useMemo(() => {
    return rawPoints.map(row => {
      const actualCr = Number((row.actual_price / 1e7).toFixed(3));
      return {
        actual_cr: actualCr,
        actual_pkr: row.actual_price,
        area: row.area,
        locality: row.locality,
        svm_cr: Number(((row.svm_pred || 0) / 1e7).toFixed(3)),
        svm_res: Number(((row.actual_price - (row.svm_pred || 0)) / 1e6).toFixed(2)),
        linear_cr: Number(((row.linear_regression_pred || 0) / 1e7).toFixed(3)),
        linear_res: Number(((row.actual_price - (row.linear_regression_pred || 0)) / 1e6).toFixed(2)),
        poly_cr: Number(((row.polynomial_regression_pred || 0) / 1e7).toFixed(3)),
        poly_res: Number(((row.actual_price - (row.polynomial_regression_pred || 0)) / 1e6).toFixed(2)),
        knn_cr: Number(((row.knn_pred || 0) / 1e7).toFixed(3)),
        knn_res: Number(((row.actual_price - (row.knn_pred || 0)) / 1e6).toFixed(2)),
        dt_cr: Number(((row.decision_tree_pred || 0) / 1e7).toFixed(3)),
        dt_res: Number(((row.actual_price - (row.decision_tree_pred || 0)) / 1e6).toFixed(2)),
      };
    });
  }, [rawPoints]);

  // Compute maximum coordinate for 45-degree reference line
  const maxCoordinate = useMemo(() => {
    let maxVal = 10;
    formattedData.forEach(d => {
      if (d.actual_cr > maxVal) maxVal = d.actual_cr;
    });
    return Math.min(15, Math.ceil(maxVal));
  }, [formattedData]);

  // Compute residual distribution histogram for the active model
  const residualHistogram = useMemo(() => {
    if (!formattedData.length) return [];
    
    const keyMap = {
      'SVM': 'svm_res',
      'Linear Regression': 'linear_res',
      'Polynomial Regression': 'poly_res',
      'KNN': 'knn_res',
      'Decision Tree': 'dt_res'
    };
    const resKey = keyMap[activeResidualModel] || 'svm_res';
    const residuals = formattedData.map(d => d[resKey]).filter(v => v !== undefined && !isNaN(v));

    // Create 15 balanced bins around 0
    const minRes = -25; // -25M PKR
    const maxRes = 25;  // +25M PKR
    const binSize = (maxRes - minRes) / 16;

    const bins = Array.from({ length: 16 }, (_, i) => {
      const start = minRes + i * binSize;
      const end = start + binSize;
      return {
        bin_label: `${start.toFixed(0)} to ${end.toFixed(0)}M`,
        bin_center: Number(((start + end) / 2).toFixed(1)),
        count: 0
      };
    });

    residuals.forEach(r => {
      for (let i = 0; i < bins.length; i++) {
        const start = minRes + i * binSize;
        const end = start + binSize;
        if (r >= start && r < end) {
          bins[i].count += 1;
          break;
        }
      }
    });

    return bins;
  }, [formattedData, activeResidualModel]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-slate-500 dark:text-slate-400 font-medium">Loading test set predictions...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      
      {/* Page Header */}
      <div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
          Actual vs. Predicted Price Scatter & Residual Diagnostics
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Visual test of model calibration on {formattedData.length} unseen test listings. Points lying directly on the dashed red diagonal represent perfect valuations.
        </p>
      </div>

      {/* Model Selection Multi-Toggles */}
      <div className="p-4 sm:p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center space-x-2 text-xs font-bold text-slate-600 dark:text-slate-300 uppercase tracking-wider">
            <Layers className="w-4 h-4 text-blue-600" />
            <span>Visible Models on Scatter Plot:</span>
          </div>

          <div className="flex flex-wrap gap-2">
            {modelsList.map(m => {
              const isChecked = selectedModels[m.name];
              return (
                <button
                  key={m.name}
                  onClick={() => toggleModel(m.name)}
                  className={`flex items-center space-x-2 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
                    isChecked
                      ? 'bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-white shadow-sm'
                      : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 text-slate-400 dark:text-slate-500 hover:border-slate-300'
                  }`}
                >
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: m.color }} />
                  <span>{m.name}</span>
                  {isChecked ? <CheckSquare className="w-3.5 h-3.5 text-blue-600" /> : <Square className="w-3.5 h-3.5" />}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Scatter Plot */}
      <div className="p-6 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              Actual Price vs. Predicted Price (in Crores PKR)
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              X-Axis: Real Ground-Truth Listing Price | Y-Axis: Model Forecasted Price
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded-full bg-rose-50 dark:bg-rose-950 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-800">
            Dashed Line = 45° Ideal (y = x)
          </span>
        </div>

        <div className="h-96 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 15, right: 25, bottom: 25, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              
              <XAxis 
                type="number" 
                dataKey="actual_cr" 
                name="Actual Price" 
                unit=" Cr" 
                domain={[0, maxCoordinate]} 
                tick={{ fontSize: 11 }}
                label={{ value: 'Actual Plot Price (Crores PKR)', position: 'insideBottom', offset: -15, fontSize: 12, fill: '#64748b' }}
              />
              <YAxis 
                type="number" 
                name="Predicted Price" 
                unit=" Cr" 
                domain={[0, maxCoordinate]} 
                tick={{ fontSize: 11 }}
                label={{ value: 'Predicted Price (Crores PKR)', angle: -90, position: 'insideLeft', offset: 0, fontSize: 12, fill: '#64748b' }}
              />

              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-700 text-white text-xs space-y-1 shadow-xl">
                        <p className="font-bold text-slate-200">{data.locality} ({data.area} Marla)</p>
                        <p className="text-emerald-400">Actual: PKR {data.actual_cr.toFixed(2)} Crore</p>
                        {selectedModels['SVM'] && <p className="text-emerald-300">SVM: PKR {data.svm_cr.toFixed(2)} Cr</p>}
                        {selectedModels['Linear Regression'] && <p className="text-blue-300">Linear: PKR {data.linear_cr.toFixed(2)} Cr</p>}
                        {selectedModels['Polynomial Regression'] && <p className="text-purple-300">Polynomial: PKR {data.poly_cr.toFixed(2)} Cr</p>}
                        {selectedModels['KNN'] && <p className="text-cyan-300">KNN: PKR {data.knn_cr.toFixed(2)} Cr</p>}
                        {selectedModels['Decision Tree'] && <p className="text-amber-300">Decision Tree: PKR {data.dt_cr.toFixed(2)} Cr</p>}
                      </div>
                    );
                  }
                  return null;
                }}
              />

              {/* 45-degree Perfect Prediction Line */}
              <ReferenceLine 
                segment={[{ x: 0, y: 0 }, { x: maxCoordinate, y: maxCoordinate }]} 
                stroke="#ef4444" 
                strokeWidth={2} 
                strokeDasharray="5 5" 
              />

              {/* Render Scatters for Selected Models */}
              {selectedModels['SVM'] && (
                <Scatter name="SVM" data={formattedData.map(d => ({ ...d, y: d.svm_cr }))} fill={MODEL_COLORS['SVM']} fillOpacity={0.6} />
              )}
              {selectedModels['Linear Regression'] && (
                <Scatter name="Linear" data={formattedData.map(d => ({ ...d, y: d.linear_cr }))} fill={MODEL_COLORS['Linear Regression']} fillOpacity={0.5} />
              )}
              {selectedModels['Polynomial Regression'] && (
                <Scatter name="Polynomial" data={formattedData.map(d => ({ ...d, y: d.poly_cr }))} fill={MODEL_COLORS['Polynomial Regression']} fillOpacity={0.5} />
              )}
              {selectedModels['KNN'] && (
                <Scatter name="KNN" data={formattedData.map(d => ({ ...d, y: d.knn_cr }))} fill={MODEL_COLORS['KNN']} fillOpacity={0.5} />
              )}
              {selectedModels['Decision Tree'] && (
                <Scatter name="Decision Tree" data={formattedData.map(d => ({ ...d, y: d.dt_cr }))} fill={MODEL_COLORS['Decision Tree']} fillOpacity={0.5} />
              )}

            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Residual Diagnostics Section */}
      <div className="p-6 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
              <BarChart3 className="w-4 h-4 text-indigo-600" />
              <span>Residual Error Distribution (Actual - Predicted)</span>
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Examining whether model prediction errors are symmetrically centered at 0 or systematically biased.
            </p>
          </div>

          {/* Select model for residual analysis */}
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-500 font-medium">Model:</span>
            <select
              value={activeResidualModel}
              onChange={(e) => setActiveResidualModel(e.target.value)}
              className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {modelsList.map(m => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={residualHistogram} margin={{ top: 10, right: 10, left: -20, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
              <XAxis 
                dataKey="bin_label" 
                tick={{ fontSize: 10 }} 
                interval={1} 
                angle={-25} 
                textAnchor="end" 
              />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip 
                formatter={(val) => [`${val} test plots`, 'Count']}
                labelFormatter={(lbl) => `Residual Range: ${lbl}`}
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px', fontSize: '12px' }}
              />
              <ReferenceLine x="0 to 3M" stroke="#ef4444" strokeWidth={2} strokeDasharray="3 3" />
              <Bar 
                dataKey="count" 
                fill={MODEL_COLORS[activeResidualModel] || '#3b82f6'} 
                radius={[4, 4, 0, 0]} 
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 text-xs text-slate-600 dark:text-slate-300 space-y-1.5">
          <p className="font-bold text-slate-900 dark:text-white flex items-center space-x-1.5">
            <Info className="w-4 h-4 text-blue-500" />
            <span>Residual Analysis Insights:</span>
          </p>
          <p>
            - A well-calibrated ML regression model displays a <strong>Gaussian bell-curve centered at 0</strong>.
          </p>
          <p>
            - Residuals &gt; 0 signify the model <em>under-estimated</em> the price (common in luxury phases with hidden amenities).
          </p>
          <p>
            - Residuals &lt; 0 signify the model <em>over-estimated</em> the price (often seen when plots have litigation issues or unpaved road access not captured in text).
          </p>
        </div>
      </div>

    </div>
  );
}
