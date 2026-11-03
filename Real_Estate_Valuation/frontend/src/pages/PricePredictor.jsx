import React, { useState, useEffect } from 'react';
import { 
  Calculator, 
  MapPin, 
  Ruler, 
  Sparkles, 
  TrendingUp, 
  BarChart3, 
  ShieldCheck, 
  AlertCircle,
  Trophy,
  ArrowRight
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell 
} from 'recharts';
import { postPrediction, fetchLocalities } from '../services/api';
import { MODEL_COLORS, formatPKR } from '../constants/models';

export default function PricePredictor() {
  const [localities, setLocalities] = useState([]);
  const [selectedLocality, setSelectedLocality] = useState('DHA Defence');
  const [areaInput, setAreaInput] = useState('10');
  const [unit, setUnit] = useState('Marla');
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState(null);

  // Load localities on mount
  useEffect(() => {
    fetchLocalities()
      .then(data => {
        if (data && data.localities) {
          setLocalities(data.localities);
          if (data.localities.includes('DHA Defence')) {
            setSelectedLocality('DHA Defence');
          } else if (data.localities.length > 0) {
            setSelectedLocality(data.localities[0]);
          }
        }
      })
      .catch(err => {
        console.error("Failed to load localities:", err);
      });
  }, []);

  // Quick preset button handler
  const handlePreset = (areaVal, unitVal) => {
    setAreaInput(String(areaVal));
    setUnit(unitVal);
  };

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setError(null);
    const numArea = parseFloat(areaInput);

    if (isNaN(numArea) || numArea <= 0) {
      setError('Please enter a valid positive land area.');
      return;
    }

    setLoading(true);
    try {
      const data = await postPrediction({
        locality: selectedLocality,
        area: numArea,
        unit: unit
      });
      setPredictionResult(data);
    } catch (err) {
      setError(err.message || 'Failed to generate valuation.');
    } finally {
      setLoading(false);
    }
  };

  // Run initial prediction for DHA Defence 10 Marla when localities load
  useEffect(() => {
    if (localities.length > 0 && !predictionResult) {
      handlePredict();
    }
  }, [localities]);

  // Format data for comparison bar chart
  const chartData = predictionResult?.regression_predictions 
    ? Object.entries(predictionResult.regression_predictions).map(([name, val]) => ({
        name: name.replace(' Regression', ''),
        price_crore: val.price_crore || (val.price_pkr / 1e7),
        formatted: val.formatted || formatPKR(val.price_pkr),
        fullName: name,
        color: MODEL_COLORS[name] || '#3b82f6'
      }))
    : [];

  const classification = predictionResult?.classification || {};
  const probs = classification.probabilities || {};

  return (
    <div className="space-y-8 animate-fadeIn">
      
      {/* Header */}
      <div>
        <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-950/70 text-blue-700 dark:text-blue-300 text-xs font-semibold mb-2">
          <Calculator className="w-3.5 h-3.5" />
          <span>Interactive Property Valuation</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
          Real-Time Multi-Model Plot Price Predictor
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Select a Lahore locality and land area to generate instantaneous valuation estimates across all 6 machine learning models.
        </p>
      </div>

      {/* Input Form & Parameters */}
      <div className="p-6 sm:p-8 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
        <form onSubmit={handlePredict} className="space-y-6">
          
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            
            {/* Locality Dropdown */}
            <div className="md:col-span-6 space-y-2">
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center space-x-1.5">
                <MapPin className="w-3.5 h-3.5 text-blue-600" />
                <span>Locality / Housing Society</span>
              </label>
              <select
                value={selectedLocality}
                onChange={(e) => setSelectedLocality(e.target.value)}
                className="w-full px-4 py-3 rounded-2xl bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 transition text-sm"
              >
                {localities.map(loc => (
                  <option key={loc} value={loc}>{loc}</option>
                ))}
              </select>
            </div>

            {/* Area Input & Unit */}
            <div className="md:col-span-4 space-y-2">
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center space-x-1.5">
                <Ruler className="w-3.5 h-3.5 text-indigo-600" />
                <span>Plot Area Quantity</span>
              </label>
              <div className="flex rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/80">
                <input
                  type="number"
                  step="0.1"
                  min="0.5"
                  max="100"
                  value={areaInput}
                  onChange={(e) => setAreaInput(e.target.value)}
                  className="w-full px-4 py-3 bg-transparent text-slate-900 dark:text-white font-semibold focus:outline-none text-sm"
                  placeholder="e.g. 5, 10, 1"
                />
                <select
                  value={unit}
                  onChange={(e) => setUnit(e.target.value)}
                  className="px-3 py-3 bg-slate-100 dark:bg-slate-700 font-bold text-xs text-slate-700 dark:text-slate-200 border-l border-slate-200 dark:border-slate-700 focus:outline-none cursor-pointer"
                >
                  <option value="Marla">Marla</option>
                  <option value="Kanal">Kanal (20 Marla)</option>
                </select>
              </div>
            </div>

            {/* Submit Button */}
            <div className="md:col-span-2 flex items-end">
              <button
                type="submit"
                disabled={loading}
                className="w-full py-3 px-4 rounded-2xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold text-sm shadow-md shadow-blue-500/25 transition disabled:opacity-50 flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Predict</span>
                  </>
                )}
              </button>
            </div>

          </div>

          {/* Quick Presets */}
          <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-100 dark:border-slate-800 text-xs">
            <span className="text-slate-500 font-medium">Standard Presets:</span>
            <button
              type="button"
              onClick={() => handlePreset(3, 'Marla')}
              className="px-3 py-1 rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-blue-50 dark:hover:bg-blue-950 font-medium transition"
            >
              3 Marla
            </button>
            <button
              type="button"
              onClick={() => handlePreset(5, 'Marla')}
              className="px-3 py-1 rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-blue-50 dark:hover:bg-blue-950 font-medium transition"
            >
              5 Marla
            </button>
            <button
              type="button"
              onClick={() => handlePreset(10, 'Marla')}
              className="px-3 py-1 rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-blue-50 dark:hover:bg-blue-950 font-medium transition"
            >
              10 Marla
            </button>
            <button
              type="button"
              onClick={() => handlePreset(1, 'Kanal')}
              className="px-3 py-1 rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-blue-50 dark:hover:bg-blue-950 font-medium transition"
            >
              1 Kanal (20 Marla)
            </button>
            <button
              type="button"
              onClick={() => handlePreset(2, 'Kanal')}
              className="px-3 py-1 rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-blue-50 dark:hover:bg-blue-950 font-medium transition"
            >
              2 Kanal (40 Marla)
            </button>
          </div>

        </form>

        {error && (
          <div className="mt-4 p-4 rounded-2xl bg-rose-50 dark:bg-rose-950/60 border border-rose-200 dark:border-rose-900 text-rose-700 dark:text-rose-300 text-xs flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Prediction Results Display */}
      {predictionResult && (
        <div className="space-y-8 animate-fadeIn">
          
          {/* Target Query Details */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-4 rounded-2xl bg-slate-100 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 text-xs text-slate-600 dark:text-slate-300">
            <div>
              <span>Property: </span>
              <strong className="text-slate-900 dark:text-white">{predictionResult.input.locality}</strong>
              <span> • Area: </span>
              <strong className="text-slate-900 dark:text-white">{predictionResult.input.original_area} {predictionResult.input.unit}</strong>
              <span> ({predictionResult.input.area_marla} Marla equivalent)</span>
            </div>
            <span className="font-mono text-slate-500">log(area) = {predictionResult.input.log_area}</span>
          </div>

          {/* Model Prediction Cards Row */}
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white mb-4">
              Regression Model Valuations (Side-by-Side)
            </h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
              {Object.entries(predictionResult.regression_predictions).map(([name, data]) => {
                const isBestModel = name === 'SVM';
                const color = MODEL_COLORS[name] || '#3b82f6';

                return (
                  <div 
                    key={name}
                    className={`p-5 rounded-3xl border transition relative ${
                      isBestModel 
                        ? 'bg-gradient-to-b from-emerald-50/50 to-white dark:from-emerald-950/30 dark:to-slate-900 border-emerald-300 dark:border-emerald-800 shadow-md ring-2 ring-emerald-500/20' 
                        : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800'
                    }`}
                  >
                    {isBestModel && (
                      <span className="absolute -top-3 right-4 px-2 py-0.5 rounded-full text-[10px] font-extrabold uppercase bg-emerald-600 text-white shadow-sm flex items-center space-x-1">
                        <Trophy className="w-3 h-3" />
                        <span>Best Model</span>
                      </span>
                    )}

                    <div className="flex items-center space-x-2 mb-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                      <span className="text-xs font-bold text-slate-700 dark:text-slate-300 truncate">
                        {name}
                      </span>
                    </div>

                    <div className="space-y-1 my-2">
                      <p className="text-xl font-black text-slate-900 dark:text-white tracking-tight">
                        {data.formatted}
                      </p>
                      <p className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
                        {data.price_pkr.toLocaleString()} PKR
                      </p>
                    </div>

                    <div className="mt-3 pt-2 border-t border-slate-100 dark:border-slate-800 flex justify-between text-[11px] text-slate-500">
                      <span>In Crores:</span>
                      <strong className="text-slate-700 dark:text-slate-300">{data.price_crore} Cr</strong>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Bar Chart Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* 2 Cols: Predicted Price Comparison Bar Chart */}
            <div className="lg:col-span-2 p-6 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                    <BarChart3 className="w-4 h-4 text-blue-600" />
                    <span>Price Forecast Comparison Across Algorithms</span>
                  </h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Comparing valuation estimates (in Crores PKR) for the chosen property.
                  </p>
                </div>
                <span className="text-xs px-2.5 py-1 rounded-full bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300 font-semibold border border-blue-200 dark:border-blue-800">
                  Valuation Spread
                </span>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 15, right: 15, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis unit=" Cr" tick={{ fontSize: 11 }} />
                    <Tooltip 
                      formatter={(val, name, item) => [item.payload.formatted, 'Predicted Price']}
                      contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px', fontSize: '12px' }}
                    />
                    <Bar dataKey="price_crore" radius={[4, 4, 0, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* 1 Col: Logistic Regression Price Band Classifier Result */}
            <div className="p-6 rounded-3xl border border-pink-200 dark:border-pink-900/60 bg-gradient-to-b from-pink-50/30 to-white dark:from-pink-950/20 dark:to-slate-900 shadow-sm flex flex-col justify-between">
              
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <span className="w-3 h-3 rounded-full bg-pink-500" />
                  <h4 className="text-sm font-bold uppercase tracking-wider text-pink-700 dark:text-pink-300">
                    Logistic Regression
                  </h4>
                </div>

                <div className="space-y-1">
                  <span className="text-xs font-medium text-slate-500">Predicted Market Tier:</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-3xl font-black text-slate-900 dark:text-white">
                      {classification.predicted_band || 'High'}
                    </span>
                    <span className="text-xs px-2.5 py-0.5 rounded-full font-bold bg-pink-100 text-pink-800 dark:bg-pink-950 dark:text-pink-300 border border-pink-300 dark:border-pink-800">
                      Market Tier
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    {classification.band_definitions?.[classification.predicted_band]}
                  </p>
                </div>

                {/* Probability Distribution Bars */}
                <div className="space-y-2 pt-2 border-t border-pink-100 dark:border-pink-900/40">
                  <span className="text-xs font-bold text-slate-700 dark:text-slate-300">
                    Confidence Distribution:
                  </span>
                  {['High', 'Medium', 'Low'].map(b => {
                    const prob = (probs[b] || 0) * 100;
                    return (
                      <div key={b} className="space-y-1">
                        <div className="flex justify-between text-xs font-medium">
                          <span className="text-slate-600 dark:text-slate-400">{b} Bracket</span>
                          <span className="font-mono font-bold text-slate-900 dark:text-white">{prob.toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${
                              b === 'High' ? 'bg-pink-500' : b === 'Medium' ? 'bg-indigo-500' : 'bg-blue-500'
                            }`}
                            style={{ width: `${prob}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <p className="text-[11px] text-slate-400 dark:text-slate-500 pt-3">
                *Logistic regression outputs softmax class probabilities across the 3 budget tiers.
              </p>

            </div>

          </div>

        </div>
      )}

    </div>
  );
}
