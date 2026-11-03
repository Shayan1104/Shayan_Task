import React from 'react';
import { 
  Building2, 
  Coins, 
  BrainCircuit, 
  Trophy, 
  MapPin, 
  BarChart, 
  Layers, 
  CheckCircle2,
  TrendingUp,
  Info
} from 'lucide-react';
import { 
  BarChart as RechartsBarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  CartesianGrid, 
  Cell 
} from 'recharts';
import KPICard from '../components/KPICard';
import { formatPKR } from '../constants/models';

export default function Overview({ edaData, metricsData, loading }) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-slate-500 dark:text-slate-400 font-medium">Loading dataset analytics...</p>
      </div>
    );
  }

  const priceSummary = edaData?.price_summary || {};
  const medianPrice = priceSummary.median || 10500000;
  const totalListings = edaData?.total_listings || 2982;
  const bestModel = edaData?.best_model || { name: 'SVM', test_r2: 0.5939 };
  const localityMedians = edaData?.median_by_locality || [];
  const priceHistogram = edaData?.price_histogram || [];
  const areaHistogram = edaData?.area_histogram || [];

  return (
    <div className="space-y-8 animate-fadeIn">
      
      {/* Hero Welcome Header */}
      <div className="p-6 sm:p-8 rounded-3xl bg-gradient-to-r from-blue-700 via-indigo-700 to-emerald-700 text-white shadow-xl shadow-blue-500/10">
        <div className="max-w-3xl">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-white/15 backdrop-blur-md text-xs font-semibold text-emerald-200 mb-3">
            <TrendingUp className="w-3.5 h-3.5" />
            <span>Zameen.com Residential Plots Dataset (Lahore, Pakistan)</span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight">
            Property Valuation & Machine Learning Analytics
          </h1>
          <p className="mt-2 text-sm sm:text-base text-blue-100/90 leading-relaxed">
            Exploratory Data Analysis and predictive modeling across 6 machine learning architectures.
            Trained to forecast fair market prices for residential plots from standard physical and geospatial features.
          </p>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <KPICard
          title="Listings Used"
          value={totalListings.toLocaleString()}
          subtitle="Cleaned residential plots for sale"
          icon={Building2}
          color="blue"
          badge="Verified"
        />
        <KPICard
          title="Median Plot Price"
          value={formatPKR(medianPrice)}
          subtitle="50th percentile valuation in PKR"
          icon={Coins}
          color="emerald"
          badge="Market Median"
        />
        <KPICard
          title="Models Evaluated"
          value="6 Algorithms"
          subtitle="5 Regressors + 1 Classifier"
          icon={BrainCircuit}
          color="indigo"
          badge="Benchmarked"
        />
        <KPICard
          title="Best Model"
          value={bestModel.name}
          subtitle={`Top R² Score: ${(bestModel.test_r2 * 100).toFixed(1)}%`}
          icon={Trophy}
          color="amber"
          badge="Rank #1"
        />
      </div>

      {/* Pipeline Summary & Data Hygiene Badge */}
      <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-xl bg-emerald-100 dark:bg-emerald-950/80 text-emerald-700 dark:text-emerald-300">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-bold text-slate-800 dark:text-slate-100">
              Rigorous Data Cleaning & Outlier Removal Pipeline Applied
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Filtered from 20,571 raw rows down to 2,982 verified residential plot listings. 95 shifted installment rows repaired; 2,156 duplicate listings and 45 Tukey IQR price-per-marla outliers eliminated.
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2 text-xs font-mono font-medium text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700">
          <span>IQR Bounds: [PKR 2.1 Lakh/M - 66.0 Lakh/M]</span>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

        {/* Chart 1: Price Distribution */}
        <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                <BarChart className="w-4 h-4 text-blue-600" />
                <span>Price Distribution (in Crores PKR)</span>
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Frequency count per price bin. Demonstrates characteristic right-skew.
              </p>
            </div>
            <span className="text-xs px-2.5 py-1 rounded-full bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300 font-semibold border border-blue-200 dark:border-blue-800">
              1 Crore = 10M PKR
            </span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsBarChart data={priceHistogram} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                <XAxis 
                  dataKey="bin_label" 
                  tick={{ fontSize: 10 }}
                  interval={2} 
                  angle={-30} 
                  textAnchor="end" 
                />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip 
                  formatter={(val) => [`${val} listings`, 'Frequency']}
                  labelFormatter={(lbl) => `Price Range: ${lbl}`}
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px', fontSize: '12px' }}
                />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </RechartsBarChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-3 p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 text-xs text-slate-600 dark:text-slate-300 flex items-start space-x-2">
            <Info className="w-4 h-4 text-blue-500 mt-0.5 shrink-0" />
            <span>
              <strong>Insight:</strong> Over 70% of residential plots fall below PKR 2 Crore (~20M PKR), with a long right tail of executive properties up to PKR 10+ Crore in luxury sectors.
            </span>
          </div>
        </div>

        {/* Chart 2: Median Price by Locality */}
        <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                <MapPin className="w-4 h-4 text-emerald-600" />
                <span>Median Price Across Top Localities</span>
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Top 12 most frequent neighborhoods ranked by median valuation.
              </p>
            </div>
            <span className="text-xs px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 font-semibold border border-emerald-200 dark:border-emerald-800">
              Lahore Sectors
            </span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsBarChart 
                data={localityMedians.slice(0, 12)} 
                layout="vertical"
                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
                <XAxis type="number" unit=" Cr" tick={{ fontSize: 11 }} />
                <YAxis 
                  dataKey="locality" 
                  type="category" 
                  width={110} 
                  tick={{ fontSize: 11, fontWeight: 500 }}
                />
                <Tooltip 
                  formatter={(val, name, item) => [item.payload.formatted, 'Median Price']}
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px', fontSize: '12px' }}
                />
                <Bar dataKey="median_price_crore" radius={[0, 4, 4, 0]}>
                  {localityMedians.slice(0, 12).map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={index === 0 ? '#10b981' : index < 4 ? '#059669' : '#0284c7'} 
                    />
                  ))}
                </Bar>
              </RechartsBarChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-3 p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 text-xs text-slate-600 dark:text-slate-300 flex items-start space-x-2">
            <Info className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
            <span>
              <strong>Insight:</strong> Established premium sectors (e.g. DHA Defence, Model Town, Gulberg) command 3x–6x higher prices per plot compared to peripheral housing societies (Central Park, Al Rehman Garden).
            </span>
          </div>
        </div>

      </div>

      {/* Plot Size Clustering Overview */}
      <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
        <h3 className="text-base font-bold text-slate-900 dark:text-white mb-2 flex items-center space-x-2">
          <Layers className="w-4 h-4 text-indigo-600" />
          <span>Standard Pakistan Plot Sizes in the Dataset</span>
        </h3>
        <p className="text-xs text-slate-500 dark:text-slate-400 mb-6">
          Unlike continuous acreage in Western datasets, residential plots in Pakistan cluster strictly into standard architectural units.
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700/50 text-center">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">Urban Compact</span>
            <p className="text-xl font-extrabold text-blue-600 dark:text-blue-400 mt-1">3 Marla</p>
            <span className="text-[11px] text-slate-500 dark:text-slate-400">~675 sq ft</span>
          </div>
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700/50 text-center">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">Standard Family</span>
            <p className="text-xl font-extrabold text-indigo-600 dark:text-indigo-400 mt-1">5 Marla</p>
            <span className="text-[11px] text-slate-500 dark:text-slate-400">~1,125 sq ft</span>
          </div>
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700/50 text-center">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">Large Residential</span>
            <p className="text-xl font-extrabold text-emerald-600 dark:text-emerald-400 mt-1">10 Marla</p>
            <span className="text-[11px] text-slate-500 dark:text-slate-400">~2,250 sq ft</span>
          </div>
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700/50 text-center">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">Executive Estate</span>
            <p className="text-xl font-extrabold text-amber-600 dark:text-amber-400 mt-1">1 Kanal (20 Marla)</p>
            <span className="text-[11px] text-slate-500 dark:text-slate-400">~4,500 sq ft</span>
          </div>
        </div>
      </div>

    </div>
  );
}
