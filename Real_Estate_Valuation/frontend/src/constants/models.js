export const MODEL_COLORS = {
  'Linear Regression': '#3b82f6',      // Blue
  'Polynomial Regression': '#8b5cf6',  // Purple / Violet
  'KNN': '#06b6d4',                    // Cyan
  'SVM': '#10b981',                    // Emerald Green
  'Decision Tree': '#f59e0b',          // Amber
  'Logistic Regression': '#ec4899',    // Pink / Rose
};

export const MODEL_DESCRIPTIONS = {
  'Linear Regression': {
    summary: 'Fits an ordinary least squares hyperplane minimizing sum of squared residuals between features and continuous plot price.',
    strengths: 'Fast to train, fully interpretable coefficients, establishes a reliable baseline.',
    weaknesses: 'Assumes linear relationships; sensitive to extreme price outliers and multi-collinearity.',
    howItWorks: 'Calculates closed-form weights (W = (X^T X)^-1 X^T y) giving each locality and unit of area an additive valuation weight.'
  },
  'Polynomial Regression': {
    summary: 'Extends linear models by creating degree-2 polynomial combinations (area, area^2, log_area^2) to capture non-linear land scaling.',
    strengths: 'Captures non-linear elasticity where marginal plot price per marla decreases for larger acreage.',
    weaknesses: 'Prone to severe overfitting at boundaries and can produce wild predictions outside training area domains.',
    howItWorks: 'Transforms input features with polynomial basis expansion before fitting ordinary least squares.'
  },
  'KNN': {
    summary: 'Predicts plot price by taking distance-weighted average of the K most similar historical listings in area and locality.',
    strengths: 'Non-parametric, intuitive for real estate ("what did neighboring plots sell for?"), adapts to local micro-markets.',
    weaknesses: 'High memory usage at inference, computationally expensive on large datasets, degraded by irrelevant features.',
    howItWorks: 'Measures Euclidean distance across standardized features and weights nearest neighbors inversely proportional to distance.'
  },
  'SVM': {
    summary: 'Finds an optimal hyperplane within an epsilon-insensitive tube using support vectors, ignoring small valuation errors.',
    strengths: 'Robust against extreme luxury outliers, regularized margin prevents overfitting on noisy listings.',
    weaknesses: 'Requires careful feature scaling and hyperparameter tuning; computationally demanding with large kernels.',
    howItWorks: 'Solves quadratic optimization maximizing margin while penalizing deviations outside the tolerance band (epsilon).'
  },
  'Decision Tree': {
    summary: 'Partitions data into orthogonal feature regions by iteratively selecting splits that minimize mean squared error.',
    strengths: 'Captures complex non-linear thresholds and feature interactions (e.g. DHA Phase 6 vs Phase 8) without scaling requirements.',
    weaknesses: 'High variance, prone to overfitting deep branches, produces step-wise discrete predictions.',
    howItWorks: 'Greedily splits training nodes by minimizing variance reduction (MSE) at each candidate threshold.'
  },
  'Logistic Regression': {
    summary: 'Classifies plots into 3 equal market tiers (Low, Medium, High) using the multinomial softmax function.',
    strengths: 'Provides calibrated probability confidence for price brackets; robust against numeric price variance.',
    weaknesses: 'Cannot predict exact monetary PKR values; requires discretizing continuous prices into arbitrary bins.',
    howItWorks: 'Maps linear combination of features through softmax function to compute class membership probabilities.'
  }
};

export function formatPKR(amount) {
  if (amount === undefined || amount === null || isNaN(amount)) return 'N/A';
  const num = Number(amount);
  if (num >= 1e7) {
    return `PKR ${(num / 1e7).toFixed(2)} Crore`;
  } else if (num >= 1e5) {
    return `PKR ${(num / 1e5).toFixed(2)} Lakh`;
  } else if (num >= 1e3) {
    return `PKR ${(num / 1e3).toFixed(2)} Thousand`;
  } else {
    return `PKR ${num.toLocaleString()}`;
  }
}
