const API_BASE_URL = 'http://localhost:8000/api';

export async function fetchMetrics() {
  const res = await fetch(`${API_BASE_URL}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.statusText}`);
  return res.json();
}

export async function fetchPredictions() {
  const res = await fetch(`${API_BASE_URL}/predictions`);
  if (!res.ok) throw new Error(`Failed to fetch test predictions: ${res.statusText}`);
  return res.json();
}

export async function fetchEdaSummary() {
  const res = await fetch(`${API_BASE_URL}/eda`);
  if (!res.ok) throw new Error(`Failed to fetch EDA data: ${res.statusText}`);
  return res.json();
}

export async function fetchLocalities() {
  const res = await fetch(`${API_BASE_URL}/localities`);
  if (!res.ok) throw new Error(`Failed to fetch localities: ${res.statusText}`);
  return res.json();
}

export async function postPrediction(payload) {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed with status: ${res.status}`);
  }
  return res.json();
}
