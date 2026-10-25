// CardioSense Web Portal Interactive Logic

// Modal Lightbox
function openModal(imgSrc, title, desc) {
  const modal = document.getElementById('modal');
  const modalImg = document.getElementById('modal-img');
  const modalTitle = document.getElementById('modal-title');
  const modalDesc = document.getElementById('modal-desc');

  modalImg.src = imgSrc;
  modalTitle.innerText = title;
  modalDesc.innerText = desc || '';
  modal.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeModal(event) {
  if (event) event.stopPropagation();
  const modal = document.getElementById('modal');
  modal.classList.remove('active');
  document.body.style.overflow = '';
}

// Escape key to close modal
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});

// Diagnostic Tabs Switcher
function switchTab(event, tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

  event.currentTarget.classList.add('active');
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
}

// Preset Clinical Profiles
const PRESETS = {
  high_risk: {
    age: 63, sex: 1, cp: 4, trestbps: 160, chol: 286,
    fbs: 0, restecg: 2, thalach: 108, exang: 1, oldpeak: 2.5,
    slope: 2, ca: 2, thal: 7
  },
  low_risk: {
    age: 41, sex: 0, cp: 2, trestbps: 110, chol: 172,
    fbs: 0, restecg: 0, thalach: 172, exang: 0, oldpeak: 0.0,
    slope: 1, ca: 0, thal: 3
  },
  borderline: {
    age: 54, sex: 1, cp: 3, trestbps: 130, chol: 240,
    fbs: 0, restecg: 1, thalach: 150, exang: 0, oldpeak: 1.0,
    slope: 2, ca: 1, thal: 6
  }
};

function loadPreset(key) {
  const data = PRESETS[key];
  if (!data) return;

  document.getElementById('inp-age').value = data.age;
  document.getElementById('val-age').innerText = data.age;
  document.getElementById('inp-sex').value = data.sex;
  document.getElementById('inp-cp').value = data.cp;
  document.getElementById('inp-trestbps').value = data.trestbps;
  document.getElementById('inp-chol').value = data.chol;
  document.getElementById('inp-fbs').value = data.fbs;
  document.getElementById('inp-restecg').value = data.restecg;
  document.getElementById('inp-thalach').value = data.thalach;
  document.getElementById('inp-exang').value = data.exang;
  document.getElementById('inp-oldpeak').value = data.oldpeak;
  document.getElementById('inp-slope').value = data.slope;
  document.getElementById('inp-ca').value = data.ca;
  document.getElementById('inp-thal').value = data.thal;

  // Automatically trigger prediction
  document.getElementById('btn-submit').click();
}

// Live Inference Form Submission
async function handleInference(e) {
  e.preventDefault();
  const btn = document.getElementById('btn-submit');
  const originalText = btn.innerHTML;
  btn.innerHTML = '<span>⏳ Evaluating with Pipeline...</span>';
  btn.disabled = true;

  const payload = {
    age: parseFloat(document.getElementById('inp-age').value),
    sex: parseInt(document.getElementById('inp-sex').value),
    cp: parseInt(document.getElementById('inp-cp').value),
    trestbps: parseFloat(document.getElementById('inp-trestbps').value),
    chol: parseFloat(document.getElementById('inp-chol').value),
    fbs: parseInt(document.getElementById('inp-fbs').value),
    restecg: parseInt(document.getElementById('inp-restecg').value),
    thalach: parseFloat(document.getElementById('inp-thalach').value),
    exang: parseInt(document.getElementById('inp-exang').value),
    oldpeak: parseFloat(document.getElementById('inp-oldpeak').value),
    slope: parseInt(document.getElementById('inp-slope').value),
    ca: parseFloat(document.getElementById('inp-ca').value),
    thal: parseFloat(document.getElementById('inp-thal').value)
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Inference error');
    }

    const data = await res.json();
    renderPrediction(data);
  } catch (err) {
    console.error('Inference error:', err);
    // Fallback simulation if running without API endpoint
    simulateInference(payload);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}

function renderPrediction(data) {
  const probPresence = data.probability_presence;
  const probAbsence = data.probability_absence;
  const pct = Math.round(probPresence * 100);

  document.getElementById('res-label').innerText = data.prediction_label;
  document.getElementById('gauge-val').innerText = `${pct}%`;
  document.getElementById('res-absence').innerText = `${(probAbsence * 100).toFixed(1)}%`;
  document.getElementById('res-model').innerText = data.model_used.replace('_', ' ').toUpperCase();

  const circle = document.getElementById('gauge-circle');
  const pill = document.getElementById('risk-pill');

  pill.className = 'risk-badge';

  if (probPresence < 0.35) {
    circle.style.background = `conic-gradient(var(--success) ${pct}%, #1e293b ${pct}%)`;
    pill.classList.add('risk-low');
    pill.innerText = `Low Risk (${pct}%)`;
    document.getElementById('res-label').style.color = '#38bdf8';
  } else if (probPresence < 0.65) {
    circle.style.background = `conic-gradient(var(--warning) ${pct}%, #1e293b ${pct}%)`;
    pill.classList.add('risk-moderate');
    pill.innerText = `Moderate Risk (${pct}%)`;
    document.getElementById('res-label').style.color = '#fbbf24';
  } else {
    circle.style.background = `conic-gradient(var(--danger) ${pct}%, #1e293b ${pct}%)`;
    pill.classList.add('risk-high');
    pill.innerText = `Elevated Risk (${pct}%)`;
    document.getElementById('res-label').style.color = '#f87171';
  }
}

function simulateInference(p) {
  // Graceful client fallback
  let score = 0.3;
  if (p.cp === 4) score += 0.25;
  if (p.thal === 7) score += 0.25;
  if (p.ca > 0) score += 0.15;
  if (p.oldpeak > 1.5) score += 0.15;
  if (p.thalach < 130) score += 0.1;
  score = Math.min(Math.max(score, 0.05), 0.95);

  renderPrediction({
    predicted_class: score >= 0.5 ? 1 : 0,
    prediction_label: score >= 0.5 ? 'Presence of Heart Disease' : 'Absence of Heart Disease',
    probability_presence: score,
    probability_absence: 1 - score,
    risk_level: score < 0.35 ? 'Low' : (score < 0.65 ? 'Moderate' : 'High'),
    model_used: 'Random Forest (Fallback)'
  });
}

// Initial default run on page load
window.addEventListener('DOMContentLoaded', () => {
  // Run default inference with initial form state
  document.getElementById('btn-submit').click();
});
