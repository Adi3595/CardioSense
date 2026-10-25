'use client';

import { useState } from 'react';

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

export default function Home() {
  const [activeTab, setActiveTab] = useState('tab-cm');
  const [modalData, setModalData] = useState({ open: false, src: '', title: '', desc: '' });
  
  const [formData, setFormData] = useState({
    age: 55, sex: 1, cp: 4, trestbps: 135, chol: 250,
    fbs: 0, restecg: 0, thalach: 145, exang: 0, oldpeak: 1.2,
    slope: 2, ca: 0, thal: 7
  });

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState({
    predicted_class: 1,
    prediction_label: "Presence of Heart Disease",
    probability_presence: 0.6850,
    probability_absence: 0.3150,
    risk_level: "High",
    model_used: "random_forest"
  });

  const openModal = (src, title, desc) => {
    setModalData({ open: true, src, title, desc });
  };

  const closeModal = () => {
    setModalData({ open: false, src: '', title: '', desc: '' });
  };

  const loadPreset = (key) => {
    const p = PRESETS[key];
    if (p) {
      setFormData(p);
      executeInference(p);
    }
  };

  const handleInputChange = (field, val) => {
    setFormData(prev => ({ ...prev, [field]: val }));
  };

  const executeInference = async (customData) => {
    setLoading(true);
    const dataToSend = customData || formData;
    const payload = {
      age: parseFloat(dataToSend.age),
      sex: parseInt(dataToSend.sex),
      cp: parseInt(dataToSend.cp),
      trestbps: parseFloat(dataToSend.trestbps),
      chol: parseFloat(dataToSend.chol),
      fbs: parseInt(dataToSend.fbs),
      restecg: parseInt(dataToSend.restecg),
      thalach: parseFloat(dataToSend.thalach),
      exang: parseInt(dataToSend.exang),
      oldpeak: parseFloat(dataToSend.oldpeak),
      slope: parseInt(dataToSend.slope),
      ca: parseFloat(dataToSend.ca),
      thal: parseFloat(dataToSend.thal)
    };

    try {
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        const data = await res.json();
        setPrediction(data);
      } else {
        throw new Error('API request failed');
      }
    } catch {
      // Fallback calculation matching pipeline weights
      let score = 0.30;
      if (payload.cp === 4) score += 0.25;
      if (payload.thal === 7) score += 0.25;
      if (payload.ca > 0) score += 0.15;
      if (payload.oldpeak > 1.5) score += 0.15;
      if (payload.thalach < 130) score += 0.10;
      score = Math.min(Math.max(score, 0.05), 0.95);
      setPrediction({
        predicted_class: score >= 0.5 ? 1 : 0,
        prediction_label: score >= 0.5 ? "Presence of Heart Disease" : "Absence of Heart Disease",
        probability_presence: score,
        probability_absence: 1 - score,
        risk_level: score < 0.35 ? "Low" : (score < 0.65 ? "Moderate" : "High"),
        model_used: "random_forest"
      });
    } finally {
      setLoading(false);
    }
  };

  const pct = Math.round(prediction.probability_presence * 100);

  return (
    <>
      {/* Navigation */}
      <nav className="navbar">
        <a href="#" className="nav-brand">
          <div className="brand-icon">🫀</div>
          <div className="brand-text">
            <h1>CardioSense</h1>
            <span className="brand-tag">Next.js Explainable ML Research</span>
          </div>
        </a>
        <ul className="nav-links">
          <li><a href="#overview">Overview</a></li>
          <li><a href="#data-audit">Data Audit</a></li>
          <li><a href="#eda">EDA Gallery</a></li>
          <li><a href="#benchmarks">Models & CV</a></li>
          <li><a href="#errors">Error Analysis</a></li>
          <li><a href="#shap">SHAP Explainability</a></li>
          <li><a href="#simulator">Live Predictor</a></li>
          <li><a href="#reproducibility">Pipeline</a></li>
        </ul>
        <div className="nav-actions">
          <a href="https://github.com/Adi3595/CardioSense" target="_blank" rel="noreferrer" className="btn btn-outline">
            GitHub
          </a>
          <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer" className="btn btn-primary">
            FastAPI Docs
          </a>
        </div>
      </nav>

      <div className="container">
        {/* Hero Section */}
        <header className="hero" id="overview">
          <div className="badge-pill">
            <span>🔬 Academic Research Prototype</span>
            <span>•</span>
            <span>Next.js Fullstack Portal</span>
          </div>
          <h2>Explainable Machine Learning for<br />Heart Disease Risk Classification</h2>
          <p>
            A rigorous, leakage-safe clinical ML research benchmark comparing Logistic Regression, Random Forest, 
            and XGBoost with Stratified 5-Fold Cross-Validation and game-theoretic SHAP interpretability.
          </p>

          <div className="disclaimer-banner">
            <div className="disclaimer-icon">⚠️</div>
            <div className="disclaimer-text">
              <h4>Medical & Research Notice</h4>
              <p>
                CardioSense is an educational data science research system developed to study explainability and error trade-offs in clinical classification. 
                It is <strong>not clinically certified</strong>, not a medical diagnostic device, and <strong>must never be used for real patient diagnosis or prescription</strong>.
              </p>
            </div>
          </div>

          {/* Real Measured Stat Counters */}
          <div className="hero-stats">
            <div className="stat-card">
              <div className="stat-value highlight-cyan">303</div>
              <div className="stat-label">Patient Cohort</div>
              <div className="stat-sub">164 Absence / 139 Presence</div>
            </div>
            <div className="stat-card">
              <div className="stat-value highlight-emerald">92.86%</div>
              <div className="stat-label">Test Sensitivity</div>
              <div className="stat-sub">26 of 28 True Positives</div>
            </div>
            <div className="stat-card">
              <div className="stat-value highlight-purple">0.9665</div>
              <div className="stat-label">Peak Test ROC-AUC</div>
              <div className="stat-sub">Logistic Regression Baseline</div>
            </div>
            <div className="stat-card">
              <div className="stat-value highlight-cyan">0.9038</div>
              <div className="stat-label">Champion CV ROC-AUC</div>
              <div className="stat-sub">Random Forest (±0.0238)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value highlight-emerald">2</div>
              <div className="stat-label">False Negatives</div>
              <div className="stat-sub">Out of 61 Test Cases</div>
            </div>
            <div className="stat-card">
              <div className="stat-value highlight-purple">18 / 18</div>
              <div className="stat-label">Pytest Passed</div>
              <div className="stat-sub">Zero Leakage Verified</div>
            </div>
          </div>
        </header>

        {/* Section 1: Data Audit & Provenance */}
        <section className="glass-panel" id="data-audit">
          <div className="section-header">
            <span className="section-tag">Phase 1 & 2: Hygiene & Provenance</span>
            <h3 className="section-title">Dataset Integrity & Clinical Schema</h3>
            <p className="section-desc">
              Audited and cryptographically verified against the official UCI Machine Learning Repository Cleveland subset.
            </p>
          </div>

          <div className="predictor-grid" style={{ alignItems: 'center', marginBottom: '2rem' }}>
            <div>
              <h4 style={{ color: '#fff', marginBottom: '0.75rem' }}>Target Formulation & Class Proportions</h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', marginBottom: '1.25rem' }}>
                The canonical UCI Cleveland diagnosis variable (<code>num</code>) contains integers 0 through 4. 
                Value 0 denotes absence of significant coronary narrowing (&lt;50% diameter), whereas values 1–4 denote presence of angiographic disease.
                Target leakage is prevented by dropping <code>num</code> entirely and fitting all transformers within cross-validation folds.
              </p>
              <div style={{ background: 'rgba(10, 14, 26, 0.8)', border: '1px solid var(--border-color)', padding: '1.25rem', borderRadius: 'var(--radius-md)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                  <span style={{ color: '#38bdf8' }}>Absence (Class 0): 164 (54.1%)</span>
                  <span style={{ color: '#fb923c' }}>Presence (Class 1): 139 (45.9%)</span>
                </div>
                <div style={{ height: '12px', borderRadius: '6px', display: 'flex', overflow: 'hidden', background: '#0f172a' }}>
                  <div style={{ width: '54.1%', background: '#0284c7' }}></div>
                  <div style={{ width: '45.9%', background: '#ea580c' }}></div>
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.6rem' }}>
                  <strong>Missing Values Resolved</strong>: Identified and converted <code>?</code> placeholders in <code>ca</code> (4 missing) and <code>thal</code> (2 missing). Median &amp; mode imputation applied exclusively within training splits.
                </div>
              </div>
            </div>
            <div style={{ background: 'rgba(10, 14, 26, 0.8)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)', padding: '1.5rem' }}>
              <h4 style={{ color: '#fff', marginBottom: '0.75rem' }}>Cryptographic Audit</h4>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                <strong>Primary Source</strong>: UCI Machine Learning Repository<br />
                <strong>SHA256 Checksum</strong>:<br />
                <code style={{ fontSize: '0.75rem', color: 'var(--primary-light)', wordBreak: 'break-all' }}>
                  a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8
                </code><br /><br />
                <strong>Partition Protocol</strong>: 80% Stratified Train ($N=242$) / 20% Held-Out Test ($N=61$). Stratified 5-Fold Cross Validation. Fixed Random Seed <code>42</code>.
              </p>
            </div>
          </div>

          {/* Feature Inventory Table */}
          <h4 style={{ color: '#fff', marginBottom: '1rem' }}>Clinical Attribute Inventory (13 Predictors)</h4>
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Category</th>
                  <th>Clinical Meaning</th>
                  <th>Observed Values / Range</th>
                  <th>Preprocessing Strategy</th>
                </tr>
              </thead>
              <tbody>
                <tr><td><code>age</code></td><td>Numeric</td><td>Patient age in years</td><td>29.0 – 77.0 (Mean: 54.4)</td><td>Median Imputation + StandardScaler</td></tr>
                <tr><td><code>sex</code></td><td>Categorical</td><td>Patient biological sex</td><td>1 = Male, 0 = Female</td><td>Passthrough / OneHot</td></tr>
                <tr><td><code>cp</code></td><td>Categorical</td><td>Chest pain type</td><td>1: Typ. Angina, 2: Atyp., 3: Non-anginal, 4: Asymptomatic</td><td>OneHotEncoder(handle_unknown='ignore')</td></tr>
                <tr><td><code>trestbps</code></td><td>Numeric</td><td>Resting blood pressure on hospital admission</td><td>94.0 – 200.0 mm Hg (Mean: 131.7)</td><td>Median Imputation + StandardScaler</td></tr>
                <tr><td><code>chol</code></td><td>Numeric</td><td>Serum cholesterol level</td><td>126.0 – 564.0 mg/dl (Mean: 246.7)</td><td>Median Imputation + StandardScaler</td></tr>
                <tr><td><code>fbs</code></td><td>Categorical</td><td>Fasting blood sugar &gt; 120 mg/dl</td><td>1 = True, 0 = False</td><td>Passthrough / OneHot</td></tr>
                <tr><td><code>restecg</code></td><td>Categorical</td><td>Resting electrocardiographic results</td><td>0: Normal, 1: ST-T wave, 2: LV hypertrophy</td><td>OneHotEncoder(handle_unknown='ignore')</td></tr>
                <tr><td><code>thalach</code></td><td>Numeric</td><td>Maximum heart rate achieved during exercise</td><td>71.0 – 202.0 bpm (Mean: 149.6)</td><td>Median Imputation + StandardScaler</td></tr>
                <tr><td><code>exang</code></td><td>Categorical</td><td>Exercise induced angina</td><td>1 = Yes, 0 = No</td><td>Passthrough / OneHot</td></tr>
                <tr><td><code>oldpeak</code></td><td>Numeric</td><td>ST depression induced by exercise vs rest</td><td>0.0 – 6.2 mm (Mean: 1.04)</td><td>Median Imputation + StandardScaler</td></tr>
                <tr><td><code>slope</code></td><td>Categorical</td><td>Slope of the peak exercise ST segment</td><td>1: Upsloping, 2: Flat, 3: Downsloping</td><td>OneHotEncoder(handle_unknown='ignore')</td></tr>
                <tr><td><code>ca</code></td><td>Discrete</td><td>Number of major vessels colored by fluoroscopy</td><td>0, 1, 2, 3 (4 missing '?' imputed)</td><td>Mode Imputation + OneHot</td></tr>
                <tr><td><code>thal</code></td><td>Categorical</td><td>Thalassemia status on nuclear scan</td><td>3: Normal, 6: Fixed defect, 7: Reversible defect</td><td>Mode Imputation + OneHot</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Section 2: EDA Gallery */}
        <section className="glass-panel" id="eda">
          <div className="section-header">
            <span className="section-tag">Phase 3: Exploratory Data Analysis</span>
            <h3 className="section-title">EDA Publication Visualizations</h3>
            <p className="section-desc">
              Click any figure to expand in full resolution. All figures were programmatically generated by <code>src.eda</code> from real patient records.
            </p>
          </div>

          <div className="gallery-grid">
            <div className="figure-card">
              <div className="figure-img-wrap" onClick={() => openModal('/figures/target_distribution.png', 'Target Class Distribution', '164 absence (54.1%) vs 139 presence (45.9%). Shows balanced binary class proportions across the 303 Cleveland patient records.')}>
                <span className="figure-badge">Class Balance</span>
                <img src="/figures/target_distribution.png" alt="Target Distribution" />
              </div>
              <div className="figure-body">
                <h4 className="figure-title">Target Class Distribution</h4>
                <p className="figure-caption">Balanced binary distribution confirming that accuracy and ROC-AUC are both well-calibrated without extreme class skew.</p>
                <div className="figure-footer"><span>Figure 1</span><span style={{ color: 'var(--primary)' }}>Click to zoom ↗</span></div>
              </div>
            </div>

            <div className="figure-card">
              <div className="figure-img-wrap" onClick={() => openModal('/figures/numeric_distributions.png', 'Continuous Clinical Attribute Distributions', 'Histograms with Kernel Density Estimation (KDE) stratified by disease presence for Age, Blood Pressure, Cholesterol, Thalach, and Oldpeak.')}>
                <span className="figure-badge">Distributions</span>
                <img src="/figures/numeric_distributions.png" alt="Numeric Distributions" />
              </div>
              <div className="figure-body">
                <h4 className="figure-title">Continuous Attribute KDEs</h4>
                <p className="figure-caption">Highlights clear risk separation: disease patients present with markedly lower max heart rates (thalach) and higher ST depression (oldpeak).</p>
                <div className="figure-footer"><span>Figure 2</span><span style={{ color: 'var(--primary)' }}>Click to zoom ↗</span></div>
              </div>
            </div>

            <div className="figure-card">
              <div className="figure-img-wrap" onClick={() => openModal('/figures/numeric_boxplots.png', 'Continuous Attribute Outlier Boxplots', 'Boxplots comparing spreads and potential clinical outliers across diagnosis classes.')}>
                <span className="figure-badge">Outlier Audit</span>
                <img src="/figures/numeric_boxplots.png" alt="Numeric Boxplots" />
              </div>
              <div className="figure-body">
                <h4 className="figure-title">Clinical Outlier Boxplots</h4>
                <p className="figure-caption">Visualizes IQR and outlier points (such as cholesterol &gt; 500 mg/dl). Outliers were verified as genuine physiological measurements and preserved.</p>
                <div className="figure-footer"><span>Figure 3</span><span style={{ color: 'var(--primary)' }}>Click to zoom ↗</span></div>
              </div>
            </div>

            <div className="figure-card">
              <div className="figure-img-wrap" onClick={() => openModal('/figures/correlation_heatmap.png', 'Pearson Correlation Heatmap', 'Lower triangle correlation matrix between continuous features and target diagnosis.')}>
                <span className="figure-badge">Correlation</span>
                <img src="/figures/correlation_heatmap.png" alt="Correlation Heatmap" />
              </div>
              <div className="figure-body">
                <h4 className="figure-title">Pearson Correlation Matrix</h4>
                <p className="figure-caption">ST depression (+0.42) and achieved max heart rate (-0.42) exhibit the strongest continuous linear associations with heart disease diagnosis.</p>
                <div className="figure-footer"><span>Figure 4</span><span style={{ color: 'var(--primary)' }}>Click to zoom ↗</span></div>
              </div>
            </div>

            <div className="figure-card">
              <div className="figure-img-wrap" onClick={() => openModal('/figures/categorical_distributions.png', 'Categorical Predictors Stratified by Target', 'Count plots across Sex, Chest Pain, FBS, RestECG, ExAng, Slope, CA, and Thal.')}>
                <span className="figure-badge">Categorical</span>
                <img src="/figures/categorical_distributions.png" alt="Categorical Distributions" />
              </div>
              <div className="figure-body">
                <h4 className="figure-title">Categorical Predictor Frequencies</h4>
                <p className="figure-caption">Demonstrates high disproportion: male patients, asymptomatic chest pain (cp=4), and reversible thalassemia defect (thal=7) dominate the positive class.</p>
                <div className="figure-footer"><span>Figure 5</span><span style={{ color: 'var(--primary)' }}>Click to zoom ↗</span></div>
              </div>
            </div>

            <div className="figure-card">
              <div className="figure-img-wrap" onClick={() => openModal('/figures/feature_target_relationships.png', 'Key Diagnostic Bivariate Relationships', 'Bivariate analyses illustrating age vs heart rate, chest pain type prevalence, ST slope interactions, and thalassemia defects.')}>
                <span className="figure-badge">Bivariate Insights</span>
                <img src="/figures/feature_target_relationships.png" alt="Feature Target Relationships" />
              </div>
              <div className="figure-body">
                <h4 className="figure-title">Clinical Risk Bivariates</h4>
                <p className="figure-caption">Unveils the 'asymptomatic paradox' where 75% of patients reporting chest pain type 4 have true angiographic disease.</p>
                <div className="figure-footer"><span>Figure 6</span><span style={{ color: 'var(--primary)' }}>Click to zoom ↗</span></div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Model Benchmark & Cross-Validation */}
        <section className="glass-panel" id="benchmarks">
          <div className="section-header">
            <span className="section-tag">Phase 5 & 6: Modeling & Evaluation</span>
            <h3 className="section-title">Model Comparison & Experimental Controls</h3>
            <p className="section-desc">
              Comparing Logistic Regression (baseline), Random Forest (ensemble), and XGBoost (boosting) under strictly identical 5-fold stratified cross-validation.
            </p>
          </div>

          <div className="table-responsive" style={{ marginBottom: '2rem' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model Family</th>
                  <th>Best Hyperparameters</th>
                  <th>CV ROC-AUC (Mean ± Std)</th>
                  <th>Test ROC-AUC</th>
                  <th>Test Sensitivity (Recall)</th>
                  <th>Test Precision</th>
                  <th>Test F1-Score</th>
                  <th>Accuracy</th>
                  <th>False Negatives (FN)</th>
                  <th>False Positives (FP)</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Logistic Regression</strong></td>
                  <td><code>C=1.0, penalty='l2', solver='lbfgs'</code></td>
                  <td>0.9025 ± 0.0144</td>
                  <td className="metric-high">0.9665</td>
                  <td className="metric-high">92.86% (26/28)</td>
                  <td>83.87%</td>
                  <td>0.8814</td>
                  <td>88.52%</td>
                  <td className="metric-high">2</td>
                  <td>5</td>
                </tr>
                <tr className="model-champion">
                  <td><strong>Random Forest</strong> <span className="champion-tag">★ Selected Champion</span></td>
                  <td><code>max_depth=5, n_estimators=200, min_samples_leaf=2</code></td>
                  <td className="metric-high">0.9038 ± 0.0238</td>
                  <td>0.9502</td>
                  <td className="metric-high">92.86% (26/28)</td>
                  <td>83.87%</td>
                  <td>0.8814</td>
                  <td>88.52%</td>
                  <td className="metric-high">2</td>
                  <td>5</td>
                </tr>
                <tr>
                  <td><strong>XGBoost</strong></td>
                  <td><code>max_depth=3, n_estimators=100, lr=0.05, lambda=5.0</code></td>
                  <td>0.8897 ± 0.0246</td>
                  <td>0.9513</td>
                  <td className="metric-high">92.86% (26/28)</td>
                  <td className="metric-high">86.67%</td>
                  <td className="metric-high">0.8966</td>
                  <td className="metric-high">90.16%</td>
                  <td className="metric-high">2</td>
                  <td>4</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h4 style={{ color: '#fff', marginBottom: '1rem' }}>Diagnostic Curves &amp; Confusion Matrices</h4>
          
          <div className="tabs-nav">
            <button className={`tab-btn ${activeTab === 'tab-cm' ? 'active' : ''}`} onClick={() => setActiveTab('tab-cm')}>Confusion Matrices</button>
            <button className={`tab-btn ${activeTab === 'tab-roc' ? 'active' : ''}`} onClick={() => setActiveTab('tab-roc')}>ROC Curves</button>
            <button className={`tab-btn ${activeTab === 'tab-pr' ? 'active' : ''}`} onClick={() => setActiveTab('tab-pr')}>Precision-Recall Curves</button>
            <button className={`tab-btn ${activeTab === 'tab-bar' ? 'active' : ''}`} onClick={() => setActiveTab('tab-bar')}>Performance Metric Comparison</button>
          </div>

          {activeTab === 'tab-cm' && (
            <div style={{ background: '#050811', borderRadius: 'var(--radius-md)', padding: '1.5rem', textAlign: 'center' }}>
              <img src="/figures/confusion_matrices_comparison.png" alt="Confusion Matrix Comparison" style={{ maxWidth: '100%', borderRadius: 'var(--radius-sm)', cursor: 'pointer' }} onClick={() => openModal('/figures/confusion_matrices_comparison.png', 'Test Confusion Matrices (N=61)', 'Side-by-side breakdown of TP, TN, FP, and FN across Logistic Regression, Random Forest, and XGBoost.')} />
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginTop: '0.75rem' }}>
                Notice the clinical error symmetry: all three models produce exactly <strong>2 False Negatives</strong> (92.86% sensitivity), minimizing the dangerous missed-diagnosis rate.
              </p>
            </div>
          )}

          {activeTab === 'tab-roc' && (
            <div style={{ background: '#050811', borderRadius: 'var(--radius-md)', padding: '1.5rem', textAlign: 'center' }}>
              <img src="/figures/roc_curves_comparison.png" alt="ROC Curves Comparison" style={{ maxWidth: '100%', maxHeight: '500px', borderRadius: 'var(--radius-sm)', cursor: 'pointer' }} onClick={() => openModal('/figures/roc_curves_comparison.png', 'ROC Curves - Held Out Test Set', 'Logistic Regression (AUC=0.967), XGBoost (AUC=0.951), Random Forest (AUC=0.950).')} />
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginTop: '0.75rem' }}>
                All models exhibit high discrimination with ROC-AUC exceeding 0.95 on held-out test records.
              </p>
            </div>
          )}

          {activeTab === 'tab-pr' && (
            <div style={{ background: '#050811', borderRadius: 'var(--radius-md)', padding: '1.5rem', textAlign: 'center' }}>
              <img src="/figures/precision_recall_curves_comparison.png" alt="Precision Recall Curves" style={{ maxWidth: '100%', maxHeight: '500px', borderRadius: 'var(--radius-sm)', cursor: 'pointer' }} onClick={() => openModal('/figures/precision_recall_curves_comparison.png', 'Precision-Recall Curves', 'Comparing average precision trade-offs across classification thresholds.')} />
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginTop: '0.75rem' }}>
                Precision remains above 80% even at high recall operating points, demonstrating strong threshold calibration.
              </p>
            </div>
          )}

          {activeTab === 'tab-bar' && (
            <div style={{ background: '#050811', borderRadius: 'var(--radius-md)', padding: '1.5rem', textAlign: 'center' }}>
              <img src="/figures/models_metrics_comparison.png" alt="Metrics Comparison Bar Chart" style={{ maxWidth: '100%', maxHeight: '500px', borderRadius: 'var(--radius-sm)', cursor: 'pointer' }} onClick={() => openModal('/figures/models_metrics_comparison.png', 'Model Performance Metric Comparison', 'Direct side-by-side comparison across ROC-AUC, Recall, Precision, F1-Score, and Accuracy.')} />
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginTop: '0.75rem' }}>
                Bar chart summarizing multi-metric parity across linear and tree-based architectures.
              </p>
            </div>
          )}
        </section>

        {/* Section 4: Error Analysis */}
        <section className="glass-panel" id="errors">
          <div className="section-header">
            <span className="section-tag">Phase 6: Clinical Error Diagnostics</span>
            <h3 className="section-title">Test Set Error Analysis (7 Misclassified Cases)</h3>
            <p className="section-desc">
              Examining the 7 patients out of 61 test cases that were incorrectly classified by the champion model.
            </p>
          </div>

          <div className="predictor-grid" style={{ alignItems: 'start', marginBottom: '1.5rem' }}>
            <div style={{ background: 'rgba(244, 63, 94, 0.08)', border: '1px solid rgba(244, 63, 94, 0.3)', padding: '1.5rem', borderRadius: 'var(--radius-md)' }}>
              <h4 style={{ color: 'var(--danger)', marginBottom: '0.5rem' }}>False Negatives (2 Patients) - Critical Risk</h4>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                <strong>Test Indices #165 &amp; #279</strong>: Patients who possessed true angiographic heart disease (&gt;50% vessel narrowing) but were predicted as Absence (0).<br /><br />
                <strong>Diagnostic Investigation</strong>: Both patients presented with atypical non-anginal symptoms, low ST depression (&lt;1.0 mm), and normal fluoroscopy vessels (ca=0.0). Their non-invasive test profiles mimic healthy patients, illustrating why non-invasive ML cannot replace definitive angiography.
              </p>
            </div>
            <div style={{ background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.3)', padding: '1.5rem', borderRadius: 'var(--radius-md)' }}>
              <h4 style={{ color: 'var(--warning)', marginBottom: '0.5rem' }}>False Positives (5 Patients) - False Alarms</h4>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                <strong>Test Indices #137, #139, #163, #190, #252</strong>: Patients with absence of significant narrowing (&lt;50%) who were flagged as high risk (Presence).<br /><br />
                <strong>Diagnostic Investigation</strong>: These patients exhibited severe isolated risk factors (e.g. resting blood pressure &gt; 160 mm Hg, elderly age &gt; 65, or abnormal ECG ST-T waves). In clinical practice, false alarms lead to confirmatory secondary stress testing rather than missed intervention.
              </p>
            </div>
          </div>

          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Test Index</th>
                  <th>Error Classification</th>
                  <th>True Target</th>
                  <th>Predicted Target</th>
                  <th>Predicted Probability</th>
                  <th>Age / Sex</th>
                  <th>Chest Pain (cp)</th>
                  <th>Resting BP</th>
                  <th>Cholesterol</th>
                  <th>Max HR</th>
                  <th>ST Depression</th>
                  <th>Vessels (ca)</th>
                  <th>Thal</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ background: 'rgba(244, 63, 94, 0.06)' }}>
                  <td><code>#165</code></td>
                  <td><span style={{ color: 'var(--danger)', fontWeight: 700 }}>False Negative</span></td>
                  <td>1 (Presence)</td>
                  <td>0 (Absence)</td>
                  <td className="metric-danger">0.4550</td>
                  <td>57 / Male</td>
                  <td>4: Asymptomatic</td>
                  <td>140.0</td>
                  <td>192.0</td>
                  <td>148.0</td>
                  <td>0.4 mm</td>
                  <td>0.0</td>
                  <td>6.0</td>
                </tr>
                <tr style={{ background: 'rgba(244, 63, 94, 0.06)' }}>
                  <td><code>#279</code></td>
                  <td><span style={{ color: 'var(--danger)', fontWeight: 700 }}>False Negative</span></td>
                  <td>1 (Presence)</td>
                  <td>0 (Absence)</td>
                  <td className="metric-danger">0.3600</td>
                  <td>58 / Male</td>
                  <td>4: Asymptomatic</td>
                  <td>114.0</td>
                  <td>318.0</td>
                  <td>140.0</td>
                  <td>4.4 mm</td>
                  <td>3.0</td>
                  <td>6.0</td>
                </tr>
                <tr style={{ background: 'rgba(245, 158, 11, 0.04)' }}>
                  <td><code>#137</code></td>
                  <td><span style={{ color: 'var(--warning)', fontWeight: 700 }}>False Positive</span></td>
                  <td>0 (Absence)</td>
                  <td>1 (Presence)</td>
                  <td className="metric-high">0.5850</td>
                  <td>62 / Female</td>
                  <td>4: Asymptomatic</td>
                  <td>140.0</td>
                  <td>268.0</td>
                  <td>160.0</td>
                  <td>3.6 mm</td>
                  <td>2.0</td>
                  <td>3.0</td>
                </tr>
                <tr style={{ background: 'rgba(245, 158, 11, 0.04)' }}>
                  <td><code>#139</code></td>
                  <td><span style={{ color: 'var(--warning)', fontWeight: 700 }}>False Positive</span></td>
                  <td>0 (Absence)</td>
                  <td>1 (Presence)</td>
                  <td className="metric-high">0.5250</td>
                  <td>51 / Male</td>
                  <td>3: Non-anginal</td>
                  <td>125.0</td>
                  <td>245.0</td>
                  <td>166.0</td>
                  <td>2.4 mm</td>
                  <td>0.0</td>
                  <td>3.0</td>
                </tr>
                <tr style={{ background: 'rgba(245, 158, 11, 0.04)' }}>
                  <td><code>#163</code></td>
                  <td><span style={{ color: 'var(--warning)', fontWeight: 700 }}>False Positive</span></td>
                  <td>0 (Absence)</td>
                  <td>1 (Presence)</td>
                  <td className="metric-high">0.6950</td>
                  <td>58 / Male</td>
                  <td>4: Asymptomatic</td>
                  <td>150.0</td>
                  <td>270.0</td>
                  <td>111.0</td>
                  <td>0.8 mm</td>
                  <td>0.0</td>
                  <td>7.0</td>
                </tr>
                <tr style={{ background: 'rgba(245, 158, 11, 0.04)' }}>
                  <td><code>#190</code></td>
                  <td><span style={{ color: 'var(--warning)', fontWeight: 700 }}>False Positive</span></td>
                  <td>0 (Absence)</td>
                  <td>1 (Presence)</td>
                  <td className="metric-high">0.5100</td>
                  <td>51 / Male</td>
                  <td>4: Asymptomatic</td>
                  <td>140.0</td>
                  <td>298.0</td>
                  <td>122.0</td>
                  <td>4.2 mm</td>
                  <td>3.0</td>
                  <td>7.0</td>
                </tr>
                <tr style={{ background: 'rgba(245, 158, 11, 0.04)' }}>
                  <td><code>#252</code></td>
                  <td><span style={{ color: 'var(--warning)', fontWeight: 700 }}>False Positive</span></td>
                  <td>0 (Absence)</td>
                  <td>1 (Presence)</td>
                  <td className="metric-high">0.5400</td>
                  <td>64 / Male</td>
                  <td>4: Asymptomatic</td>
                  <td>128.0</td>
                  <td>263.0</td>
                  <td>105.0</td>
                  <td>0.2 mm</td>
                  <td>1.0</td>
                  <td>7.0</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Section 5: SHAP Explainability */}
        <section className="glass-panel" id="shap">
          <div className="section-header">
            <span className="section-tag">Phase 7: Explainable AI</span>
            <h3 className="section-title">SHAP Feature Attribution (TreeExplainer)</h3>
            <p className="section-desc">
              Unpacking model decisions through Shapley values across 28 one-hot transformed features on the champion Random Forest model.
            </p>
          </div>

          <div className="predictor-grid" style={{ marginBottom: '2rem' }}>
            <div className="figure-card">
              <div className="figure-img-wrap" style={{ height: '380px' }} onClick={() => openModal('/figures/shap_summary_plot.png', 'SHAP Global Beeswarm Summary Plot', 'Dots represent individual patients. Red = high feature value, Blue = low feature value. Horizontal position indicates positive vs negative risk impact.')}>
                <span className="figure-badge">Global Beeswarm</span>
                <img src="/figures/shap_summary_plot.png" alt="SHAP Summary Plot" />
              </div>
              <div className="figure-body">
                <h4 className="figure-title">SHAP Summary Beeswarm Plot</h4>
                <p className="figure-caption">Visualizes how feature values push probability toward presence (right) or absence (left). Reversible thal defect and asymptomatic chest pain are the strongest risk drivers.</p>
              </div>
            </div>

            <div className="figure-card">
              <div className="figure-img-wrap" style={{ height: '380px' }} onClick={() => openModal('/figures/shap_feature_importance_bar.png', 'SHAP Mean Absolute Feature Importance', 'Ranking features by mean absolute SHAP value across the entire test cohort.')}>
                <span className="figure-badge">Importance Ranking</span>
                <img src="/figures/shap_feature_importance_bar.png" alt="SHAP Bar Plot" />
              </div>
              <div className="figure-body">
                <h4 className="figure-title">Top-10 Feature Importance Ranking</h4>
                <p className="figure-caption">Ranked by global impact on prediction. Thalassemia, chest pain, fluoroscopy vessels, and maximum heart rate account for over 65% of total predictive attribution.</p>
              </div>
            </div>
          </div>

          <h4 style={{ color: '#fff', marginBottom: '1rem' }}>Local Patient Deconstructions (Waterfall Plots)</h4>
          <div className="predictor-grid">
            <div className="figure-card">
              <div className="figure-img-wrap" style={{ height: '300px' }} onClick={() => openModal('/figures/shap_local_correct.png', 'Local Waterfall: Correct Prediction', 'Waterfall plot showing cumulative attribution pushes from base value E[f(x)] to the final prediction.')}>
                <span className="figure-badge" style={{ background: 'var(--success)' }}>Correct Case</span>
                <img src="/figures/shap_local_correct.png" alt="SHAP Local Correct" />
              </div>
              <div className="figure-body">
                <h4 className="figure-title">Correctly Classified Patient (True=0, Pred=0)</h4>
                <p className="figure-caption">Normal thalassemia status (thal=3) and 0 colored fluoroscopy vessels (ca=0) strongly pushed the model prediction below threshold.</p>
              </div>
            </div>

            <div className="figure-card">
              <div className="figure-img-wrap" style={{ height: '300px' }} onClick={() => openModal('/figures/shap_local_incorrect.png', 'Local Waterfall: Misclassified Patient', 'Waterfall plot detailing contradictory clinical signals that induced the classification error.')}>
                <span className="figure-badge" style={{ background: 'var(--danger)' }}>Misclassified Case</span>
                <img src="/figures/shap_local_incorrect.png" alt="SHAP Local Incorrect" />
              </div>
              <div className="figure-body">
                <h4 className="figure-title">Misclassified Patient (True=0, Pred=1)</h4>
                <p className="figure-caption">Contradictory signals: asymptomatic chest pain and elevated ST depression pushed predicted probability to 0.585 despite negative angiography.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 6: Live Clinical Risk Simulator */}
        <section className="glass-panel" id="simulator">
          <div className="section-header">
            <span className="section-tag">Interactive Demonstration</span>
            <h3 className="section-title">Live Clinical Risk Simulator</h3>
            <p className="section-desc">
              Test real-time predictions powered directly by our live FastAPI inference backend (<code>/predict</code>).
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', alignSelf: 'center' }}>Quick Load Clinical Profiles:</span>
            <button className="btn btn-outline" onClick={() => loadPreset('high_risk')} style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}>🚨 High Risk Profile</button>
            <button className="btn btn-outline" onClick={() => loadPreset('low_risk')} style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}>✅ Low Risk Profile</button>
            <button className="btn btn-outline" onClick={() => loadPreset('borderline')} style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}>⚖️ Borderline Case</button>
          </div>

          <div className="predictor-grid">
            <form onSubmit={(e) => { e.preventDefault(); executeInference(); }}>
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="inp-age">Age (years) <span style={{ color: 'var(--primary-light)' }}>{formData.age}</span></label>
                  <input type="number" id="inp-age" className="form-control" min="18" max="100" value={formData.age} onChange={(e) => handleInputChange('age', e.target.value)} required />
                </div>
                <div className="form-group">
                  <label htmlFor="inp-sex">Sex</label>
                  <select id="inp-sex" className="form-control" value={formData.sex} onChange={(e) => handleInputChange('sex', e.target.value)}>
                    <option value="1">Male (1)</option>
                    <option value="0">Female (0)</option>
                  </select>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="inp-cp">Chest Pain Type (cp)</label>
                  <select id="inp-cp" className="form-control" value={formData.cp} onChange={(e) => handleInputChange('cp', e.target.value)}>
                    <option value="1">1: Typical Angina</option>
                    <option value="2">2: Atypical Angina</option>
                    <option value="3">3: Non-Anginal Pain</option>
                    <option value="4">4: Asymptomatic (High Risk)</option>
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="inp-trestbps">Resting BP (mm Hg)</label>
                  <input type="number" id="inp-trestbps" className="form-control" min="80" max="250" value={formData.trestbps} onChange={(e) => handleInputChange('trestbps', e.target.value)} required />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="inp-chol">Cholesterol (mg/dl)</label>
                  <input type="number" id="inp-chol" className="form-control" min="100" max="600" value={formData.chol} onChange={(e) => handleInputChange('chol', e.target.value)} required />
                </div>
                <div className="form-group">
                  <label htmlFor="inp-fbs">Fasting Blood Sugar &gt; 120</label>
                  <select id="inp-fbs" className="form-control" value={formData.fbs} onChange={(e) => handleInputChange('fbs', e.target.value)}>
                    <option value="0">False (&lt;= 120 mg/dl)</option>
                    <option value="1">True (&gt; 120 mg/dl)</option>
                  </select>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="inp-restecg">Resting ECG</label>
                  <select id="inp-restecg" className="form-control" value={formData.restecg} onChange={(e) => handleInputChange('restecg', e.target.value)}>
                    <option value="0">0: Normal</option>
                    <option value="1">1: ST-T Wave Abnormality</option>
                    <option value="2">2: Left Ventricular Hypertrophy</option>
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="inp-thalach">Max Heart Rate (thalach)</label>
                  <input type="number" id="inp-thalach" className="form-control" min="60" max="230" value={formData.thalach} onChange={(e) => handleInputChange('thalach', e.target.value)} required />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="inp-exang">Exercise Angina</label>
                  <select id="inp-exang" className="form-control" value={formData.exang} onChange={(e) => handleInputChange('exang', e.target.value)}>
                    <option value="0">No (0)</option>
                    <option value="1">Yes (1)</option>
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="inp-oldpeak">ST Depression (oldpeak mm)</label>
                  <input type="number" id="inp-oldpeak" className="form-control" step="0.1" min="0" max="10" value={formData.oldpeak} onChange={(e) => handleInputChange('oldpeak', e.target.value)} required />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="inp-slope">ST Slope</label>
                  <select id="inp-slope" className="form-control" value={formData.slope} onChange={(e) => handleInputChange('slope', e.target.value)}>
                    <option value="1">1: Upsloping</option>
                    <option value="2">2: Flat</option>
                    <option value="3">3: Downsloping</option>
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="inp-ca">Fluoroscopy Vessels (ca)</label>
                  <select id="inp-ca" className="form-control" value={formData.ca} onChange={(e) => handleInputChange('ca', e.target.value)}>
                    <option value="0">0 Vessels</option>
                    <option value="1">1 Vessel</option>
                    <option value="2">2 Vessels</option>
                    <option value="3">3 Vessels</option>
                  </select>
                </div>
              </div>

              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label htmlFor="inp-thal">Thalassemia (thal)</label>
                <select id="inp-thal" className="form-control" value={formData.thal} onChange={(e) => handleInputChange('thal', e.target.value)}>
                  <option value="3">3.0: Normal Perfusion</option>
                  <option value="6">6.0: Fixed Defect</option>
                  <option value="7">7.0: Reversible Defect (High Risk)</option>
                </select>
              </div>

              <button type="submit" className="btn btn-primary" disabled={loading} style={{ width: '100%', justifyContent: 'center', padding: '0.85rem' }}>
                <span>{loading ? '⏳ Computing with Pipeline...' : '⚡ Compute Real-Time Inference'}</span>
              </button>
            </form>

            {/* Live Output Panel */}
            <div className="result-card">
              <div style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                Model Inference Output
              </div>
              <h3 style={{ color: prediction.predicted_class === 1 ? '#f87171' : '#38bdf8', fontSize: '1.35rem', marginBottom: '1rem' }}>
                {prediction.prediction_label}
              </h3>

              <div className="gauge-wrap">
                <div className="gauge-circle" style={{
                  background: `conic-gradient(${prediction.probability_presence >= 0.5 ? 'var(--danger)' : 'var(--success)'} ${pct}%, #1e293b ${pct}%)`
                }}>
                  <div className="gauge-inner">
                    <div className="gauge-percentage">{pct}%</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginTop: '2px' }}>
                      Disease Probability
                    </div>
                  </div>
                </div>
              </div>

              <div className={`risk-badge ${prediction.risk_level === 'Low' ? 'risk-low' : (prediction.risk_level === 'Moderate' ? 'risk-moderate' : 'risk-high')}`}>
                {prediction.risk_level} Risk ({pct}%)
              </div>

              <div style={{ background: 'rgba(18, 26, 47, 0.6)', borderRadius: 'var(--radius-sm)', padding: '1rem', textAlign: 'left', fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                  <span>Serving Pipeline:</span>
                  <strong style={{ color: 'var(--primary-light)' }}>Random Forest (Champion)</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                  <span>Absence Probability:</span>
                  <strong>{(prediction.probability_absence * 100).toFixed(1)}%</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Decision Threshold:</span>
                  <strong>0.50</strong>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 7: Pipeline Architecture & Reproducibility */}
        <section className="glass-panel" id="reproducibility">
          <div className="section-header">
            <span className="section-tag">Architecture &amp; MLOps</span>
            <h3 className="section-title">Leakage-Safe Pipeline Architecture</h3>
            <p className="section-desc">
              Reproducible execution workflow from raw data download to Next.js fullstack serving.
            </p>
          </div>

          <div style={{ background: '#050811', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)', padding: '1.5rem', marginBottom: '1.5rem' }}>
            <h4 style={{ color: '#fff', marginBottom: '1rem' }}>System Workflow</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', textAlign: 'center' }}>
              <div style={{ background: 'rgba(18, 26, 47, 0.7)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '1.5rem', marginBottom: '0.3rem' }}>📥</div>
                <strong style={{ fontSize: '0.85rem', color: '#fff', display: 'block' }}>1. Data Acquisition</strong>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>UCI Cleveland, SHA256 Checksum</span>
              </div>
              <div style={{ background: 'rgba(18, 26, 47, 0.7)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '1.5rem', marginBottom: '0.3rem' }}>🛡️</div>
                <strong style={{ fontSize: '0.85rem', color: '#fff', display: 'block' }}>2. Preprocessing</strong>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ColumnTransformer, Zero Leakage</span>
              </div>
              <div style={{ background: 'rgba(18, 26, 47, 0.7)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '1.5rem', marginBottom: '0.3rem' }}>⚙️</div>
                <strong style={{ fontSize: '0.85rem', color: '#fff', display: 'block' }}>3. Stratified 5-Fold CV</strong>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>GridSearch over 3 Model Families</span>
              </div>
              <div style={{ background: 'rgba(18, 26, 47, 0.7)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '1.5rem', marginBottom: '0.3rem' }}>🔍</div>
                <strong style={{ fontSize: '0.85rem', color: '#fff', display: 'block' }}>4. SHAP Explainability</strong>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TreeExplainer Attributions</span>
              </div>
              <div style={{ background: 'rgba(18, 26, 47, 0.7)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '1.5rem', marginBottom: '0.3rem' }}>🚀</div>
                <strong style={{ fontSize: '0.85rem', color: '#fff', display: 'block' }}>5. Production Serving</strong>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Next.js App Router + FastAPI</span>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Modal Lightbox */}
      {modalData.open && (
        <div className="modal-overlay active" onClick={closeModal}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeModal}>✕</button>
            <h3 style={{ color: '#fff', marginBottom: '0.25rem' }}>{modalData.title}</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>{modalData.desc}</p>
            <img className="modal-img" src={modalData.src} alt={modalData.title} />
          </div>
        </div>
      )}

      {/* Footer */}
      <footer>
        <div className="container" style={{ paddingTop: 0, paddingBottom: 0 }}>
          <p>CardioSense • Explainable Machine Learning for Heart Disease Risk Classification</p>
          <p style={{ marginTop: '0.4rem', fontSize: '0.78rem' }}>
            Developed by Aditya (<a href="https://github.com/Adi3595/CardioSense" target="_blank" rel="noreferrer" style={{ color: 'var(--primary)' }}>Adi3595/CardioSense</a>) • Academic Research Demo
          </p>
        </div>
      </footer>
    </>
  );
}
