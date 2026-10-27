# Dubstrata Institutional Quantitative Evaluation Kit (`v3.2`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuzikayiseKhuzwayo/dubstrata-quant-eval-kit/blob/main/quant-evaluation-kit.ipynb)

> **Target Audience:** Systematic Quantitative Researchers, Portfolio Managers, Risk Engineers, and Autonomous AI Assistants (Claude Code, Cursor, Windsurf, Copilot).  
> **Objective:** Deterministic, zero-dependency evaluation of Dubstrata's **16-Channel Multivariate Causal Tensor Stream ("The Fuel")**, **Sovereign Causal Foundation Model ("The Engine")**, and **Horizon-Resolved Attribution ("HRX Glass Box")**.  
> **Portability Guarantee:** Zero local repository dependencies. Connects directly over TLS to the live Dubstrata production gateway (`https://api.dubstrata.com`) with in-memory Apache Parquet deserialization.

---

## 📦 Repository Contents & Technical Dossiers

| File | Description |
| :--- | :--- |
| **[`quant-evaluation-kit.ipynb`](./quant-evaluation-kit.ipynb)** | **Interactive Evaluation Notebook:** Pulls a live 4,096-bar (14.2-day) 5-minute Parquet partition into memory, verifies timestamp monotonicity ($t_i < t_{i+1}$) and fractional differentiation stationarity ($d^* = 0.42$), queries 9-quantile forward distributions (`/api/v1/predict/quantiles`), and audits 4-quadrant HRX attribution. |
| **[`DUBSTRATA_16_CHANNEL_TENSOR_SCHEMA_SPEC.md`](./DUBSTRATA_16_CHANNEL_TENSOR_SCHEMA_SPEC.md)** | **16-Channel Schema & Mathematical Specification:** Full data dictionary for $\mathbf{X}(t) \in \mathbb{R}^{16 \times 512}$ across **Q1 (Microstructure & Derivatives)**, **Q2 (Sybil-Scrubbed Prediction Markets)**, **Q3 (Causal Knowledge Graph)**, and **Q4 (Macro Solvency & Event Clocks)**. |
| **[`DUBSTRATA_DATA_VERIFICATION_METHODOLOGY.md`](./DUBSTRATA_DATA_VERIFICATION_METHODOLOGY.md)** | **Four-Layer Verification Methodology:** Documents Bi-Temporal Point-in-Time Stamping ($t_{\text{valid}} \le t_{\text{event}} \le t_{\text{ingest}}$), Polygon ERC-1155 CTF Disjoint-Set Union (DSU) Sybil wallet contraction ($HHI_{\text{clean}}$), Shannon entropy conviction ($\Psi_{\text{CSI}}$), and byte-span grounded claim extraction. |
| **[`DUBSTRATA_FORWARD_EVALUATION_CASE_STUDIES.md`](./DUBSTRATA_FORWARD_EVALUATION_CASE_STUDIES.md)** | **Out-of-Sample Empirical Case Studies:** Production out-of-sample verification (`68.46%` 1h DHR, `+19.56 bps` HRX deletion gain) and real-world macro/prediction-market divergence studies. |
| **[`DUBSTRATA_SIGNAL_TO_DECISION_OBSERVABILITY_GUIDE.md`](./DUBSTRATA_SIGNAL_TO_DECISION_OBSERVABILITY_GUIDE.md)** | **Signal-to-Decision Observability Trace:** End-to-end walkthrough tracing how raw orderbook toxicity (OFI/VPIN), de-Sybiled prediction market flow, and causal graph states propagate into conviction-gated risk brackets ($q_{10}$ VaR-95 stop / $q_{90}$ target). |

---

## 🚀 Quick Start (Local Jupyter / VS Code / Cursor)

### 1. Clone & Install the 5 Standard PyData Libraries

#### macOS / Linux / Bash
```bash
git clone https://github.com/MuzikayiseKhuzwayo/dubstrata-quant-eval-kit.git
cd dubstrata-quant-eval-kit

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests pandas numpy matplotlib pyarrow jupyter
```

#### Windows (PowerShell)
```powershell
git clone https://github.com/MuzikayiseKhuzwayo/dubstrata-quant-eval-kit.git
Set-Location dubstrata-quant-eval-kit

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install requests pandas numpy matplotlib pyarrow jupyter
```

### 2. Configure Your Sandbox API Key

Set your provisioned `DUBSTRATA_API_KEY` either as an environment variable or directly in **Cell 1** of [`quant-evaluation-kit.ipynb`](./quant-evaluation-kit.ipynb):

```bash
export DUBSTRATA_API_BASE="https://api.dubstrata.com"
export DUBSTRATA_API_KEY="your_provisioned_api_key_here"
export ASSET_SYMBOL="BTC/USD"
```

### 3. Run the Notebook

Open [`quant-evaluation-kit.ipynb`](./quant-evaluation-kit.ipynb) in Jupyter, VS Code, Google Colab, or Kaggle (ensure **Internet** is enabled in Kaggle/Colab) and select **Run All Cells**.

---

## 🔬 Overview of the 16-Channel Stationary Tensor ($\mathbb{R}^{16 \times 512}$)

| Quadrant | Channels | Core Features & Mathematical Formulation |
| :--- | :---: | :--- |
| **Q1: Microstructure & Derivatives** | `Ch 01 – 04` | • `target_fracdiff_price`: Stationary log-price ($d^* = 0.42$, $K = 1,332$ lags, ADF $p < 0.005$, Pearson $r > 0.90$)<br>• `target_log_volatility`: Rolling Yang-Zhang / Garman-Klass realized log-volatility<br>• `microstructure_ofi_vpin`: Multi-level CLOB Order Flow Imbalance scaled by VPIN toxicity<br>• `derivatives_surface_state`: 25-delta options risk reversal skew & dealer Gamma Exposure (GEX) |
| **Q2: Scrubbed Prediction Markets** | `Ch 05 – 08` | • `informed_market_prob`: Decoupled Polymarket & Kalshi event consensus ($\beta \equiv 0$)<br>• `entropy_conviction_psi`: Shannon entropy conviction index ($\Psi_{\text{CSI}}$) scaled by Kyle's $\lambda$<br>• `manipulation_haircut_mu`: On-chain Polygon CTF DSU Sybil cluster & wash-trading penalty ($\mu_{\text{risk}}$)<br>• `informed_capital_imbalance`: KL divergence between top-decile Brier-rated wallets and retail flow |
| **Q3: Causal Knowledge Graph** | `Ch 09 – 12` | • `friedkin_johnsen_belief`: Multi-hop network equilibrium belief state ($Y_i(t)$)<br>• `financial_impact_flow`: Decoupled CRH financial impact flow ($W_s$) over `:CAUSALLY_EXPOSED` edges<br>• `perception_vector_drift`: 3072-d semantic embedding cosine velocity across regulatory filings<br>• `graph_conflict_entropy`: Dual-agent claim contradiction Shannon entropy |
| **Q4: Solvency & Macro Regimes** | `Ch 13 – 16` | • `fundamental_solvency_score`: Point-in-time balance sheet reserve & credit spread health<br>• `dynamic_primary_correlate`: Highest instantaneous Granger/graph-elasticity cross-asset correlate<br>• `macro_liquidity_factor`: Global central bank & sovereign yield curve PC1 liquidity factor<br>• `scheduled_event_clock`: Deterministic exponential countdown ($\tau_{\text{event}}$) to FOMC, CPI, NFP & OpEx |

---

## ✅ Pass/Fail Acceptance Criteria Matrix

When running [`quant-evaluation-kit.ipynb`](./quant-evaluation-kit.ipynb), the notebook automatically verifies the following institutional invariants:

| Check ID | Product Offering | Target Metric | Assertion Condition | Expected Result |
| :--- | :--- | :--- | :--- | :---: |
| **A-01** | Offering A (Fuel) | Canonical Feature Channels | All 16 stationary channels present in Parquet slice | **PASS** |
| **A-02** | Offering A (Fuel) | Timestamp Monotonicity | `timestamps.is_monotonic_increasing == True` ($t_v \le t_t$) | **PASS** |
| **A-03** | Offering A (Fuel) | Optimal FracDiff Stationarity | $d^* = 0.42$, zero leading-zero padding, variance $> 0$ | **PASS** |
| **B-01** | Offering B (Engine) | Directional Conviction Gate | $\Psi_{\text{CSI}} \in [0.0, 1.0]$ (Automated capital brake in chop) | **PASS** |
| **B-02** | Offering B (Engine) | 9-Quantile Monotonicity | $q_{10} \le q_{20} \le \dots \le q_{90}$ ($q_{10} = \text{VaR}_{95}$ support) | **PASS** |
| **B-03** | Offering B (Engine) | Multi-Horizon Uncertainty Cone | $q_{10}^{(h)} < q_{50}^{(h)} < q_{90}^{(h)}$ across $15\text{m}, 1\text{h}, 4\text{h}, 24\text{h}$ | **PASS** |
| **C-01** | Offering C (HRX) | 4-Quadrant Attribution | $Q_1 + Q_2 + Q_3 + Q_4 = 100\%$ | **PASS** |
| **C-02** | Offering C (HRX) | Deletion Gain Proof | $\Delta\text{Gain} = \mathcal{L}(\mathbf{X}_{\setminus\text{top}}) - \mathcal{L}(\mathbf{X}) > 0$ | **PASS** |

---

## 🛠️ Diagnostic & HTTP Status Reference

| HTTP Status | Meaning | Resolution |
| :---: | :--- | :--- |
| **`200 OK`** | Successful evaluation response | Proceed to next notebook cell. |
| **`401 Unauthorized`** | Missing or invalid API key | Verify `DUBSTRATA_API_KEY` in Cell 1 matches your provisioned bearer token. |
| **`402 Payment Required`** | Sandbox evaluation quota exhausted | Contact `muzi@dubstrata.com` to top up sandbox evaluation credits. |
| **`429 Too Many Requests`** | Rate limit throttled | Sandbox keys are capped at 30–300 req/min; add a 1s delay or request Tier-2 limits. |

---

## 📬 Technical Contact & Research Portal

* **Research & Whitepapers:** [https://dubstrata.com/research](https://dubstrata.com/research)
* **HRX Command Center Portal:** [https://partners.dubstrata.com](https://partners.dubstrata.com)
* **Lead Systems & Quant Contact:** `muzi@dubstrata.com`
