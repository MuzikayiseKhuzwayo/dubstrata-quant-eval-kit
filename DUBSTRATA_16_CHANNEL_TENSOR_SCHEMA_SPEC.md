# Dubstrata Institutional Quantitative Evaluation Kit: Technical Architecture & 16-Channel Tensor Schema

> **Document Identifier:** `DUB-SPEC-QUANT-SCHEMA-v3.0`  
> **Target Audience:** Chief Risk Officers (CRO), Heads of Quantitative Research, Systematic Portfolio Managers, Evaluation Engineers  
> **Companion Artifact:** [`notebooks/quant_evaluation_kit.ipynb`](../../../../notebooks/quant_evaluation_kit.ipynb)  
> **System Architecture:** 16-Channel Multivariate Causal Tensor Engine & Sovereign Causal Foundation Model (Causal-FM v2)  
> **Network Protocol:** Pure REST API over TLS (`https://api.dubstrata.com`) with Zero-Dependency In-Memory Parquet Deserialization  
> **Audit Status:** Production Certified & Verified (Zero-Lookahead Guaranteed)

---

## 1. Executive Summary & Purpose of the Evaluation Kit

Systematic hedge funds and institutional asset managers face a structural bottleneck when evaluating proprietary alpha or risk data: **data vendor lock-in, unverified lookahead bias, and complex multi-gigabyte dependency trees**.

The **Dubstrata Institutional Quantitative Evaluation Kit** (`quant_evaluation_kit.ipynb`) was engineered to solve this friction. It provides an institutional evaluation harness with **zero local repository dependencies**. An evaluation engineer or autonomous AI agent (e.g., Claude Code, Cursor, Windsurf) can launch the notebook directly in **Google Colab**, **Kaggle**, or an internal institutional Jupyter server with five standard PyData libraries (`requests`, `pandas`, `numpy`, `matplotlib`, `pyarrow`).

The notebook validates the three commercial tiers of Dubstrata's institutional risk infrastructure:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DUBSTRATA INSTITUTIONAL PRODUCT SUITE                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  [OFFERING A: THE FUEL]                                                                │
│  • 16-Channel Multivariate Stationary Causal Tensors                                   │
│  • High-throughput Parquet streams delivered via REST/S3/GCS                            │
│  • Bi-temporally indexed (t_valid <= t_event <= t_ingest) with fractional diff d*=0.42  │
│                                                                                        │
│  [OFFERING B: THE ENGINE]                                                              │
│  • Sovereign Causal Foundation Model (TimesFM 2D-Grid Attention Backbone)              │
│  • Multi-horizon parametric quantile forecasts (q10 to q90 across 15m, 1h, 4h, 24h)    │
│  • Calibrated Value-at-Risk (VaR-95) downside bounds & directional conviction (Psi)   │
│                                                                                        │
│  [OFFERING C: THE GLASS BOX (HRX TERMINAL)]                                            │
│  • Holistic Relevance eXplanation (HRX) 4-Quadrant Attribution Decomposition          │
│  • Deletion Gain Proof: Delta Gain = L(X_{-top}) - L(X) > 0 (+0.1956)                  │
│  • Microstructure, Prediction Markets, Causal Knowledge Graph, and Macro Solvency      │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Offering A: The 16-Channel Multivariate Causal Tensor Specification

Offering A exposes point-in-time, stationary tensor partitions at a standardized **5-minute bar frequency** ($L = 512$ lookback window). Unlike legacy pricing feeds that supply only raw OHLCV bars, Dubstrata tensors synthesize physical orderbook microstructure, decentralized prediction market dynamics, knowledge graph entity drift, and macroeconomic solvency.

### 2.1 Complete 16-Channel Data Dictionary

| Ch # | Canonical Column Identifier | Physical Units / Range | Primary Category | Mathematical Definition & Source Description |
| :---: | :--- | :---: | :--- | :--- |
| **01** | `target_fracdiff_price` | $\mathbb{R}$ (Mean $\approx 0.36$) | Target Price Series | **Fractionally Differentiated Price Series ($d^* = 0.42$):** Preserves long memory ($r > 0.90$ with raw price) while achieving ADF stationarity ($p < 0.01$). Expands across $K=1,332$ convolution lags: $(1 - B)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} B^k$. |
| **02** | `target_log_volatility` | $[0.01, 0.50]$ | Volatility Regime | **Rolling Yang-Zhang / Parkinson Log-Volatility:** Minimum-variance estimator capturing continuous diffusion and discrete jump variance over rolling 24-hour windows. |
| **03** | `microstructure_ofi_vpin` | $[-1.0, +1.0]$ | CLOB Microstructure | **Normalized Order Flow Imbalance (OFI) & VPIN:** Measures volume-synchronized probability of toxicity and cumulative bid-ask quote pressure across Top-10 orderbook levels. |
| **04** | `derivatives_surface_state` | $[-1.0, +1.0]$ | Derivatives Surface | **Options Volatility Skew & Gamma Exposure:** Normalized 25-delta Risk Reversal $(IV_{\text{25d Call}} - IV_{\text{25d Put}})$ combined with net dealer Gamma Exposure (GEX) inflection. |
| **05** | `informed_market_prob` | $[0.0, 1.0]$ | Prediction Markets | **Decoupled Prediction Market Implied Probability:** Real-time event consensus from Polymarket & Kalshi resolved via decoupled Ornstein-Uhlenbeck jump-diffusion ($\beta \equiv 0$ uncoupled). |
| **06** | `entropy_conviction_psi` | $[-5.0, +5.0]$ | Information Theory | **Shannon Directional Conviction Index ($\Psi_{\text{CSI}}$):** Information entropy drift scaled by capital velocity: $\Psi = \text{sign}(\Delta P) \cdot \frac{1 - H(p)}{H_{\max}} \cdot \ln(1 + \text{Vol})$. |
| **07** | `manipulation_haircut_mu` | $[0.05, 1.00]$ | Forensics / Risk | **Sybil & Wash-Trading Penalty Multiplier ($\mu_{\text{risk}}$):** Dynamic risk haircut derived from on-chain Disjoint-Set Union (DSU) wallet clustering and contracted Herfindahl-Hirschman Index ($HHI$). |
| **08** | `informed_capital_imbalance` | $[-1.0, +1.0]$ | Prediction Markets | **Smart Money Flow ($I_{\text{flow}}$):** Net capital delta originating from top-decile historical PnL and Brier-score rated wallets, isolating informed traders from retail noise. |
| **09** | `friedkin_johnsen_belief` | $[-1.0, +1.0]$ | Social / Agent Networks | **Friedkin-Johnsen Equilibrium Belief Vector:** Dynamic social network consensus tracking how stubborn institutional nodes and news entities anchor systemic market expectations. |
| **10** | `financial_impact_flow` | $[-2.0, +2.0]$ | Causal Knowledge Graph | **Directed Graph Exposure Flow:** Breadth-first PageRank diffusion through 3,639 `CAUSALLY_EXPOSED` edges in the live ArcadeDB knowledge graph, tracing credit and contagion risks. |
| **11** | `perception_vector_drift` | $[0.0, 1.0]$ | Semantic AI | **Semantic Vector Centroid Velocity:** Cosine drift velocity ($1 - \cos(\mathbf{e}_t, \mathbf{e}_{t-1})$) across high-dimensional embeddings of regulatory, court, and sovereign filings. |
| **12** | `graph_conflict_entropy` | $[0.0, 1.0]$ | Causal Knowledge Graph | **Graph Extraction Contradiction Entropy:** Shannon entropy across conflicting claim extractions evaluated by our dual-agent (Extractor vs. Skeptic) verification pipeline. |
| **13** | `fundamental_solvency_score` | $[0.0, 1.0]$ | Macro / Solvency | **Balance Sheet Solvency & Collateral Ratio:** Point-in-time scoring of reserve coverage, sovereign fiscal stress, and high-yield credit spreads. |
| **14** | `dynamic_primary_correlate` | $[-1.0, +1.0]$ | Cross-Asset Lead-Lag | **Highest Dynamic Granger-Causal Asset:** Real-time normalized return of the single highest causal cross-asset correlate (e.g., MSTR, QQQ, US10Y, or DXY). |
| **15** | `macro_liquidity_factor` | $[-3.0, +3.0]$ | Global Macro | **Global Liquidity Principal Component (PC1):** First principal component across Federal Reserve repo facilities, Treasury General Account (TGA) drawdowns, and global central bank balance sheets. |
| **16** | `scheduled_event_clock` | $[0.0, 1.0]$ | Event Microstructure | **Deterministic Macro Event Countdown ($\tau$):** Normalized time decay to scheduled macro announcements (FOMC, CPI, NFP, Options OpEx): $\tau = \exp(-\lambda \Delta t)$. |

### 2.2 Mathematical Stationarity & Memory Guarantee

A fundamental dilemma in quantitative finance is that integer differencing ($d=1$, log-returns) eliminates trend memory ($r \approx 0$), while raw prices ($d=0$) possess infinite memory but violate stationarity assumptions required by neural attention networks.

Dubstrata solves this via **fractional differentiation with memory conservation**:
$$\tilde{X}_t = \sum_{k=0}^{\infty} \omega_k X_{t-k}, \quad \text{where } \omega_0 = 1, \; \omega_k = -\omega_{k-1} \frac{d - k + 1}{k}$$
- **Optimal Parameter:** $d^* = 0.42 \pm 0.02$
- **Augmented Dickey-Fuller (ADF) Test:** $p\text{-value} < 0.005$ (Certified Stationary)
- **Memory Retention:** Pearson correlation with raw series $r(\tilde{X}, X) = 0.9412 > 0.90$
- **Memory Convolution Length:** $K = 1,332$ historical lags required for convergence ($\text{tol} = 10^{-5}$)

---

## 3. Offering B: Sovereign Causal-FM 9-Quantile Engine Schema

Offering B queries the live inference engine (`POST /api/v1/predict/quantiles` and `POST /api/v1/predict/distribution`). Instead of outputting an uncalibrated point estimate, Causal-FM uses a **TimesFM 2D-Grid Attention Backbone** with a quantile projection head emitting **9 parametric quantile bands** and distribution cones across multiple investment horizons.

### 3.1 Quantile Request Payload Schema

```json
{
  "asset_symbol": "BTC/USD",
  "horizon_minutes": 60,
  "include_quantiles": true,
  "confidence_level": 0.80
}
```

### 3.2 Quantile Response Schema (`/api/v1/predict/quantiles`)

```json
{
  "status": "OPERATIONAL_CERTIFIED",
  "asset_symbol": "BTC/USD",
  "horizon_minutes": 60,
  "timestamp_utc": "2026-09-22T04:45:00Z",
  "current_price": 82813.22,
  "point_forecast": 83161.04,
  "directional_conviction": 0.8245,
  "quantiles": {
    "q10": 82382.59,
    "q20": 82523.37,
    "q30": 82639.32,
    "q40": 82722.13,
    "q50": 82813.22,
    "q60": 82896.03,
    "q70": 82987.14,
    "q80": 83094.83,
    "q90": 83243.72
  },
  "manipulation_risk_multiplier": 1.00,
  "var_95_downside": 82382.59,
  "upside_resistance": 83243.72,
  "expected_shortfall_cvar": 82190.15
}
```

### 3.3 Multi-Horizon Quantile Cone Schema (`/api/v1/predict/distribution`)

Used to plot expanding fan charts (cones of uncertainty) over standardized institutional holding periods ($15\text{m}, 1\text{h}, 4\text{h}, 24\text{h}$):

```json
{
  "status": "OPERATIONAL_CERTIFIED",
  "asset_symbol": "BTC/USD",
  "current_price": 82813.22,
  "cone_of_uncertainty": [
    {
      "horizon_minutes": 15,
      "horizon_label": "15m",
      "lower_bound_q10": 82564.78,
      "median_q50": 82896.03,
      "upper_bound_q90": 83061.66
    },
    {
      "horizon_minutes": 60,
      "horizon_label": "1h",
      "lower_bound_q10": 82316.34,
      "median_q50": 83144.47,
      "upper_bound_q90": 83475.72
    },
    {
      "horizon_minutes": 240,
      "horizon_label": "4h",
      "lower_bound_q10": 81819.46,
      "median_q50": 83641.35,
      "upper_bound_q90": 84303.85
    },
    {
      "horizon_minutes": 1440,
      "horizon_label": "24h",
      "lower_bound_q10": 80742.88,
      "median_q50": 84883.55,
      "upper_bound_q90": 86291.37
    }
  ]
}
```

---

## 4. Offering C: Horizon-Resolved Attribution (HRX) & Deletion Gain

Institutional compliance and risk committees prohibit "black-box" models. **Offering C** solves this with our **Holistic Relevance eXplanation (HRX)** engine.

### 4.1 4-Quadrant Decomposition

Every forecast $\hat{y}$ is decomposed into four observable, economically grounded quadrants summing to $100\%$:

```
┌────────────────────────────────────────┬────────────────────────────────────────┐
│  QUADRANT 1: Microstructure Dynamics   │  QUADRANT 2: Autonomous Prediction Mkt │
│  • Order Flow Imbalance (OFI)          │  • Polymarket / Kalshi CLOB Skew       │
│  • Toxicity (VPIN) & Quote Depth       │  • Informed Capital Imbalance (I_flow) │
│  • Derivatives Volatility Skew (GEX)   │  • Sybil Wash-Trading Scrubber         │
├────────────────────────────────────────┼────────────────────────────────────────┤
│  QUADRANT 3: Causal Knowledge Graph    │  QUADRANT 4: Macro Solvency & Regimes  │
│  • ArcadeDB 3,639 Entity Edges         │  • Global Liquidity Factor (PC1)       │
│  • Friedkin-Johnsen Belief Equilibria  │  • Real-World Asset (RWA) Elasticities │
│  • Dual-Agent Triplet Verification     │  • Options Expiration Event Clock      │
└────────────────────────────────────────┴────────────────────────────────────────┘
```

### 4.2 HRX Attribution Response Schema (`/api/v1/forecast/latest-attribution`)

```json
{
  "asset_symbol": "BTC/USD",
  "horizon_minutes": 60,
  "attribution_breakdown": {
    "Q1_Microstructure": 38.5,
    "Q2_PredictionMarkets": 29.5,
    "Q3_CausalKnowledgeGraph": 21.0,
    "Q4_MacroSolvency": 11.0
  },
  "top_feature_drivers": [
    {"channel": "target_fracdiff_price", "rank": 1, "weight_pct": 32.1},
    {"channel": "informed_capital_imbalance", "rank": 2, "weight_pct": 18.4},
    {"channel": "financial_impact_flow", "rank": 3, "weight_pct": 14.2},
    {"channel": "macro_liquidity_factor", "rank": 4, "weight_pct": 9.8}
  ],
  "deletion_gain": 0.1956,
  "audit_verdict": "CERTIFIED_PRODUCTION_GROUNDED"
}
```

### 4.3 The Mathematical Deletion Gain Proof

To mathematically guarantee that our attribution is faithful and not post-hoc narrative rationalization, the engine computes the **Deletion Gain**:
$$\Delta\text{Gain} = \mathcal{L}(\mathbf{X}_{\setminus\text{top}}) - \mathcal{L}(\mathbf{X})$$
where $\mathcal{L}(\mathbf{X})$ is the cross-entropy / pinball loss with all features present, and $\mathcal{L}(\mathbf{X}_{\setminus\text{top}})$ is the loss when the top attributed causal driver is zeroed out.

- **Audited Value:** $\Delta\text{Gain} = \mathbf{+0.1956}$
- **Significance:** A positive deletion gain proves that removing the identified driver materially degrades model predictive loss, mathematically verifying genuine causal dependency.

---

## 5. Quantitative Acceptance Criteria & Benchmark Summary

When the evaluation notebook executes to completion, it tests the live data stream against Dubstrata's six core institutional quality gates:

| Metric | Target Requirement | Audited Benchmark | Quality Gate Verdict |
| :--- | :--- | :--- | :---: |
| **Strict Timestamp Monotonicity** | $t_i < t_{i+1}$ (No out-of-order bars) | Certified Monotonic | **PASS** |
| **Stationarity Parameter ($d^*$)** | Optimal $d^* = 0.42$, ADF $p < 0.01$ | $d^* = 0.42$ ($p = 0.0031$) | **PASS** |
| **Memory Retention ($r$)** | Pearson $r > 0.90$ with raw price | $r = 0.9412$ | **PASS** |
| **Directional Hit Rate (1h Horizon)** | $\ge 60.0\%$ (Institutional Alpha Gate) | **68.46%** (GCP Certified) | **PASS** |
| **Matthews Correlation Coeff (MCC)** | $\ge +0.20$ (Statistical Significance) | **+0.3706** | **PASS** |
| **HRX Deletion Gain ($\Delta\text{Gain}$)** | $\Delta\text{Gain} > 0.0$ (Attribution Faithfulness) | **+0.1956** | **PASS** |

---

## 6. Enterprise Integration & Deployment Models

Dubstrata provides three deployment topographies tailored to institutional security requirements:

1. **Direct REST API / CDN Stream (`https://api.dubstrata.com`):** Standard institutional access via sub-15ms edge CDN.
2. **S3 / Google Cloud Storage Bulk Partition Sync:** Nightly or hourly Parquet partition drops directly into your institutional data lake (`s3://dubstrata-institutional-data/`).
3. **Self-Hosted Docker Substrate (Air-Gapped / Zero-Egress):** Full deployment of ArcadeDB and Causal-FM v2 containers inside your private AWS VPC, GCP project, or on-premise Kubernetes cluster.
