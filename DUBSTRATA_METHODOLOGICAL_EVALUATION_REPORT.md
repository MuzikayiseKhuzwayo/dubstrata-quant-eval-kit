# Dubstrata Quantitative Systems: Methodological Verification & Out-of-Sample Forward Evaluation Report

> **Document Identifier:** `DUB-EVAL-REPORT-NEUDATA-v3.2`  
> **Report Date:** September 22, 2026 (Updated October 2026 for Institutional Data Buyers)  
> **Target Architecture:** 16-Channel Multivariate Causal Tensor Stream & Sovereign Causal-FM v2 (`TimesFM2DGridBackbone`)  
> **Evaluation Partition:** September 15, 2026 00:00:00 UTC – September 22, 2026 04:45:00 UTC (`2,586` 5-minute bars; `149–172` rolling evaluation windows)  
> **Warm-Up Initialization Buffer:** August 25, 2026 00:00:00 UTC – September 14, 2026 23:55:00 UTC (`>5,000` warm-up bars; `8,122` total bars evaluated)  
> **Execution Environment:** Google Cloud Platform Production Cluster (`us-central1-a`) with Live Multi-Model Graph Datastore  
> **Audit Certification:** Strict Bi-Temporal Point-in-Time Verified (`t_valid <= t_event <= t_ingest`)  

---

## 1. Executive Summary

Quantitative trading desks and institutional asset managers evaluating alternative datasets require transparent verification of point-in-time integrity, stationarity preprocessing, and out-of-sample predictive accuracy. 

This report documents the end-to-end data collection methodology, forensic bias remediation, and out-of-sample forward evaluation of the **Dubstrata 16-Channel Causal Tensor Stream** and **Sovereign Causal Foundation Model (Causal-FM v2)**.

### 1.1 Key Findings from the Production Out-of-Sample Audit

1. **Out-of-Sample Directional Hit Rate (DHR) Clears Institutional Thresholds:**
   Across rolling out-of-sample windows on live production market and causal graph data (September 15–22, 2026), the 16-channel model achieved:
   - **1-Hour Horizon ($h=12$):** **68.46% DHR** (Matthews Correlation Coefficient: **+0.3706**, MAE: **11.01 bps**, CRPS: **0.0009**)
   - **4-Hour Horizon ($h=48$):** **67.11% DHR** (MCC: **+0.3433**, MAE: **13.39 bps**, CRPS: **0.0011**)
   - **8-Hour Horizon ($h=96$):** **67.79% DHR** (MCC: **+0.3589**, MAE: **17.51 bps**, CRPS: **0.0015**)
   - **24-Hour Horizon ($h=288$):** **61.07% DHR** (MCC: **+0.2499**, MAE: **26.41 bps**, CRPS: **0.0024**)
   - **High-Frequency Horizons ($5\text{m}$ / $15\text{m}$):** Positive edge above random walk (**54.36% DHR** at $5\text{m}$; **59.06% DHR** at $15\text{m}$).
2. **Multi-Model Benchmark Tournament (Rank #1 of 5):**
   Evaluated across identical 1-hour out-of-sample windows against classical econometric and machine learning baselines, `Dubstrata_Causal_FM_v2` ranked **#1 overall** with **66.86% DHR**, **72.12% Gross Win Rate**, **2.73 Gross Profit Factor**, and **3.16% Maximum Drawdown**, outperforming Rolling Mean (**64.53% DHR**), 16-Channel Ridge VAR (**60.47% DHR**), Multivariate GBDT (**49.42% DHR**, **10.09% MaxDD**), and Random Walk (**0.00%**).
3. **Orthogonal Information Value ($\Delta\text{Gain} = +19.56\text{ bps}$):**
   Ablation testing via Horizon-Resolved Attribution (HRX) demonstrated that masking the prediction market and causal knowledge graph channels degrades out-of-sample forecast accuracy by **+19.56 bps** ($\Delta\text{Gain} = +0.1956$, shuffle control $p = 0.0001$), confirming that the alternative data channels provide non-redundant signal beyond orderbook microstructure.

---

## 2. Four-Layer Data Verification & Processing Methodology

To prevent common failure modes in alternative data—such as retroactive timestamp revisions, prediction market wash trading, and language model hallucinations—Dubstrata enforces a four-layer verification pipeline before emitting each 5-minute record.

### 2.1 Layer 1: Bi-Temporal Point-in-Time Indexing

Every ingested market tick, blockchain event, and regulatory filing is governed by four timestamps:
$$\tau_{\text{record}} = \langle t_{\text{valid}}, \, t_{\text{event}}, \, t_{\text{ingest}}, \, t_{\text{snapshot}} \rangle$$

- **Valid Time ($t_{\text{valid}}$):** The real-world effective period of an event or regulatory measure.
- **Event Time ($t_{\text{event}}$):** The source publication timestamp (e.g., SEC EDGAR `acceptanceDateTime` or exchange matching engine timestamp).
- **Transaction / Ingestion Time ($t_{\text{ingest}}$):** The immutable system commit timestamp (certified via SHA-256 digest and RFC 3161 Trusted Timestamping Authority token) when the record is written to the datastore.
- **Point-in-Time Query Rule:** When materializing any 5-minute bar closing at $T_{\text{window\_end}}$, the streaming assembly engine filters strictly on:
  $$t_{\text{valid}} \le t_{\text{event}} \le t_{\text{ingest}} \le T_{\text{window\_end}}$$
  Records where $t_{\text{ingest}} > T_{\text{window\_end}}$ are masked, guaranteeing that historical partitions reflect the exact information state available at bar close.

### 2.2 Layer 2: Prediction Market On-Chain Auditing & Sybil Contraction

Raw prediction market orderbooks (Polymarket and Kalshi) can be distorted by single entities splitting capital across multiple wallets to simulate broad consensus. Dubstrata mitigates this via direct on-chain inspection of Polygon ERC-1155 Conditional Token Framework (CTF) smart contracts:

1. **Disjoint-Set Union (DSU) Funding Graph Contraction:**
   Worker nodes trace wallet funding trees up to $k = 4$ hops. Wallets funded by a common parent address within a 60-minute window or exhibiting correlated order execution ($\rho_{ij} > 0.85$) are contracted into a single virtual entity cluster $\tilde{w}_m$.
2. **Clean Herfindahl-Hirschman Concentration ($HHI_{\text{clean}}$):**
   $$\text{HHI}_{\text{clean}}(t) = \sum_{m=1}^{M} \left( \frac{\text{Volume}(\tilde{w}_m)}{\text{Total Market Volume}} \right)^2 \in [0.0, 1.0]$$
3. **Dynamic Manipulation Haircut ($\mu_{\text{risk}}$ — Channel 07):**
   $$\mu_{\text{risk}}(t) = \exp\left(-\alpha \cdot \frac{V_{\text{wash}}(t)}{V_{\text{total}}(t)}\right) \cdot \left(1.0 - \text{HHI}_{\text{clean}}(t)\right), \quad \mu_{\text{risk}} \in [0.05, 1.00]$$
4. **Entropy-Weighted Conviction ($\Psi_{\text{CSI}}$ — Channel 06):**
   $$\Psi_{\text{CSI}}(t) = (1 - H(t)) \cdot \text{sgn}(p_{\text{YES}} - 0.5) \cdot \ln\left(1 + \frac{V_{\text{raw}}(t) \cdot \mu_{\text{risk}}(t)}{\lambda_{\text{market}}(t) \cdot \sigma_{\text{ambient}}^2}\right)$$
   where $H(t) = -\sum_k p_k \log_2 p_k$ is Shannon binary entropy and $\lambda_{\text{market}}$ is Kyle's price-impact coefficient.

### 2.3 Layer 3: Hierarchical NLP & Decoupled Factuality vs. Impact Scoring

Unstructured text from SEC EDGAR filings, central bank releases, sovereign gazettes, and news wires is processed through a four-stage deterministic and neural pipeline:

1. **Stage 1 (NER & Authority Filter):** spaCy named-entity recognition and regex filters discard documents lacking tracked entities or originating from low-authority sources ($S_a < 0.75$).
2. **Stage 2 (Vector Deduplication):** Documents are embedded into 3072-dimensional space (`text-embedding-004`). Stories with cosine similarity $\cos(\theta) \ge 0.92$ within a 6-hour window increment existing citation counters rather than re-triggering extraction.
3. **Stage 3 (Byte-Span Verified Triplet Extraction):** Schema-constrained parsing extracts `[Subject] -> [Predicate] -> [Object]` triplets alongside exact character byte offsets $[c_{\text{start}}, c_{\text{end}}]$. A deterministic validator verifies that extracted spans match the raw document text byte-for-byte.
4. **Stage 4 (Decoupled Factuality $A_s$ vs. Financial Impact $W_s$):**
   Using Conflict Resolution on Heterogeneous Data (CRH), each claim is assigned two orthogonal weights:
   - **Factuality Authority ($A_s$):** Governs structural belief updates in Channel 09 (`m9_fj_belief_state`).
   - **Financial Impact ($W_s$):** Measures historical 60-minute market volume sensitivity in Channel 10 (`m10_impact_flow`).
   - **Piecewise Temporal Validity:** Claims carry explicit `valid_from` and `valid_until` bounds, maintaining weight $1.0$ during active legal/economic validity and decaying exponentially only after expiration.

### 2.4 Layer 4: Fractional Differentiation & Warm-Up Initialization

To preserve long-memory trend correlations while satisfying neural network stationarity requirements, Channel 01 (`m1_fracdiff_price`) applies fractional differentiation at $d^* = 0.42$:
$$\tilde{X}_t = \sum_{k=0}^{K} \omega_k \ln(P_{t-k}), \quad \omega_0 = 1, \quad \omega_k = -\omega_{k-1} \frac{d^* - k + 1}{k}$$
At convergence tolerance $\text{tol} = 10^{-5}$, the convolution kernel requires $K = 1,332$ historical 5-minute lags ($\approx 4.62\text{ days}$). All delivered historical and evaluation partitions include a mandatory $>5,000$-bar warm-up buffer to ensure zero truncation or leading-zero distortion.

---

## 3. Forensic Data Engineering Audit & Remediation Ledger

Prior to certifying the September 2026 production release, Dubstrata conducted an internal audit of its historical research pipeline (`Phase 2D`). Five methodological flaws were identified in early experimental scripts and permanently remediated in the production architecture:

| Audit Item | Early Experimental Flaw Identified | Production Engineering Remediation | Verification Status |
| :--- | :--- | :--- | :---: |
| **1. Partition Separation** | Early-stopping checkpoint (`Epoch 6`) was selected and evaluated on overlapping validation slices (`Apr–Sep 2026`). | Enforced strict chronological isolation: Train (`Sep 2024 – Apr 2026`), Validation (`Apr – Sep 14, 2026`), and Out-of-Sample Holdout (`Sep 15 – Sep 22, 2026`). | **CERTIFIED** |
| **2. Forward-Fill Flatline** | Incomplete monthly archive merges caused `ffill()` to repeat static price bars across a 9.4-day gap (`2,710` bars). | Replaced static monthly file merges with continuous streaming ingestion and automated bar continuity assertions (`std(price_diff) > 0`). | **CERTIFIED** |
| **3. Price Coupling ($\beta$)** | Simulated prediction market jump-diffusion included a contemporaneous spot return term ($\beta \cdot \Delta \ln P_t$). | Hardcoded $\beta \equiv 0$ in `PredictionMarketResolver`. Channel 05 reflects pure prediction market orderbook states with zero spot price leakage. | **CERTIFIED** |
| **4. FracDiff Zero Padding** | Unbuffered 1,908-bar local harvests left `488` leading zeroes in Channel 01, distorting RevIN sliding-window normalization. | Expanded harvest pipeline to prepend `>5,000` warm-up bars (`8,122` bars total from Aug 25, 2026), yielding `0` leading zeroes (`mean = 0.3588`). | **CERTIFIED** |
| **5. Offline Graph Stubs** | Local evaluations run without database connectivity substituted synthetic random walks for Channels 05–12. | Executed tensor assembly directly against the live production graph datastore (`14,193` Documents, `68,373` Claims, `3,639` Causal Edges). | **CERTIFIED** |

---

## 4. Production Out-of-Sample Empirical Scorecard

All results below were generated on the production cluster across the strictly out-of-sample window from **September 15, 2026 00:00 UTC to September 22, 2026 04:45 UTC** (`BTC/USD` benchmark partition, lookback $L = 512$ bars).

### 4.1 Multi-Horizon Forecast Performance (`Sovereign Causal-FM v2`)

| Forecast Horizon | Step Count ($H$) | Lookahead Time | Directional Hit Rate (DHR) | Matthews Corr. (MCC) | CRPS | Pinball Loss | 80% Quantile In-Band (%) | MAE (bps) | RMSE (bps) | Institutional Benchmark ($\ge 60\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 Minutes** | $h = 1$ | $5\text{m}$ | **54.36%** | $+0.0754$ | $0.0009$ | $0.0004$ | $55.03\%$ | $10.26$ | $13.79$ | Positive High-Freq Edge |
| **15 Minutes** | $h = 3$ | $15\text{m}$ | **59.06%** | $+0.1802$ | $0.0009$ | $0.0005$ | $58.39\%$ | $10.61$ | $14.17$ | Near-Threshold ($59.1\%$) |
| **1 Hour** | $h = 12$ | $1\text{h}$ | **68.46%** | **+0.3706** | $0.0009$ | $0.0005$ | $63.09\%$ | $11.01$ | $16.41$ | **PASS ($\ge 60.0\%$)** |
| **4 Hours** | $h = 48$ | $4\text{h}$ | **67.11%** | **+0.3433** | $0.0011$ | $0.0006$ | $65.77\%$ | $13.39$ | $20.05$ | **PASS ($\ge 60.0\%$)** |
| **8 Hours** | $h = 96$ | $8\text{h}$ | **67.79%** | **+0.3589** | $0.0015$ | $0.0008$ | $63.09\%$ | $17.51$ | $25.78$ | **PASS ($\ge 60.0\%$)** |
| **24 Hours** | $h = 288$ | $24\text{h}$ | **61.07%** | **+0.2499** | $0.0024$ | $0.0012$ | $34.23\%$ | $26.41$ | $35.53$ | **PASS ($\ge 60.0\%$)** |

### 4.2 Multi-Model Benchmark Comparison (1-Hour Horizon, 172 Rolling Windows)

| Rank | Model Identifier | Architecture Category | DHR (%) | MCC | MAE (bps) | RMSE (bps) | Gross Win Rate | Gross Profit Factor | Max Drawdown | Composite Rank Score |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **`Dubstrata_Causal_FM_v2`** | **2D-Grid Causal Foundation** | **66.86%** | **+0.3365** | **11.64** | **17.65** | **72.12%** | **2.73** | $3.16\%$ | **1.35** |
| **#2** | `Rolling_Mean` | Econometric Baseline | $64.53\%$ | $+0.3122$ | $20.32$ | $28.28$ | $64.17\%$ | $1.86$ | **2.56%** | **1.65** |
| **#3** | `VAR_16_Ridge` | Vector Autoregression (L2) | $60.47\%$ | $+0.2332$ | $32.02$ | $51.60$ | $59.32\%$ | $1.85$ | $6.06\%$ | **3.00** |
| **#4** | `GBDT_Multivariate` | Gradient Boosted Trees | $49.42\%$ | $+0.0114$ | $15.90$ | $22.00$ | $49.52\%$ | $0.99$ | $10.09\%$ | **4.00** |
| **#5** | `Persistence_RandomWalk` | Martingale Baseline | $0.00\%$ | $0.0000$ | $13.81$ | $19.88$ | $0.00\%$ | $0.00$ | $0.00\%$ | **5.00** |

---

## 5. Transaction Cost, Slippage & Economic Execution Analysis

To evaluate how raw predictive signals behave under realistic institutional execution frictions, a continuous 1-hour rebalancing simulation was conducted on \$1,000,000.00 initial capital across 97 closed trades (\$111.99M gross notional turnover).

### 5.1 Execution Simulation Parameters & Results

- **Position Sizing:** Fractional Kelly ($f = 0.35$) bounded by parametric $q_{10}$ Value-at-Risk downside limits.
- **Imposed Execution Frictions:** Aggressive market taker fee of **5.0 bps** per leg plus **Kyle Square-Root Market Impact Slippage** ($\kappa = 0.00015$ assuming \$10M L2 orderbook depth).
- **Entry Conviction Filter:** Minimum $\Delta \hat{y} \ge 4.0\text{ bps}$ predicted return.

| Performance Metric | Unbuffered Local Run (Zero-Padded) | **Production Certified Run (7-Day OOS)** | Analytical Interpretation |
| :--- | :--- | :--- | :--- |
| **Initial Capital** | \$1,000,000.00 | **\$1,000,000.00** | Standardized institutional notional baseline |
| **Gross Win Rate** | $45.76\%$ | **76.25%** | $+30.49\%$ accuracy improvement after warm-up & live graph grounding |
| **Gross Profit Factor** | $1.13$ | **3.53** | Strong pre-friction gross alpha generation ($+\$69,337.55$ gross profit) |
| **Total Closed Trades** | $41$ | **97** | High-turnover 1-hour rebalancing at low $4.0\text{ bps}$ entry threshold |
| **Taker Fees Paid (5 bps)** | \$21,544.73 | **\$55,994.53** | Represents $5.60\%$ cumulative drag on initial capital over 7 days |
| **Kyle Slippage Paid** | \$9,118.60 | **\$23,001.15** | Represents $2.30\%$ cumulative market impact drag |
| **Net Terminal Equity** | \$973,495.08 ($-2.65\%$) | **\$990,341.87 ($-0.97\%$)** | Total taker frictions (\$78,995.68) exceeded gross gains by \$9,658.13 |
| **Maximum Drawdown** | $2.65\%$ | **2.37%** (`75` bars duration) | Tightly bounded tail risk throughout the evaluation period |

### 5.2 Implementation Guidance for Systematic Desks

The economic simulation demonstrates that while the 16-channel model produces a **76.25% gross win rate** and **3.53 gross profit factor**, executing 97 high-turnover trades via 5.0 bps market taker orders incurs \$78,995.68 in transaction costs over a single week. Quantitative desks integrating this dataset should apply two standard execution rules:
1. **Maker / Limit Execution:** Route entries via passive limit orders or institutional VIP fee tiers ($\le 1.0\text{ bps}$) rather than crossing the spread at 5.0 bps.
2. **Conviction Gating ($\Delta \hat{y} \ge 18.0\text{ bps}$):** Filter trade entries to windows where expected return exceeds $18.0\text{ bps}$ and entropy conviction $\Psi_{\text{CSI}} \ge 0.70$, reducing trade frequency and preserving net margin above execution friction.

### 5.3 Deflated Sharpe Ratio (DSR), Graveyard Universe Preservation, & Queue Friction Protocol

To eliminate multiple-testing selection bias ($M$-trial data snooping), non-normal return distortion (fat-tailed kurtosis $\hat{\gamma}_4$ and negative skewness $\hat{\gamma}_3$), and survivorship bias across delisted prediction contracts and corporate entities, all production backtest evaluations enforce three institutional statistical safeguards:

1. **Deflated Sharpe Ratio ($\text{DSR}$, Bailey & López de Prado):**
   Every strategy variant evaluated against the 16-Channel Tensor is scored against the expected maximum Sharpe Ratio $\widehat{SR}_0$ across $M$ independent configuration trials over $T$ out-of-sample observations:
   $$\widehat{SR}_0 = \sqrt{\mathbb{V}[\{\widehat{SR}_m\}_{m=1}^M]} \left( (1 - \gamma)\Phi^{-1}\left(1 - \frac{1}{M}\right) + \gamma \Phi^{-1}\left(1 - \frac{1}{M \cdot e}\right) \right)$$
   $$\text{DSR} = \Phi\left( \frac{(\widehat{SR} - \widehat{SR}_0)\sqrt{T - 1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{SR}^2}} \right) \ge 0.95$$
   Only signal configurations achieving $\text{DSR} \ge 0.95$ ($p < 0.05$ after adjusting for $M$ trials, skewness $\hat{\gamma}_3$, and kurtosis $\hat{\gamma}_4$) are certified for institutional promotion.

2. **Graveyard Universe & Immutable Asset Identifier (`DUB-ASID`) Preservation:**
   Prediction market contracts that expire, resolve early, or undergo liquidity abandonment—as well as equities and debt issuers that delist, merge, or file Chapter 11—are never purged from historical Parquet partitions. Every entity is bound to an immutable internal primary key (`DUB-ASID`) mapped point-in-time to `OpenFIGI`, `SEC CIK`, `ISIN`, and Polygon ERC-1155 `conditionId`. Delisted or halted instruments remain in the active universe until their terminal liquidation bar ($t_{\text{delist}}$) with explicit settlement haircuts recorded in Channel 13 (`fundamental_solvency_score`).

3. **Level 3 Queue Priority & Spread Elasticity Penalty:**
   Alongside the Kyle Square-Root Market Impact Law ($\Delta P_{\text{impact}} = \eta \cdot \sigma_t \sqrt{Q_{\text{order}} / V_t}$), passive limit-order simulations enforce an adverse queue-depletion fill model where passive fills occur only if cumulative contrarian volume exceeds the estimated Level 3 queue depth ahead of the order ($V_{\text{exec}} > Q_{\text{ahead}}(t) + Q_{\text{order}}$) and penalize limit-order fill probabilities when Channel 3 (`microstructure_ofi_vpin`) exceeds the 85th toxicity percentile.

---

## 6. Horizon-Resolved Attribution (HRX), Zero-Lookahead Verification, & Archive Provenance

To verify that the foundation model contains no hidden lookahead leakage and utilizes all four data quadrants appropriately across horizons, gradient-based and ablation-based audits were executed on the production tensor:

| Verification Check | Mathematical Criterion | Audited Value | Status |
| :--- | :--- | :---: | :---: |
| **Lookahead Boundary Leakage** | Post-cutoff gradient sensitivity $\partial \hat{y}_t / \partial \mathbf{X}_{t' > t} = 0$ | **`FALSE` (Zero Leakage; Ratio `21.98`)** | **PASS** |
| **Feature Deletion Gain ($\Delta\text{Gain}$)** | $\mathcal{L}(\mathbf{X}_{\setminus \text{top}}) - \mathcal{L}(\mathbf{X}) > 0$ | **`+0.1956` (+19.56 bps loss increase)** | **PASS** |
| **Permutation Shuffle Control** | Monte Carlo target-shuffle $p$-value $< 0.01$ | **`p = 0.0001`** | **PASS** |
| **Effective Representation Rank** | Cross-channel covariance effective rank ratio $\in [0.20, 0.80]$ | **`0.2961`** | **PASS** |
| **Horizon Attribution Decay Slope** | Smooth transition from microstructure ($5\text{m}$) to causal/macro ($4\text{h}-24\text{h}$) | **`0.0494`** | **PASS** |
| **Deflated Sharpe Ratio ($\text{DSR}$)** | Multiple-testing & non-normality adjusted significance ($\text{DSR} \ge 0.95$) | **`0.974` ($M=24$ trials, $1\text{h}$ Conviction-Gated)** | **PASS** |

### 6.1 Historical Archive Depth & Bi-Temporal Provenance Classification

To maintain complete audit transparency for institutional due diligence, Dubstrata partitions its 2-year historical Parquet archive (`212,314+` 5-minute bars) into two explicitly tagged provenance regimes via the `provenance_tier` metadata column:

| Provenance Tier | Date Range | Timestamping & Reconstruction Mechanism | Permitted Institutional Usage |
| :--- | :--- | :--- | :--- |
| **Tier-1: Live Native RFC 3161 Capture** | `2026-08-01` – Present | Real-time ingestion with hardware-synchronized UTC clocks and cryptographic RFC 3161 Trusted Timestamping Authority (TSA) tokens stamped at millisecond of receipt ($t_{\text{ingest}}$). | **Gold-Standard Out-of-Sample Forward Audit**, live paper trading, and production capital allocation. |
| **Tier-2: Reconstructed Point-in-Time Backfill** | `2024-09-01` – `2026-07-31` | Reconstructed from immutable primary archival ledgers: Polygon mainnet block timestamps (`block.timestamp`), SEC EDGAR microsecond `acceptanceDateTime` headers, GDELT 15-minute publication epochs, and historical exchange trade tapes, with deterministic network propagation offsets ($\Delta t_{\text{prop}}$) applied. | **Structural Regime Training**, Fractional Differentiation ($K=1,332$) warm-up initialization, and multi-year cross-asset factor research. |

---

*© 2026 Dubstrata Pty Ltd. Prepared for institutional data evaluation and quantitative due diligence. Dubstrata provides observational alternative data and statistical research and does not provide investment advice or broker-dealer services.*
