# Dubstrata 16-Channel Causal Tensor Stream & Predictive Forecast Feed — Institutional Data Dictionary

> **Document Identifier:** `DUB-DATA-DICT-NEUDATA-v3.2`  
> **Product Name:** Dubstrata 16-Channel Causal Tensor Stream and Forward Forecast Feed  
> **Target Audience:** Quantitative Researchers, Alternative Data Scouts, Portfolio Managers, Data Engineering Teams  
> **Sampling Cadence:** 5-Minute Uniform Stationary Bars (`300s` intervals, 24/7/365 continuous)  
> **Historical Availability:** September 15, 2024 – Present (`>210,000` 5-minute intervals per core asset)  
> **Point-in-Time Guarantee:** Strict Bi-Temporal Transaction-Time Indexing (`last_seen_at <= window_end`)  

---

## 1. Dataset Overview & Delivery Architecture

Dubstrata provides a point-in-time synchronized 5-minute time-series dataset ("The Fuel") and multi-horizon probabilistic forecast feed ("The Engine") combining market orderbook microstructure, decentralized and regulated prediction market consensus (Polymarket and Kalshi), structured event claims extracted from primary regulatory filings, and macroeconomic liquidity indicators.

### 1.1 Delivery Channels & Formats

| Delivery Channel | Protocol / Location | Format | Cadence & Latency SLA |
| :--- | :--- | :--- | :--- |
| **Cloud Data Lake (Historical & Daily)** | Google Cloud Storage (`gs://dubstrata-alt-data/tenant={id}/`) or AWS S3 | Apache Parquet (Snappy compressed, PyArrow v15+ compatible) | Hourly & Daily partitions synced at `00:05 UTC` (`<= 8 min` completion) |
| **Streaming Message Bus** | Confluent Cloud Apache Kafka (`topic: alt-data-time-series`) | UTF-8 JSON (Schema Registry validated) | Every 5 minutes (`<= 250ms` lag after bar close) |
| **Real-Time WebSocket Stream** | `wss://api.dubstrata.com/api/v1/ws/tensor` & `/ws/alerts` | UTF-8 JSON | Sub-150ms broadcast upon bar close or dislocation event |
| **REST Query Endpoints** | `https://api.dubstrata.com/api/v1/predict/*` | UTF-8 JSON / In-Memory Parquet | `<= 15ms` gateway response time (`99.9%` uptime SLA) |

---

## 2. Primary Key & Bi-Temporal Metadata Fields

Every record in the 5-minute tensor dataset is uniquely indexed by `(window_end, asset_ticker)`. To prevent lookahead bias in historical backtesting, all features within a bar `[window_start, window_end]` are computed strictly from market ticks, blockchain states, and document claims whose system commit timestamp (`t_ingest` / `last_seen_at`) occurred at or before `window_end`.

| Field Name | Parquet / SQL Type | Nullable | Format / Example | Description & Point-in-Time Rule |
| :--- | :--- | :---: | :--- | :--- |
| `window_start` | `TIMESTAMP(3) UTC` | `NOT NULL` | `2026-09-22T04:40:00.000Z` | Inclusive start timestamp of the 5-minute observation bar (`window_end - 300s`). |
| `window_end` (`timestamp`) | `TIMESTAMP(3) UTC` | `NOT NULL` | `2026-09-22T04:45:00.000Z` | **Primary Point-in-Time Index.** Inclusive close timestamp of the 5-minute observation bar (`t_ingest <= window_end`). Exposed as both `window_end` and `timestamp` in Parquet partitions. |
| `asset_ticker` (`asset_symbol`) | `VARCHAR(32)` | `NOT NULL` | `BTC/USD`, `SPY.US`, `CL=F` | Canonical instrument identifier. Exposed as both `asset_ticker` and `asset_symbol` in Parquet partitions. |
| `dub_asid` | `VARCHAR(16)` | `NOT NULL` | `DUB-ASID-000101` | Immutable Dubstrata Security Master primary key (survives ticker changes, corporate actions, and delistings). |
| `openfigi` | `VARCHAR(12)` | `NOT NULL` | `BBG0013HG9X2` | Point-in-time OpenFIGI 12-character instrument identifier mapped in the Security Master. |
| `provenance_tier` | `VARCHAR(32)` | `NOT NULL` | `TIER_1_REDISTRIBUTABLE` | Archive provenance and licensing classification (`TIER_1_REDISTRIBUTABLE` / `TIER_1_LIVE_RFC3161`, `TIER_2_PIT_RECONSTRUCTED`, or `TIER_2_BYOL_JOIN`). |
| `bar_sequence_id` | `INT64` | `NOT NULL` | `184920` | Monotonically increasing integer index per asset partition (`t_i < t_{i+1}` invariant enforced). |
| `is_warmup_certified` | `BOOLEAN` | `NOT NULL` | `true` | Indicates whether the bar is preceded by at least `K = 1,332` valid historical lags required to initialize the fractional differentiation convolution filter without zero padding. |

---

## 3. Offering A: 16-Channel Multivariate Causal Tensor Schema ("The Fuel")

The core feature tensor $\mathbf{X}(t) \in \mathbb{R}^{16 \times 512}$ is organized into four orthogonal domain quadrants (4 channels per quadrant). All 16 channels are stored as IEEE 754 double-precision floats (`DOUBLE` / `FLOAT64` in Parquet and SQL; `float32` in compressed model checkpoints) and are guaranteed free of `NaN` or `Inf` values in production partitions.

### 3.1 Quadrant 1: Target Asset Microstructure & Derivatives (Channels 01–04)

| Ch | Field Name (`SQL` / `Parquet`) | Canonical Alias | Typical Range / Units | Stationarity & Mathematical Formulation |
| :---: | :--- | :--- | :---: | :--- |
| **01** | `m1_fracdiff_price` | `target_fracdiff_price` | $\mathbb{R}$ (Mean $\approx 0.36$) | **Fractionally Differentiated Log-Price ($d^* = 0.42$):** Computed via fixed-window fractional differentiation over $K = 1,332$ lags ($\text{tol} = 10^{-5}$): $\tilde{X}_t = \sum_{k=0}^{K} \omega_k \ln(P_{t-k})$. Preserves long-memory trend correlation ($r = 0.9412$ with raw price) while passing the Augmented Dickey-Fuller stationarity test ($p < 0.005$). |
| **02** | `m2_log_volatility` | `target_log_volatility` | $[0.01, 0.50]$ (Annualized Log-Vol) | **Garman-Klass / Yang-Zhang Realized Volatility:** Minimum-variance estimator combining overnight/inter-bar jumps and continuous intra-bar high-low-open-close range variance over a rolling 96-hour (1,152-bar) window. |
| **03** | `m3_ofi_vpin` | `microstructure_ofi_vpin` | $[-1.00, +1.00]$ (Normalized Index) | **Order Flow Imbalance Scaled by VPIN:** Multi-level Central Limit Order Book (CLOB) bid-ask queue depth imbalance across the top 10 price levels, weighted by Volume-Synchronized Probability of Informed Trading (VPIN). |
| **04** | `m4_derivatives_state` | `derivatives_surface_state` | $[-1.00, +1.00]$ (Z-Score Normalized) | **Options Volatility Skew & Funding Regime:** Composite index combining normalized 25-delta options risk reversal ($IV_{\text{25d Call}} - IV_{\text{25d Put}}$), net dealer Gamma Exposure (GEX), and perpetual futures funding rate basis. |

### 3.2 Quadrant 2: Informed Crowd & Prediction Markets (Channels 05–08)

| Ch | Field Name (`SQL` / `Parquet`) | Canonical Alias | Typical Range / Units | Stationarity & Mathematical Formulation |
| :---: | :--- | :--- | :---: | :--- |
| **05** | `m5_informed_market_prob` | `informed_market_prob` | $[0.00, 1.00]$ (Implied Probability) | **Decoupled Prediction Market Consensus Probability:** Aggregate implied probability across mapped Polymarket and CFTC-regulated Kalshi event contracts, weighted by wallet historical Brier score and scrubbed of wash trading. Strictly decoupled from contemporaneous target asset returns ($\beta \equiv 0$). |
| **06** | `m6_entropy_conviction_psi` | `entropy_conviction_psi` | $[-5.00, +5.00]$ (Bits $\times$ Log-Vol) | **Information-Theoretic Conviction Index ($\Psi_{\text{CSI}}$):** Scales prediction market probability deltas by Shannon entropy distance from a uniform prior: $\Psi_{\text{CSI}}(t) = (1 - H(t)) \cdot \text{sgn}(p_{\text{YES}} - 0.5) \cdot \ln(1 + V_{\text{clean}}(t) / (\lambda_{\text{market}} \cdot \sigma_{\text{ambient}}^2))$. |
| **07** | `m7_manipulation_haircut_mu` | `manipulation_haircut_mu` | $[0.05, 1.00]$ (Risk Multiplier) | **On-Chain Sybil & Concentration Haircut ($\mu_{\text{risk}}$):** Derived by auditing Polygon ERC-1155 Conditional Token Framework (CTF) smart contracts via Disjoint-Set Union (DSU) wallet funding graph contraction. Computes clean Herfindahl-Hirschman Index ($HHI_{\text{clean}}$) and penalizes circular wash volume: $\mu_{\text{risk}} = \exp(-\alpha V_{\text{wash}}/V_{\text{total}}) \cdot (1 - HHI_{\text{clean}})$. |
| **08** | `m8_informed_capital_imbalance` | `informed_capital_imbalance` | $[-1.00, +1.00]$ (Divergence Score) | **Informed vs. Retail Order Flow Divergence:** Normalized net capital flow delta and Kullback-Leibler (KL) divergence between top-decile historical PnL/accuracy institutional wallets and unrated retail order flow. |

### 3.3 Quadrant 3: Causal Knowledge Graph Dynamics (Channels 09–12)

| Ch | Field Name (`SQL` / `Parquet`) | Canonical Alias | Typical Range / Units | Stationarity & Mathematical Formulation |
| :---: | :--- | :--- | :---: | :--- |
| **09** | `m9_fj_belief_state` | `friedkin_johnsen_belief` | $[-1.00, +1.00]$ (Equilibrium Score) | **Friedkin-Johnsen Network Belief State ($Y_i(t)$):** Synthesized directional sentiment and risk score for the target entity, computed via iterative equilibrium propagation across multi-hop knowledge graph connections while accounting for source authority inertia ($\Theta_E$). |
| **10** | `m10_impact_flow` | `financial_impact_flow` | $[-2.00, +2.00]$ (Impact-Weighted Flow) | **Decoupled Financial Impact Flow ($W_s$):** Breadth-first causal exposure flow propagated across directed `:CAUSALLY_EXPOSED` graph edges, weighted by source historical 60-minute market volume correlation ($W_s$) and gated by piecewise temporal validity envelopes (`valid_from <= t <= valid_until`). |
| **11** | `m11_vector_drift` | `perception_vector_drift` | $[0.00, 1.00]$ (Cosine Displacement) | **Semantic Embedding Drift Velocity:** Cosine distance displacement ($1 - \cos(\mathbf{e}_t, \mathbf{e}_{t-1})$) of 3072-dimensional `text-embedding-004` centroids across newly ingested regulatory filings, central bank releases, and corporate disclosures. |
| **12** | `m12_conflict_entropy` | `graph_conflict_entropy` | $[0.00, 1.00]$ (Normalized Shannon) | **Dual-Agent Claim Contradiction Entropy:** Shannon entropy calculated across concurrent `:SUPPORTS` versus `:CONTRADICTS` relational claims attached to the target entity's knowledge graph node within the active validity window. |

### 3.4 Quadrant 4: Solvency, Dynamic Correlates & Macro Regimes (Channels 13–16)

| Ch | Field Name (`SQL` / `Parquet`) | Canonical Alias | Typical Range / Units | Stationarity & Mathematical Formulation |
| :---: | :--- | :--- | :---: | :--- |
| **13** | `m13_solvency_score` | `fundamental_solvency_score` | $[0.00, 1.00]$ (Health Score) | **Fundamental Solvency & Credit Stress Score:** Composite point-in-time indicator measuring corporate balance sheet liquidity, collateral reserve coverage, credit default swap (CDS) spreads, and sovereign fiscal stress. |
| **14** | `m14_dynamic_correlate` | `dynamic_primary_correlate` | $[-1.00, +1.00]$ (Normalized Return) | **Dynamic Bipartite Primary Correlate Return:** Standardized 5-minute return of the highest-ranked causally connected cross-asset instrument, dynamically selected at each bar via multi-hop graph elasticity scoring: $\text{Elasticity}(A, E_k) = \sum_{p} \prod_{e \in p} w(e) S_a(e) e^{-\lambda \Delta t}$. |
| **15** | `m15_macro_liquidity` | `macro_liquidity_factor` | $[-3.00, +3.00]$ (PC1 Z-Score) | **Global Macro Liquidity Principal Component (PC1):** First principal component extracted from US 2Y/10Y Treasury yield curves, the US Dollar Index (DXY), High-Yield Corporate Credit Spreads (CDX HY), and central bank net liquidity balances. |
| **16** | `m16_scheduled_event_clock` | `scheduled_event_clock` | $[0.00, 1.00]$ (Proximity Decay) | **Scheduled Macro & Expiry Event Clock ($\tau_{\text{event}}$):** Deterministic exponential countdown ($\tau = \exp(-\lambda \Delta t)$) measuring temporal proximity to scheduled macroeconomic releases (FOMC rate decisions, CPI, NFP), SEC statutory deadlines, earnings calls, and major options expirations. |

---

## 4. Offering B & C: Predictive Forecast & HRX Attribution Fields ("The Engine")

For clients subscribing to the Sovereign Causal-FM Predictive Engine and Horizon-Resolved Attribution (HRX) feed, the following output fields are delivered alongside the 16-channel tensor at each 5-minute interval.

| Field Name | Data Type | Range / Units | Description |
| :--- | :--- | :---: | :--- |
| `forecast_horizon` | `VARCHAR(8)` | `5m`, `15m`, `1h`, `4h`, `8h`, `24h` | Forward lookahead horizon corresponding to $H \in \{1, 3, 12, 48, 96, 288\}$ 5-minute steps. |
| `current_price` | `DOUBLE` | $> 0.0$ (Asset Quote Currency) | Last traded reference price at `window_end`. |
| `point_forecast` | `DOUBLE` | $> 0.0$ (Asset Quote Currency) | Conditional median / expected price forecast at `window_end + forecast_horizon`. |
| `directional_conviction` | `DOUBLE` | $[0.00, 1.00]$ | Calibrated probability of directional movement combined with entropy conviction gating. |
| `q10` .. `q90` | `DOUBLE` (9 fields) | Monotonic ($q_{10} \le q_{20} \dots \le q_{90}$) | Nine parametric price forecast quantiles ($10\text{th}$ to $90\text{th}$ percentiles) providing calibrated uncertainty bands. |
| `var_95_downside` | `DOUBLE` | $> 0.0$ (Asset Quote Currency) | Parametric $95\%$ Value-at-Risk downside support boundary over the target horizon. |
| `expected_shortfall_cvar` | `DOUBLE` | $> 0.0$ (Asset Quote Currency) | Conditional Value-at-Risk (Expected Shortfall) in the $5\%$ left tail. |
| `hrx_q1_microstructure_pct` | `DOUBLE` | $[0.0, 100.0]\%$ | Percentage of current forecast attribution driven by Quadrant 1 (Microstructure & Derivatives). |
| `hrx_q2_prediction_mkt_pct` | `DOUBLE` | $[0.0, 100.0]\%$ | Percentage of current forecast attribution driven by Quadrant 2 (Prediction Markets). |
| `hrx_q3_causal_graph_pct` | `DOUBLE` | $[0.0, 100.0]\%$ | Percentage of current forecast attribution driven by Quadrant 3 (Causal Knowledge Graph). |
| `hrx_q4_macro_solvency_pct` | `DOUBLE` | $[0.0, 100.0]\%$ | Percentage of current forecast attribution driven by Quadrant 4 (Solvency & Macro Regimes). |
| `hrx_deletion_gain` | `DOUBLE` | $\mathbb{R}$ (Benchmark: $+0.1956$) | Verified loss increase $\Delta\text{Gain} = \mathcal{L}(\mathbf{X}_{\setminus \text{top}}) - \mathcal{L}(\mathbf{X})$ when top causal drivers are masked. |

---

## 5. Coverage Universe & Identifier Mapping Reference

Neudata's automated ticker parser accepts Bloomberg `.US` equity tickers in the portal text field. The table below provides the complete cross-asset identifier mapping across all four asset tiers included in the dataset.

| Asset Tier | Dubstrata `asset_ticker` | Instrument Name | Bloomberg Ticker | SEC CIK / Contract Standard | Primary Dynamic Correlates (Ch 14) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Digital Assets** | `BTC/USD` | Bitcoin Spot / Perpetual Composite | `XBTUSD Curncy` | Polygon CTF / CME Globex | `MSTR.US`, `QQQ.US`, `SPY.US`, `GLD.US` |
| **Tier 1: Digital Assets** | `ETH/USD` | Ethereum Spot / Perpetual Composite | `XETUSD Curncy` | Polygon CTF / CME Globex | `BTC/USD`, `COIN.US`, `QQQ.US` |
| **Tier 1: Digital Assets** | `SOL/USD` | Solana Spot / Perpetual Composite | `XSOUSD Curncy` | Polygon CTF | `BTC/USD`, `ETH/USD`, `QQQ.US` |
| **Tier 2: US Equities** | `SPY.US` | SPDR S&P 500 ETF Trust | `SPY.US` | `0000884394` | `QQQ.US`, `NVDA.US`, `US10Y`, `DXY` |
| **Tier 2: US Equities** | `QQQ.US` | Invesco QQQ Trust (Nasdaq 100) | `QQQ.US` | `0001067839` | `NVDA.US`, `AAPL.US`, `SPY.US`, `US10Y` |
| **Tier 2: US Equities** | `MSTR.US` | MicroStrategy Inc. (Strategy) | `MSTR.US` | `0001050446` | `BTC/USD`, `COIN.US`, `QQQ.US` |
| **Tier 2: US Equities** | `NVDA.US` | NVIDIA Corporation | `NVDA.US` | `0001045810` | `TSM.US`, `QQQ.US`, `SPY.US` |
| **Tier 2: US Equities** | `AAPL.US` | Apple Inc. | `AAPL.US` | `0000320193` | `QQQ.US`, `TSM.US`, `SPY.US` |
| **Tier 2: US Equities** | `TSLA.US` | Tesla, Inc. | `TSLA.US` | `0001318605` | `QQQ.US`, `NVDA.US`, `BTC/USD` |
| **Tier 2: US Equities** | `COIN.US` | Coinbase Global, Inc. | `COIN.US` | `0001679788` | `BTC/USD`, `ETH/USD`, `MSTR.US` |
| **Tier 2: US Equities** | `BLK.US` | BlackRock, Inc. | `BLK.US` | `0001364742` | `SPY.US`, `BTC/USD`, `JPM.US` |
| **Tier 2: US Equities** | `TSM.US` | Taiwan Semiconductor Mfg. Co. | `TSM.US` | `0001046179` | `NVDA.US`, `AAPL.US`, `QQQ.US` |
| **Tier 2: US Equities** | `XOM.US` | Exxon Mobil Corporation | `XOM.US` | `0000034088` | `CL=F`, `BZ=F`, `CVX.US` |
| **Tier 2: US Equities** | `CVX.US` | Chevron Corporation | `CVX.US` | `0000093410` | `CL=F`, `BZ=F`, `XOM.US` |
| **Tier 2: US Equities** | `JPM.US` | JPMorgan Chase & Co. | `JPM.US` | `0000019617` | `GS.US`, `SPY.US`, `US10Y` |
| **Tier 2: US Equities** | `GS.US` | The Goldman Sachs Group, Inc. | `GS.US` | `0000886982` | `JPM.US`, `SPY.US`, `US10Y` |
| **Tier 3: Commodities** | `CL=F` | WTI Crude Oil Front-Month Futures | `CL1 Comdty` | NYMEX / Kalshi Energy | `BZ=F`, `XOM.US`, `DXY` |
| **Tier 3: Commodities** | `BZ=F` | Brent Crude Oil Front-Month Futures | `CO1 Comdty` | ICE Futures / Tanker Freight | `CL=F`, `XOM.US`, `CVX.US` |
| **Tier 3: Commodities** | `XAU/USD` | Spot Gold / COMEX Gold Composite | `XAU Curncy` | COMEX / Sovereign Reserves | `DXY`, `US10Y`, `SPY.US` |
| **Tier 3: Commodities** | `HG=F` | High-Grade Copper Futures | `HG1 Comdty` | COMEX | `CL=F`, `XAU/USD`, `SPY.US` |
| **Tier 4: Macro & RWAs** | `BUIDL` | Tokenized US Treasury Liquidity Pool | `USBMMF Index` | ERC-20 / US 3M T-Bills | `US02Y`, `US10Y`, `DXY` |

---

## 6. Canonical Streaming SQL & Parquet DDL Specification

> **Schema Naming Convention:** Streaming Flink SQL tables and Kafka JSON topics (`alt-data-time-series`) use compact indexed column names (`m1_fracdiff_price` .. `m16_scheduled_event_clock`), whereas REST/GCS Parquet evaluation partitions (`GET /api/v1/data/parquet`) expose the verbose canonical column identifiers (`target_fracdiff_price` .. `scheduled_event_clock`) alongside the 10 bi-temporal & Security Master metadata columns (`timestamp`, `window_start`, `window_end`, `asset_symbol`, `asset_ticker`, `dub_asid`, `openfigi`, `provenance_tier`, `bar_sequence_id`, `is_warmup_certified`).

```sql
CREATE TABLE alt_data_time_series_mt (
    window_start TIMESTAMP(3) NOT NULL,
    window_end TIMESTAMP(3) NOT NULL,
    asset_ticker VARCHAR(32) NOT NULL,
    dub_asid VARCHAR(16) NOT NULL,
    openfigi VARCHAR(12),
    provenance_tier VARCHAR(32) NOT NULL,
    bar_sequence_id BIGINT NOT NULL,
    is_warmup_certified BOOLEAN NOT NULL,
    m1_fracdiff_price DOUBLE NOT NULL,
    m2_log_volatility DOUBLE NOT NULL,
    m3_ofi_vpin DOUBLE NOT NULL,
    m4_derivatives_state DOUBLE NOT NULL,
    m5_informed_market_prob DOUBLE NOT NULL,
    m6_entropy_conviction_psi DOUBLE NOT NULL,
    m7_manipulation_haircut_mu DOUBLE NOT NULL,
    m8_informed_capital_imbalance DOUBLE NOT NULL,
    m9_fj_belief_state DOUBLE NOT NULL,
    m10_impact_flow DOUBLE NOT NULL,
    m11_vector_drift DOUBLE NOT NULL,
    m12_conflict_entropy DOUBLE NOT NULL,
    m13_solvency_score DOUBLE NOT NULL,
    m14_dynamic_correlate DOUBLE NOT NULL,
    m15_macro_liquidity DOUBLE NOT NULL,
    m16_scheduled_event_clock DOUBLE NOT NULL,
    PRIMARY KEY (window_end, dub_asid) NOT ENFORCED
) WITH (
    'connector' = 'confluent-kafka',
    'topic' = 'alt-data-time-series',
    'value.format' = 'json'
);
```

### 6.1 Sample 5-Minute JSON Streaming Record

```json
{
  "window_start": "2026-09-22T04:40:00Z",
  "window_end": "2026-09-22T04:45:00Z",
  "asset_ticker": "BTC/USD",
  "dub_asid": "DUB-ASID-000101",
  "openfigi": "BBG0013HG9X2",
  "provenance_tier": "TIER_1_REDISTRIBUTABLE",
  "bar_sequence_id": 184920,
  "is_warmup_certified": true,
  "m1_fracdiff_price": 0.358412,
  "m2_log_volatility": 0.042819,
  "m3_ofi_vpin": 0.314092,
  "m4_derivatives_state": 0.189204,
  "m5_informed_market_prob": 0.642000,
  "m6_entropy_conviction_psi": 1.842109,
  "m7_manipulation_haircut_mu": 0.941200,
  "m8_informed_capital_imbalance": 0.412890,
  "m9_fj_belief_state": 0.289104,
  "m10_impact_flow": 0.512049,
  "m11_vector_drift": 0.118402,
  "m12_conflict_entropy": 0.084120,
  "m13_solvency_score": 0.912000,
  "m14_dynamic_correlate": 0.224190,
  "m15_macro_liquidity": 0.641029,
  "m16_scheduled_event_clock": 0.142857
}
```

---

## 7. Security Master Symbology, Archive Provenance, & Dual-Mode Market Data Licensing

### 7.1 Point-in-Time Security Master & Graveyard Universe (`DUB-ASID`)

To eliminate ticker-reuse errors and survivorship bias across delisted equities, restructured sovereigns, and expired prediction contracts, every tensor partition enforces a bi-temporal Security Master mapping:
- **`dub_asid` (Immutable Primary Key):** Permanent Dubstrata Asset & Entity Identifier that never changes across corporate name changes, ticker symbol reassignments, or delistings.
- **Cross-Identifier Mapping:** Point-in-time resolution (`[effective_from, effective_to]`) to `OpenFIGI`, `SEC CIK`, `ISIN`, `CUSIP`, and Polygon ERC-1155 `conditionId`.
- **Graveyard Retention:** Delisted corporate issuers and resolved/abandoned prediction contracts remain in historical Parquet partitions through their exact terminal settlement bar ($t_{\text{delist}}$).

### 7.2 Archive Depth Provenance Tiers (`provenance_tier`)

Every record in the 2-year historical Parquet lake (`212,314+` 5-minute bars) is tagged with its exact temporal provenance tier:
- **`TIER_1_LIVE_RFC3161` (`2026-08-01` – Present):** Native live capture stamped at the millisecond of ingestion ($t_{\text{ingest}}$) by Dubstrata's ingestion daemons with cryptographic RFC 3161 Trusted Timestamping Authority tokens. Used for all certified out-of-sample forward evaluations.
- **`TIER_2_PIT_RECONSTRUCTED` (`2024-09-01` – `2026-07-31`):** Point-in-time historical backfill reconstructed from immutable primary archival timestamps (Polygon mainnet `block.timestamp`, SEC EDGAR microsecond `acceptanceDateTime`, and GDELT 15-minute publication epochs) with conservative propagation lag buffers ($\Delta t_{\text{prop}}$) applied for Fractional Differentiation ($K=1,332$) warm-up and long-horizon regime calibration.

### 7.3 Dual-Mode Q1 Market Data Licensing Boundary (Zero Exchange Liability)

Dubstrata is a **Causal Knowledge Graph and Prediction Market Alternative Data Provider (Quadrants 2–4)**, not a raw Traditional Finance (TradFi) Level-3 tick redistributor. To protect buy-side clients from unauthorized exchange redistribution audits (NYSE/CTA, Nasdaq/UTP, CME, OPRA), Quadrant 1 (Channels 1–4) operates under a strict **Dual-Mode Licensing Architecture**:
1. **Mode A — Turnkey Direct Delivery (Digital Assets & Prediction Markets + BYOL TradFi Join):**
   - For API-redistributable public orderbooks (Digital Assets `BTC/USD`, `ETH/USD`, `SOL/USD`, Polymarket CTF, Kalshi public endpoints), Dubstrata delivers all 16 channels turnkey (`m1`–`m16`).
   - For license-restricted TradFi equities, futures, and options (`SPY`, `QQQ`, `NVDA`, `CL=F`), Dubstrata delivers the **12-Channel Alternative & Causal Tensor (`m5`–`m16`)** plus open-source Python/Flink **BYOL Q1 Join Adapters** (`dubstrata.JoinAdapter`) that compute `m1`–`m4` locally inside the client's VPC using the client's existing licensed feeds (**BMLL Technologies, Databento, algoseek, LSEG, or Bloomberg B-PIPE**). Zero exchange-licensed TradFi tick data crosses Dubstrata's network boundary.
2. **Mode B — Managed Derived-Data & Pass-Through Venue Licensing (Enterprise Add-On):**
   - For desks requesting turnkey 16-channel pre-materialized tensors on TradFi underlyings without running local Q1 joiners, Dubstrata provisions the underlying feed via authorized **Vendor-of-Record pass-through billing** (via Databento Enterprise / Cboe Hanweck / Exchange Derived-Data Agreements) itemized transparently as a pass-through COGS schedule on the client Order Form.

---

*© 2026 Dubstrata Pty Ltd. All metrics are provided strictly as observational alternative data and statistical research for institutional modeling purposes.*
