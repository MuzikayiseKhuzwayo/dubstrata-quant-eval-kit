# Dubstrata Quantitative Evaluation Kit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuzikayiseKhuzwayo/dubstrata-quant-eval-kit/blob/main/quant-evaluation-kit.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/MuzikayiseKhuzwayo/dubstrata-quant-eval-kit/blob/main/quant-evaluation-kit.ipynb)

Independent point-in-time (PIT) signal verification and walk-forward evaluation workbench for the **Dubstrata 16-Channel Multivariate Tensor** and **De-Sybiled Prediction Market (PM) Layer**.

---

## Why This Repo Exists

To properly validate an alternative signal, a quantitative researcher should not have to trust pre-computed API outputs or fixed targets. This repository includes a **raw point-in-time 5-minute historical sample** (`data/dubstrata_btc_pit_5m_sample.parquet`, **25,920 bars / 90 days**: `2026-06-24T04:50:00Z` to `2026-09-22T04:45:00Z`) so you can verify the signal **100% offline**:

1. **All 4 Bi-Temporal Timestamps on Every 5-Minute Row**:
   - `t_valid` (`window_start`): Start of the 5-minute observation validity interval ($T_{\text{bar\_close}} - 300\text{s}$).
   - `t_event`: Final source matching-engine / Polygon CTF block / SEC EDGAR event timestamp inside the 5-minute bar ($T_{\text{valid}} < T_{\text{event}}$).
   - `t_ingest`: Immutable datastore transaction commit timestamp ($T_{\text{event}} < T_{\text{ingest}} \le T_{\text{bar\_close}}$). Any observation committed after `bar_close_time` is hard-gated out of bar $t$ and deferred to bar $t+1$.
   - `bar_close_time` (`window_end`): The 5-minute decision cutoff $T_{\text{bar\_close}}$.
2. **Raw Un-Differentiated BTC Spot OHLCV Prices**:
   - `btc_open`, `btc_high`, `btc_low`, `btc_close`, `btc_volume` aligned at each 5-minute bar close so you can construct your own forward return targets $r_{t \to t+h} = \frac{\text{btc\_close}_{t+h}}{\text{btc\_close}_t} - 1$ over any horizon $h$.
3. **All 16 Point-in-Time Stationary Channels**:
   - **Q1 Microstructure**: `target_fracdiff_price` ($d^* = 0.42$), `target_log_volatility` (Garman-Klass), `microstructure_ofi_vpin`, `derivatives_surface_state`
   - **Q2 De-Sybiled Prediction Markets**: `informed_market_prob`, `entropy_conviction_psi` ($\Psi_{\text{CSI}}$), `manipulation_haircut_mu` ($\mu_{\text{risk}}$ Sybil defense down-scaler), `informed_capital_imbalance`
   - **Q3 Causal Knowledge Graph**: `friedkin_johnsen_belief`, `financial_impact_flow`, `perception_vector_drift`, `graph_conflict_entropy`
   - **Q4 Macro Solvency & Catalyst Clock**: `fundamental_solvency_score`, `dynamic_primary_correlate`, `macro_liquidity_factor`, `scheduled_event_clock`

---

## Quick Start (Zero API Key Required)

```bash
git clone https://github.com/MuzikayiseKhuzwayo/dubstrata-quant-eval-kit.git
cd dubstrata-quant-eval-kit
pip install numpy pandas pyarrow scipy scikit-learn matplotlib jupyter
jupyter notebook quant-evaluation-kit.ipynb
```

Inside `quant-evaluation-kit.ipynb`:
- **Module 1 (Window Selector)**: Set `START_TIME`, `END_TIME`, and `HORIZON_BARS` (e.g., `3` = 15m, `12` = 1h, `48` = 4h) to slice any window from the local 90-day Parquet file (`data/dubstrata_btc_pit_5m_sample.parquet`) or pull a custom range via `GET /api/v1/tensor/sample?start_time=...&end_time=...`.
- **Module 2 (4-Timestamp PIT & Zero-Lookahead Audit)**: Verifies `t_valid < t_event < t_ingest <= bar_close_time` on 100% of rows (`0` violations across 25,920 bars) and cross-correlates $\Delta \text{target\_fracdiff\_price}_t$ against past and future BTC returns ($r = -0.0009$ at lag $k=-1$, confirming zero future target leakage).
- **Module 3 (Independent Target Builder)**: Constructs forward spot returns $r_{t \to t+h} = \frac{\text{btc\_close}_{t+h}}{\text{btc\_close}_t} - 1$ directly from `btc_close` with a strict right-edge purge boundary.
- **Module 4 (Purged Walk-Forward Out-of-Sample Ablation)**: Compares `Macro/Micro Baseline (No PM)` vs. `De-Sybiled PM Layer Only` vs. `Combined (Baseline + PM Layer)` using a strictly purged expanding walk-forward loop (`tr_slice = slice(0, t_start - HORIZON_BARS)`).
- **Module 5 (Walk-Forward Realistic Cost & Slippage Stress-Test)**: Evaluates Gross vs. Net equity curves, Sharpe ratios, drawdowns, and profit factors under configurable `TAKER_FEE_BPS` (`5.0 bps`), `MAKER_FEE_BPS` (`1.0 bp`), `BLENDED_FEE_BPS` (`2.0 bps`), `HALF_SPREAD_BPS` (`1.0 bp`), and `KYLE_SLIPPAGE_BPS` (`0.5 bp`).
- **Module 6 (Optional Live API Appendix)**: Queries live `/api/v1/predict/quantiles` forecast cones if `DUBSTRATA_API_KEY` is set.

---

## Summary of Walk-Forward Out-of-Sample Results (Built from Raw `btc_close`)

| Window | Horizon | Macro/Micro Baseline OOS IC | De-Sybiled PM Layer OOS IC | Combined OOS IC | IC Uplift ($\Delta\text{IC}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **30-Day Window** (`2026-08-23` – `2026-09-22`) | **1h** ($h=12$) | `+0.0151` ($p=0.22$) | **`+0.0766`** ($p=4.6\times 10^{-10}$) | **`+0.0724`** ($p=3.8\times 10^{-9}$) | **`+0.0573`** |
| **30-Day Window** (`2026-08-23` – `2026-09-22`) | **4h** ($h=48$) | `+0.0416` ($p=7.7\times 10^{-4}$) | **`+0.1801`** ($p=8.8\times 10^{-49}$) | **`+0.1587`** ($p=4.0\times 10^{-38}$) | **`+0.1171`** |
| **90-Day Partition** (`2026-06-24` – `2026-09-22`) | **1h** ($h=12$) | `-0.0019` ($p=0.77$) | **`+0.0326`** ($p=1.4\times 10^{-6}$) | **`+0.0301`** ($p=8.6\times 10^{-6}$) | **`+0.0320`** |
| **90-Day Partition** (`2026-06-24` – `2026-09-22`) | **4h** ($h=48$) | `+0.0064` ($p=0.34$) | **`+0.0552`** ($p=3.7\times 10^{-16}$) | **`+0.0543`** ($p=1.1\times 10^{-15}$) | **`+0.0479`** |

---

## References: Prediction Market Microstructure from a Systematic Trading Perspective

1. **Wolfers, J., & Zitzewitz, E. (2004 / 2006)** — *"Prediction Markets"* (*Journal of Economic Perspectives*, 18(2), 107–126) & *"Interpreting Prediction Market Prices as Probabilities"* (NBER Working Paper w12200). Foundational derivation of when binary contract prices converge to risk-neutral vs. objective probabilities under heterogeneous beliefs and wealth constraints.
2. **Hanson, R. (2003 / 2007)** — *"Combinatorial Information Market Design"* (*Information Systems Frontiers*, 5(1), 107–119) & *"Logarithmic Market Scoring Rules for Modular Combinatorial Information Aggregation"*. Establishes the information-theoretic equivalence between market scoring rules, Kullback-Leibler divergence ($\Psi$), and inventory-driven automated market makers vs. central limit order books (CLOBs).
3. **Manski, C. F. (2006) & Ottaviani, M., & Sørensen, P. N. (2007)** — *"Interpreting the Predictions of Prediction Markets"* (*Economics Letters*, 91(3), 425–429) & *"Outcome Manipulation in Corporate Prediction Markets"* (*Journal of the European Economic Association*, 5(2–3), 554–563). Analyzes strategic distortion and wash-volume spoofing in binary contracts—the theoretical motivation behind our on-chain Union-Find wallet clustering and manipulation haircut $\mu_{\text{risk}}$.
4. **Kyle, A. S. (1985) & Easley, D., López de Prado, M. M., & O'Hara, M. (2012)** — *"Continuous Auctions and Insider Trading"* (*Econometrica*, 53(6), 1315–1335) & *"Flow Toxicity and Liquidity in a High-Frequency World"* (*Review of Financial Studies*, 25(5), 1457–1493). Connects informed order flow toxicity (VPIN / capital imbalance) in binary event books to price discovery lead-times ahead of underlying spot orderbooks.
