# Dubstrata Quantitative Evaluation Kit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuzikayiseKhuzwayo/dubstrata-quant-eval-kit/blob/main/quant-evaluation-kit.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/MuzikayiseKhuzwayo/dubstrata-quant-eval-kit/blob/main/quant-evaluation-kit.ipynb)

Independent point-in-time (PIT) signal verification and 12-month multi-regime walk-forward evaluation workbench for the **Dubstrata 16-Channel Multivariate Tensor** and **De-Sybiled Prediction Market (PM) Layer**.

---

## Why This Repo Exists

To rigorously validate an alternative signal, a quantitative researcher should not have to trust pre-computed API outputs, static structural priors, or zero-lag execution assumptions. This repository bundles a **raw 12-month multi-regime point-in-time 5-minute historical dataset** (`data/dubstrata_btc_pit_5m_sample.parquet`, **105,120 bars / 365 days**: `2025-09-22T04:50:00Z` to `2026-09-22T04:45:00Z`) so you can stress-test the signal **100% offline**:

1. **All 4 Bi-Temporal Timestamps on Every 5-Minute Row (`105,120` Bars)**:
   - `t_valid` (`window_start`): Start of the 5-minute observation validity interval ($T_{\mathrm{bar\_close}} - 300\text{ s}$).
   - `t_event`: Final source matching-engine / Polygon CTF block / SEC EDGAR event timestamp inside the 5-minute bar ($T_{\mathrm{valid}} < T_{\mathrm{event}}$).
   - `t_ingest`: Immutable datastore transaction commit timestamp ($T_{\mathrm{event}} < T_{\mathrm{ingest}} \le T_{\mathrm{bar\_close}}$). Any observation committed after `bar_close_time` is hard-gated out of bar $t$ and deferred to bar $t+1$.
   - `bar_close_time` (`window_end`): The 5-minute decision cutoff $T_{\mathrm{bar\_close}}$.
2. **Raw Un-Differentiated BTC Spot OHLCV & 1-Bar Execution Delay (`btc_open` at $t+1$)**:
   - `btc_open`, `btc_high`, `btc_low`, `btc_close`, `btc_volume` are aligned at each 5-minute bar close.
   - To eliminate zero-lag close-to-close inflation, all forward return targets and backtest fills enter at the **next bar's open (`btc_open` at $t+1$)**:
     $$r_{t+1 \to t+1+h}^{\mathrm{exec}} = \frac{\mathrm{btc\_open}_{t+1+h}}{\mathrm{btc\_open}_{t+1}} - 1$$
3. **Zero Static Priors ($\mathbf{w}_{\mathrm{prior}} = \mathbf{0}$) & Monotonic Rank Correlation Audit**:
   - Walk-forward estimation uses **100% unconstrained out-of-sample Ridge regression** (`StandardScaler + Ridge`) fitted strictly on purged historical training windows (`purge = HORIZON_BARS + 1`) with zero hardcoded weights.
   - Both **Pearson IC** (linear) and **Spearman Rank IC** (monotonic rank) are reported side by side across all horizons (`15m`, `1h`, `4h`) and across all 12 monthly regimes (`2025-10` to `2026-09`).
4. **No EMA Position Smoothing & Explicit Lo (2002) / Newey-West Sharpe Adjustment**:
   - Module 5 uses a discrete conviction-gated hysteresis state machine (no `ewm` position smoothing) and reports both unadjusted annualized Sharpe and **Lo (2002) / Newey-West autocorrelation-adjusted Sharpe ($\mathrm{SR}_{\mathrm{NW}}(q=6)$)** across **Blended (`3.5 bps`/leg)** and **Taker (`6.5 bps`/leg)** cost schedules.

---

## Quick Start (Zero API Key Required)

```bash
git clone [https://github.com/MuzikayiseKhuzwayo/dubstrata-quant-eval-kit.git](https://github.com/MuzikayiseKhuzwayo/dubstrata-quant-eval-kit.git)
cd dubstrata-quant-eval-kit
pip install numpy pandas pyarrow scipy scikit-learn matplotlib jupyter
jupyter notebook quant-evaluation-kit.ipynb

```

---

## Summary of 12-Month Unconstrained Walk-Forward Results ($\mathbf{w}_{\mathrm{prior}} = \mathbf{0}$, 1-Bar Execution Delay)

### 1. Full 12-Month Walk-Forward Out-of-Sample Ablation (`N = 96,467` OOS 5m Bars after 30-Day Burn-In)

| Model / Feature Set | Horizon | OOS Bars | OOS Pearson IC | OOS Spearman Rank IC | Rank IC $t$-stat | Rank IC $p$-value | Rank IC Uplift ($\Delta\mathrm{Rank\ IC}$) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Macro/Micro Baseline (No PM)** | **1h** ($h=12$) | `96,467` | `-0.0031` | `+0.0088` | `+2.73` | $6.32 \times 10^{-3}$ | `0.0000` |
| **De-Sybiled PM Layer Only** | **1h** ($h=12$) | `96,467` | **`+0.0402`** | **`+0.0396`** | **`+12.30`** | $9.13 \times 10^{-35}$ | **`+0.0308`** |
| **Combined (`Baseline + PM Layer`)** | **1h** ($h=12$) | `96,467` | **`+0.0343`** | **`+0.0423`** | **`+13.16`** | $1.69 \times 10^{-39}$ | **`+0.0335`** |
| **Macro/Micro Baseline (No PM)** | **4h** ($h=48$) | `96,431` | `-0.0070` | `-0.0215` | `-6.69` | $2.22 \times 10^{-11}$ | `0.0000` |
| **De-Sybiled PM Layer Only** | **4h** ($h=48$) | `96,431` | **`+0.0150`** | **`+0.0105`** | **`+3.25`** | $1.14 \times 10^{-3}$ | **`+0.0320`** |
| **Combined (`Baseline + PM Layer`)** | **4h** ($h=48$) | `96,431` | **`+0.0130`** | **`+0.0074`** | **`+2.29`** | $2.17 \times 10^{-2}$ | **`+0.0289`** |

### 2. 1-Bar Delayed Walk-Forward Execution Backtest (No EMA Smoothing, Lo / Newey-West Adjusted Sharpe)

| Strategy | Cost Schedule | Gross Return (%) | Net Return (%) | Unadjusted Net Sharpe | Lo / Newey-West Sharpe ($q=6$) | Max Drawdown (%) | Gated Hit Rate (%) | Net Profit Factor | Rebalance Events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Macro/Micro Baseline (No PM)** | `3.5 bps` (Blended) | `+19.56%` | `+5.55%` | `+0.47` | `+0.48` | `10.22%` | `51.93%` | `1.11` | `346` |
| **De-Sybiled PM Layer Only** | `3.5 bps` (Blended) | **`+51.78%`** | **`+31.04%`** | **`+2.16`** | **`+2.37`** | **`5.26%`** | **`53.10%`** | **`1.38`** | `413` |
| **Combined (`Baseline + PM Layer`)** | `3.5 bps` (Blended) | **`+37.51%`** | **`+20.39%`** | **`+1.54`** | **`+1.70`** | **`7.28%`** | **`52.77%`** | **`1.30`** | `369` |
| **Macro/Micro Baseline (No PM)** | `6.5 bps` (Taker) | `+19.56%` | `-5.15%` | `-0.31` | `-0.32` | `11.45%` | `50.09%` | `1.06` | `346` |
| **De-Sybiled PM Layer Only** | `6.5 bps` (Taker) | **`+51.78%`** | **`+15.53%`** | **`+1.18`** | **`+1.31`** | **`6.07%`** | **`51.42%`** | **`1.30`** | `413` |
| **Combined (`Baseline + PM Layer`)** | `6.5 bps` (Taker) | **`+37.51%`** | **`+7.42%`** | **`+0.63`** | **`+0.70`** | **`10.62%`** | **`51.33%`** | **`1.23`** | `369` |

---

## References: Prediction Market Microstructure from a Systematic Trading Perspective

1. **Wolfers, J., & Zitzewitz, E. (2004 / 2006)** — *"Prediction Markets"* (*Journal of Economic Perspectives*, 18(2), 107–126) & *"Interpreting Prediction Market Prices as Probabilities"* (NBER Working Paper w12200). Foundational derivation of when binary contract prices converge to risk-neutral vs. objective probabilities under heterogeneous beliefs and wealth constraints.
2. **Hanson, R. (2003 / 2007)** — *"Combinatorial Information Market Design"* (*Information Systems Frontiers*, 5(1), 107–119) & *"Logarithmic Market Scoring Rules for Modular Combinatorial Information Aggregation"*. Establishes the information-theoretic equivalence between market scoring rules, Kullback-Leibler divergence ($\Psi$), and inventory-driven automated market makers vs. central limit order books (CLOBs).
3. **Manski, C. F. (2006) & Ottaviani, M., & Sørensen, P. N. (2007)** — *"Interpreting the Predictions of Prediction Markets"* (*Economics Letters*, 91(3), 425–429) & *"Outcome Manipulation in Corporate Prediction Markets"* (*Journal of the European Economic Association*, 5(2–3), 554–563). Analyzes strategic distortion and wash-volume spoofing in binary contracts—the theoretical motivation behind our on-chain Union-Find wallet clustering and manipulation haircut $\mu_{\mathrm{risk}}$.
4. **Kyle, A. S. (1985), Lo, A. W. (2002), & Easley, D., López de Prado, M. M., & O'Hara, M. (2012)** — *"Continuous Auctions and Insider Trading"* (*Econometrica*, 53(6), 1315–1335), *"The Statistics of Sharpe Ratios"* (*Financial Analysts Journal*, 58(4), 36–52), & *"Flow Toxicity and Liquidity in a High-Frequency World"* (*Review of Financial Studies*, 25(5), 1457–1493). Connects informed order flow toxicity in binary event books to price discovery lead-times and establishes autocorrelation-adjusted Sharpe inference.

