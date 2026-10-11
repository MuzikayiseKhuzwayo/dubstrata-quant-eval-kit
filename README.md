# Dubstrata Institutional Quantitative Evaluation Guide

## Autonomous AI Agent & Quantitative Engineer Setup Specification

> **Target Audience:** Autonomous AI Assistants (Claude Code, Cursor, Windsurf, Copilot, ChatGPT), Systematic Quantitative Researchers, Risk Engineers.  
> **Objective:** Deterministic, zero-drift environment setup and execution of the Dubstrata 30-Day Institutional Evaluation Suite.  
> **Portability Guarantee:** Zero repository or local framework dependencies. Connects via pure REST over mutual TLS to the Dubstrata API CDN (`https://api.dubstrata.com`).

---

## 🤖 Instructions for AI Coding Assistants

If you are an AI assistant (Claude Code, Cursor, Windsurf, Aider, etc.) tasked with evaluating Dubstrata:

1. **Execute Phase 1:** Create an isolated directory and install the 5 required PyData packages (`requests`, `pandas`, `numpy`, `matplotlib`, `pyarrow`).
2. **Execute Phase 2:** Prompt the user for their `DUBSTRATA_API_KEY` (or read it from their environment) and create `.env`.
3. **Execute Phase 3:** Run the evaluation notebook (`quant_evaluation_kit.ipynb`).
4. **Compare Outputs:** Validate test outputs against the **Acceptance Criteria Matrix** in Phase 4. Report the audited scores to the user.

---

## Phase 1: Environment Provisioning (One-Liner Setup)

Run the following commands in your shell to establish an isolated evaluation environment:

### macOS / Linux / Bash

```bash
# 1. Create and enter evaluation workspace
mkdir -p dubstrata-eval && cd dubstrata-eval

# 2. Initialize virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install core evaluation libraries
pip install --upgrade pip
pip install requests pandas numpy matplotlib pyarrow
```

### Windows (PowerShell)

```powershell
# 1. Create and enter evaluation workspace
New-Item -ItemType Directory -Force -Path dubstrata-eval; Set-Location dubstrata-eval

# 2. Initialize virtual environment (Python 3.10+)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install core evaluation libraries
python -m pip install --upgrade pip
pip install requests pandas numpy matplotlib pyarrow
```

---

## Phase 2: Authentication & Environment Configuration

Request an institutional evaluation API key from the Dubstrata team (`partners@dubstrata.com` or via your partner portal).

Create a `.env` file in your `dubstrata-eval` directory:

```bash
# Dubstrata Production API Base URL
DUBSTRATA_API_BASE="https://api.dubstrata.com"

# Institutional Bearer Token (Provided by Dubstrata)
DUBSTRATA_API_KEY="your_provisioned_api_key_here"

# Canonical Evaluation Asset (Default: BTC/USD)
ASSET_SYMBOL="BTC/USD"
```

Alternatively, export them directly in your shell session:

```bash
export DUBSTRATA_API_BASE="https://api.dubstrata.com"
export DUBSTRATA_API_KEY="your_provisioned_api_key_here"
export ASSET_SYMBOL="BTC/USD"
```

---

## Phase 3.1: Run the Evaluation Notebook (`quant_evaluation_kit.ipynb`) Locally

If you prefer an interactive notebook interface with live Matplotlib visualizations:

1. Open [`quant_evaluation_kit.ipynb`](../../../../notebooks/quant_evaluation_kit.ipynb) in your Jupyter session.
2. Toggle **Settings &rarr; Internet &rarr; Internet On**.
3. Run all the cells in the notebook.

---

## Phase 3.2: Interactive Jupyter / Kaggle / Colab Evaluation in the Cloud

If they prefer an interactive notebook interface with live Matplotlib visualizations without having to install Python, they can use the following:

1. Copy [`quant_evaluation_kit.ipynb`](../../../../notebooks/quant_evaluation_kit.ipynb) into your Jupyter, Kaggle, or Google Colab session.
2. In Kaggle: Toggle **Settings &rarr; Internet &rarr; Internet On**.
3. Set your `DUBSTRATA_API_KEY` in the top configuration cell.
4. Select **Run All Cells**.

---

## Phase 4: Pass/Fail Acceptance Criteria Matrix

| Check ID | Product Offering | Target Metric | Assertion Condition | Required Result |
| :--- | :--- | :--- | :--- | :---: |
| **A-01** | Offering A (Fuel) | Canonical Feature Channels | Exactly 16 channels present | **PASS** |
| **A-02** | Offering A (Fuel) | Timestamp Monotonicity | `timestamps.is_monotonic_increasing == True` | **PASS** |
| **A-03** | Offering A (Fuel) | Optimal FracDiff Parameter | $d^* = 0.42$ with ADF test $p < 0.01$ | **PASS** |
| **A-04** | Offering A (Fuel) | Memory Retention | Pearson $r > 0.90$ with raw price | **PASS** |
| **B-01** | Offering B (Engine) | Directional Conviction | $\Psi_{\text{CSI}} \in [0.0, 1.0]$ | **PASS** |
| **B-02** | Offering B (Engine) | 9-Quantile Monotonicity | $q_{10} \le q_{20} \le \dots \le q_{90}$ | **PASS** |
| **B-03** | Offering B (Engine) | Multi-Horizon Uncertainty | $q_{10}^{(h)} < q_{50}^{(h)} < q_{90}^{(h)}$ across all horizons $h$ | **PASS** |
| **C-01** | Offering C (HRX) | 4-Quadrant Decomposition | Q1, Q2, Q3, Q4 sum to $100\%$ | **PASS** |
| **C-02** | Offering C (HRX) | Deletion Gain Proof | $\Delta\text{Gain} = \mathcal{L}(\mathbf{X}_{\setminus\text{top}}) - \mathcal{L}(\mathbf{X}) > 0$ ($+0.1956$) | **PASS** |

---

## Phase 5: Diagnostic & Status Code Reference

| HTTP Status | Meaning | Remediation |
| :---: | :--- | :--- |
| **`200 OK`** | Successful evaluation response | Proceed to next assertion. |
| **`401 Unauthorized`** | Missing or invalid API key | Verify `DUBSTRATA_API_KEY` matches your assigned bearer credential. |
| **`402 Payment Required`** | Evaluation quota exhausted | Contact `partners@dubstrata.com` to top up institutional evaluation credits. |
| **`429 Too Many Requests`** | Rate limit throttled | Add a 1-second sleep between requests or request Tier-2 institutional rate limits. |
| **`503 Degraded Source`** | Fail-closed data protection | Upstream exchange data or prediction market feed degraded; fail-closed engine prohibits synthetic extrapolation during live market dislocations. |

---

## Technical Support & Institutional Onboarding

* **Technical Lead:** `muzi@dubstrata.com`
* **Command Center Portal:** [`Dubstrata Partners`](https://partners.dubstrata.com)
* **API Documentation:** To be provided on request
