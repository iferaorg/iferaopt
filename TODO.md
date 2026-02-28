# TODO — iferaopt

## Step 1: Storage Layer
- [ ] Implement Parquet read/write helpers with Hive partitioning
- [ ] Implement Zarr tensor save/load for GPU-ready data
- [ ] Add DuckDB/Polars query utilities for exploration
- [ ] Set up DVC for data version control (optional)

## Step 2: Data Acquisition & Preprocessing
- [ ] Implement ThetaData REST API client for fetching historical SPX option quotes
- [ ] Build intraday 1-min OHLC pipeline for SPX (9:30–16:00 ET)
- [ ] Store raw data as Hive-partitioned Parquet (zstd) under `data/raw/`
- [ ] Compute 60-minute Opening Range (OR) per trading day
- [ ] Detect first post-10:30 breakout above OR High
- [ ] Fetch option chain at entry (bid/ask, delta, IV) for candidate strikes
- [ ] Build economic calendar integration (FOMC, CPI, holidays)
- [ ] Convert processed data to GPU-ready tensors (Zarr or torch)

## Step 3: Metrics & Evaluation
- [ ] Implement vectorized Sortino, max drawdown, and Calmar computation on GPU
- [ ] Implement aggregated OOS go-live checks (Sortino > 1.8, Calmar > 1.2, maxDD < 18%)
- [ ] Implement slippage sensitivity analysis (+10c, +20c scenarios)
- [ ] Implement parameter and feature stability diagnostics across windows

## Step 4: Base Strategy Implementation
- [ ] Implement put credit spread construction (short leg by delta, long leg by width)
- [ ] Implement realistic slippage model (% of bid-ask spread with minimum floor)
- [ ] Implement minute-level stop-loss checking on re-priced spread
- [ ] Implement position sizing (% of equity risk per trade)
- [ ] Implement cash-settlement exit at 16:00 ET

## Step 5: Purged Cross-Validation
- [ ] Implement `PurgedKFold` splitter with configurable embargo (days)
- [ ] Integrate purged CV into core parameter optimization (Step 7)
- [ ] Integrate purged CV into RF threshold tuning (Step 8)

## Step 6: GPU Backtester
- [ ] Build `GPUBacktester` class (PyTorch module) for vectorized strategy simulation
- [ ] Implement batched parameter evaluation (Latin-Hypercube sampling, 10k–25k combos)
- [ ] Validate GPU results against a simple CPU reference implementation

## Step 7: Walk-Forward Optimization (WFO) Framework
- [ ] Implement rolling window generator (18-month IS / 3-month OOS / 1-month step)
- [ ] Implement per-window calibration loop (core params → filter selection → refit)
- [ ] Implement OOS equity curve aggregation across all windows
- [ ] Compute WFO metrics: WFE, % profitable windows, parameter stability

## Step 8: Feature Selection via Shallow Random Forest
- [ ] Train shallow RF classifier (depth 1–3, 1500 trees) on IS trade labels
- [ ] Implement probability-threshold grid search using purged inner validation
- [ ] Log top feature importances and threshold stability per window
- [ ] Add safeguards: `min_samples_leaf`, `class_weight='balanced'`

## Step 9: Live Deployment
- [ ] Build monthly re-calibration script (re-run WFO on latest 18-month IS)
- [ ] Export params/filters/threshold as JSON for trading bot
- [ ] Implement daily execution logic: OR calculation, filter check, spread placement
- [ ] Build monitoring dashboard for live vs. expected OOS metrics

## Step 10: Extended Features
- [ ] Add OR duration as an optimizable parameter (60, 75, 90 minutes)
- [ ] Add expanding (anchored) window variant for WFO
- [ ] Add regime detection (e.g., VIX-based) for dynamic window adjustment
- [ ] Add false-breakout rate monitoring for crowding detection
