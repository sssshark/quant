# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python environment (mandatory)

**Run everything with `~/miniconda3/bin/python`** (Python 3.13, all deps installed). This is the only supported interpreter here.

- Do **not** use termux/system python (cmake fake-hangs compiling deps).
- Do **not** `pip install akshare` (IP rate-limiting, poor quality). `bt.py` imports akshare and therefore does **not** run in this env — the working backtest path is `engine.py` (pulls data over tushare HTTP, no akshare).
- Run scripts directly: `~/miniconda3/bin/python engine.py real`. Use `-u` for unbuffered output when running in the background (`python -u diag/x.py 2>&1`), else block-buffering hides output.

## Commands

```bash
~/miniconda3/bin/python engine.py                # default "real": 等权+风控 vs 基准 + chart
~/miniconda3/bin/python engine.py <mode>         # see main() for full list; key modes below
~/miniconda3/bin/python test_momentum.py         # runs all tests, prints npass/N, exit code
~/miniconda3/bin/python -m pytest test_momentum.py::test_multi_k1_equivariance   # single test
~/miniconda3/bin/python diag/diag_sector.py      # a diagnostic in diag/
~/miniconda3/bin/python live.py                  # paper/DRY_RUN live skeleton (miniQMT)
```

`engine.py` modes (dispatched in `main()`): `real` (default), `nomomentum`, `regime`, `multi`, `attrib`/`mfat` (CAPM/multi-factor attribution), `sweep`, `robust`, `wf`, `pcv`, `pbo`, `boot`, `bootmom`, `dsr`, `universe`, `bondstress`, `freezetest`, `turnover`, `review`. Modes `real/regime/attrib/mfat/nomomentum/bondstress/freezetest/multi` load data with `with_limits=True` (T+1 fills + limit masks); the rest keep the legacy no-limit path for historical comparability.

## Architecture

The central design rule: **strategy logic lives in exactly one place** — `cta.decide_targets()` — and the backtest (`engine.py`), Backtrader cross-check (`bt.py`), and live trader (`live.py`) all call it. "回测怎么选、实盘就怎么选" — change the strategy once, it propagates everywhere. Do not duplicate the ~20 `decide_targets` parameters into another module; if you need a new knob, add it to `decide_targets` and thread it through.

- **`cta.py`** — the "brain". Defines `POOL` (7 ETFs, the *only* universe source), `DEFENSE` (国债), all strategy params, `decide_targets` (single blended-momentum signal + layered risk controls: vol-target → trend filter → crash protection → drawdown control), and the A-share limit table `_limit()`.
- **`engine.py`** (1700+ lines) — vectorized backtest + the entire research/robustness suite. `backtest()` is the core loop: month-end signal on T, **fill T+1, returns accrue from T+2** (`weights.shift(1)`), costs = turnover × (commission+slippage). The `strategy=None` path calls `decide_targets` directly; passing a `Strategy` object delegates decisions to it.
- **`multi_strategy.py`** — the multi-strategy (B型) composition layer. `Strategy` protocol (`name`/`universe`/`target`) → `CTAStrategy` (wraps `decide_targets`, default `hold_all=True`), `BondMomentumStrategy` (国债时序动量, the validated second strategy, Sharpe ~1.5, corr −0.06~−0.11 with CTA), `MultiStrategy` (equal-weight combiner), `RiskParityMulti` (rolling 1/σ allocation, Sharpe ~1.91). **Multiple alphas = multiple `Strategy` objects stacked at the `MultiStrategy` layer — never stuff extra factors inside a single strategy.**
- **`bt.py`** — Backtrader event-driven cross-check of `engine.py` (broker-managed fills). Requires backtrader+akshare → not runnable in this env; kept as an alternate-execution reference.
- **`live.py`** — miniQMT live skeleton. `BROKER="paper"` uses `paper_broker.py` (local drop-in); `"qmt"` connects real miniQMT. Reads `hold_all` from `live_config.json`. `DRY_RUN=True` prints the plan without trading.
- **`paper_broker.py`** — stateful local paper broker (JSON-persisted, accumulates across runs).
- **`crossmarket.py`** — out-of-sample US-ETF validation: `_us_globals()` context manager swaps `cta.POOL` to a US universe, runs the *same* strategy with *same* params, restores on exit. Proves the edge isn't A-share-pool-specific.
- **`diag/`** — standalone diagnostic scripts (each gates its own `sys.path` to the repo root). Each new candidate strategy must pass "三道关" (three gates) before promotion: ① 互补 `|corr|<0.3` with the base, ② 自身 Sharpe `>0.5`, ③ 组合增益 50/50 > max(individuals). Negative results (均值回归/配对/板块选股/供需期限结构) are documented there and kept — they are the methodology record.
- **`test_momentum.py`** — pytest-compatible regression suite. Plain `python test_momentum.py` also runs it.
- **`量化笔记/`** — zero-base quant notes (Chinese). See writing-style rule below.

## Critical invariants (don't break these)

- **K=1 equivariance** (`test_multi_k1_equivariance`): a single-strategy `MultiStrategy` via `backtest(strategy=…)` must reproduce the `decide_targets` direct-path NAV **pointwise** (atol 1e-9). This is the refactor's lifeline — any change to decision/execution must preserve it. Equivalent: K=2 stacking two identical CTAs == one CTA.
- **No lookahead in pool selection.** Universe members are chosen on priors (asset class, listing year) only — **never** by backtest return. (B1 pool-expansion and B2 tiering were tested harmful.)
- **`target` flat-position gotcha**: returning `{}` from `target()` is treated by `backtest` as "no-op, keep old position". To go flat you must return `{code: 0.0}` (non-empty dict). `BondMomentumStrategy` documents this explicitly.
- **Adjusted prices (前复权)**: use `engine._qfq_from_pre_close(close, pre_close)` — aligns splits/dividends via the `pre_close` column. tushare `pro_bar`'s `adj` does **not** work for ETFs; pull `fund_daily` with `close,pre_close` and adjust manually.
- **Limit table** (`cta._limit`): 159 segment is *mixed* — `159915` (创业板) = ±20% but cross-border `159941` = ±10%; `588` prefix (科创板) = ±20%; everything else default ±10%. Any new code added to POOL must be classified correctly, or `_apply_target_with_limits` will mis-handle fills.
- **`diag/` scripts**: each must `sys.path.insert(0, repo_root)` at import and compute `HERE` **up two levels** (`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`) to find the root `.tushare_token`. `runpy.run_path` does *not* add the script's dir to `sys.path` (unlike `python x.py`) — verify with a real interpreter invocation.

## Data & secrets

- tushare over proxy `https://fastapic.stockai888.top`; token read from `.tushare_token` (or `TUSHARE_API`/`TUSHARE_TOKEN` env). Retry with backoff on the proxy's intermittent SSL/read timeouts.
- **Gitignored, never commit**: `.tushare_token`, `live_config.json`, `paper_account.json`, `last_rebalance.txt`, `usdata/`, `ETF动量轮动_回测结果.png`. Verify before every commit with `git check-ignore <file>`.

## Writing & conventions

- Module names are lowercase snake (`cta`, `engine`, `bt`, `live`, `multi_strategy`); the entry points were renamed from the old `etf_momentum_*` prefix (do not reintroduce that name).
- **量化笔记 style (per user)**: term explanations must be extremely detailed and aimed at a zero-base reader — but **plain and direct, no metaphors/analogies**. Explain the concept itself, don't reach for a "it's like…" comparison.
- **`IMPROVEMENTS.md` is the canonical change log** (A/C/D/E/F/G/H/J/M prefixes, e.g. `F8 多因子归因`, `M3 国债时序动量`). Cite these IDs in commits. When you close/open an item, flip its `- [ ]`→`- [x]`, add a 解决记录 row, and update the 总/已完/待办 counters in the 进度 block. As of last update: 84 total, 61 done; all [高]/[中] cleared, the open 23 are [低] (out-of-scope / pending live / stage-2).
- Live default recommendation is `hold_all=true`: cross-market/out-of-sample tests (J1/J2/J5/G3) found momentum stock-selection has no reliable alpha (PBO 0.67); equal-weight-all-with-risk-control is the robust deployment. `hold_all=false` (selection) is kept only as a demonstration.
