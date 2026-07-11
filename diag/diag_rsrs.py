# -*- coding: utf-8 -*-
"""诊断:RSRS(阻力支撑相对强度)作为大盘择时信号的有效性。

来源动机:RSRS(光大证券《再探技术分析与量化》系列)在本仓库 grep 0 命中——大盘择时
信号层是空白。现有趋势过滤是"沪深300 跌破 SMA(200)"(cta.decide_targets 第四步)。
RSRS 用价格高点(high)对低点(low)的 OLS 斜率刻画"阻力位/支撑位的相对强度",再标准化
成 z-score、用 r² 加权,是另一种信息源的大盘择时信号。本诊断回答:RSRS 作为大盘择时,
相对 SMA(200) 有无边际。

⚠ 这是诊断脚本(信号有效性检验),不改 cta.py。只有 RSRS 显著优于 SMA 才值得进入第二步
(替换 decide_targets 趋势过滤——那需走"decide_targets 新增参数 + 三道关"的正式流程,
而非直接改)。定位对齐 A7(EMA vs SMA 大盘择时信号检验):换一个大盘择时信号,隔离地看
信号本身的质量。预判参考 A6/A7/C6:大盘择时换更复杂的信号往往不如 SMA,RSRS 大概率
是又一个边际不显著的候选——但 RSRS 机制(阻力支撑强度)与均线方向不完全同源,值得一测。

口径:大盘择时直接作用在沪深300 ETF(BENCH=510300,可交易)上——看多持有、看空空仓
(持现金),不含 CTA 的 vol_target/选股,纯测"择时信号本身"。与 A7 同构(隔离信号层)。

RSRS 算法(光大研报口径):
  1. 取沪深300 指数(000300)近 N=18 日的 high/low,OLS: high ~ low,斜率 β。
  2. 滚动算 β 序列(每个交易日一个 β)。
  3. 修正标准分 = zscore(β, 过去 M=600 日) × r²(该次回归判定系数)。
     r² 加权:拟合越好(支撑/阻力关系越线性)的 β 越可信(M=600 取研报长窗口口径)。
  4. 信号:修正标准分 > 阈值 THR(默认 0.7)→ 看多;否则看空。

数据:沪深300 指数(000300)的 high/low(akshare index_zh_a_hist)。用指数而非 ETF:指数
无除权跳变,high/low 干净;ETF 除权日 high/low 会断、扭曲 RSRS。与 CTA 趋势过滤用
510300 close 同源(ETF 跟踪指数,差异极小)。择时回测的可交易标的用 510300 ETF close
(来自 engine.load_real),与 CTA 执行口径一致(月末调仓、T+1、含手续费/滑点)。

对比三组大盘择时(均在 510300 上):
  A. SMA(200):价 > MA200 持有,否则空仓(现有趋势过滤的纯信号版)
  B. RSRS 修正标准分 > THR:持有,否则空仓
  C. RSRS + SMA 共振:两者都看多才持有(双重确认)
配对 circular block bootstrap(block=21≈1月,n=2000):Δ=SR(候选)−SR(SMA),p=P(Δ≤0)。
分 regime + 关键熊市段(2015 股灾/2018 贸易战/2022)看避开大跌的能力。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo 根(找 cta/engine)
import numpy as np
import pandas as pd

import cta as mc
import engine as e
from engine import BENCH, COMMISSION, SLIPPAGE

N_RSRS = 18        # high~low 回归窗口(研报值)
M_STD = 600        # β 标准化窗口(研报长窗口口径)
THR = 0.7          # 修正标准分开仓阈值(研报值)
N_BOOT = 2000
BLOCK = 21


def load_hs300_index_ohlc():
    """沪深300 指数(000300)日 OHLC。双源容错:akshare index_zh_a_hist → fund_etf_hist_em(510300)。
    返回 DataFrame(high, low, close),index 为 Timestamp。失败给清晰错误(东财限频/网络)。"""
    import akshare as ak
    end = pd.Timestamp.today().strftime("%Y%m%d")
    last = None
    # 源1:000300 指数(首选,无除权扭曲)
    try:
        df = ak.index_zh_a_hist(symbol="000300", period="daily", start_date="20110101", end_date=end)
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期").sort_index()
        out = df[["最高", "最低", "收盘"]].astype(float)
        out.columns = ["high", "low", "close"]
        if len(out) > 252:
            return out
        last = "000300 指数数据不足"
    except Exception as ex:
        last = repr(ex)[:160]
    # 源2:510300 ETF OHLC(退路;除权日 high/low 可能跳变,标注此局限)
    try:
        df = ak.fund_etf_hist_em(symbol="510300", period="daily", start_date="20110101",
                                 end_date=end, adjust="hfq")
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期").sort_index()
        out = df[["最高", "最低", "收盘"]].astype(float)
        out.columns = ["high", "low", "close"]
        print("  ⚠ 000300 指数拉取失败,退回 510300 ETF 后复权 OHLC(除权日 high/low 可能扭曲 RSRS)")
        return out
    except Exception as ex:
        raise RuntimeError(
            f"akshare 拉沪深300 OHLC 双源均失败(东财限频/网络;建议配置 TUSHARE_TOKEN 走 tushare)。\n"
            f"  源1(000300 指数): {last}\n  源2(510300 ETF): {repr(ex)[:160]}")


def rsrs_beta_r2(high, low, N=N_RSRS):
    """RSRS:近 N 日 high 对 low 的 OLS 斜率 β 与判定系数 r² 序列。
    每个 t 用 [t-N+1, t] 的 (low=x, high=y) 做 OLS high ~ low。返回 (beta, r2) Series。"""
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    T = len(h)
    beta = np.full(T, np.nan)
    r2 = np.full(T, np.nan)
    for t in range(N - 1, T):
        x = l[t - N + 1:t + 1]
        y = h[t - N + 1:t + 1]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        xm, ym = x.mean(), y.mean()
        sxx = ((x - xm) ** 2).sum()
        if sxx < 1e-12:
            continue
        b = ((x - xm) * (y - ym)).sum() / sxx
        a = ym - b * xm
        ss_res = ((y - (a + b * x)) ** 2).sum()
        ss_tot = ((y - ym) ** 2).sum()
        beta[t] = b
        r2[t] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return pd.Series(beta, index=high.index), pd.Series(r2, index=high.index)


def rsrs_modified_score(beta, r2, M=M_STD):
    """修正标准分 = zscore(β, 过去 M 日) × r²。β 序列前 M-1 个为 NaN(标准化窗口不足)。"""
    mu = beta.rolling(M).mean()
    sd = beta.rolling(M).std()
    z = (beta - mu) / sd
    return (z * r2).astype(float)


def timing_backtest(px_bench, signal):
    """大盘择时回测:signal(bool Series,看多)→ 月末采样持有到下月末、T+2 吃收益口径对齐
    engine.backtest(signal T→execute T+1→w_lag=shift(1)→收益 T+2)、含成本。
    px_bench: 可交易标的(510300 ETF)close Series。返回 (nav, ret),从首个有效调仓日起。"""
    month_ends = px_bench.resample("ME").last().index
    rebal = [px_bench.index[px_bench.index <= me][-1] for me in month_ends if (px_bench.index <= me).any()]
    need = max(M_STD, mc.TREND_MA, 126)           # 信号预热:RSRS 的 M 日标准化 + SMA200 取大者
    rebal = [d for d in rebal if d >= px_bench.index[need]]
    if not rebal:
        return None, None
    rd_set = set(rebal)
    pos = signal.reindex(px_bench.index).where(signal.index.isin(rd_set)).ffill().fillna(0.0)
    pos_exec = pos.shift(1).fillna(0.0)             # T+1 成交(防收盘瞬时成交前视)
    w_lag = pos_exec.shift(1).fillna(0.0)           # 再 shift = T+2 吃收益(对齐 engine.backtest 口径)
    ret = px_bench.pct_change().fillna(0.0)
    gross = w_lag * ret
    turnover = (pos_exec - pos_exec.shift(1).fillna(0.0)).abs()
    cost = turnover * (COMMISSION + SLIPPAGE)
    net = (gross - cost).fillna(0.0)
    start = rebal[0]
    nav = (1 + net).cumprod().loc[start:]
    return nav / nav.iloc[0], net.loc[start:]


def _boot_vs_ref(cand, ref, n_boot=N_BOOT, block=BLOCK, seed=7):
    T = len(ref)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        idx = e._circ_block_idx(T, block, rng)
        boots[k] = e._ann_sharpe(cand[idx]) - e._ann_sharpe(ref[idx])
    return boots


def _row(name, nav, daily):
    if nav is None:
        return (name, float("nan"), float("nan"), float("nan"), float("nan"))
    q = e.perf(nav, daily)
    return name, q["年化"], q["夏普"], q["回撤"], q["Sortino"]


def main():
    print("加载数据...")
    res = e.load_real(with_limits=True)
    px = res[0]
    cb, cs = (res[1], res[2]) if isinstance(res, tuple) else (None, None)
    px_bench = px[BENCH].dropna()
    print(f"  ETF(510300) {px_bench.index[0].date()} ~ {px_bench.index[-1].date()}, {len(px_bench)} 日")
    print("拉沪深300 指数(000300) OHLC(算 RSRS 信号)...")
    ohlc = load_hs300_index_ohlc()
    print(f"  指数 {ohlc.index[0].date()} ~ {ohlc.index[-1].date()}, {len(ohlc)} 日")

    # RSRS 信号(基于 000300 指数 high/low),对齐到 ETF 时间轴
    beta, r2 = rsrs_beta_r2(ohlc["high"], ohlc["low"])
    score = rsrs_modified_score(beta, r2, M_STD)
    sig_rsrs = (score > THR).reindex(px_bench.index).fillna(False)
    sig_sma = (px_bench > px_bench.rolling(mc.TREND_MA).mean()).fillna(False)
    sig_both = sig_rsrs & sig_sma

    nav_sma, ret_sma = timing_backtest(px_bench, sig_sma)
    nav_rsrs, ret_rsrs = timing_backtest(px_bench, sig_rsrs)
    nav_both, ret_both = timing_backtest(px_bench, sig_both)

    # 买入持有基准(同口径区间)
    common = ret_sma.index.intersection(ret_rsrs.index).intersection(ret_both.index)
    bh = px_bench.pct_change().fillna(0.0).loc[common]
    nav_bh = (1 + bh).cumprod()
    nav_bh = nav_bh / nav_bh.iloc[0]

    print(f"\n=== 大盘择时绩效(标的:510300, {common[0].date()}~{common[-1].date()}) ===")
    print(f"  {'择时信号':<22}{'年化':>8}{'夏普':>7}{'回撤':>8}{'Sortino':>9}{'多头占比':>10}")
    for name, nav, ret, sig in [("买入持有(B&H)", nav_bh, bh, pd.Series(True, index=common)),
                                 ("A. SMA(200)", nav_sma, ret_sma, sig_sma),
                                 ("B. RSRS 修正分", nav_rsrs, ret_rsrs, sig_rsrs),
                                 ("C. RSRS+SMA 共振", nav_both, ret_both, sig_both)]:
        r = _row(name, nav, ret)
        on = sig.reindex(common).fillna(False).mean() * 100
        print(f"  {r[0]:<22}{r[1]*100:>7.1f}%{r[2]:>7.2f}{r[3]*100:>7.1f}%{r[4]:>9.2f}{on:>9.1f}%")

    print(f"\n=== block bootstrap:Δ=SR(候选)−SR(SMA)  n={N_BOOT} block={BLOCK} ===")
    for label, ret_c, seed in [("B. RSRS vs A.SMA", ret_rsrs.loc[common], 7),
                               ("C. 共振 vs A.SMA", ret_both.loc[common], 11)]:
        a = ret_sma.loc[common].to_numpy(float)
        c = ret_c.to_numpy(float)
        boots = _boot_vs_ref(c, a, seed=seed)
        ci = np.percentile(boots, [2.5, 97.5])
        p = float((boots <= 0).mean())
        obs = e._ann_sharpe(c) - e._ann_sharpe(a)
        print(f"  {label:<20}Δ(obs)={obs:+.3f}  95%CI=[{ci[0]:+.3f},{ci[1]:+.3f}]  P(Δ≤0)={p:.3f}")

    print("\n=== 关键熊市段:择时是否避开大跌(累计收益) ===")
    segs = [("2015 股灾", "2015-06", "2015-09"), ("2018 贸易战熊", "2018-01", "2018-12"),
            ("2022 调整", "2022-01", "2022-10"), ("2024 小盘危机", "2024-01", "2024-02")]
    print(f"  {'区间':<16}{'B&H':>10}{'SMA':>10}{'RSRS':>10}{'共振':>10}")
    for label, lo, hi in segs:
        m = (common >= lo) & (common <= hi)
        if m.sum() < 5:
            continue
        cum = lambda r: np.prod(1 + r.loc[common][m]) - 1
        print(f"  {label:<16}{cum(bh)*100:>9.1f}%{cum(ret_sma)*100:>9.1f}%"
              f"{cum(ret_rsrs)*100:>9.1f}%{cum(ret_both)*100:>9.1f}%")

    boots = _boot_vs_ref(ret_rsrs.loc[common].to_numpy(float),
                         ret_sma.loc[common].to_numpy(float), seed=7)
    sig = (np.percentile(boots, 2.5) > 0) and (float((boots <= 0).mean()) < 0.05)
    print(f"\n=== 判定 ===")
    print(f"  {'✓ RSRS 作为大盘择时显著优于 SMA(200) → 值得进入第二步(替换 CTA 趋势过滤,走 decide_targets 新参流程)'
          if sig else '✗ RSRS 未显著优于 SMA(200) → 保持现有 SMA(200) 趋势过滤'}")
    print("  注:与 A7(EMA/KAMA/FRAMA 自适应均线均不如 SMA)同源教训——大盘择时换更复杂信号往往不胜 SMA。")
    print("     若显著,下一步不是改本脚本,而是给 cta.decide_targets 加 trend_mode∈{sma,rsrs} 参数 + 三道关。")


if __name__ == "__main__":
    main()
