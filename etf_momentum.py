# -*- coding: utf-8 -*-
"""
A股 ETF 动量轮动 —— 向量化回测 / 研究工具。

策略逻辑统一来自 momentum_core.decide_targets（与 backtrader 回测、实盘共用一份），
这里只负责“喂历史数据 + 算净值 + 出指标/图”，速度快，适合调参研究。

运行:
  python etf_momentum.py            # 真实数据：改进版 vs 无风控版 vs 基准 + 出图
  python etf_momentum.py sweep      # 参数稳健性扫描（不同回看窗口/持仓数）
  python etf_momentum.py robust     # 单参数扰动稳健性（看是不是"平台"，查过拟合）
  python etf_momentum.py wf         # walk-forward 滚动样本外（量化样本外夏普衰减）
  python etf_momentum.py boot       # RISK_ADJ 夏普提升的 bootstrap 显著性检验
  python etf_momentum.py universe   # 标的池消融：踢掉黄金/纳指，量化 alpha 对池子的依赖
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from momentum_core import (POOL, DEFENSE, BENCH, MAX_LOOKBACK, LOOKBACKS, TOP_N,
                           VOL_TARGET, VOL_WINDOW, TREND_MA, TREND_CUT, SKIP_RECENT,
                           RISK_ADJ, COMMISSION, SLIPPAGE, WEIGHTING, INV_VOL_WINDOW,
                           CRASH_PROT, CRASH_LOOKBACK, CRASH_THR, CRASH_CUT,
                           decide_targets)

ALL_CODES = list(POOL) + [DEFENSE[0]]


# ---------------- 数据 ----------------
def load_real():
    """akshare 拉真实日线（前复权收盘价），各 ETF 按自身上市日，起点锚定基准+防守。

    改进项 E2：每只 ETF 带重试 + 指数退避，应对东财接口临时限频。注意 akshare 内 ETF
    复权数据仅东财(fund_etf_hist_em)一家可靠——新浪源(fund_etf_hist_sina)不复权、
    baostock 与股票接口均不支持 ETF，故无等价复权备源；长期根治需引入 tushare 等付费源。"""
    import akshare as ak
    import time
    end_date = pd.Timestamp.today().strftime("%Y%m%d")   # 动态截止日：跑到哪天拉到哪天，不再锁死 2025 年底
    series = {}
    for c in ALL_CODES:
        df = None
        for attempt in range(4):                          # 最多重试 4 次，指数退避
            try:
                df = ak.fund_etf_hist_em(symbol=c, period="daily",
                                         start_date="20140101", end_date=end_date, adjust="qfq")
                if len(df):
                    break
                df = None                                  # 空表视为失败
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(
                        f"{c} 拉取失败（东财接口可能限频，请稍后重试或更换网络）: {e}") from e
            time.sleep(5 * (attempt + 1))                  # 失败退避 5/10/15/20s
        time.sleep(0.3)                                    # 成功也小间隔，降低触发限频概率
        df["日期"] = pd.to_datetime(df["日期"])
        series[c] = df.set_index("日期")["收盘"].rename(c)
    # 各 ETF 上市日不同，按日期外连接成一张宽表（缺失处为 NaN）
    px = pd.concat(series.values(), axis=1).sort_index()
    px = px.dropna(how="all").ffill()                       # 仅向前填上市后的停牌缺口，上市前仍为 NaN
    # 起点锚定到“基准 + 防守资产都已上市”之后，保证基准曲线和防守切换全程有效；
    # 晚上市的标的（如中证1000）在有数据后才会进入动量排序。
    ready = px.index[px[[BENCH, DEFENSE[0]]].notna().all(axis=1)]
    return px.loc[ready[0]:]


# ---------------- 回测引擎（调用核心大脑） ----------------
def backtest(px, lookbacks=LOOKBACKS, top_n=TOP_N, vol_target=VOL_TARGET, trend_ma=None,
             trend_cut=TREND_CUT, vol_window=VOL_WINDOW, skip_recent=SKIP_RECENT,
             risk_adj=RISK_ADJ, weighting=WEIGHTING, inv_vol_window=INV_VOL_WINDOW,
             crash_prot=CRASH_PROT, crash_lookback=CRASH_LOOKBACK,
             crash_thr=CRASH_THR, crash_cut=CRASH_CUT):
    """月末调仓，权重由 decide_targets 决定。返回 (策略净值, 日收益, 调仓次数, 持仓日志)。
    trend_ma:    大盘趋势过滤均线天数（None=关闭），用于对比加/不加趋势择时的效果。
    trend_cut:   趋势下行时股票仓的"保留比例"（透传给 decide_targets，供 robust 扫描）。
    vol_target:  年化波动目标（None=关闭）；vol_window: 估计近期波动的回看窗口。
    skip_recent: 跳过最近 N 日再算动量（21≈1个月）；risk_adj: 是否用风险调整动量。两者用于 A/B 测试。"""
    rets = px.pct_change().fillna(0.0)                      # 每只标的的每日收益率矩阵

    # 启动期所需历史 = max(全局 MAX_LOOKBACK, 本次最长回看窗口 + skip_recent)。
    #   不写死 MAX_LOOKBACK：否则含 252 日窗口或大 skip 的组合在 day127~need 段历史不足被判
    #   None、凑不齐 top_n → 系统性空仓，年化被不公平拉低（sweep/wf 的长窗口候选即此症）。
    #   默认 (21,63,126)+skip0 时 need=126，与原行为完全一致。
    need = max(MAX_LOOKBACK, max(lookbacks) + skip_recent)

    # 1) 找出每个月最后一个交易日作为调仓日；并跳过历史不足 need 的初期
    month_ends = px.resample("ME").last().index
    rebal_days = [px.index[px.index <= me][-1] for me in month_ends if (px.index <= me).any()]
    rebal_days = [d for d in rebal_days if d >= px.index[need]]
    rd_set = set(rebal_days)

    # 2) 逐日推进，构造一张“每天持有什么权重”的表。非调仓日沿用上次权重（cur 不变）
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    cur = pd.Series(0.0, index=px.columns)                  # 当前持仓权重，跨日延续
    holdings_log = []
    for d in px.index:
        if d in rd_set:                                    # 到调仓日才重新决策
            # 把截至当日的收盘价喂给核心大脑（与实盘喂法一致）
            recent = {c: px[c].loc[:d].dropna().tolist() for c in POOL}
            target, picks = decide_targets(recent, lookbacks=lookbacks,
                                           top_n=top_n, vol_target=vol_target,
                                           vol_window=vol_window, trend_ma=trend_ma,
                                           trend_cut=trend_cut, skip_recent=skip_recent,
                                           risk_adj=risk_adj, weighting=weighting,
                                           inv_vol_window=inv_vol_window,
                                           crash_prot=crash_prot,
                                           crash_lookback=crash_lookback,
                                           crash_thr=crash_thr, crash_cut=crash_cut)
            if target:
                cur = pd.Series(0.0, index=px.columns)
                for code, w in target.items():
                    cur[code] += w                         # 写入新目标权重
                holdings_log.append((d.date(), picks))
        weights.loc[d] = cur.values

    # 3) 关键防未来函数：今天的收益用“昨天收盘时持有的权重”（shift(1)）来计，
    #    即调仓日当天还按旧权重，次日才用新权重。
    w_lag = weights.shift(1).fillna(0.0)
    gross = (w_lag * rets).sum(axis=1)                      # 组合每日毛收益

    # 4) 交易成本 = 换手率 ×（手续费+滑点）。换手率 = 权重变化的绝对值之和
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (COMMISSION + SLIPPAGE)
    net = gross - cost                                      # 净收益

    # 5) 从有足够历史的起点开始累乘成净值曲线，并归一化到 1.0
    start = px.index[need]
    nav = (1 + net).cumprod().loc[start:]
    nav = nav / nav.iloc[0]
    return nav, net.loc[start:], len(rebal_days), holdings_log


def bench_nav(px):
    start = px.index[MAX_LOOKBACK]
    nav = (1 + px[BENCH].pct_change().fillna(0.0)).cumprod().loc[start:]
    return nav / nav.iloc[0]


# ---------------- 绩效指标 ----------------
def perf(nav, daily):
    """根据净值曲线 nav 和日收益 daily 算常用绩效指标。"""
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1/years) - 1               # 年化收益（几何平均）
    vol = daily.std() * np.sqrt(252)                   # 年化波动
    sharpe = (daily.mean() * 252) / (daily.std() * np.sqrt(252) + 1e-12)  # 夏普（无风险利率按0）
    mdd = ((nav / nav.cummax()) - 1).min()             # 最大回撤 = 距历史最高点的最大跌幅
    return dict(总收益=nav.iloc[-1]-1, 年化=cagr, 波动=vol, 夏普=sharpe, 回撤=mdd, 年数=years)


def rolling_metrics(daily, window=252):
    """滚动 window 日（默认1年）的年化夏普/波动 + 逐日回撤（水下曲线）序列。
    全期单个夏普会掩盖“某段靠运气”，看滚动序列才能判断策略是否随时间稳定（改进项 F2）。"""
    roll_sharpe = daily.rolling(window).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252) + 1e-12), raw=True)
    roll_vol = daily.rolling(window).std() * np.sqrt(252)
    nav = (1 + daily).cumprod()
    drawdown = nav / nav.cummax() - 1                         # 水下曲线：距历史最高点的跌幅
    return roll_sharpe, roll_vol, drawdown


def _print_table(rows):
    """rows: [(名称, perf字典)]"""
    print(f"{'策略':<22}{'总收益':>9}{'年化':>8}{'波动':>8}{'最大回撤':>9}{'夏普':>7}")
    for name, p in rows:
        print(f"{name:<22}{p['总收益']*100:>8.1f}%{p['年化']*100:>7.1f}%"
              f"{p['波动']*100:>7.1f}%{p['回撤']*100:>8.1f}%{p['夏普']:>7.2f}")


# ---------------- 主流程 ----------------
def run_sweep(px):
    print("参数稳健性扫描（年化% / 最大回撤% / 夏普）:\n")
    lb_opts = {"单60日": (60,), "单126日": (126,), "混合1/3/6月": (21, 63, 126), "混合3/6/12月": (63, 126, 252)}
    print(f"{'回看窗口 \\ 持仓数':<18}" + "".join(f"{'TopN='+str(n):>20}" for n in (1, 2, 3)))
    for lname, lb in lb_opts.items(): # lb是回看多少天
        cells = []
        for n in (1, 2, 3):
            nav, daily, _, _ = backtest(px, lookbacks=lb, top_n=n)
            p = perf(nav, daily)
            cells.append(f"{p['年化']*100:5.1f}/{p['回撤']*100:6.1f}/{p['夏普']:.2f}")
        print(f"{lname:<18}" + "".join(f"{c:>20}" for c in cells))
    bn = bench_nav(px)
    pb = perf(bn, bn.pct_change().fillna(0))
    print(f"\n基准 买入持有沪深300: 年化 {pb['年化']*100:.1f}% / 回撤 {pb['回撤']*100:.1f}% / 夏普 {pb['夏普']:.2f}")


# =====================================================================
# 过拟合检验工具集（robust / wf / boot）—— 三个互补视角：
#   robust: 单参数扰动，看指标是不是"平台"（孤峰 = 过拟合）
#   wf    : 滚动样本外，量化样本外相对全样本内的夏普衰减
#   boot  : 配对 block bootstrap，检验 RISK_ADJ 的微提升是否显著
# =====================================================================

# ---- 通用：用一段日收益重建从 1.0 起的子段净值并算指标 ----
def _seg_perf(daily_seg):
    """子段切片不是从 1.0 起，直接 perf() 会算错年化/总收益；这里重建归一化净值再算。"""
    nav = (1 + daily_seg).cumprod()
    nav = nav / nav.iloc[0]
    return perf(nav, daily_seg)


# ---------------- robust：单参数扰动稳健性 ----------------
def run_robust(px):
    """
    one-at-a-time 稳健性扫描：固定当前默认参数为基准，每次只扰动一个维度，
    看年化/回撤/夏普是否随参数"平滑"变化。相邻取值接近 = 稳健（平台）；
    某个取值是明显孤峰 = 过拟合警告。相比 sweep 的二维网格更易读、组合不爆炸。
    注：各维度共用同一段历史；lookbacks 候选均 ≤ MAX_LOOKBACK(126) 日，避免长窗口
    版本因头部历史不足空仓而被系统性低估。
    """
    base = dict(lookbacks=LOOKBACKS, top_n=TOP_N, vol_target=VOL_TARGET,
                trend_ma=TREND_MA, trend_cut=TREND_CUT,
                skip_recent=SKIP_RECENT, risk_adj=RISK_ADJ)
    # (展示名, 参数 key, 候选取值, 基准值)
    sweeps = [
        ("大盘趋势砍仓比例 trend_cut", "trend_cut",
            [0.0, 0.3, 0.5, 0.7, 0.9], TREND_CUT),
        ("年化波动目标 vol_target", "vol_target",
            [None, 0.10, 0.15, 0.20, 0.25], VOL_TARGET),
        ("持仓数 top_n", "top_n",
            [1, 2, 3, 4, 5], TOP_N),
        ("动量回看窗口 lookbacks", "lookbacks",
            [(60,), (126,), (21, 63, 126), (21, 126), (63, 126)], LOOKBACKS),
        ("跳过最近 N 日 skip_recent", "skip_recent",
            [0, 5, 10, 21], SKIP_RECENT),
        ("风险调整动量 risk_adj", "risk_adj",
            [True, False], RISK_ADJ),
        ("趋势均线天数 trend_ma", "trend_ma",
            [None, 120, 200, 250], TREND_MA),
    ]
    for label, key, vals, base_v in sweeps:
        print(f"\n=== 扰动 {label}（基准 = {base_v}）===")
        print(f"{'取值':<24}{'年化':>9}{'最大回撤':>10}{'夏普':>8}  备注")
        for v in vals:
            p = dict(base)
            p[key] = v
            nav, daily, _, _ = backtest(px, **p)
            pf = perf(nav, daily)
            tag = "← 基准" if v == base_v else ""
            shown = ("None" if v is None
                     else ("/".join(map(str, v)) if isinstance(v, tuple) else str(v)))
            print(f"{shown:<24}{pf['年化']*100:>8.1f}%{pf['回撤']*100:>9.1f}%"
                  f"{pf['夏普']:>8.2f}  {tag}")
    print("\n判读：相邻取值指标接近 → 稳健（平台）；某个取值明显跳成孤峰 → 该参数被过拟合。")


# ---------------- wf：walk-forward 滚动样本外 ----------------
# 训练段选参用的候选网格（控制在 8 个，覆盖关键维度，避免组合爆炸）。
# 每条都显式带 trend_ma=TREND_MA —— 与 main 部署版同口径；否则 backtest 默认 trend_ma=None，
# WF 评估的就成了“无趋势过滤”策略族，样本外衰减率无法挂到实际部署的策略上。
# 基线用 risk_adj=False，与 momentum_core.RISK_ADJ 当前默认一致。
WF_GRID = [
    dict(lookbacks=(21, 63, 126), top_n=3, trend_cut=0.5, risk_adj=False, trend_ma=TREND_MA),  # 部署默认
    dict(lookbacks=(21, 63, 126), top_n=3, trend_cut=0.5, risk_adj=True,  trend_ma=TREND_MA),  # 开风险调整对照
    dict(lookbacks=(126,),        top_n=3, trend_cut=0.5, risk_adj=False, trend_ma=TREND_MA),  # 单窗口
    dict(lookbacks=(63, 126, 252),top_n=3, trend_cut=0.5, risk_adj=False, trend_ma=TREND_MA),  # 更长混合(3/6/12月)
    dict(lookbacks=(21, 63, 126), top_n=2, trend_cut=0.5, risk_adj=False, trend_ma=TREND_MA),  # 更集中
    dict(lookbacks=(21, 63, 126), top_n=3, trend_cut=0.3, risk_adj=False, trend_ma=TREND_MA),  # 轻择时
    dict(lookbacks=(21, 63, 126), top_n=3, trend_cut=0.7, risk_adj=False, trend_ma=TREND_MA),  # 重择时
    dict(lookbacks=(21, 63, 126), top_n=3, trend_cut=0.5, risk_adj=False, trend_ma=None),      # 关趋势过滤对照
]


def _near_trading_day(dates, target):
    """把 target 日期对齐到 dates 中 ≤ target 的最后一个交易日；越界返回 None。"""
    sub = dates[dates <= target]
    return sub[-1] if len(sub) else None


def walk_forward(px, train_years=5, test_years=1, grid=None, metric="夏普"):
    """
    滚动样本外：每个测试窗口前用其专属"训练窗口"在 grid 上按 metric 选最优参数，
    冻结参数在测试窗口上跑；拼接所有测试窗口的日收益得到"样本外净值"。再对比
    "全样本内 grid 最优"——样本外相对全样本内的夏普衰减，就是过拟合的直接量化：
    衰减越多，回测越不可信。

    性能：每个 grid 参数只在全期上回测一次（cached），训练/测试段都从这份结果切片，
    所以总成本 ≈ len(grid) 次回测，而非 (段数 × grid) 次。

    返回 (oos_nav, oos_daily, report, full_best, oos_metrics)：
      report    : [(测试起, 测试止, 选中参数, 训练 metric, 测试 metric), ...]
      full_best : (metric 值, 参数, 全样本内 perf) —— 数据窥探上限
    """
    if grid is None:
        grid = WF_GRID
    dates = px.index
    start = dates[MAX_LOOKBACK]
    end = dates[-1]

    # 生成滚动 (train_start, train_end, test_start, test_end)。步长 = test_years：窗口整体每年
    # 前进一段、训练/测试逐年滑动重叠。切勿写成 t0=test_start——那样步长会变成 train_years，
    # 12 年数据只能切出 2 段测试，衰减率失去意义（曾因此误报“衰减 1.35”）。
    windows = []
    t0 = start
    while True:
        train_end = _near_trading_day(dates, t0 + pd.DateOffset(years=train_years))
        test_start = _near_trading_day(dates, t0 + pd.DateOffset(years=train_years)
                                       + pd.Timedelta(days=1))
        test_end = _near_trading_day(dates, t0 + pd.DateOffset(years=train_years + test_years))
        if test_start is None or test_end is None or test_start >= end:
            break
        windows.append((t0, train_end, test_start, test_end))
        t0 = t0 + pd.DateOffset(years=test_years)   # 整体前进 test_years（重叠滚动）

    # 每个 grid 参数全期回测一次并缓存（nav 已从 1.0 归一化；daily 是日净收益）
    cached = [(p, ) + backtest(px, **p)[:2] for p in grid]   # [(params, nav, daily), ...]

    oos_pieces, report = [], []
    for tr_s, tr_e, te_s, te_e in windows:
        # 训练段：按 metric 选最优参数
        best = None
        for params, nav, daily in cached:
            score = _seg_perf(daily.loc[tr_s:tr_e])[metric]
            if best is None or score > best[0]:
                best = (score, params, daily)
        tr_score, best_params, best_daily = best
        # 测试段：冻结参数，切测试段评估
        te_score = _seg_perf(best_daily.loc[te_s:te_e])[metric]
        oos_pieces.append(best_daily.loc[te_s:te_e])
        report.append((te_s, te_e, best_params, tr_score, te_score))

    oos_daily = pd.concat(oos_pieces)
    oos_nav = (1 + oos_daily).cumprod()
    oos_nav = oos_nav / oos_nav.iloc[0]
    oos_metrics = perf(oos_nav, oos_daily)

    # 全样本内 grid 最优（数据窥探上限）
    full_best = None
    for params, nav, daily in cached:
        fp = perf(nav, daily)
        if full_best is None or fp[metric] > full_best[0]:
            full_best = (fp[metric], params, fp)
    return oos_nav, oos_daily, report, full_best, oos_metrics


def _wf_report(oos_nav, oos_daily, report, full_best, oos_metrics):
    fb_score, fb_params, fb_perf = full_best
    print("Walk-forward 滚动样本外（训练 5 年选参 / 测试 1 年 / 滚动）:\n")
    print(f"{'测试期':<24}{'训练夏普':>9}{'测试夏普':>9}  选中参数")
    for te_s, te_e, params, tr_score, te_score in report:
        period = f"{te_s.date()}~{te_e.date()}"
        ps = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"{period:<24}{tr_score:>9.2f}{te_score:>9.2f}  {ps}")
    print(f"\n全样本内 grid 最优（数据窥探上限）: {fb_params}")
    print(f"  → 年化 {fb_perf['年化']*100:.1f}%  回撤 {fb_perf['回撤']*100:.1f}%  夏普 {fb_perf['夏普']:.2f}")
    print(f"\n样本外（拼接所有测试段，{oos_metrics['年数']:.1f} 年）:")
    print(f"  → 年化 {oos_metrics['年化']*100:.1f}%  回撤 {oos_metrics['回撤']*100:.1f}%  夏普 {oos_metrics['夏普']:.2f}")
    decay = oos_metrics['夏普'] / fb_perf['夏普'] if fb_perf['夏普'] > 0 else float("nan")
    print(f"\n夏普衰减 = 样本外 / 全样本内 = {decay:.2f}")
    print("  （≥0.6 算扛得住过拟合；<0.4 说明回测明显拟合到样本内）")


# ---------------- boot：RISK_ADJ 提升的配对 block bootstrap ----------------
def _circ_block_idx(T, block, rng):
    """circular block bootstrap 索引：把序列当成环，随机起点取连续 block 个，
    拼到长度 T。保留时序自相关（朴素重抽样会破坏收益序列的自相关结构）。"""
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
    return idx[:T]


def bootstrap_risk_adj(px, n_boot=2000, block=21, seed=7):
    """
    检验"风险调整动量"带来的夏普提升是否统计显著。risk_adj=True / False 共享除该开关
    外的一切，两条日收益高度相关，故对配对 (d1, d0) 做 circular block bootstrap
    （block≈1 个月保留自相关）。Δ = SR(d1) − SR(d0)；p = 重抽样里 Δ≤0 的比例
    （单边，越小越显著）。返回 (delta_obs, boot_mean, ci_lo, ci_hi, p)。
    """
    _, d1, _, _ = backtest(px, risk_adj=True)
    _, d0, _, _ = backtest(px, risk_adj=False)
    common = d1.index.intersection(d0.index)
    a1 = d1.loc[common].to_numpy()
    a0 = d0.loc[common].to_numpy()
    T = len(a1)

    def _sr(x):
        sd = x.std(ddof=1)
        return (x.mean() / (sd + 1e-12)) * np.sqrt(252) if sd > 0 else 0.0

    delta_obs = _sr(a1) - _sr(a0)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _circ_block_idx(T, block, rng)
        boots[b] = _sr(a1[idx]) - _sr(a0[idx])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return delta_obs, float(boots.mean()), float(ci_lo), float(ci_hi), float((boots <= 0).mean())


def _boot_report(res):
    delta_obs, mean, lo, hi, p = res
    print("RISK_ADJ（风险调整动量）夏普提升的 bootstrap 显著性检验:\n")
    print(f"  观测夏普差 Δ = SR(risk_adj=True) − SR(False) = {delta_obs:+.4f}")
    print(f"  配对 block bootstrap（n=2000, block=21 日）:")
    print(f"    均值 {mean:+.4f}    95% CI [{lo:+.4f}, {hi:+.4f}]    P(Δ≤0) = {p:.3f}")
    if lo > 0:
        verdict = "显著为正（CI 整体 > 0）→ RISK_ADJ 的提升可信"
    elif hi < 0:
        verdict = "显著为负 → RISK_ADJ 反而拖累，建议关闭"
    else:
        verdict = ("不显著（CI 跨 0）→ 那 ~0.04 的夏普差更可能是噪声，\n        "
                   "建议关掉 RISK_ADJ 省一个过拟合自由度")
    print(f"\n  结论：{verdict}")


# ---------------- universe：标的池消融（alpha 对池子构成的依赖）----------------
def _with_pool(sub_dict, fn):
    """临时把 momentum_core.POOL 和本模块 POOL 都换成 sub_dict、跑完 fn() 恢复。
    backtest 与 decide_targets 内部都按各自模块的全局 POOL 遍历标的，故两处都要换。"""
    import momentum_core as mc
    g = globals()
    saved = (g["POOL"], mc.POOL)
    g["POOL"] = sub_dict
    mc.POOL = sub_dict
    try:
        return fn()
    finally:
        g["POOL"], mc.POOL = saved


def run_universe(px, drop=("518880", "513100")):
    """
    universe 消融：对比“默认池”与“踢掉指定顺风资产”的指标，量化策略 alpha 对池子构成的依赖
    （selection / universe bias）——这是 robust / wf 都查不到的一类过拟合（它们扫描的全是
    固定池子）。黄金(518880)、纳指(513100) 恰是 2014-2025 最强的两类趋势资产，踢掉它们能
    直接量出“回测好看有多少来自池子里恰好放了事后赢家”。
    """
    full = dict(POOL)
    sub = {c: n for c, n in full.items() if c not in drop}
    px_sub = px[list(sub) + [DEFENSE[0]]]
    print(f"universe 消融：踢掉 {[full[c] for c in drop]}，池中剩 {len(sub)} 只\n")
    rows = []
    nav, daily, _, _ = backtest(px, trend_ma=TREND_MA)
    rows.append((f"默认池(全{len(full)}只)", perf(nav, daily)))
    nav2, daily2, _, _ = _with_pool(sub, lambda: backtest(px_sub, trend_ma=TREND_MA))
    rows.append(("踢掉黄金+纳指", perf(nav2, daily2)))
    bn = bench_nav(px)
    rows.append(("买入持有沪深300", perf(bn, bn.pct_change().fillna(0))))
    _print_table(rows)
    d_sharpe = rows[0][1]["夏普"] - rows[1][1]["夏普"]
    d_cagr = (rows[0][1]["年化"] - rows[1][1]["年化"]) * 100
    print(f"\n解读：踢掉这两只，夏普降 {d_sharpe:.2f}、年化降 {d_cagr:.1f} 个百分点。"
          f"\n      这部分就是 alpha 对“池子里恰好有顺风资产”的依赖。"
          f"\n      但踢掉后仍明显跑赢基准（沪深300 夏普 {rows[2][1]['夏普']:.2f}），"
          f"说明动量逻辑本身有 alpha，不是纯靠池子。")


def main():
    # 读命令行参数决定运行模式：传 "sweep" 走参数扫描，不传则默认 "real" 跑回测出图。
    # sys.argv[0] 是脚本名，argv[1] 才是用户传的第一个参数，故需先判断长度防越界。
    mode = sys.argv[1] if len(sys.argv) > 1 else "real"
    px = load_real()

    if mode == "sweep":
        run_sweep(px)
        return
    if mode == "robust":
        run_robust(px)
        return
    if mode == "wf":
        oos_nav, oos_daily, report, full_best, oos_metrics = walk_forward(px)
        _wf_report(oos_nav, oos_daily, report, full_best, oos_metrics)
        return
    if mode == "boot":
        _boot_report(bootstrap_risk_adj(px))
        return
    if mode == "universe":
        run_universe(px)
        return

    # 风控逐步叠加（等权）+ C1 反向波动加权对照，全部对比基准
    nav_base, ret_base, _, _ = backtest(px, vol_target=None)
    nav_vt, ret_vt, n_vt, _ = backtest(px, vol_target=VOL_TARGET)
    nav_tr, ret_tr, _, _ = backtest(px, vol_target=VOL_TARGET, trend_ma=TREND_MA)              # 等权（默认）
    nav_iv, ret_iv, _, log = backtest(px, vol_target=VOL_TARGET, trend_ma=TREND_MA,
                                      weighting="inv_vol")                                      # C1 反向波动
    bn = bench_nav(px)

    p_base, p_vt = perf(nav_base, ret_base), perf(nav_vt, ret_vt)
    p_tr, p_iv = perf(nav_tr, ret_tr), perf(nav_iv, ret_iv)
    pb = perf(bn, bn.pct_change().fillna(0))

    print(f"回测区间: {nav_tr.index[0].date()} ~ {nav_tr.index[-1].date()}  "
          f"共{p_tr['年数']:.1f}年  调仓{n_vt}次  （真实数据）\n")
    _print_table([
        ("改进版(无回撤控制)", p_base),
        (f"改进版+波动目标{int(VOL_TARGET*100)}%", p_vt),
        (f"+大盘趋势过滤{TREND_MA}日(等权)", p_tr),
        (f"+大盘趋势过滤(反向波动C1)", p_iv),
        ("买入持有沪深300", pb),
    ])
    print(f"\n[C1 反向波动 vs 等权] 夏普 {p_tr['夏普']:.2f}→{p_iv['夏普']:.2f} "
          f"({p_iv['夏普']-p_tr['夏普']:+.2f})  年化 {p_tr['年化']*100:.1f}%→{p_iv['年化']*100:.1f}% "
          f"({(p_iv['年化']-p_tr['年化'])*100:+.1f}pp)  回撤 {p_tr['回撤']*100:.1f}%→{p_iv['回撤']*100:.1f}%")
    if log:
        print(f"最近一次调仓 {log[-1][0]}: {log[-1][1]}")

    # 画图：净值 / 水下曲线 / 滚动夏普 三子图（F2 滚动指标）
    rs_tr, _, dd_tr = rolling_metrics(ret_tr)
    rs_iv, _, dd_iv = rolling_metrics(ret_iv)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(nav_tr.index, nav_tr.values, label="等权 + TrendFilt", lw=1.8, color="#1f4e79")
    axes[0].plot(nav_iv.index, nav_iv.values, label="反向波动 + TrendFilt (C1)", lw=1.8, color="#2e7d32")
    axes[0].plot(bn.index, bn.values, label="Buy & Hold CSI300", lw=1.1, color="#c0504d", alpha=0.8)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Net Value (log)")
    axes[0].legend(loc="upper left"); axes[0].grid(alpha=0.3)
    axes[0].set_title("A-share ETF Momentum Rotation [REAL] — C1 反向波动加权 vs 等权")
    axes[1].fill_between(dd_tr.index, dd_tr.values * 100, 0, color="#1f4e79", alpha=0.25, label="等权")
    axes[1].plot(dd_iv.index, dd_iv.values * 100, color="#2e7d32", lw=1.0, label="反向波动")
    axes[1].set_ylabel("Drawdown %"); axes[1].legend(loc="lower left"); axes[1].grid(alpha=0.3)
    axes[2].plot(rs_tr.index, rs_tr.values, color="#1f4e79", lw=1.0, label="等权")
    axes[2].plot(rs_iv.index, rs_iv.values, color="#2e7d32", lw=1.0, label="反向波动")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("Rolling Sharpe (1y)"); axes[2].legend(loc="upper left"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    out = "ETF动量轮动_回测结果.png"
    fig.savefig(out, dpi=130)
    print("\n图已保存:", out)

    print("\n[F2 滚动指标(1年窗口)摘要]")
    for name, rs in (("等权", rs_tr), ("反向波动", rs_iv)):
        rs = rs.dropna()
        if len(rs):
            print(f"  {name}: 滚动夏普 均值{rs.mean():.2f} / 最差{rs.min():.2f} / "
                  f"占比>0 {(rs > 0).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
