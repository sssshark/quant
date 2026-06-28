# -*- coding: utf-8 -*-
"""
A股 ETF 动量轮动 —— 向量化回测 / 研究工具。

策略逻辑统一来自 momentum_core.decide_targets（与 backtrader 回测、实盘共用一份），
这里只负责“喂历史数据 + 算净值 + 出指标/图”，速度快，适合调参研究。

运行:
  python etf_momentum.py            # 真实数据：改进版 vs 无风控版 vs 基准 + 出图
  python etf_momentum.py sweep      # 参数稳健性扫描（不同回看窗口/持仓数）
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from momentum_core import (POOL, DEFENSE, BENCH, MAX_LOOKBACK, LOOKBACKS, TOP_N,
                           VOL_TARGET, TREND_MA, COMMISSION, SLIPPAGE, decide_targets)

ALL_CODES = list(POOL) + [DEFENSE[0]]


# ---------------- 数据 ----------------
def load_real():
    """akshare 拉真实日线（前复权收盘价），各 ETF 按自身上市日，起点锚定基准+防守。"""
    import akshare as ak
    series = {}
    for c in ALL_CODES:
        df = ak.fund_etf_hist_em(symbol=c, period="daily",
                                 start_date="20140101", end_date="20251231", adjust="qfq")
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
def backtest(px, lookbacks=LOOKBACKS, top_n=TOP_N, vol_target=VOL_TARGET, trend_ma=None):
    """月末调仓，权重由 decide_targets 决定。返回 (策略净值, 日收益, 调仓次数, 持仓日志)。
    trend_ma: 大盘趋势过滤均线天数（None=关闭），用于对比加/不加趋势择时的效果。"""
    rets = px.pct_change().fillna(0.0)                      # 每只标的的每日收益率矩阵

    # 1) 找出每个月最后一个交易日作为调仓日；并跳过历史不足 MAX_LOOKBACK 的初期
    month_ends = px.resample("ME").last().index
    rebal_days = [px.index[px.index <= me][-1] for me in month_ends if (px.index <= me).any()]
    rebal_days = [d for d in rebal_days if d >= px.index[MAX_LOOKBACK]]
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
                                           trend_ma=trend_ma)
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
    start = px.index[MAX_LOOKBACK]
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


def main():
    # 读命令行参数决定运行模式：传 "sweep" 走参数扫描，不传则默认 "real" 跑回测出图。
    # sys.argv[0] 是脚本名，argv[1] 才是用户传的第一个参数，故需先判断长度防越界。
    mode = sys.argv[1] if len(sys.argv) > 1 else "real"
    px = load_real()

    if mode == "sweep":
        run_sweep(px)
        return

    # 改进版（无回撤控制） vs +波动率目标 vs +波动率目标+大盘趋势过滤 vs 基准
    nav_base, ret_base, _, _ = backtest(px, vol_target=None)
    nav_vt, ret_vt, n_vt, log = backtest(px, vol_target=VOL_TARGET)
    nav_tr, ret_tr, _, log_tr = backtest(px, vol_target=VOL_TARGET, trend_ma=TREND_MA)
    bn = bench_nav(px)

    p_base, p_vt, p_tr = perf(nav_base, ret_base), perf(nav_vt, ret_vt), perf(nav_tr, ret_tr)
    pb = perf(bn, bn.pct_change().fillna(0))

    print(f"回测区间: {nav_vt.index[0].date()} ~ {nav_vt.index[-1].date()}  "
          f"共{p_vt['年数']:.1f}年  调仓{n_vt}次  （真实数据）\n")
    _print_table([
        ("改进版(无回撤控制)", p_base),
        (f"改进版+波动目标{int(VOL_TARGET*100)}%", p_vt),
        (f"+大盘趋势过滤{TREND_MA}日", p_tr),
        ("买入持有沪深300", pb),
    ])
    if log:
        print(f"\n最近一次调仓 {log[-1][0]}: {log[-1][1]}")

    # 画图：趋势过滤版 vs 波动目标版 vs 无控制版 vs 基准
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(nav_tr.index, nav_tr.values, label=f"+ Trend Filter MA{TREND_MA}", lw=2.0, color="#2e7d32")
    ax.plot(nav_vt.index, nav_vt.values, label=f"Improved + VolTarget {int(VOL_TARGET*100)}% (Top3)", lw=1.6, color="#1f4e79")
    ax.plot(nav_base.index, nav_base.values, label="Improved, no risk control", lw=1.2, color="#7f7f7f", alpha=0.8)
    ax.plot(bn.index, bn.values, label="Buy & Hold CSI300", lw=1.3, color="#c0504d", alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("A-share ETF Momentum Rotation [REAL]")
    ax.set_ylabel("Net Value (log, start=1.0)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    txt = (f"TrendFilt CAGR {p_tr['年化']*100:.1f}%  MaxDD {p_tr['回撤']*100:.1f}%  Sharpe {p_tr['夏普']:.2f}\n"
           f"VolTarget CAGR {p_vt['年化']*100:.1f}%  MaxDD {p_vt['回撤']*100:.1f}%  Sharpe {p_vt['夏普']:.2f}\n"
           f"CSI300    CAGR {pb['年化']*100:.1f}%  MaxDD {pb['回撤']*100:.1f}%  Sharpe {pb['夏普']:.2f}")
    ax.text(0.99, 0.02, txt, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=9, family="monospace", bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))
    fig.tight_layout()
    out = "ETF动量轮动_回测结果.png"
    fig.savefig(out, dpi=130)
    print("\n图已保存:", out)


if __name__ == "__main__":
    main()
