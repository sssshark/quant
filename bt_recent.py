"""近几年窗口回测（独立于 engine.main 的全周期口径）。

为什么单独写：engine.main 的 real 模式硬从 2013-03-25 起锚定、跑全 12.8 年。
想只看近几年的表现时，不能简单 px.tail()——动量排序窗口(126日)、趋势均线(200日)、
波动目标窗 都依赖 warmup 历史，切掉 warmup 会让前 need 天被判空仓、指标失真。

本脚的正确做法：load_real 拉全量数据(保留 warmup)→ backtest 全程跑一次(数据驱动
自动跳过 need 日 warmup)→ 只在统计阶段把 nav/daily 截到近 N 年重新归一化算 perf。
这样信号生成完全等价于全周期，只是观察窗口不同，没有重定基线带来的前视/口径切换。

用法：
    ~/miniconda3/bin/python bt_recent.py            # 默认近 3 年
    ~/miniconda3/bin/python bt_recent.py 5          # 近 5 年
    ~/miniconda3/bin/python bt_recent.py 3 eq       # 近 3 年 + 等权(非反向波动)
    ~/miniconda3/bin/python bt_recent.py 3 iv       # 近 3 年 + 反向波动C1(默认)

与全周期对照一并打印，看近几年相对全期的衰减(过拟合/近期失效的直接信号)。
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt

from engine import (
    load_real, backtest, perf, bench_nav, rolling_metrics, run_regime,
    _print_table, VOL_TARGET, TREND_MA,
)
from cta import POOL


def _slice(nav, daily, years):
    """把净值/日收益截到最近 N 年并重新归一化到 1.0。

    warmup 已在 backtest 内完成；这里只截统计窗口、重置基点。"""
    cut = nav.index[-1] - pd.DateOffset(years=years)
    start = nav.index[nav.index >= cut][0]
    nav_s = nav.loc[start:]
    daily_s = daily.loc[start:]
    nav_s = nav_s / nav_s.iloc[0]                    # 归一化到窗口起点 = 1.0
    return nav_s, daily_s, start


def main():
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    mode = sys.argv[2] if len(sys.argv) > 2 else "iv"      # iv=反向波动选股 / eq=等权选股 / hold=等权全池+风控(实盘推荐)
    hold_all = (mode == "hold")
    weighting = None if mode in ("eq", "hold") else "inv_vol"
    wlabel = "等权全池+风控(hold_all,实盘推荐)" if hold_all else (
        "反向波动C1" if weighting == "inv_vol" else "等权")

    # 全量拉数据(带涨跌停掩码)，保留 warmup 历史
    px, cant_buy, cant_sell = load_real(with_limits=True)
    lim = dict(cant_buy=cant_buy, cant_sell=cant_sell)
    bn_full = bench_nav(px)

    # 全周期跑一次(信号生成等价于 engine.main real 口径)
    nav_full, ret_full, n, log = backtest(
        px, vol_target=VOL_TARGET, trend_ma=TREND_MA, weighting=weighting,
        hold_all=hold_all, **lim)
    nav_b_full = bn_full.loc[nav_full.index]

    # 近 N 年切片
    nav_s, ret_s, start = _slice(nav_full, ret_full, years)
    nav_b_s = (bn_full.loc[nav_s.index] / bn_full.loc[start])

    p_full = perf(nav_full, ret_full)
    p_s = perf(nav_s, ret_s)
    pb_full = perf(nav_b_full, nav_b_full.pct_change().fillna(0.0))
    pb_s = perf(nav_b_s, nav_b_s.pct_change().fillna(0.0))

    print(f"\n===== 近 {years} 年窗口回测 ({wlabel}) =====")
    print(f"窗口: {start.date()} ~ {nav_s.index[-1].date()}  "
          f"共{(nav_s.index[-1]-start).days/365.25:.1f}年  "
          f"(全期调仓{n}次,信号生成同全周期,仅统计窗口不同)\n")
    _print_table([
        (f"策略-近{years}年", p_s),
        ("策略-全周期", p_full),
        (f"基准-近{years}年", pb_s),
        ("基准-全周期", pb_full),
    ])

    print(f"\n[近几年 vs 全周期 衰减] "
          f"夏普 {p_full['夏普']:.2f}→{p_s['夏普']:.2f} ({p_s['夏普']-p_full['夏普']:+.2f})  "
          f"年化 {p_full['年化']*100:.1f}%→{p_s['年化']*100:.1f}% "
          f"({(p_s['年化']-p_full['年化'])*100:+.1f}pp)  "
          f"回撤 {p_full['回撤']*100:.1f}%→{p_s['回撤']*100:.1f}%")
    print(f"[超额 vs 基准] 近{years}年: 年化 {p_s['年化']*100:.1f}% vs {pb_s['年化']*100:.1f}% "
          f"(超额 {(p_s['年化']-pb_s['年化'])*100:+.1f}pp)  "
          f"全期: {p_full['年化']*100:.1f}% vs {pb_full['年化']*100:.1f}% "
          f"(超额 {(p_full['年化']-pb_full['年化'])*100:+.1f}pp)")
    if log:
        print(f"最近一次调仓 {log[-1][0]}: {log[-1][1]}")

    # 滚动夏普(窗口内 1 年滚动)
    rs, _, dd = rolling_metrics(ret_s)
    rs = rs.dropna()
    if len(rs):
        print(f"\n[F2 滚动指标(1年窗口,窗口内)] "
              f"滚动夏普 均值{rs.mean():.2f} / 最差{rs.min():.2f} / 占比>0 {(rs>0).mean()*100:.0f}%")

    # 分 regime(只对窗口内日收益)
    run_regime(px, ret_s, f"近{years}年 {wlabel}+趋势")

    # 画图:窗口净值/水下/滚动夏普
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(nav_s.index, nav_s.values, label=f"Strategy {wlabel}", lw=1.8, color="#2e7d32")
    axes[0].plot(nav_b_s.index, nav_b_s.values, label="Buy&Hold CSI300", lw=1.1, color="#c0504d", alpha=0.8)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Net Value (log)")
    axes[0].legend(loc="upper left"); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"A-share ETF Momentum Rotation [REAL] — 近{years}年窗口 ({wlabel})")
    axes[1].fill_between(dd.index, dd.values*100, 0, color="#2e7d32", alpha=0.25)
    axes[1].set_ylabel("Drawdown %"); axes[1].grid(alpha=0.3)
    axes[2].plot(rs.index, rs.values, color="#2e7d32", lw=1.0)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("Rolling Sharpe (1y)"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    out = f"ETF动量轮动_近{years}年_{wlabel}.png"
    fig.savefig(out, dpi=130)
    print("\n图已保存:", out)


if __name__ == "__main__":
    main()
