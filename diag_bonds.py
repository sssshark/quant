# -*- coding: utf-8 -*-
"""诊断:利率/债券策略作为第二策略的可行性 —— 与 CTA 的相关性诊断。

理论定位:CTA 主要是风险资产多头(A股权益 + 黄金 + 纳指)。债券(国债)与权益历史上
低相关甚至负相关(股债跷跷板)→ 债券策略可能是天然的分散 beta 来源。本诊断验证这一点。

⚠️ 简化前提:第一轮只用池里已有的 511010(国债 ETF,长债)做诊断(数据现成)。
  - 若国债方向显示低相关 + 正收益 → 值得扩展到短债+长债组合 / 期限策略(需额外拉债券 ETF 数据)。
  - 若高相关或无 alpha → 国债方向也悬。
  单只长债 ETF 代表性有限(只代表长债),结论标注此局限。

诊断三个口径:
  1. 国债买入持有(buy&hold 511010):纯债 beta 和 CTA 关系的基线(最干净的股债跷跷板检验)。
  2. 国债时序动量(混合动量>0 持有、<=0 空仓,月末调仓):加了择时 alpha 的版本。
  3. 国债时序动量 + 趋势过滤(价>MA200):更稳健版。

vs CTA 相关性(全样本/分regime)+ 自身绩效 + 50/50 组合增益。

判定:与 CTA 低相关(股债跷跷板)+ 国债策略正收益 + 组合增益 → 国债方向值得扩展多债券。
"""
import numpy as np
import pandas as pd

import momentum_core as mc
import etf_momentum as e
from etf_momentum import DEFENSE, COMMISSION, SLIPPAGE

BOND = DEFENSE[0]   # "511010" 国债 ETF(池里已有的长债)


def cta_returns(px, cb, cs):
    nav, ret, _, _ = e.backtest(px, hold_all=True, vol_target=mc.VOL_TARGET,
                                trend_ma=mc.TREND_MA, cant_buy=cb, cant_sell=cs)
    return nav, ret


def bond_buyhold(px):
    """国债买入持有(纯债 beta 基线)。"""
    p = px[BOND].dropna()
    need = 126
    start = p.index[need]
    ret = p.pct_change().fillna(0.0)
    nav = (1 + ret).cumprod().loc[start:]
    nav = nav / nav.iloc[0]
    return nav, ret.loc[start:]


def bond_momentum(px, with_trend=False):
    """国债时序动量:混合动量(21/63/126 日)>0 持有、<=0 空仓,月末调仓,T+1 成交。
    with_trend:额外要求价>MA200(同 CTA trend filter 口径)。"""
    p = px[BOND]
    month_ends = px.resample("ME").last().index
    rebal = [px.index[px.index <= me][-1] for me in month_ends if (px.index <= me).any()]
    need = 126
    rebal = [d for d in rebal if d >= px.index[need]]
    rd_set = set(rebal)
    mom = (p / p.shift(21) - 1 + p / p.shift(63) - 1 + p / p.shift(126) - 1) / 3   # 同 CTA blended 口径
    sig = mom > 0
    if with_trend:
        sig = sig & (p > p.rolling(200).mean())
    pos = sig.where(sig.index.isin(rd_set)).ffill().fillna(0.0)    # 月末采样 + 持有到下月末
    pos = pos.shift(1).fillna(0.0)                                 # T+1 成交防前视
    ret = p.pct_change().fillna(0.0)
    gross = pos * ret
    turnover = pos.diff().abs().fillna(0.0)
    cost = turnover * (COMMISSION + SLIPPAGE)
    net = (gross - cost).fillna(0.0)
    start = rebal[0]
    nav = (1 + net).cumprod().loc[start:]
    nav = nav / nav.iloc[0]
    return nav, net.loc[start:]


def corr_by_regime(a, b, px):
    lab = e.regime_labels(px).reindex(a.index).fillna("震荡")
    rows = []
    for r in ["牛市", "熊市", "震荡", "全样本"]:
        if r == "全样本":
            xa, ya = a, b
        else:
            m = lab == r
            xa, ya = a[m], b[m]
        xv = xa.to_numpy(dtype=float)
        yv = ya.to_numpy(dtype=float)
        mask = ~(np.isnan(xv) | np.isnan(yv))     # 某些切片含 NaN 边界,剔除后再算
        xv, yv = xv[mask], yv[mask]
        if len(xv) < 5 or np.std(xv) < 1e-9 or np.std(yv) < 1e-9:
            rows.append((r, len(xv), float("nan")))
        else:
            rows.append((r, len(xv), float(np.corrcoef(xv, yv)[0, 1])))
    return rows


def main():
    print("加载数据...")
    res = e.load_real(with_limits=True)
    px = res[0]
    cb, cs = (res[1], res[2]) if isinstance(res, tuple) else (None, None)
    print(f"  {px.index[0].date()} ~ {px.index[-1].date()}, {len(px)} 个交易日")
    print(f"  国债 ETF {BOND} 数据:{px[BOND].dropna().shape[0]} 个交易日")

    print("\n[1] CTA 口径回测(等权全池 + vol_target + 趋势过滤)...")
    nav_cta, ret_cta = cta_returns(px, cb, cs)

    print("[2] 国债策略回测(买入持有 / 时序动量 / 动量+趋势)...")
    nav_bh, ret_bh = bond_buyhold(px)
    nav_mom, ret_mom = bond_momentum(px, with_trend=False)
    nav_mt, ret_mt = bond_momentum(px, with_trend=True)

    common = ret_cta.index.intersection(ret_bh.index).intersection(ret_mom.index).intersection(ret_mt.index)
    rc = ret_cta.loc[common]

    print("\n=== 绩效对比 ===")
    def row(name, nav, daily):
        q = e.perf(nav, daily)
        return (name, f"{q['年化']*100:.1f}%", f"{q['夏普']:.2f}",
                f"{q['回撤']*100:.1f}%", f"{q['Sortino']:.2f}")
    print(f"{'策略':<28}{'年化':>8}{'夏普':>7}{'回撤':>8}{'Sortino':>9}")
    for r in [row("CTA(等权+风控)", nav_cta, ret_cta),
              row("国债买入持有", nav_bh, ret_bh),
              row("国债时序动量", nav_mom, ret_mom),
              row("国债动量+趋势", nav_mt, ret_mt)]:
        print(f"{r[0]:<28}{r[1]:>8}{r[2]:>7}{r[3]:>8}{r[4]:>9}")

    print("\n=== 50/50 组合(CTA + 国债策略) ===")
    for blabel, rb in [("国债买入持有", ret_bh.loc[common]), ("国债动量+趋势", ret_mt.loc[common])]:
        comb = 0.5 * rc + 0.5 * rb
        q = e.perf((1 + comb).cumprod(), comb)
        print(f"  CTA+{blabel}: 年化{q['年化']*100:.1f}% 夏普{q['夏普']:.2f} 回撤{q['回撤']*100:.1f}%")

    print("\n=== 相关性:CTA vs 国债策略(分 regime) ===")
    print("  (股债跷跷板:低/负相关=互补;正相关=股债同向)")
    for label, rb in [("国债买入持有", ret_bh.loc[common]), ("国债动量+趋势", ret_mt.loc[common])]:
        print(f"\n  CTA vs {label}:")
        for r, n, c in corr_by_regime(rc, rb, px):
            print(f"    {r:<6} n={n:>5}  corr={c:+.3f}")

    print("\n=== 判定 ===")
    for label, rb in [("买入持有", ret_bh.loc[common]), ("动量+趋势", ret_mt.loc[common])]:
        cv = rc.to_numpy(dtype=float)
        rv = rb.to_numpy(dtype=float)
        mask = ~(np.isnan(cv) | np.isnan(rv))
        c = float(np.corrcoef(cv[mask], rv[mask])[0, 1]) if mask.sum() >= 5 else float("nan")
        print(f"  CTA vs 国债{label}: 全样本相关性 {c:+.3f}")
    print("  (|corr|<0.3 = 真互补,股债跷跷板成立;对比均值回归 +0.71、配对 -0.04)")


if __name__ == "__main__":
    main()
