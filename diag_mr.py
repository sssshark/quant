# -*- coding: utf-8 -*-
"""诊断:均值回归(震荡套利 / 网格思想)作为第二策略的可行性 —— 与 CTA 的相关性诊断。

目的(阶段2 前置):在投入实现"第二策略"之前,先用数据回答最关键的问题——
均值回归和现有 CTA(趋势 / 做多波动率)到底有多互补?

核心洞察(理论定位):
  CTA       = 做多波动率(趋势 / 危机 alpha):趋势市、崩盘避损赚钱,震荡市被反复假突破洗。
  均值回归   = 做空波动率(逆势调仓 / 网格):震荡市反复收割波段,单边趋势被打穿。
  若两者收益低相关(尤其熊市负相关)→ 组合后平滑曲线,值得做第二策略。
  若高相关(同受市场因子驱动)→ 加了无分散收益,放弃。

A股实操约束:ETF 融券几乎不可行 → 不能做空一头 → 经典市场中性配对交易做不了。
故诊断用"单标的 z-score 均值回归"(布林带思想,多头逆势调仓):
  超卖(z 低)加仓、超买(z 高)减仓,仓位外配国债。这贴合 A股 可落地的网格 / 震荡套利形态。

两个版本对照:
  MR-nogate:纯均值回归(无风控)——看裸 short-vol 风险特征(预期熊市崩)。
  MR-gate  :大盘 MA200 下方时停手(全配国债)——可行化版本,看 gate 能否救活熊市。

判定标准(三道关,任一不过即不值得做):
  1. 互补性:CTA vs MR 相关性(全样本 <0.3 量级,熊市负相关尤佳)。
  2. MR 自身:夏普站得住(>0.5)、回撤可控(<40%)、gate 版熊市不全崩。
  3. 组合增益:50/50 组合夏普 > max(单 CTA, 单 MR),分散确实提升风险调整收益。

注意:这是可行性诊断,非生产策略。MR 参数固定经典值(布林带 20 日、线性逆势映射),
不调参(避免 PBO 过拟合),只看"够不够互补"这个二阶问题。
"""
import numpy as np
import pandas as pd

import momentum_core as mc
import etf_momentum as e
from etf_momentum import DEFENSE, BENCH, COMMISSION, SLIPPAGE

W = 20                       # 布林带 / z-score 回看窗口(经典月度)
GATE_MA = mc.TREND_MA        # regime gate 用的大盘均线(与 CTA trend filter 同源,公平)
MR_CODES = ["510300", "510500", "159915", "512100"]   # 4 只 A股宽基(震荡性强、流动性好、可多头调仓)
WEIGHT_PER = 1.0 / len(MR_CODES)   # 每只宽基满仓权重(等权)


def cta_returns(px, cb, cs):
    """CTA 瘦策略口径(= CTAStrategy 默认):等权全池 + vol_target + 趋势过滤。返回 (nav, 日收益)。"""
    nav, ret, _, _ = e.backtest(
        px, hold_all=True, vol_target=mc.VOL_TARGET, trend_ma=mc.TREND_MA,
        cant_buy=cb, cant_sell=cs)
    return nav, ret


def mr_weights(px, use_gate):
    """向量化均值回归目标权重 DataFrame(index=date, cols=MR_CODES+国债)。
    每只宽基:z=(P-MA)/STD(20日),仓位 w=clip(0.5-0.5z,0,1)*WEIGHT_PER。
      超卖(z=-1)→满仓该宽基的份额;超买(z=+1)→清仓;中位→半仓。剩余配国债。
    use_gate:大盘价<MA200 时股票仓清零(趋势市停手,全国债)——避免单边下跌一路接刀。"""
    p = px[MR_CODES]
    ma = p.rolling(W).mean()
    std = p.rolling(W).std()
    z = (p - ma) / (std + 1e-12)
    w = (0.5 - 0.5 * z).clip(0, 1) * WEIGHT_PER
    if use_gate:
        gate = (px[BENCH] > px[BENCH].rolling(GATE_MA).mean()).astype(float)
        w = w.mul(gate, axis=0)
    out = w.copy()
    out[DEFENSE[0]] = (1.0 - w.sum(axis=1)).clip(0, 1)
    return out


def mr_backtest(px, use_gate):
    """跑均值回归回测,返回 (nav, daily_net)。T+1 成交防前视,换手扣 COMMISSION+SLIPPAGE。"""
    weights = mr_weights(px, use_gate)
    cols = MR_CODES + [DEFENSE[0]]
    w_lag = weights.shift(1).fillna(0.0)            # 昨天收盘的权重吃今天的收益(防前视)
    rets = px[cols].pct_change().fillna(0.0)
    gross = (w_lag * rets).sum(axis=1)
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (COMMISSION + SLIPPAGE)
    net = gross - cost
    start = weights.dropna().index[0]               # z 与(若开)gate 都有效的首日
    nav = (1 + net).cumprod().loc[start:]
    nav = nav / nav.iloc[0]
    return nav, net.loc[start:]


def corr_by_regime(a, b, px):
    """分 regime 算 a/b 日收益相关性(用基准 regime 标签切片)+ 全样本。"""
    lab = e.regime_labels(px).reindex(a.index).fillna("震荡")
    rows = []
    for r in ["牛市", "熊市", "震荡", "全样本"]:
        if r == "全样本":
            x, y = a, b
        else:
            m = lab == r
            x, y = a[m], b[m]
        if len(x) < 5 or x.std() < 1e-9 or y.std() < 1e-9:
            rows.append((r, len(x), float("nan")))
        else:
            rows.append((r, len(x), float(np.corrcoef(x, y)[0, 1])))
    return rows


def main():
    print("加载数据...")
    res = e.load_real(with_limits=True)
    px = res[0]
    cb, cs = (res[1], res[2]) if isinstance(res, tuple) else (None, None)
    print(f"  日期范围 {px.index[0].date()} ~ {px.index[-1].date()}, {len(px)} 个交易日")

    print("\n[1] CTA 口径回测(等权全池 + vol_target + 趋势过滤)...")
    nav_cta, ret_cta = cta_returns(px, cb, cs)

    print("[2] 均值回归回测(无 gate / 带 MA200 gate)...")
    nav_mr0, ret_mr0 = mr_backtest(px, use_gate=False)
    nav_mr1, ret_mr1 = mr_backtest(px, use_gate=True)

    # 对齐到三序列公共区间,确保相关性 / 组合口径一致
    common = ret_cta.index.intersection(ret_mr0.index).intersection(ret_mr1.index)
    rc = ret_cta.loc[common]
    r0 = ret_mr0.loc[common]
    r1 = ret_mr1.loc[common]
    comb0 = 0.5 * rc + 0.5 * r0
    comb1 = 0.5 * rc + 0.5 * r1

    print("\n=== 绩效对比 ===")
    def row(name, nav, daily):
        p = e.perf(nav, daily)
        return (name, f"{p['年化']*100:.1f}%", f"{p['夏普']:.2f}",
                f"{p['回撤']*100:.1f}%", f"{p['Sortino']:.2f}")
    print(f"{'策略':<26}{'年化':>8}{'夏普':>7}{'回撤':>8}{'Sortino':>9}")
    for r in [row("CTA(等权+风控)", nav_cta, ret_cta),
              row("均值回归(无gate)", nav_mr0, ret_mr0),
              row("均值回归(MA200gate)", nav_mr1, ret_mr1),
              row("50/50 CTA+MR(无gate)", (1 + comb0).cumprod(), comb0),
              row("50/50 CTA+MR(gate)", (1 + comb1).cumprod(), comb1)]:
        print(f"{r[0]:<26}{r[1]:>8}{r[2]:>7}{r[3]:>8}{r[4]:>9}")

    print("\n=== 相关性:CTA vs 均值回归(分 regime) ===")
    print("  (低/负相关=互补强;接近1=无互补。熊市相关性最关键:CTA避损 vs MR被short-vol打穿)")
    for label, mr in [("MR 无gate", r0), ("MR MA200gate", r1)]:
        print(f"\n  CTA vs {label}:")
        for r, n, c in corr_by_regime(rc, mr, px):
            print(f"    {r:<6} n={n:>5}  corr={c:+.3f}")

    print("\n=== 均值回归自身分 regime 表现(看熊市是否崩 = short-vol 命门) ===")
    lab = e.regime_labels(px).reindex(common).fillna("震荡")
    for label, mr in [("无gate", r0), ("MA200gate", r1)]:
        print(f"\n  {label}:")
        for r in ["牛市", "熊市", "震荡"]:
            seg = mr[lab == r]
            if len(seg) < 5:
                continue
            seg_nav = (1 + seg).cumprod()
            seg_nav = seg_nav / seg_nav.iloc[0]
            p = e.perf(seg_nav, seg)
            print(f"    {r:<6} 占比{len(seg)/len(common)*100:4.1f}%  年化{p['年化']*100:6.1f}%  "
                  f"夏普{p['夏普']:5.2f}  回撤{p['回撤']*100:6.1f}%")

    print("\n=== 判定 ===")
    print(f"  CTA vs MR(无gate) 全样本相关性: {float(np.corrcoef(rc, r0)[0, 1]):+.3f}")
    print(f"  CTA vs MR(gate)   全样本相关性: {float(np.corrcoef(rc, r1)[0, 1]):+.3f}")
    print("  → 重点看上方「熊市」一行:若为负,说明两者在崩盘期互补最强(MR的核心理由)。")


if __name__ == "__main__":
    main()
