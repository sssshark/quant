# -*- coding: utf-8 -*-
"""诊断:多债券组合(不同久期/品类)vs 单只长债(511010)vs CTA。

回答"扩展多债券是否值得"(用户选:诊断 + 暂停,不落地)。
候选(去货币 ETF):511010(5年国债·中端利率)/ 511260(十年国债·长端利率)/ 511220(城投·信用)。
三者覆盖「中端利率 + 长端利率 + 信用利差」,久期与风险源有分散。

方法:每只债券各自时序动量(混合动量>0 且价>MA200→满仓,否则空仓;同 BondMomentumStrategy
口径),等权合成「多债券动量」。对比单只 511010 动量,看加长债+信用是否提升夏普/降回撤。
再算 vs CTA 相关性 + 50/50 组合增益(多债券 vs 单只),判断扩展价值。

⚠ 诊断性质,不落地。预期:债券间高相关(都受利率驱动),分散有限;若多债券≈单只则单只够用(简单优先)。
公共区间从 511260 上市(2017-08)起,约 9 年。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo 根(找 cta/engine/multi_strategy)
import os, time
import numpy as np
import pandas as pd
import requests

import cta as mc
import engine as e
from engine import COMMISSION, SLIPPAGE

EXTRA_BONDS = ["511260", "511220"]          # 额外拉(511010 已在 load_real)
ALL_BONDS = ["511010"] + EXTRA_BONDS        # 中端利率 + 长端利率 + 信用


def fetch_bond(code, start="20120101"):
    """自写拉单只 ETF 前复权(复用 e._qfq_from_pre_close,口径同 _load_via_tushare)。"""
    token = open(os.path.join(os.path.dirname(__file__), ".tushare_token"), encoding="utf-8").read().strip()
    api = os.environ.get("TUSHARE_API", "https://fastapic.stockai888.top")
    tc = code + (".SH" if code[0] in "5" else ".SZ")
    r = requests.post(api, json={"api_name": "fund_daily", "token": token,
                      "params": {"ts_code": tc, "start_date": start, "end_date": "20260708"},
                      "fields": "trade_date,close,pre_close"},
                      headers={"Accept-Encoding": "gzip"}, timeout=30)
    j = r.json()
    d = j["data"]
    df = pd.DataFrame(d["items"], columns=d["fields"])
    df["date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("date").sort_index()
    close = df["close"].astype(float)
    pre = df["pre_close"].astype(float) if "pre_close" in df else close.shift(1)
    return e._qfq_from_pre_close(close, pre)


def bond_mom_ret(px, code, with_trend=True):
    """单只债券时序动量日收益(月末调仓、T+2 吃收益口径对齐 backtest)。返回 (nav, daily) 或 None。"""
    p = px[code].dropna()
    if len(p) < 250:
        return None
    month_ends = p.resample("ME").last().index
    rebal = [p.index[p.index <= me][-1] for me in month_ends if (p.index <= me).any()]
    rebal = [d for d in rebal if d >= p.index[200]]
    if not rebal:
        return None
    rd_set = set(rebal)
    mom = (p / p.shift(21) - 1 + p / p.shift(63) - 1 + p / p.shift(126) - 1) / 3
    sig = ((mom > 0) & (p > p.rolling(200).mean())).astype(float) if with_trend else (mom > 0).astype(float)
    weights = sig.where(sig.index.isin(rd_set)).ffill().shift(1).fillna(0.0)   # = backtest weights
    w_lag = weights.shift(1).fillna(0.0)
    ret = p.pct_change().fillna(0.0)
    gross = w_lag * ret
    turnover = (weights - weights.shift(1).fillna(0.0)).abs()
    net = (gross - turnover * (COMMISSION + SLIPPAGE)).fillna(0.0)
    start = rebal[0]
    nav = (1 + net).cumprod().loc[start:]
    nav = nav / nav.iloc[0]
    return nav, net.loc[start:]


def main():
    print("加载 CTA + 511010(load_real)...")
    res = e.load_real(with_limits=True)
    px = res[0]
    cb, cs = (res[1], res[2]) if isinstance(res, tuple) else (None, None)
    nav_cta, ret_cta, _, _ = e.backtest(px, hold_all=True, vol_target=mc.VOL_TARGET,
                                        trend_ma=mc.TREND_MA, cant_buy=cb, cant_sell=cs)

    print("拉额外债券:", EXTRA_BONDS)
    px = px.copy()
    for c in EXTRA_BONDS:
        px[c] = fetch_bond(c).reindex(px.index)
        time.sleep(0.6)

    print("\n[单只 vs 多债券组合 时序动量]")
    singles = {}
    for c in ALL_BONDS:
        r = bond_mom_ret(px, c)
        if r:
            singles[c] = r

    common = None
    for _, (_, net) in singles.items():
        common = net.index if common is None else common.intersection(net.index)
    multi_ret = None
    for _, (_, net) in singles.items():
        multi_ret = net.loc[common].copy() if multi_ret is None else multi_ret + net.loc[common]
    multi_ret = multi_ret / len(singles)
    multi_nav = (1 + multi_ret).cumprod()
    multi_nav = multi_nav / multi_nav.iloc[0]

    def row(name, nav, daily):
        p = e.perf(nav, daily)
        return (f"{name:<26}{p['年化']*100:>7.1f}%{p['夏普']:>7.2f}"
                f"{p['回撤']*100:>7.1f}%{p['Sortino']:>9.2f}")
    print(f"{'策略':<26}{'年化':>8}{'夏普':>7}{'回撤':>8}{'Sortino':>9}")
    for c, (nav, net) in singles.items():
        n_c = nav.loc[common[0]:]
        n_c = n_c / n_c.iloc[0]
        print(row(f"单只 {c}", n_c, net.loc[common]))
    print(row("多债券组合(等权动量)", multi_nav, multi_ret))

    print("\n[vs CTA 相关性](公共区间)", common[0].date(), "~", common[-1].date())
    ret_cta_a = ret_cta.reindex(common).fillna(0.0)
    for c, (_, net) in singles.items():
        x = ret_cta_a.to_numpy(float)
        y = net.loc[common].to_numpy(float)
        m = ~(np.isnan(x) | np.isnan(y))
        corr = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 5 else float("nan")
        print(f"  CTA vs 单只 {c}: {corr:+.3f}")
    x = ret_cta_a.to_numpy(float)
    y = multi_ret.to_numpy(float)
    m = ~(np.isnan(x) | np.isnan(y))
    print(f"  CTA vs 多债券组合: {float(np.corrcoef(x[m], y[m])[0, 1]):+.3f}")

    print("\n[50/50 组合增益:多债券 vs 单只 511010]")
    s010 = singles["511010"][1].loc[common]
    comb_single = 0.5 * ret_cta_a + 0.5 * s010
    comb_multi = 0.5 * ret_cta_a + 0.5 * multi_ret
    print(f"{'策略':<26}{'年化':>8}{'夏普':>7}{'回撤':>8}{'Sortino':>9}")
    print(row("50/50 CTA+单只511010", (1 + comb_single).cumprod(), comb_single))
    print(row("50/50 CTA+多债券组合", (1 + comb_multi).cumprod(), comb_multi))


if __name__ == "__main__":
    main()
