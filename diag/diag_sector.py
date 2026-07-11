# -*- coding: utf-8 -*-
"""诊断:板块策略 A(行业轮动·横截面) vs B(板块内时序) vs 基本盘(CTA+国债 K=2)。

回答"板块基金上做量化策略值不值得落地为第三/四个 Strategy"。A/B 各跑三道关
(同国债/均值回归/配对那套方法论,只有全过才进 multi_strategy):
  ① 互补性   |corr(板块策略, 基本盘)| < 0.3   (低/负 = 提供新分散,非同源 beta)
  ② 自身     板块策略独立夏普 > 0.5
  ③ 组合增益 50/50(板块 + 基本盘) 夏普 > max(各自)

A 行业轮动(横截面):5 行业 ETF 按混合动量(21/63/126 日)排序,月末选 top_n=3 等权。
  —— 结构同 CTA 选股(decide_targets 的 top-N),但池子是行业 ETF(消费/医药/证券/银行/地产,
  互相低相关)而非宽基指数(沪深300/中证500/创业板…互相高相关)。J2 在宽基上证 PBO=0.67
  (选股无效,因宽基高度同质、选谁差别小);本诊断测"换到低相关行业池,选股 alpha 是否复活"。
B 板块内时序:每个行业各自 blended mom>0 且价>MA200→满仓,否则空仓;5 行业等权合成。
  —— 结构同 BondMomentumStrategy(国债时序动量),标的换成行业 ETF。测"行业级时序择时"
  能否像国债那样提供股债跷跷板式的低相关分散(行业 ETF 也是权益,但有择时→熊市空仓)。

⚠ 池子选择严格先验(资产类别=行业 + 上市年限≤2013,13 年历史对齐基本盘),绝不用回测
收益挑(B1 扩池 / B2 分档的前视教训)。消费/医药/证券/银行/地产 均于 2010-2013 上市。
军工(2016)/传媒(2017)/新能源(2019)等上市晚,为保证 13 年公共区间对齐基本盘,不纳入。

口径:vectorized 月末调仓、T+2 吃收益(双 shift,对齐 diag_bonds2 / backtest)、含
COMMISSION+SLIPPAGE。基本盘(CTA+国债 K=2)走 engine.backtest(strategy=…)真口径取日收益。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo 根(找 cta/engine/multi_strategy)
import os
import time
import numpy as np
import pandas as pd
import requests

import cta as mc
import engine as e
from engine import COMMISSION, SLIPPAGE
from multi_strategy import CTAStrategy, BondMomentumStrategy, MultiStrategy

# 先验选池:行业分散 + 上市≤2013(13 年历史对齐基本盘)。绝不用回测收益挑。
SECTORS = {
    "159928": "消费",
    "512010": "医药",
    "512880": "证券",
    "512800": "银行",
    "512200": "地产",
}
LOOKBACKS = (21, 63, 126)
TREND_MA = 200
TOP_N = 3
START = "20130901"          # 5 行业均已上市的公共起点


def fetch_etf(code, start="20120101"):
    """tushare HTTP 拉单只 ETF 前复权(复用 engine._qfq_from_pre_close,口径同 _load_via_tushare)。"""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo 根(找 .tushare_token)
    token = open(os.path.join(here, ".tushare_token"), encoding="utf-8").read().strip()
    api = os.environ.get("TUSHARE_API", "http://47.116.63.181:8000/dataapi")
    tc = code + (".SH" if code[0] in "5" else ".SZ")
    r = requests.post(f"{api.rstrip('/')}/fund_daily",
                      json={"token": token,
                      "params": {"ts_code": tc, "start_date": start, "end_date": pd.Timestamp.today().strftime("%Y%m%d")},
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


def _month_rebal(p):
    """月末调仓日序列(对齐 diag_bonds2):每月最后一个交易日,且晚于 TREND_MA 起算。"""
    month_ends = p.resample("ME").last().index
    rebal = [p.index[p.index <= me][-1] for me in month_ends if (p.index <= me).any()]
    return [d for d in rebal if d >= p.index[TREND_MA]]


def _blended_mom(p):
    """混合动量(21/63/126 日涨幅均值),对齐 CTA blended_momentum。"""
    return sum(p.shift(lb).pipe(lambda s, pp=p: pp / s - 1) for lb in LOOKBACKS) / len(LOOKBACKS)


def sector_rotation_A(px_sec, top_n=TOP_N):
    """A 行业轮动:横截面混合动量选 top_n、等权。月末调仓、T+2 吃收益。返回 daily net ret。"""
    moms = pd.DataFrame({c: _blended_mom(px_sec[c].dropna()) for c in px_sec})
    all_rebal = sorted(set().union(*[_month_rebal(px_sec[c].dropna()) for c in px_sec]))
    # 调仓日:top_n 设 1/top_n、其余行业设 0;非调仓日留 NaN → ffill 跨日持仓(避开 where 形状坑)
    weight_df = pd.DataFrame(np.nan, index=px_sec.index, columns=px_sec.columns)
    for d in all_rebal:
        if d not in moms.index:
            continue
        row = moms.loc[d].dropna()
        if len(row) >= top_n:
            picks = list(row.nlargest(top_n).index)
            weight_df.loc[d, picks] = 1.0 / top_n
            others = [c for c in weight_df.columns if c not in picks]
            weight_df.loc[d, others] = 0.0
    weights = weight_df.ffill().shift(1).fillna(0.0)   # ffill 持仓 + shift T+1
    w_lag = weights.shift(1).fillna(0.0)                # 再 shift = T+2 吃收益(对齐 diag_bonds2/backtest)
    ret = px_sec.pct_change().fillna(0.0)
    gross = (w_lag * ret).sum(axis=1)
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = (gross - turnover * (COMMISSION + SLIPPAGE)).fillna(0.0)
    return net.loc[all_rebal[0]:] if all_rebal else net.loc[START:]


def sector_timing_B(px_sec):
    """B 板块内时序:每行业 blended mom>0 且>MA200→满仓,否则空仓;5 行业等权(各占 1/N 槽,空仓贡献 0)。"""
    n = px_sec.shape[1]
    slots = []
    for c in px_sec:
        p = px_sec[c].dropna()
        if len(p) < TREND_MA + max(LOOKBACKS):
            continue
        rebal = _month_rebal(p)
        if not rebal:
            continue
        rd_set = set(rebal)
        mom = _blended_mom(p)
        sig = ((mom > 0) & (p > p.rolling(TREND_MA).mean())).astype(float)
        weights = sig.where(sig.index.isin(rd_set)).ffill().shift(1).fillna(0.0)
        w_lag = weights.shift(1).fillna(0.0)
        r = p.pct_change().fillna(0.0)
        gross = w_lag * r
        turnover = (weights - weights.shift(1).fillna(0.0)).abs()
        net = (gross - turnover * (COMMISSION + SLIPPAGE)).fillna(0.0)
        slots.append(net / n)            # 每行业占 1/N 槽
    if not slots:
        return None
    return pd.concat(slots, axis=1).sum(axis=1).loc[START:]


def sector_equalweight(px_sec):
    """语境基准:5 行业等权、月再平衡、无选股无择时(纯 B&H 行业池本身)。用于区分
    『A/B 差是行业池没收益(dead money)』还是『动量在这市场被 whipsaw』。"""
    n = px_sec.shape[1]
    all_rebal = sorted(set().union(*[_month_rebal(px_sec[c].dropna()) for c in px_sec]))
    weight_df = pd.DataFrame(np.nan, index=px_sec.index, columns=px_sec.columns)
    for d in all_rebal:
        weight_df.loc[d, :] = 1.0 / n
    weights = weight_df.ffill().shift(1).fillna(0.0)
    w_lag = weights.shift(1).fillna(0.0)
    ret = px_sec.pct_change().fillna(0.0)
    gross = (w_lag * ret).sum(axis=1)
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = (gross - turnover * (COMMISSION + SLIPPAGE)).fillna(0.0)
    return net.loc[all_rebal[0]:] if all_rebal else net.loc[START:]


def _sharpe(r):
    r = r.dropna()
    return e._ann_sharpe(r.to_numpy(dtype=float)) if len(r) > 1 else float("nan")


def _metrics(r):
    r = r.dropna()
    nav = (1 + r).cumprod()
    nav = nav / nav.iloc[0]
    return e.perf(nav, r)


def main():
    print("加载基本盘(CTA hold_all=True + 国债 K=2)...")
    res = e.load_real(with_limits=True)
    px_broad = res[0]
    cb, cs = (res[1], res[2]) if isinstance(res, tuple) else (None, None)
    base = MultiStrategy([CTAStrategy(vol_target=mc.VOL_TARGET, trend_ma=mc.TREND_MA, hold_all=True),
                          BondMomentumStrategy()])
    _, ret_base, _, _ = e.backtest(px_broad, strategy=base, cant_buy=cb, cant_sell=cs)
    sr_base_full = _sharpe(ret_base)
    print(f"  基本盘 全样本 夏普 {sr_base_full:.2f}")

    print("\n拉行业 ETF:", [f"{c}({n})" for c, n in SECTORS.items()])
    px_sec = pd.DataFrame(index=px_broad.index)
    for c in SECTORS:
        px_sec[c] = fetch_etf(c).reindex(px_broad.index)
        time.sleep(0.6)
    px_sec = px_sec.loc[START:]
    print(f"  行业面板: {px_sec.index[0].date()} ~ {px_sec.index[-1].date()} 共 {len(px_sec)} 交易日")

    print("\n[行业间日收益相关矩阵](低 = 横截面有分散可挖,选股可能复活;高 = 同质、选股仍无效)")
    corr_mat = px_sec.pct_change().corr().round(2)
    corr_mat.index = [SECTORS[c] for c in corr_mat.index]
    corr_mat.columns = [SECTORS[c] for c in corr_mat.columns]
    print(corr_mat.to_string())
    offdiag = corr_mat.values[~np.eye(len(corr_mat), dtype=bool)]
    print(f"  行业间平均相关 {offdiag.mean():.2f}(对照:宽基 POOL 高度同质 → 选股 PBO=0.67)")

    ret_A = sector_rotation_A(px_sec)
    ret_B = sector_timing_B(px_sec)
    ret_ew = sector_equalweight(px_sec)

    print("\n[语境基准] 5 行业等权 B&H(月再平衡,无选股无择时)——区分『行业池没收益』vs『动量 whipsaw』:")
    mt_ew = _metrics(ret_ew)
    print(f"  等权 B&H: 夏普 {_sharpe(ret_ew):.2f}  年化 {mt_ew['年化']*100:.1f}%  回撤 {mt_ew['回撤']*100:.1f}%")
    print(f"  → 若 B&H 也有收益而 A/B≈0:动量在这市场被来回打脸(政策驱动的急涨急跌杀动量);")
    print(f"     若 B&H 也≈0:行业池本身是 dead money(地产拖累、消费/医药见顶回落),标的选错。")

    print(f"\n{'='*74}")
    print(f"[三道关] 板块策略 vs 基本盘(CTA+国债 K=2)")
    print(f"{'='*74}")
    hdr = (f"{'策略':<20}{'夏普':>6}{'年化':>7}{'回撤':>7}{'vs基本盘':>10}"
           f"{'①互补':>7}{'②自身':>7}{'③组合增益':>14}{'判定':>6}")
    print(hdr)
    print("-" * 74)
    for name, rs in [("A 行业轮动(横面top3)", ret_A), ("B 板块内时序(择时)", ret_B)]:
        common = ret_base.index.intersection(rs.index)
        rb = ret_base.loc[common].to_numpy(dtype=float)
        rs_c = rs.loc[common].to_numpy(dtype=float)
        mask = ~(np.isnan(rb) | np.isnan(rs_c))
        corr = float(np.corrcoef(rb[mask], rs_c[mask])[0, 1]) if mask.sum() > 5 else float("nan")
        sr_s = _sharpe(rs)
        sr_base = _sharpe(ret_base.loc[common])
        comb = pd.Series(0.5 * rb + 0.5 * rs_c, index=common)
        sr_comb = _sharpe(comb)
        mt = _metrics(rs)
        g1 = abs(corr) < 0.3
        g2 = sr_s > 0.5
        g3 = sr_comb > max(sr_s, sr_base)
        verdict = "✓过" if (g1 and g2 and g3) else "✗"
        g3s = f"{sr_comb:.2f}>{max(sr_s, sr_base):.2f}" if g3 else f"{sr_comb:.2f}≤{max(sr_s,sr_base):.2f}"
        print(f"{name:<20}{sr_s:>6.2f}{mt['年化']*100:>6.1f}%{mt['回撤']*100:>6.1f}%"
              f"{corr:>+10.3f}{('✓' if g1 else '✗'):>7}{('✓' if g2 else '✗'):>7}{g3s:>14}{verdict:>6}")

    print("\n判读:")
    print("  A 行业轮动:若 vs 基本盘高相关 → 只是权益 beta、非新 alpha(J2 宽基选股无效的延伸到行业);")
    print("     若行业间相关低且组合增益成立,才说明『低相关池子上选股复活』,值得落地。")
    print("  B 板块时序:有熊市空仓择时 → 可能提供低相关分散(同国债时序逻辑);自身夏普看行业动量有效性。")
    print("  三关全过才落地进 multi_strategy;否则记一笔、不落地(同均值回归/配对)。")


if __name__ == "__main__":
    main()
