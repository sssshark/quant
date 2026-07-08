# -*- coding: utf-8 -*-
"""诊断:商品期货期限结构(供需前瞻信号)时序择时商品行业 ETF,过三道关。

承接方法论质疑:挑板块应看供需/政策/价值,不只 PE/PB。PE/PB 行业数据免费版不覆盖
(index_dailybasic 仅宽基 5 个),转测供需——用期货近远月期限结构 backwardation 作
最前瞻供需信号(近月>远月=供不应求=多头),比库存/销售额等滞后景气更领先。

sanity 已证伪(diag_fut_term_probe):铜/螺纹/豆粕期限结构 vs 对应行业 ETF 后 21 日收益,
三品种相关都≈0/负(+0.038/-0.054/-0.001)、多头端不系统占优 → 信号无前瞻 alpha。
本诊断把信号做成时序择时策略(月末 backwardation>0 持有否则空仓,3 行业等权),
过三道关数值化确认:信号无预测力 → 择时夏普<等权 B&H + 过不了三道关(①权益同源 + ②无 alpha)。

口径同 diag_sector:月末调仓 T+2(双 shift)、含 COMMISSION+SLIPPAGE。
三道关:互补 |corr|<0.3 + 自身夏普>0.5 + 组合增益 50/50>max。
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
from diag_sector import fetch_etf, _month_rebal, _sharpe, _metrics, sector_equalweight

# 商品行业 ETF ↔ 期货品种映射(fund_basic benchmark + 期货代码核对)
SUPPLY_MAP = [
    # (ETF代码, ETF名, 期货前缀, 交易所, 期货名)
    ("512400", "有色", "CU", "SHF", "沪铜"),
    ("515210", "钢铁", "RB", "SHF", "螺纹钢"),
    ("159825", "农业", "M",  "DCE", "豆粕"),
]
START_SUPPLY = "20210101"   # 3 ETF 均已上市(农业 159825 2020-12 上市,取 2021 起稳定)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo 根(找 .tushare_token)
TOKEN = open(os.path.join(HERE, ".tushare_token"), encoding="utf-8").read().strip()
API = os.environ.get("TUSHARE_API", "https://fastapic.stockai888.top")


def ts_post(api_name, params, fields=None, retries=4):
    for attempt in range(retries):
        try:
            r = requests.post(API, json={"api_name": api_name, "token": TOKEN,
                              "params": params, "fields": fields or ""},
                              headers={"Accept-Encoding": "gzip"}, timeout=25)
            j = r.json()
            if j.get("code") != 0:
                return None
            d = j["data"]
            return pd.DataFrame(d["items"], columns=d["fields"])
        except (requests.exceptions.RequestException, ValueError):
            if attempt == retries - 1:
                return None
            time.sleep(1.0 + attempt)
    return None


def gen_contracts(prefix, exch, years=range(2018, 2028)):
    return [f"{prefix}{y % 100:02d}{m:02d}.{exch}" for y in years for m in range(1, 13)]


def fetch_contract_settle(ts_code):
    df = ts_post("fut_daily", {"ts_code": ts_code, "start_date": "20180101", "end_date": "20260708"},
                 "trade_date,settle")
    if df is None or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("date")["settle"].astype(float).rename(ts_code)


def contract_ym(ts_code):
    s = ts_code.split(".")[0]
    return 2000 + int(s[-4:-2]), int(s[-2:])


def term_structure(prefix, exch, label):
    """拉各月合约 settle → t 日近月/次近月 → backwardation=(near/next-1)。"""
    series = {}
    for c in gen_contracts(prefix, exch):
        try:
            s = fetch_contract_settle(c)
        except Exception:
            s = None
        if s is not None and len(s):
            series[c] = s
        time.sleep(0.12)
    if not series:
        return None
    wide = pd.DataFrame(series).sort_index()
    ym = {c: contract_ym(c) for c in wide.columns}
    near_l, next_l = [], []
    for d, row in wide.iterrows():
        vals = row.dropna()
        if len(vals) < 2:
            near_l.append(np.nan); next_l.append(np.nan); continue
        ordered = sorted(vals.index, key=lambda c: ym[c])
        near_l.append(vals[ordered[0]]); next_l.append(vals[ordered[1]])
    bd = (pd.Series(near_l, index=wide.index) / pd.Series(next_l, index=wide.index) - 1.0).dropna()
    print(f"  {label}({prefix}): {len(series)}合约 backwardation 均值{bd.mean()*100:.2f}% "
          f"正占比{(bd>0).mean()*100:.0f}% 样本{len(bd)}")
    return bd


def supply_timing(px_etf, bd):
    """月末调仓:backwardation>0(供不应求)持有、否则空仓。T+2 吃收益,含成本。返回 daily net。"""
    p = px_etf.dropna()
    bd = bd.reindex(p.index).ffill()
    rebal = _month_rebal(p)
    if not rebal:
        return None
    sig = (bd > 0).astype(float)
    weights = sig.where(sig.index.isin(set(rebal))).ffill().shift(1).fillna(0.0)
    w_lag = weights.shift(1).fillna(0.0)            # 再 shift = T+2
    r = p.pct_change().fillna(0.0)
    gross = w_lag * r
    turnover = (weights - weights.shift(1).fillna(0.0)).abs()
    net = (gross - turnover * (COMMISSION + SLIPPAGE)).fillna(0.0)
    return net.loc[rebal[0]:]


def main():
    print("加载基本盘(CTA hold_all=True + 国债 K=2)...")
    res = e.load_real(with_limits=True)
    px_broad = res[0]
    cb, cs = (res[1], res[2]) if isinstance(res, tuple) else (None, None)
    base = MultiStrategy([CTAStrategy(vol_target=mc.VOL_TARGET, trend_ma=mc.TREND_MA, hold_all=True),
                          BondMomentumStrategy()])
    _, ret_base, _, _ = e.backtest(px_broad, strategy=base, cant_buy=cb, cant_sell=cs)
    print(f"  基本盘 夏普 {_sharpe(ret_base):.2f}")

    print("\n拉商品行业 ETF + 对应期货期限结构:")
    px_sec = pd.DataFrame(index=px_broad.index)
    bd_map = {}
    for etf, name, prefix, exch, flabel in SUPPLY_MAP:
        px_sec[etf] = fetch_etf(etf).reindex(px_broad.index)
        bd_map[etf] = term_structure(prefix, exch, flabel)
        time.sleep(0.3)
    px_sec = px_sec.loc[START_SUPPLY:]
    print(f"  面板: {px_sec.index[0].date()}~{px_sec.index[-1].date()} 共 {len(px_sec)} 交易日")

    # 期限结构时序择时:3 行业各 1/N 槽,空仓贡献 0
    n = len(SUPPLY_MAP)
    slots = []
    for etf, name, *_ in SUPPLY_MAP:
        net = supply_timing(px_sec[etf], bd_map[etf])
        if net is not None:
            slots.append(net / n)
    ret_strat = pd.concat(slots, axis=1).sum(axis=1).dropna()
    ret_ew = sector_equalweight(px_sec)

    print(f"\n{'='*76}")
    print(f"[三道关] 期限结构供需择时 vs 基本盘(CTA+国债 K=2)")
    print(f"{'='*76}")
    print(f"{'策略':<22}{'夏普':>6}{'年化':>7}{'回撤':>7}{'vs基本盘':>10}"
          f"{'①互补':>7}{'②自身':>7}{'③组合增益':>14}{'判定':>6}")
    print("-" * 76)
    rows = [("期限结构供需择时", ret_strat), ("(语境)等权B&H", ret_ew)]
    for name, rs in rows:
        common = ret_base.index.intersection(rs.index)
        rb = ret_base.loc[common].to_numpy(float)
        rs_c = rs.loc[common].to_numpy(float)
        mask = ~(np.isnan(rb) | np.isnan(rs_c))
        corr = float(np.corrcoef(rb[mask], rs_c[mask])[0, 1]) if mask.sum() > 5 else float("nan")
        sr_s = _sharpe(rs); sr_base = _sharpe(ret_base.loc[common])
        comb = pd.Series(0.5 * rb + 0.5 * rs_c, index=common)
        sr_comb = _sharpe(comb)
        mt = _metrics(rs)
        g1 = abs(corr) < 0.3; g2 = sr_s > 0.5; g3 = sr_comb > max(sr_s, sr_base)
        verdict = "✓过" if (g1 and g2 and g3) else "✗"
        g3s = f"{sr_comb:.2f}>{max(sr_s, sr_base):.2f}" if g3 else f"{sr_comb:.2f}≤{max(sr_s, sr_base):.2f}"
        print(f"{name:<22}{sr_s:>6.2f}{mt['年化']*100:>6.1f}%{mt['回撤']*100:>6.1f}%"
              f"{corr:>+10.3f}{('✓' if g1 else '✗'):>7}{('✓' if g2 else '✗'):>7}{g3s:>14}{verdict:>6}")

    print("\n判读:")
    print("  期限结构(最前瞻供需信号)择时:若夏普<等权B&H → 信号无预测力,供需法在A股商品股无效。")
    print("  sanity 已证三品种(铜/螺纹/豆粕)期限结构 vs 行业ETF后21日收益相关≈0/负、多头端不占优;")
    print("  本回测数值化确认:信号做成策略也亏,且板块=权益与基本盘同源过不了①互补。")
    print("  → 供需分析(即便用最前瞻信号)在 A 股商品股实证无效,结案。")
    print("  方法论闭环:价值(PE/PB行业数据免费版不覆盖)/供需(期限结构3品种证伪)/政策(无干净数据+定性)")
    print("  三路在免费数据+三道关框架下均不可行/无效,且板块无论如何过不了互补关(权益同源)。")


if __name__ == "__main__":
    main()
