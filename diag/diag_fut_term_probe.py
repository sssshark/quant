# -*- coding: utf-8 -*-
"""探查+原型:商品期货期限结构 backwardation(供需前瞻信号)可算性 + 对商品股前瞻性 sanity。

期限结构 = 市场对现货紧张的前瞻定价:近月>次近月 = backwardation = 供不应求(多头)。
比库存/销售额等滞后景气数据更领先——部分绕开"股价领先景气"坑。

验证算法(铜 CU):拉 2018-2026 各月合约 settle → 合成 date×contract →
t 日取有数据合约按交割月排序取近/次近月 → backwardation=(near-next)/next。
sanity: 铜期限结构(当期) vs 有色 ETF(512400)后 21 日收益相关性——
正相关 + 多头端收益更高 = 期限结构对有色股有前瞻 alpha;否则商品股不跟期货。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo 根(找 cta/engine/multi_strategy)
import os
import time
import numpy as np
import pandas as pd
import requests

from diag_sector import fetch_etf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo 根(找 .tushare_token)
TOKEN = open(os.path.join(HERE, ".tushare_token"), encoding="utf-8").read().strip()
API = os.environ.get("TUSHARE_API", "http://47.116.63.181:8000/dataapi")


def ts_post(api_name, params, fields=None, retries=4):
    """代理偶发 SSL/读超时,重试 + 容错(失败返回 None,不抛——单合约失败跳过不中断)。"""
    for attempt in range(retries):
        try:
            r = requests.post(f"{API.rstrip('/')}/{api_name}",
                              json={"token": TOKEN,
                              "params": params, "fields": fields or ""},
                              headers={"Accept-Encoding": "gzip"}, timeout=25)
            j = r.json()
            if j.get("code") != 0:
                return None
            d = j["data"]
            return pd.DataFrame(d["items"], columns=d["fields"])
        except (requests.exceptions.RequestException, ValueError) as ex:
            if attempt == retries - 1:
                print(f"    ✗ {params.get('ts_code', api_name)} 重试{retries}次仍失败: {type(ex).__name__}")
                return None
            time.sleep(1.0 + attempt)        # 退避
    return None


def gen_contracts(prefix, exch, years=range(2018, 2028)):
    return [f"{prefix}{y % 100:02d}{m:02d}.{exch}" for y in years for m in range(1, 13)]


def fetch_contract_settle(ts_code):
    df = ts_post("fut_daily", {"ts_code": ts_code, "start_date": "20180101", "end_date": pd.Timestamp.today().strftime("%Y%m%d")},
                 "trade_date,settle")
    if df is None or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("date")["settle"].astype(float).rename(ts_code)


def contract_ym(ts_code):
    s = ts_code.split(".")[0]            # CU2401.SHF -> CU2401
    return 2000 + int(s[-4:-2]), int(s[-2:])   # (2024, 1)


def term_structure(prefix, exch, label):
    print(f"\n--- {label} {prefix}.{exch}: 拉各月合约 ---")
    contracts = gen_contracts(prefix, exch)
    series = {}
    n_fail = 0
    for c in contracts:
        try:
            s = fetch_contract_settle(c)
        except Exception:
            s = None
        if s is not None and len(s):
            series[c] = s
        else:
            n_fail += 1
        time.sleep(0.12)                 # tushare 限速留余量
    if n_fail:
        print(f"  ({n_fail} 个合约拉取失败/无数据,已跳过)")
    if not series:
        print(f"  {label}: 无合约数据")
        return None
    wide = pd.DataFrame(series).sort_index()
    print(f"  {label}: {len(series)} 个有数据合约, {wide.index.min().date()}~{wide.index.max().date()}, {len(wide)} 交易日")
    ym = {c: contract_ym(c) for c in wide.columns}
    near_l, next_l = [], []
    for d, row in wide.iterrows():
        vals = row.dropna()
        if len(vals) < 2:
            near_l.append(np.nan); next_l.append(np.nan); continue
        ordered = sorted(vals.index, key=lambda c: ym[c])   # 交割月升序
        near_l.append(vals[ordered[0]])                     # 近月=最早交割月
        next_l.append(vals[ordered[1]])                     # 次近月
    near = pd.Series(near_l, index=wide.index)
    nxt = pd.Series(next_l, index=wide.index)
    bd = (near / nxt - 1.0).dropna()
    print(f"  backwardation(近/次近-1): 均值{bd.mean() * 100:.2f}%  正占比(供不应求){(bd > 0).mean() * 100:.0f}%  样本{len(bd)}")
    return bd


MAPPING = [
    # (期货前缀, 交易所, 期货名, ETF代码, ETF名)
    ("CU", "SHF", "沪铜",   "512400", "有色"),
    ("RB", "SHF", "螺纹钢", "515210", "钢铁"),
    ("M",  "DCE", "豆粕",   "159825", "农业"),
]


def sanity(bd, etf_code, etf_name, horizon=21):
    """期限结构(当期) vs 对应行业ETF后 horizon 日收益:看多头端是否占优。"""
    px = fetch_etf(etf_code).dropna()
    common = bd.index.intersection(px.index)
    if len(common) < 100:
        print(f"    {etf_name}({etf_code}): 公共点太少 {len(common)}")
        return
    bd_a = bd.loc[common]
    fwd = px.loc[common].pct_change(horizon).shift(-horizon)   # 当期信号 → 后 horizon 日 ETF 收益
    m = ~(bd_a.isna() | fwd.isna())
    if m.sum() < 50:
        print(f"    {etf_name}({etf_code}): 有效样本不足")
        return
    corr = np.corrcoef(bd_a[m], fwd[m])[0, 1]
    pos = fwd[m][bd_a[m] > 0]
    neg = fwd[m][bd_a[m] <= 0]
    tag = "✓有alpha" if (corr > 0.05 and pos.mean() > neg.mean()) else "✗无alpha"
    print(f"    {etf_name}({etf_code}): 相关{corr:+.3f}  多头端{pos.mean() * 100:+.2f}%(n={len(pos)})"
          f"  空仓端{neg.mean() * 100:+.2f}%(n={len(neg)})  {tag}")


def main():
    print("=== 期限结构 backwardation 对 A 股商品行业股的前瞻 alpha sanity ===")
    print("    (backwardation>0=供不应求→持该行业ETF;看后21日收益多头端是否占优)\n")
    for prefix, exch, flabel, etf, elabel in MAPPING:
        bd = term_structure(prefix, exch, flabel)
        if bd is None:
            continue
        sanity(bd, etf, elabel)
        print()
    print("判读:若三品种相关都≈0/负 + 多头端不占优 → 期货期限结构(最前瞻供需信号)")
    print("      对 A 股商品股普遍无 alpha → 供需分析选板块在 A 股实证无效(即便用最前瞻信号)。")


if __name__ == "__main__":
    main()
