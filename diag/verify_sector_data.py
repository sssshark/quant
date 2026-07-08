# -*- coding: utf-8 -*-
"""核验 A 股行业 ETF 数据与 diag_sector/_factors 结果是否可信(临时核验脚本)。

用户质疑:A 股内部卫星(行业等权 B&H 夏普 0.26、各因子 0.12~0.25)是否数据有误?正常不该这么差。
查 5 件事:
  ① 每个 ETF 的除权日数(pre_close≠昨收 的天数)→ 确认分红被 _qfq_from_pre_close 捕获
  ② 最大日幅 >15% → 疑似未复权拆分(工件)
  ③ 复权总收益 vs 纯价收益 → 分红贡献量级(应 ~1-2%/yr 累积)
  ④ 等权 B&H 静态(无成本无再平衡) vs diag(月再平衡+成本) → 成本/再平衡影响
  ⑤ 沪深300 同复权法 → 方法本身是否给 ~7%(real 报告值)
  ⑥ 因子策略 gross(成本=0) vs net → underperformance 是成本还是选股破坏
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo 根(找 cta/engine/multi_strategy)
import os
import time
import numpy as np
import pandas as pd
import requests

import engine as e
from engine import _qfq_from_pre_close
import diag_sector as ds
import diag_sector_factors as dsf

SECTORS = ds.SECTORS
START = ds.START


def fetch_raw(code, start="20120101"):
    """raw close + pre_close(未复权),同 fetch_etf 的取数但保留 pre_close。"""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo 根(找 .tushare_token)
    token = open(os.path.join(here, ".tushare_token"), encoding="utf-8").read().strip()
    api = os.environ.get("TUSHARE_API", "https://fastapic.stockai888.top")
    tc = code + (".SH" if code[0] in "5" else ".SZ")
    r = requests.post(api, json={"api_name": "fund_daily", "token": token,
                      "params": {"ts_code": tc, "start_date": start, "end_date": "20260708"},
                      "fields": "trade_date,close,pre_close"},
                      headers={"Accept-Encoding": "gzip"}, timeout=30)
    j = r.json(); d = j["data"]
    df = pd.DataFrame(d["items"], columns=d["fields"])
    df["date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("date").sort_index()
    return df["close"].astype(float), df["pre_close"].astype(float)


def main():
    print("=== ①②③ 单 ETF 数据完整性 + 分红捕获 ===")
    print(f"{'代码':<8}{'名称':<6}{'起点':>12}{'日数':>6}{'NaN':>5}{'最大日幅':>8}"
          f"{'除权日':>7}{'价收益':>9}{'复权收益':>9}{'分红贡献':>9}")
    adj = {}
    for c, name in SECTORS.items():
        rc, rp = fetch_raw(c)
        adj[c] = _qfq_from_pre_close(rc, rp)
        time.sleep(0.5)
        s = rc.dropna()
        f = rp / rc.shift(1)
        n_exdiv = int(((f - 1).abs() > 0.001).sum())     # pre_close≠昨收 = 除权/拆分
        max_day = float(rc.pct_change().abs().max())
        price_ret = float(s.iloc[-1] / s.iloc[0] - 1)
        a = adj[c].dropna()
        adj_ret = float(a.iloc[-1] / a.iloc[0] - 1)
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        print(f"{c:<8}{name:<6}{str(s.index[0].date()):>12}{len(s):>6}{int(rc.isna().sum()):>5}"
              f"{max_day*100:>7.1f}%{n_exdiv:>7}{price_ret*100:>8.1f}%{adj_ret*100:>8.1f}%"
              f"{(adj_ret-price_ret)*100:>8.1f}%   ({(adj_ret-price_ret)/yrs*100:.2f}%/yr)")
    print("  解读:除权日>0 → 分红被捕获(复权收益含分红);最大日幅<15% → 无未复权拆分。")

    # ④ 等权 B&H:静态 vs diag
    print("\n=== ④ 等权 B&H 分解:静态(无成本无再平衡) vs diag(月再平衡+成本) ===")
    px = pd.DataFrame({c: adj[c] for c in SECTORS}).dropna(how="all").loc[START:]
    rets = px.pct_change().fillna(0.0)
    bh_static = (rets * (1.0 / len(SECTORS))).sum(axis=1)         # 静态 1/5,买入持有不调
    bh_diag = ds.sector_equalweight(px)                            # 月再平衡回 1/5 + 成本
    print(f"  COMMISSION={ds.COMMISSION}  SLIPPAGE={ds.SLIPPAGE}")
    print(f"  静态1/5 B&H:  夏普{ds._sharpe(bh_static):.2f}  年化{ds._metrics(bh_static)['年化']*100:.1f}%  回撤{ds._metrics(bh_static)['回撤']*100:.1f}%")
    print(f"  diag月再平衡: 夏普{ds._sharpe(bh_diag):.2f}  年化{ds._metrics(bh_diag)['年化']*100:.1f}%  回撤{ds._metrics(bh_diag)['回撤']*100:.1f}%")

    # ⑤ 沪深300 同方法对照
    print("\n=== ⑤ 沪深300(510300)同复权法 → 方法本身对不对 ===")
    rc3, rp3 = fetch_raw("510300"); time.sleep(0.3)
    a3 = _qfq_from_pre_close(rc3, rp3).loc[START:]
    r3 = a3.pct_change().fillna(0.0)
    print(f"  沪深300 B&H(同复权法): 夏普{ds._sharpe(r3):.2f}  年化{ds._metrics(r3)['年化']*100:.1f}%  回撤{ds._metrics(r3)['回撤']*100:.1f}%")
    print(f"  real 模式报告:        夏普0.43  年化7.1%   (吻合=方法对)")

    # ⑥ 因子策略 gross(成本=0) vs net → underperformance 是成本还是选股
    print("\n=== ⑥ 因子策略 underperformance 分解:gross(无成本) vs net(含成本) ===")
    print(f"{'策略':<12}{'net夏普':>9}{'gross夏普':>10}{'net年化':>9}{'gross年化':>10}{'等权B&H':>9}")
    bh_sharpe = ds._sharpe(bh_static)
    # 备份并置零成本
    sav = (ds.COMMISSION, ds.SLIPPAGE, dsf.COMMISSION, dsf.SLIPPAGE, e.COMMISSION, e.SLIPPAGE)
    ds.COMMISSION = ds.SLIPPAGE = dsf.COMMISSION = dsf.SLIPPAGE = e.COMMISSION = e.SLIPPAGE = 0.0
    gross = {
        "低波": dsf.sector_lowvol(px),
        "反转": dsf.sector_reversal(px),
        "动量A": ds.sector_rotation_A(px),
    }
    (ds.COMMISSION, ds.SLIPPAGE, dsf.COMMISSION, dsf.SLIPPAGE, e.COMMISSION, e.SLIPPAGE) = sav
    net = {
        "低波": dsf.sector_lowvol(px),
        "反转": dsf.sector_reversal(px),
        "动量A": ds.sector_rotation_A(px),
    }
    for k in gross:
        g, n = gross[k], net[k]
        print(f"{k:<12}{ds._sharpe(n):>9.2f}{ds._sharpe(g):>10.2f}"
              f"{ds._metrics(n)['年化']*100:>8.1f}%{ds._metrics(g)['年化']*100:>9.1f}%{bh_sharpe:>9.2f}")
    print("  解读:gross≈等权B&H → underperformance 主要是成本(降换手可救);gross<<B&H → 选股本身破坏价值(whipsaw,真)。")


if __name__ == "__main__":
    main()
