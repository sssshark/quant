# -*- coding: utf-8 -*-
"""全行业篮子 + 多选股法综合诊断:『怎么挑板块』到底有没有用。

用户:C 肯定要把篮子补全,核心是『如何挑板块』。目标:钉死 basket caveat——
若在一个**全行业篮子**(不只旧经济 5 个)上,动量/低波/反转**全都输给等权(不挑)**,
则『选股无效』与篮子无关(J2 宽基 + M5/M6 行业 三重证伪 → 四重)。

设计:
  · 候选篮子 ~16 个行业 ETF,**先验按行业覆盖选**(消费/医药/金融/地产/军工/传媒/有色 +
    新经济:半导体/科技/新能源/新能源车/食品饮料/煤炭/计算机/通信/农业/光伏),绝不用收益挑。
    fetch 过滤无效/历史太短(< 3 年)的,保留的有效篮子透明打印。
  · 两面板:
      LONG  = 上市 ≤ 2016 的(≥10 年,统计力强)
      FULL  = 全部有效(从 2019,~7 年,breadth 全)
  · 方法(价格可算):等权(基准=不挑)、动量(top-N)、低波(top-N)、反转(top-N)。
  · 核心比较:**有没有任何选法跑赢等权?**(standalone 夏普 + vs 等权)
  · 跑赢等权的 → 再过三道关 vs 基本盘(互补 |corr|<0.3 + 自身>0.5 + 组合增益)。

价值/质量因子需行业指数 PE/PB(index_dailybasic + ETF→指数映射),先跑价格法;若有信号再补。
口径同 diag_sector:月末调仓 T+2、含 COMMISSION+SLIPPAGE。top_n 按篮子大小取 ~1/3。
"""
import time
import numpy as np
import pandas as pd

import cta as mc
import engine as e
from multi_strategy import CTAStrategy, BondMomentumStrategy, MultiStrategy
from diag_sector import (fetch_etf, _month_rebal, _blended_mom, _sharpe, _metrics,
                         sector_equalweight)
from diag_sector_factors import _sector_select

# 候选篮子(先验按行业覆盖,不用收益挑)。fetch 后按有效性过滤。
CANDIDATES = {
    "159928": "消费", "512010": "医药", "512880": "证券", "512800": "银行",
    "512200": "地产", "512660": "军工", "512980": "传媒", "512400": "有色",
    "512480": "半导体", "515000": "科技", "516160": "新能源", "515030": "新能源车",
    "515170": "食品饮料", "515210": "煤炭", "512720": "计算机", "515880": "通信",
    "159825": "农业", "516710": "光伏", "159995": "芯片",
}
MIN_DAYS = 750            # 至少 ~3 年历史才纳入
LONG_CUTOFF = "2017-10-01"   # LONG 面板:上市 ≤ 2017-10(旧经济为主,~9 年,统计力强)
FULL_CUTOFF = "2020-03-01"   # FULL 面板:从 2020-03(含新经济:半导体/科技/计算机/通信/芯片等)


def sector_momentum(px_sec, top_n):
    moms = pd.DataFrame({c: _blended_mom(px_sec[c].dropna()) for c in px_sec})
    return _sector_select(px_sec, moms, top_n=top_n)


def sector_lowvol(px_sec, top_n, lookback=126):
    vol = px_sec.pct_change().rolling(lookback).std() * np.sqrt(252)
    return _sector_select(px_sec, -vol, top_n=top_n)


def sector_reversal(px_sec, top_n, lookback=21):
    trailing = px_sec / px_sec.shift(lookback) - 1.0
    return _sector_select(px_sec, -trailing, top_n=top_n)


def load_base():
    res = e.load_real(with_limits=True)
    px_broad = res[0]
    cb, cs = (res[1], res[2]) if isinstance(res, tuple) else (None, None)
    base = MultiStrategy([CTAStrategy(vol_target=mc.VOL_TARGET, trend_ma=mc.TREND_MA, hold_all=True),
                          BondMomentumStrategy()])
    _, ret_base, _, _ = e.backtest(px_broad, strategy=base, cant_buy=cb, cant_sell=cs)
    return ret_base, px_broad


def fetch_panel(px_broad):
    """拉全部候选,返回 {code: (adj_series, first_date)};过滤无效/太短。"""
    print("拉候选行业 ETF:", len(CANDIDATES), "个...")
    out = {}
    for c, name in CANDIDATES.items():
        try:
            s = fetch_etf(c).reindex(px_broad.index).dropna()
        except Exception as ex:
            print(f"  {c} {name}: ✗ 拉取失败 {ex}")
            continue
        if len(s) < MIN_DAYS:
            print(f"  {c} {name}: ✗ 历史太短({len(s)} 日 < {MIN_DAYS})")
            continue
        out[c] = (s, s.index[0])
        print(f"  {c} {name}: ✓ {s.index[0].date()}~{s.index[-1].date()} ({len(s)} 日)")
        time.sleep(0.5)
    return out


def build_panel(data, codes, start):
    px = pd.DataFrame({c: data[c][0] for c in codes}).loc[start:]
    return px.dropna(how="all")


def run_panel(label, px_sec, ret_base, top_n):
    n = px_sec.shape[1]
    print(f"\n{'='*78}")
    print(f"[{label}] {n} 行业, {px_sec.index[0].date()}~{px_sec.index[-1].date()} "
          f"({len(px_sec)} 日), top_n={top_n}")
    print(f"{'='*78}")
    methods = {
        "等权(不挑,基准)": sector_equalweight(px_sec),
        f"动量(top{top_n})": sector_momentum(px_sec, top_n),
        f"低波(top{top_n})": sector_lowvol(px_sec, top_n),
        f"反转(top{top_n})": sector_reversal(px_sec, top_n),
    }
    print(f"{'方法':<18}{'夏普':>7}{'年化':>8}{'回撤':>8}{'vs等权':>9}{'跑赢等权':>10}")
    print("-" * 78)
    ew_sharpe = _sharpe(methods["等权(不挑,基准)"])
    winners = []
    for name, rs in methods.items():
        sr = _sharpe(rs)
        mt = _metrics(rs)
        # vs 等权 相关 + 是否跑赢
        common = methods["等权(不挑,基准)"].index.intersection(rs.index)
        a = methods["等权(不挑,基准)"].loc[common].to_numpy(float)
        b = rs.loc[common].to_numpy(float)
        m = ~(np.isnan(a) | np.isnan(b))
        corr_ew = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 5 else float("nan")
        beats = sr > ew_sharpe + 0.02     # 跑赢等权 0.02 以上算"胜"(噪声容差)
        tag = "✓" if beats else "✗"
        print(f"{name:<18}{sr:>7.2f}{mt['年化']*100:>7.1f}%{mt['回撤']*100:>7.1f}%"
              f"{corr_ew:>+9.3f}{tag:>10}")
        if beats and "等权" not in name:
            winners.append((name, rs))
    print(f"  → 等权夏普 {ew_sharpe:.2f};{'没有选法跑赢等权(选股无效,与篮子无关)' if not winners else '有选法跑赢等权,进三道关'}")

    # 三道关 vs 基本盘(对胜者)
    if winners:
        print(f"\n  [三道关 vs 基本盘(夏普{_sharpe(ret_base):.2f})]")
        print(f"  {'方法':<16}{'夏普':>7}{'vs基本盘':>10}{'①互补':>7}{'②自身':>7}{'③增益':>10}{'判定':>6}")
        for name, rs in winners:
            common = ret_base.index.intersection(rs.index)
            rb = ret_base.loc[common].to_numpy(float)
            rs_c = rs.loc[common].to_numpy(float)
            mask = ~(np.isnan(rb) | np.isnan(rs_c))
            corr = float(np.corrcoef(rb[mask], rs_c[mask])[0, 1]) if mask.sum() > 5 else float("nan")
            sr_s = _sharpe(rs); sr_base = _sharpe(ret_base.loc[common])
            comb = pd.Series(0.5 * rb + 0.5 * rs_c, index=common)
            sr_comb = _sharpe(comb)
            g1 = abs(corr) < 0.3; g2 = sr_s > 0.5; g3 = sr_comb > max(sr_s, sr_base)
            verdict = "✓过" if (g1 and g2 and g3) else "✗"
            g3s = f"{sr_comb:.2f}>{max(sr_s,sr_base):.2f}" if g3 else f"{sr_comb:.2f}≤{max(sr_s,sr_base):.2f}"
            print(f"  {name:<16}{sr_s:>7.2f}{corr:>+10.3f}{('✓' if g1 else '✗'):>7}"
                  f"{('✓' if g2 else '✗'):>7}{g3s:>10}{verdict:>6}")
    return winners


def main():
    print("加载基本盘(CTA hold_all=True + 国债 K=2)...")
    ret_base, px_broad = load_base()
    print(f"  基本盘 夏普 {_sharpe(ret_base):.2f}")

    data = fetch_panel(px_broad)
    valid = list(data.keys())
    print(f"\n有效行业 {len(valid)} 个: {[f'{c}({CANDIDATES[c]})' for c in valid]}")

    # LONG 面板:上市 ≤ 2016
    long_codes = [c for c in valid if data[c][1] <= pd.Timestamp(LONG_CUTOFF)]
    if len(long_codes) >= 5:
        px_long = build_panel(data, long_codes, LONG_CUTOFF)
        run_panel(f"LONG(上市≤2016, {len(long_codes)}行业)", px_long, ret_base, top_n=3)

    # FULL 面板:从 2019-06,全部有效
    full_codes = [c for c in valid if data[c][1] <= pd.Timestamp(FULL_CUTOFF)]
    if len(full_codes) >= 8:
        px_full = build_panel(data, full_codes, FULL_CUTOFF)
        run_panel(f"FULL(全行业, {len(full_codes)}行业)", px_full, ret_base, top_n=4)

    print("\n" + "=" * 78)
    print("判读:")
    print("  若两面板都没有选法跑赢等权 → 『选股无效』与篮子无关(J2+M5+M6+本诊断四重证伪),")
    print("    『怎么挑板块』的答案=不挑(等权全行业)。板块策略的正确形态可能是『被动等权配置』而非主动选股。")
    print("  即便有选法跑赢等权,仍需过三道关(互补+自身+增益)才值得作卫星——而板块=权益,与基本盘")
    print("  同源高相关,①互补大概率挂(同 M5/M6)。")


if __name__ == "__main__":
    main()
