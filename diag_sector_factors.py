# -*- coding: utf-8 -*-
"""诊断:非动量因子选行业(低波 / 反转)三道关 vs 基本盘(CTA+国债 K=2)。

承接 diag_sector(动量选行业 A 夏普 0.12、三关全挂)。J2 只证伪了『动量选股』(PBO=0.67);
本诊断测**非动量因子**在行业 ETF 上的横截面选股是否有效:
  · 低波(防御):选 trailing 实现波动最低的行业 —— 最强防御因子,最贴用户『资金安全』目标。
  · 反转(逆向):选 trailing 收益最差的行业(博跌深反弹)。

⚠ 价值/质量因子需行业指数 PE/PB/ROE 基本面(tushare index_dailybasic + ETF→指数映射),
作为 phase 2:**若低波(最强防御因子)都挂,则行业选股无论什么因子都死,省去拉基本面功夫**;
若有信号再补价值/质量。

口径同 diag_sector:5 行业(消费/医药/证券/银行/地产,先验选池)、2013 起、月末调仓 T+2、
含 COMMISSION+SLIPPAGE。三道关:互补 |corr|<0.3 + 自身夏普>0.5 + 组合增益 50/50>max。
语境基准:等权 B&H(夏普 0.26)——因子策略比它差即说明选股主动亏钱。
"""
import time
import numpy as np
import pandas as pd

import cta as mc
import engine as e
from engine import COMMISSION, SLIPPAGE
from multi_strategy import CTAStrategy, BondMomentumStrategy, MultiStrategy
from diag_sector import (fetch_etf, _month_rebal, _sharpe, _metrics, sector_equalweight,
                         SECTORS, TOP_N, START)

VOL_LOOKBACK = 126          # 低波:trailing 126 日(~半年)实现波动
REV_LOOKBACK = 21           # 反转:trailing 21 日(~1 月)收益


def _sector_select(px_sec, score_df, top_n=TOP_N):
    """通用横截面选股:每月选 score 最高的 top_n 行业、等权。月末调仓 T+2(对齐 diag_sector)。
    score_df[date, sector] 越大越想配。返回 daily net ret。"""
    all_rebal = sorted(set().union(*[_month_rebal(px_sec[c].dropna()) for c in px_sec]))
    weight_df = pd.DataFrame(np.nan, index=px_sec.index, columns=px_sec.columns)
    for d in all_rebal:
        if d not in score_df.index:
            continue
        row = score_df.loc[d].dropna()
        if len(row) >= top_n:
            picks = list(row.nlargest(top_n).index)
            weight_df.loc[d, picks] = 1.0 / top_n
            others = [c for c in weight_df.columns if c not in picks]
            weight_df.loc[d, others] = 0.0
    weights = weight_df.ffill().shift(1).fillna(0.0)
    w_lag = weights.shift(1).fillna(0.0)
    ret = px_sec.pct_change().fillna(0.0)
    gross = (w_lag * ret).sum(axis=1)
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = (gross - turnover * (COMMISSION + SLIPPAGE)).fillna(0.0)
    return net.loc[all_rebal[0]:] if all_rebal else net.loc[START:]


def sector_lowvol(px_sec, top_n=TOP_N, lookback=VOL_LOOKBACK):
    """低波:选 trailing 实现波动最低的行业(防御)。score = -vol(波动越低分越高)。"""
    vol = px_sec.pct_change().rolling(lookback).std() * np.sqrt(252)
    return _sector_select(px_sec, -vol)


def sector_reversal(px_sec, top_n=TOP_N, lookback=REV_LOOKBACK):
    """反转:选 trailing 收益最差的行业(逆向,博反弹)。score = -trailing_return。"""
    trailing = px_sec / px_sec.shift(lookback) - 1.0
    return _sector_select(px_sec, -trailing)


def main():
    print("加载基本盘(CTA hold_all=True + 国债 K=2)...")
    res = e.load_real(with_limits=True)
    px_broad = res[0]
    cb, cs = (res[1], res[2]) if isinstance(res, tuple) else (None, None)
    base = MultiStrategy([CTAStrategy(vol_target=mc.VOL_TARGET, trend_ma=mc.TREND_MA, hold_all=True),
                          BondMomentumStrategy()])
    _, ret_base, _, _ = e.backtest(px_broad, strategy=base, cant_buy=cb, cant_sell=cs)
    print(f"  基本盘 全样本 夏普 {_sharpe(ret_base):.2f}")

    print("\n拉行业 ETF:", [f"{c}({n})" for c, n in SECTORS.items()])
    px_sec = pd.DataFrame(index=px_broad.index)
    for c in SECTORS:
        px_sec[c] = fetch_etf(c).reindex(px_broad.index)
        time.sleep(0.6)
    px_sec = px_sec.loc[START:]
    print(f"  行业面板: {px_sec.index[0].date()} ~ {px_sec.index[-1].date()} 共 {len(px_sec)} 交易日")

    ret_lowvol = sector_lowvol(px_sec)
    ret_reversal = sector_reversal(px_sec)
    ret_ew = sector_equalweight(px_sec)

    print(f"\n{'='*76}")
    print(f"[三道关] 非动量因子选行业 vs 基本盘(CTA+国债 K=2)")
    print(f"{'='*76}")
    print(f"{'策略':<24}{'夏普':>6}{'年化':>7}{'回撤':>7}{'vs基本盘':>10}"
          f"{'①互补':>7}{'②自身':>7}{'③组合增益':>14}{'判定':>6}")
    print("-" * 76)
    for name, rs in [("低波(选低波动·防御)", ret_lowvol), ("反转(选跌多·逆向)", ret_reversal)]:
        common = ret_base.index.intersection(rs.index)
        rb = ret_base.loc[common].to_numpy(float)
        rs_c = rs.loc[common].to_numpy(float)
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
        g3s = f"{sr_comb:.2f}>{max(sr_s, sr_base):.2f}" if g3 else f"{sr_comb:.2f}≤{max(sr_s, sr_base):.2f}"
        print(f"{name:<24}{sr_s:>6.2f}{mt['年化']*100:>6.1f}%{mt['回撤']*100:>6.1f}%"
              f"{corr:>+10.3f}{('✓' if g1 else '✗'):>7}{('✓' if g2 else '✗'):>7}{g3s:>14}{verdict:>6}")
    mt_ew = _metrics(ret_ew)
    print(f"{'(语境)等权B&H':<24}{_sharpe(ret_ew):>6.2f}{mt_ew['年化']*100:>6.1f}%"
          f"{mt_ew['回撤']*100:>6.1f}%   ← 无选股基准(因子比它差=主动亏钱)")
    print(f"{'(对照)动量A(diag_sector)':<24}{'0.12':>6}{'0.5%':>7}{'-54.4%':>7}{'+0.60':>10}")

    print("\n判读:")
    print("  低波:选防御行业。低波是最强防御因子、最贴『资金安全』;若它都挂,行业选股无论因子都死。")
    print("  反转:博跌深反弹;A股政策驱动下可能抓政策底,但易接落刀(地产长期下跌)。")
    print("  两因子 vs 等权B&H(夏普0.26):比它差=选股主动亏钱(同 diag_sector 动量 A 0.12)。")
    print("  价值/质量(phase 2):需行业指数 PE/PB/ROE;若低波/反转全挂则不必再花功夫拉基本面。")


if __name__ == "__main__":
    main()
