# -*- coding: utf-8 -*-
"""诊断:配对交易(协整套利)作为第二策略的可行性 —— 与 CTA 的相关性诊断(理论上界)。

⚠️ 重要前提:这是"假设做空可行"的理论上界诊断。
  经典配对交易 = 多 A 空 B 的市场中性组合,其低相关性正来自 beta 对冲(对冲掉市场因子,
  只赚价差回归 alpha)。剥夺做空,它就退化成轮动(同 CTA 同源,见 diag_mr.py 的 +0.7 结论)。
  A股 ETF 融券几乎不可行(费率高、标的少),实操做空成本会大幅侵蚀甚至吞掉收益。
  故本诊断回答的是:"配对交易这个策略品类,在 A股 ETF 标的上,理论上(假设做空)与 CTA 互补吗?"
    - 若理论上界就很差(高相关/负收益)→ 配对交易彻底放弃。
    - 若理论上界很好(近零相关/正收益)→ 是好方向,但需解决做空实现(期权合成空头/股指期货对冲/融券)。

方法:
  1. 协整筛选:POOL 全部两两 Engle-Granger coint 检验,选 p<阈值的协整对。
     (注:全样本选对含前视,仅用于判断互补性是否存在,非实盘表现。)
  2. 配对交易回测(每对 dollar-neutral + beta-hedge):滚动 z-score(60日),entry±2/exit±0.5,
     T+1 成交,换手扣 COMMISSION+SLIPPAGE。⚠️ 未计做空融券成本 → 结果是上界。
  3. vs CTA 相关性(全样本/分regime)+ 组合增益(50/50)。

判定:与 CTA 近零相关(市场中性应≈0)+ 配对自身正收益 + 组合增益 → 值得探索做空实现路径。
"""
import numpy as np
import pandas as pd
from itertools import combinations

import cta as mc
import engine as e
from engine import POOL, BENCH, COMMISSION, SLIPPAGE

WINDOW = 60        # 价差 z-score 回看窗口
ENTRY = 2.0        # z 超过 ±2 开仓
EXIT = 0.5         # z 回到 ±0.5 内平仓
COINT_P = 0.10     # 协整筛选阈值(宽松,纳入候选)


def cta_returns(px, cb, cs):
    nav, ret, _, _ = e.backtest(px, hold_all=True, vol_target=mc.VOL_TARGET,
                                trend_ma=mc.TREND_MA, cant_buy=cb, cant_sell=cs)
    return nav, ret


def pair_positions(z, entry, exit_):
    """经典 entry/exit 规则生成持仓状态(+1 多A空B / -1 空A多B / 0 平仓)。仅平仓后重新进场,不直接翻转。"""
    pos = pd.Series(0, index=z.index)
    state = 0
    vals = z.values
    for i in range(len(vals)):
        zi = vals[i]
        if np.isnan(zi):
            continue
        if state == 0:
            if zi < -entry:
                state = 1
            elif zi > entry:
                state = -1
        elif state == 1:
            if zi > -exit_:
                state = 0
        elif state == -1:
            if zi < exit_:
                state = 0
        pos.iloc[i] = state
    return pos


def pair_backtest(px, a, b):
    """单对配对交易回测(假设做空可行,dollar-neutral + beta-hedge)。返回 (nav, daily_net, n_trades)。
    收益 = pos × [0.5·ret_a − 0.5·β·ret_b],β=全样本 OLS log price 回归(诊断口径,含前视)。"""
    s = px[[a, b]].dropna()
    if len(s) < 300:
        return None
    la, lb = np.log(s[a]), np.log(s[b])
    beta = float(np.polyfit(lb.values, la.values, 1)[0])     # log(A) ~ β·log(B)
    spread = la - beta * lb
    m = spread.rolling(WINDOW).mean()
    sd = spread.rolling(WINDOW).std()
    z = (spread - m) / (sd + 1e-12)
    pos = pair_positions(z, ENTRY, EXIT)
    ra = s[a].pct_change().fillna(0.0)
    rb = s[b].pct_change().fillna(0.0)
    gross = pos.shift(1).fillna(0) * (0.5 * ra - 0.5 * beta * rb)     # T+1 成交防前视
    turnover = pos.diff().abs().fillna(0) * 0.5 * (1 + abs(beta))     # 双边换手近似
    cost = turnover * (COMMISSION + SLIPPAGE)
    net = (gross - cost).fillna(0.0)
    nav = (1 + net).cumprod()
    nav = nav / nav.iloc[0]
    n_trades = int((pos.diff().abs() > 0).sum())
    return nav, net, n_trades


def corr_by_regime(a, b, px):
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
    print(f"  {px.index[0].date()} ~ {px.index[-1].date()}, {len(px)} 个交易日")

    print("\n[1] CTA 口径回测(等权全池 + vol_target + 趋势过滤)...")
    nav_cta, ret_cta = cta_returns(px, cb, cs)

    print("\n[2] 协整筛选(Engle-Granger coint,全样本,含选对前视——仅判互补性)...")
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        print("  [跳过] statsmodels 未安装。")
        coint = None
    coint_rows = []
    for a, b in combinations(list(POOL.keys()), 2):
        s = px[[a, b]].dropna()
        if len(s) < 300:
            continue
        pval = float(coint(s[a].values, s[b].values)[1]) if coint else 1.0
        coint_rows.append((a, POOL[a], b, POOL[b], pval, len(s)))
    coint_rows.sort(key=lambda x: x[4])
    print(f"  {'对':<30}{'coint p':>10}{'样本':>8}")
    for a, na, b, nb, p, n in coint_rows:
        flag = " *" if p < COINT_P else ""
        print(f"  {na}({a}) vs {nb}({b})".ljust(30) + f"{p:>10.3f}{n:>8}{flag}")

    coint_pairs = [(r[0], r[2]) for r in coint_rows if r[4] < COINT_P]
    if not coint_pairs:
        coint_pairs = [(r[0], r[2]) for r in coint_rows[:2]]
        print(f"\n  ⚠️ 无对通过 p<{COINT_P}(协整性弱);取 p 最小 2 对作对照: "
              f"{[(POOL[a], POOL[b]) for a, b in coint_pairs]}")
    else:
        print(f"\n  协整对(p<{COINT_P}): {[(POOL[a], POOL[b]) for a, b in coint_pairs]}")

    print("\n[3] 配对交易回测(假设做空可行,dollar-neutral + beta-hedge;⚠️ 未计融券成本 = 上界)...")
    pair_rets = []
    for a, b in coint_pairs:
        r = pair_backtest(px, a, b)
        if r is None:
            continue
        nav_p, net_p, nt = r
        p = e.perf(nav_p, net_p)
        print(f"  {POOL[a]}({a}) vs {POOL[b]}({b}): 年化{p['年化']*100:6.2f}% 夏普{p['夏普']:5.2f} "
              f"回撤{p['回撤']*100:6.1f}% 开平仓{nt:>4}次")
        pair_rets.append(net_p)

    if not pair_rets:
        print("  无可回测对。")
        return

    df = pd.concat(pair_rets, axis=1)
    ret_pairs = df.mean(axis=1)                                 # 多对等权合成
    nav_pairs = (1 + ret_pairs).cumprod()
    nav_pairs = nav_pairs / nav_pairs.iloc[0]

    common = ret_cta.index.intersection(ret_pairs.index)
    rc = ret_cta.loc[common]
    rp = ret_pairs.loc[common]
    comb = 0.5 * rc + 0.5 * rp

    print("\n=== 绩效对比 ===")
    def row(name, nav, daily):
        q = e.perf(nav, daily)
        return (name, f"{q['年化']*100:.1f}%", f"{q['夏普']:.2f}",
                f"{q['回撤']*100:.1f}%", f"{q['Sortino']:.2f}")
    print(f"{'策略':<30}{'年化':>8}{'夏普':>7}{'回撤':>8}{'Sortino':>9}")
    for r in [row("CTA(等权+风控)", nav_cta, ret_cta),
              row("配对组合(等权·假设做空)", nav_pairs, ret_pairs),
              row("50/50 CTA+配对", (1 + comb).cumprod(), comb)]:
        print(f"{r[0]:<30}{r[1]:>8}{r[2]:>7}{r[3]:>8}{r[4]:>9}")

    print("\n=== 相关性:CTA vs 配对组合(分 regime) ===")
    print("  (市场中性配对应≈0;接近±1 = 仍有共同市场因子)")
    for r, n, c in corr_by_regime(rc, rp, px):
        print(f"    {r:<6} n={n:>5}  corr={c:+.3f}")

    print("\n=== 判定 ===")
    full = float(np.corrcoef(rc, rp)[0, 1])
    pp = e.perf(nav_pairs, ret_pairs)
    pc = e.perf((1 + comb).cumprod(), comb)
    pca = e.perf(nav_cta, ret_cta)
    print(f"  全样本相关性:     {full:+.3f}   (|corr|<0.3 = 真互补;对比均值回归 +0.71)")
    print(f"  配对组合夏普:     {pp['夏普']:.2f}    (>0.5 = 自身站得住)")
    print(f"  50/50 组合夏普:   {pc['夏普']:.2f}    vs 单CTA {pca['夏普']:.2f}  (>单策略 = 分散增益)")
    print("  ⚠️ 以上为假设做空的理论上界;实操 A股 融券费率高/标的少,成本会大幅侵蚀,甚至转负。")


if __name__ == "__main__":
    main()
