# -*- coding: utf-8 -*-
"""诊断:EPO(Enhanced Portfolio Optimization,Ledoit-Wolf 收缩协方差最小方差)
vs 等权 / 1σ RiskParity 资金分配 + block bootstrap 显著性。

来源动机:聚宽社区 EPO 策略(Ledoit-Wolf 收缩协方差求权重)在本仓库 grep 0 命中——
组合权重层是空白。现有 RiskParityMulti 是"对角 1/σ"(只用各策略自己的波动、忽略策略间
相关性);EPO 用收缩后的全协方差求最小方差权重,理论上利用了相关性信息,收缩又压住
小样本(滚动窗口)下协方差估计的噪声。本诊断回答:EPO 相对 1/σ 有无边际。

⚠ 这是诊断脚本(走三道关),不改 multi_strategy.py / cta.py。判定通过才考虑落地 EPOMulti。

本仓库组合层范式:CTA 策略(等权全池+风控)+ 国债策略,在"策略层"组合(B 型,见
multi_strategy.py)。故 EPO 也在策略层做(2 资产:CTA/国债),与 diag_riskparity 严格可比,
而非资产层(7 ETF)。N=2 收缩空间有限——若不显著是诚实结论(2 资产相关性参数仅 1 个,
1/σ 与 min-var 差别本就小);若要更大 N 的 EPO 需扩到资产层,那是另一架构、另开诊断。

三层诊断(对齐 diag_riskparity.py 的三层):
  1. 静态全样本协方差(有前视,标注):LW 收缩 min-var vs 1/σ vs 等权
  2. 滚动无前视:每 t 用过去 VOL_WINDOW 日收益算 t-1 的 LW 收缩 min-var 权重
  3. 月频 backtest:EPOMulti(滚动 LW min-var)vs RiskParityMulti(1/σ)vs 等权 MultiStrategy
每层配对 circular block bootstrap(block=21≈1月,n=2000):
  Δ=SR(EPO)−SR(对照),p=P(Δ≤0)。核心对比是 EPO vs 1/σ(现有部署候选)。

Ledoit-Wolf(2004)收缩协方差(手写,不依赖 sklearn):
  Σ_shrunk = δ·F + (1−δ)·S
  S = (1/T) X'X           样本协方差(X 已去均值)
  F = mu·I(mu=trace(S)/N)  缩放单位阵目标(sklearn/OAS 默认的 well-conditioned estimator)
                          ※ 不用常数相关目标:N=2 时 F 恒等于 S(gamma=0、delta=0、未收缩),
                            使"LW-shrunk min-var"退化为未收缩样本协方差 min-var(负结论被
                            "无法收缩的目标"污染)。identity 目标 N=2 下 F≠S、gamma>0、delta>0,
                            "LW-shrunk"标签名副其实。
  δ = clip((π−ρ)/(γ·T), 0, 1)
  π = Σ_ij Var(s_ij)      样本协方差各元素的总方差
  γ = ||F−S||²_F          目标与样本的 Frobenius 距离
  ρ = 0                   identity 目标下 f_ij = mu·δ_ij 是常数(非随机的),故
                          Cov(f_ij, s_ij)=0 ∀ij → ρ=0(数值验证:rho=0 严格复现 sklearn
                          LedoitWolf.shrinkage_;rho=trace(pi_mat) 会把 delta 低估约 40-70%)。
min-var 权重:w ∝ Σ_shrunk^{-1} 1,归一化到和=1。负权→ long-only clip+归一化;
  Sigma 奇异(LinAlgError)→ 加对角抖动重解(而非退等权,避免人为压低 EPO 与 1/σ 的差异)。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo 根(找 cta/engine/multi_strategy)
import numpy as np
import pandas as pd

import cta as mc
import engine as e
from multi_strategy import CTAStrategy, BondMomentumStrategy, MultiStrategy, RiskParityMulti

N_BOOT = 2000
BLOCK = 21
VOL_WINDOW = 63   # 滚动协方差窗口(对齐 RiskParityMulti.vol_window 默认)


# ---------------- Ledoit-Wolf 收缩协方差 + 最小方差权重 ----------------
def ledoit_wolf_cov(R):
    """Ledoit-Wolf(2004)收缩协方差估计(缩放单位阵目标,对齐 sklearn/OAS well-conditioned estimator)。

    R: (T, N) 日收益矩阵(每列一个资产),未去均值(内部去)。
    返回 (Sigma_shrunk[N,N], delta)。delta∈[0,1]=收缩强度(0=纯样本协方差,1=纯目标)。

    目标用 F=mu*I(mu=trace(S)/N)而非常数相关:常数相关目标在 N=2 下 F≡S(仅 1 个非对角
    corr,F[0,1]=r̄·sd0·sd1=corr[0,1]·sd0·sd1=S[0,1];对角 d_i=S_ii),导致 gamma=0、delta=0,
    "LW-shrunk min-var"实为未收缩样本协方差 min-var。identity 目标保证 F≠S、gamma>0、delta>0。
    identity 目标下 f_ij=mu·δ_ij 是常数(非随机的),故 Cov(f_ij,s_ij)=0 ∀ij → rho=0。
    数值验证:rho=0 严格复现 sklearn LedoitWolf.shrinkage_;若用 rho=trace(pi_mat) 会把
    delta 低估约 40-70%(误把对角项 Var(s_ii) 当成 f_ii 与 s_ii 的协方差,但 f_ii=mu 常数→协方差=0)。
    """
    R = np.asarray(R, dtype=float)
    T, N = R.shape
    R = R - R.mean(axis=0)
    S = (R.T @ R) / T                          # 样本协方差(除以 T,ML 口径)
    mu = np.trace(S) / N                       # 缩放常数(=平均方差)
    F = mu * np.eye(N)                         # identity 收缩目标(well-conditioned):F≠S(N=2 亦然)
    # π = Σ_ij Var(s_ij),其中 Var(s_ij)=(1/T)Σ_t(x_it x_jt)² − s_ij²
    #   因为 (x_i x_j)² = x_i²·x_j²,故 (1/T)Σ(x_it x_jt)² = (X²ᵀ X²)/T
    X2 = R ** 2
    pi_mat = (X2.T @ X2) / T - S ** 2
    pi = pi_mat.sum()
    gamma = np.sum((F - S) ** 2)               # γ=||F−S||²_F(identity 目标下非零)
    # rho: LW2004 定义 Σ_ij Cov(f_ij, s_ij)。identity 目标 f_ij=mu·δ_ij 是常数 → Cov(常数, s_ij)=0,
    #   故 rho=0(数值验证严格等于 sklearn LedoitWolf.shrinkage_ 的隐含 rho)。pi_mat 仅用于 π。
    rho = 0.0
    kappa = (pi - rho) / gamma if gamma > 1e-12 else 0.0
    delta = float(np.clip(kappa / T, 0.0, 1.0))
    Sigma = delta * F + (1.0 - delta) * S
    return Sigma, delta


def min_var_weights(Sigma):
    """最小方差权重 w ∝ Σ^{-1} 1,归一化和=1。

    不再一律退等权(原逻辑在 15.7% rolling rebalance 触发、人为把 EPO 压成等权、
    压低它与 1/σ 的差异)。两个触发分别治:
      (a) 负权重分支:Σ 正定但无约束 min-var 给负权(corr>0.19 时)→ long-only:clip 负权
          到 0 再归一化(N=2 下退化干净),而非退等权。
      (b) LinAlgError 分支:Σ 奇异(rolling bond 空仓期全 0 方差→ det=0)→ 加对角抖动
          1e-8 重解;若仍奇异才退等权。"""
    N = Sigma.shape[0]
    ones = np.ones(N)
    try:
        w = np.linalg.solve(Sigma, ones)
    except np.linalg.LinAlgError:
        # Σ 奇异:加对角抖动后重解(对齐 sklearn OAS 的正则化思路),避免直接退等权
        try:
            w = np.linalg.solve(Sigma + 1e-8 * np.eye(N), ones)
        except np.linalg.LinAlgError:
            return ones / N
    if np.any(~np.isfinite(w)):
        return ones / N
    if np.any(w <= 0):
        # long-only 约束 min-var:clip 负权到 0 再归一化(N=2 退化干净)
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            return ones / N
        return w / w.sum()
    return w / w.sum()


def inv_vol_weights(sig_annual):
    """对角 1/σ 权重(等价 RiskParityMulti 静态版;忽略相关性)。sig_annual:各资产年化波动数组。"""
    inv = 1.0 / np.maximum(sig_annual, 1e-12)
    return inv / inv.sum()


# ---------------- bootstrap / 报告 辅助 ----------------
def _boot_vs_ref(cand, ref, n_boot=N_BOOT, block=BLOCK, seed=7):
    """配对 circular block bootstrap:Δ=SR(cand)−SR(ref) 的分布。返回 boots 数组。"""
    T = len(ref)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        idx = e._circ_block_idx(T, block, rng)
        boots[k] = e._ann_sharpe(cand[idx]) - e._ann_sharpe(ref[idx])
    return boots


def _row(name, r, idx):
    nav = pd.Series(np.cumprod(1.0 + r), index=idx)
    nav = nav / nav.iloc[0]
    q = e.perf(nav, pd.Series(r, index=idx))
    return name, q['年化'], q['夏普'], q['回撤'], q['Sortino']


def _print_delta(label, cand, ref, boots):
    ci = np.percentile(boots, [2.5, 97.5])
    p = float((boots <= 0).mean())
    obs = e._ann_sharpe(cand) - e._ann_sharpe(ref)
    print(f"    {label:<18}Δ(obs)={obs:+.3f}  95%CI=[{ci[0]:+.3f},{ci[1]:+.3f}]  P(Δ≤0)={p:.3f}")


def main():
    print("加载数据(若无 tushare token 走东财回退,较慢)...")
    res = e.load_real(with_limits=True)
    px = res[0]
    cb, cs = (res[1], res[2]) if isinstance(res, tuple) else (None, None)
    print(f"  {px.index[0].date()} ~ {px.index[-1].date()}, {len(px)} 个交易日")

    print("跑 CTA / 国债 单策略回测(取日收益)...")
    cfg = dict(vol_target=mc.VOL_TARGET, trend_ma=mc.TREND_MA, hold_all=True)
    cta = CTAStrategy(**cfg)
    bond = BondMomentumStrategy()
    _, ret_cta, _, _ = e.backtest(px, strategy=cta, cant_buy=cb, cant_sell=cs)
    _, ret_bond, _, _ = e.backtest(px, strategy=bond, cant_buy=cb, cant_sell=cs)

    common = ret_cta.index.intersection(ret_bond.index)
    rc = ret_cta.loc[common].to_numpy(dtype=float)
    rb = ret_bond.loc[common].to_numpy(dtype=float)
    R = np.column_stack([rc, rb])              # (T, 2)

    # ===== 第1层:静态全样本协方差(有前视,概念诊断) =====
    print(f"\n{'='*70}\n[第1层] 静态全样本协方差(有前视,概念诊断)\n{'='*70}")
    Sigma, delta = ledoit_wolf_cov(R)
    w_eq = np.array([0.5, 0.5])
    w_iv = inv_vol_weights(np.array([rc.std() * np.sqrt(252), rb.std() * np.sqrt(252)]))
    w_mv = min_var_weights(Sigma)
    print(f"  LW 收缩强度 δ={delta:.3f}  (0=纯样本协方差, 1=纯 identity 目标 mu·I)")
    print(f"  {'权重':<18}{'CTA':>10}{'国债':>10}")
    print(f"  {'等权':<18}{w_eq[0]*100:>9.1f}%{w_eq[1]*100:>9.1f}%")
    print(f"  {'1/σ risk-parity':<18}{w_iv[0]*100:>9.1f}%{w_iv[1]*100:>9.1f}%")
    print(f"  {'LW min-var(EPO)':<18}{w_mv[0]*100:>9.1f}%{w_mv[1]*100:>9.1f}%")

    r_eq = R @ w_eq
    r_iv = R @ w_iv
    r_mv = R @ w_mv
    print(f"\n  {'组合':<22}{'年化':>8}{'夏普':>7}{'回撤':>8}{'Sortino':>9}")
    for name, r in [("等权 50/50", r_eq), ("1/σ risk-parity", r_iv), ("LW min-var(EPO)", r_mv)]:
        m = _row(name, r, common)
        print(f"  {m[0]:<22}{m[1]*100:>7.1f}%{m[2]:>7.2f}{m[3]*100:>7.1f}%{m[4]:>9.2f}")

    print(f"\n  [block bootstrap] n={N_BOOT}  block={BLOCK}≈1月")
    _print_delta("1/σ vs 等权", r_iv, r_eq, _boot_vs_ref(r_iv, r_eq, seed=7))
    _print_delta("EPO vs 等权", r_mv, r_eq, _boot_vs_ref(r_mv, r_eq, seed=11))
    _print_delta("EPO vs 1/σ ★", r_mv, r_iv, _boot_vs_ref(r_mv, r_iv, seed=23))
    print("    ★ = 核心:EPO(LW 收缩)相对现有 1/σ risk-parity 的边际")

    # ===== 第2层:滚动 LW 收缩(无前视) =====
    print(f"\n{'='*70}\n[第2层] 滚动 LW 收缩(无前视, vol_window={VOL_WINDOW} 日)\n{'='*70}")
    s_rc = pd.Series(rc, index=common)
    s_rb = pd.Series(rb, index=common)
    w_mv_c = pd.Series(np.nan, index=common)
    w_mv_b = pd.Series(np.nan, index=common)
    for i in range(VOL_WINDOW, len(common)):
        Rw = np.column_stack([rc[i - VOL_WINDOW:i], rb[i - VOL_WINDOW:i]])
        Sw, _ = ledoit_wolf_cov(Rw)
        w = min_var_weights(Sw)
        w_mv_c.iloc[i] = w[0]
        w_mv_b.iloc[i] = w[1]
    w_mv_c = w_mv_c.shift(1).fillna(0.5)       # t-1 权重(无前视)
    w_mv_b = w_mv_b.shift(1).fillna(0.5)
    valid = common[VOL_WINDOW:]
    r_mv_roll = (w_mv_c.loc[valid] * s_rc.loc[valid] + w_mv_b.loc[valid] * s_rb.loc[valid]).to_numpy(float)
    r_eq_roll = (0.5 * s_rc.loc[valid] + 0.5 * s_rb.loc[valid]).to_numpy(float)
    # 滚动 1/σ 对照(复现 diag_riskparity 滚动段)
    sig_c = s_rc.rolling(VOL_WINDOW).std() * np.sqrt(252)
    sig_b = s_rb.rolling(VOL_WINDOW).std() * np.sqrt(252)
    tot = sig_c + sig_b
    w_iv_c = (sig_b / tot).shift(1).fillna(0.5)   # w_cta ∝ 1/σ_c = σ_b/(σ_b+σ_c)
    w_iv_b = (sig_c / tot).shift(1).fillna(0.5)
    r_iv_roll = (w_iv_c.loc[valid] * s_rc.loc[valid] + w_iv_b.loc[valid] * s_rb.loc[valid]).to_numpy(float)
    print(f"  滚动权重均值: CTA {w_mv_c.loc[valid].mean()*100:.1f}%  国债 {w_mv_b.loc[valid].mean()*100:.1f}%")
    print(f"  {'组合':<22}{'年化':>8}{'夏普':>7}{'回撤':>8}")
    for name, r in [("等权", r_eq_roll), ("滚动 1/σ", r_iv_roll), ("滚动 LW min-var", r_mv_roll)]:
        m = _row(name, r, valid)
        print(f"  {m[0]:<22}{m[1]*100:>7.1f}%{m[2]:>7.2f}{m[3]*100:>7.1f}%")
    _print_delta("EPO vs 1/σ(滚动)", r_mv_roll, r_iv_roll, _boot_vs_ref(r_mv_roll, r_iv_roll, seed=31))

    # ===== 第3层:月频 backtest(EPOMulti vs RiskParityMulti vs 等权) =====
    print(f"\n{'='*70}\n[第3层] 月频 backtest(现实可部署口径,项目调仓频率)\n{'='*70}")

    class EPOMulti:
        """滚动 LW 收缩 min-var 组合器(诊断用,未入 multi_strategy.py)。
        实现 Strategy 协议;照 RiskParityMulti,把"1/σ"换成"LW 收缩协方差→min-var 权重"。
        每次 target 用各子策略「按上期持仓」的近期日收益序列拼成 (L, n) 矩阵做 LW 收缩;
        历史不足 VOL_WINDOW/2 或首次(无上期)或协方差非正定 → 退等权。"""
        def __init__(self, strategies, vol_window=VOL_WINDOW, min_weight=0.05):
            self.strategies = strategies
            self.vol_window = vol_window
            self.min_weight = min_weight
            self.name = "EPO(" + "+".join(s.name for s in strategies) + ")"
            self.universe = sorted({c for s in strategies for c in s.universe})
            self._prev = None

        def target(self, recent, t):
            sub = [s.target(recent, t) for s in self.strategies]
            n = len(self.strategies)
            allocs = [1.0 / n] * n
            if self._prev is not None:
                codes = set()
                for pt in self._prev:
                    codes |= set(pt.keys())
                lens = [len(recent[c]) for c in codes if c in recent]
                if lens:
                    L = min(min(lens) - 1, self.vol_window)
                    strat_rets = []
                    ok = True
                    for ci in range(n):
                        pt = self._prev[ci]
                        cols = []
                        for c, w in pt.items():
                            arr = recent.get(c)
                            if arr is None or len(arr) <= L:
                                continue
                            tail = arr[-L - 1:]
                            cols.append([w * (tail[k] / tail[k - 1] - 1.0) for k in range(1, L + 1)])
                        if not cols:
                            ok = False
                            break
                        strat_rets.append([sum(col) for col in zip(*cols)])
                    if ok and len(strat_rets) == n:
                        Sw, _d = ledoit_wolf_cov(np.column_stack(strat_rets))
                        a = min_var_weights(Sw)
                        a = [max(x, self.min_weight) for x in a]
                        s2 = sum(a)
                        allocs = [x / s2 for x in a]
            self._prev = sub
            out = {}
            for st, a in zip(sub, allocs):
                for c, w in st.items():
                    out[c] = out.get(c, 0.0) + a * w
            return out

    epo = EPOMulti([CTAStrategy(**cfg), BondMomentumStrategy()])
    rp = RiskParityMulti([CTAStrategy(**cfg), BondMomentumStrategy()])
    eq = MultiStrategy([CTAStrategy(**cfg), BondMomentumStrategy()], allocs=[0.5, 0.5])
    _, req_bt, _, _ = e.backtest(px, strategy=eq, cant_buy=cb, cant_sell=cs)
    _, rrp_bt, _, _ = e.backtest(px, strategy=rp, cant_buy=cb, cant_sell=cs)
    _, rpo_bt, _, _ = e.backtest(px, strategy=epo, cant_buy=cb, cant_sell=cs)
    cm = req_bt.index.intersection(rrp_bt.index).intersection(rpo_bt.index)
    a = req_bt.loc[cm].to_numpy(float)
    b = rrp_bt.loc[cm].to_numpy(float)
    c = rpo_bt.loc[cm].to_numpy(float)
    print(f"  {'组合':<24}{'年化':>8}{'夏普':>7}{'回撤':>8}")
    for name, r in [("等权", a), ("RiskParity(1/σ)", b), ("EPO(LW min-var)", c)]:
        m = _row(name, r, cm)
        print(f"  {m[0]:<24}{m[1]*100:>7.1f}%{m[2]:>7.2f}{m[3]*100:>7.1f}%")
    boots = _boot_vs_ref(c, b, seed=37)
    sig = (np.percentile(boots, 2.5) > 0) and (float((boots <= 0).mean()) < 0.05)
    _print_delta("EPO vs RiskParity(月频)", c, b, boots)
    print(f"\n  判定: {'✓ EPO(LW 收缩 min-var)月频显著优于 1/σ → 值得落地 EPOMulti(滚动版)' if sig else '✗ 月频不显著 → 保持 RiskParity(1/σ)'}")
    print("  ⚠ 注:N=2(CTA/国债)相关性参数仅 1 个,LW 收缩边际本就小;1/σ 已接近此 N 下的最优。")
    print("     若要更大 N 的 EPO,需扩到资产层(7+1 ETF),那是另一架构,另开诊断。")


if __name__ == "__main__":
    main()
