# -*- coding: utf-8 -*-
"""诊断:risk-parity(风险平价)vs 等权 资金分配 + block bootstrap 显著性。

risk-parity:权重 ∝ 1/σ_i(波动反比),让每个策略对组合总风险的贡献相等。国债低波
(σ≈2.3%)、CTA 高波(σ≈11.7%)→ risk-parity 自动给国债高权重(~84%),落在 allocs 扫描的
高夏普区(75% 国债夏普 1.69)。

⚠ 静态权重(全样本 σ)有前视——用整个回测期的波动算权重。作为"risk-parity 概念是否有效"
的诊断可接受;若显著有效,再考虑动态滚动版(无前视,但引入波动窗口参数)。risk-parity 是
**先验规则(波动反配,非收益优化)**,故 bootstrap 的 p 值无需多重比较打折(不像"挑最优窗口")。

方法:CTA / 国债 单策略日收益(common 区间)线性组合:
  等权      ret_eq = 0.5·ret_cta + 0.5·ret_bond
  风险平价  ret_rp = w_cta·ret_cta + w_bond·ret_bond,  w_i ∝ 1/σ_i(全样本年化波动)
配对 circular block bootstrap(block=21≈1月,n=2000):Δ=SR(rp)−SR(eq),p=P(Δ≤0)。
"""
import numpy as np
import pandas as pd

import momentum_core as mc
import etf_momentum as e
from multi_strategy import CTAStrategy, BondMomentumStrategy

N_BOOT = 2000
BLOCK = 21


def main():
    print("加载数据...")
    res = e.load_real(with_limits=True)
    px = res[0]
    cb, cs = (res[1], res[2]) if isinstance(res, tuple) else (None, None)

    print("跑 CTA / 国债 单策略回测(取日收益)...")
    cfg = dict(vol_target=mc.VOL_TARGET, trend_ma=mc.TREND_MA, hold_all=True)
    cta = CTAStrategy(**cfg)
    bond = BondMomentumStrategy()
    _, ret_cta, _, _ = e.backtest(px, strategy=cta, cant_buy=cb, cant_sell=cs)
    _, ret_bond, _, _ = e.backtest(px, strategy=bond, cant_buy=cb, cant_sell=cs)

    common = ret_cta.index.intersection(ret_bond.index)
    rc = ret_cta.loc[common].to_numpy(dtype=float)
    rb = ret_bond.loc[common].to_numpy(dtype=float)

    # 全样本年化波动 → 静态 risk-parity 权重(有前视,标注)
    sig_c = rc.std() * np.sqrt(252)
    sig_b = rb.std() * np.sqrt(252)
    inv_c, inv_b = 1.0 / sig_c, 1.0 / sig_b
    w_c = inv_c / (inv_c + inv_b)
    w_b = inv_b / (inv_c + inv_b)
    print(f"\n全样本年化波动: CTA {sig_c*100:.1f}%   国债 {sig_b*100:.1f}%")
    print(f"risk-parity 权重: CTA {w_c*100:5.1f}%   国债 {w_b*100:5.1f}%   (波动反比)")
    print(f"等权权重:         CTA  50.0%   国债  50.0%")

    ret_eq = 0.5 * rc + 0.5 * rb
    ret_rp = w_c * rc + w_b * rb

    def metrics(r):
        nav = pd.Series(np.cumprod(1 + r), index=common)
        nav = nav / nav.iloc[0]
        return e.perf(nav, pd.Series(r, index=common))

    print(f"\n{'组合':<24}{'年化':>8}{'夏普':>7}{'回撤':>8}{'Sortino':>9}")
    for name, r in [("等权 50/50", ret_eq), ("风险平价(RP)", ret_rp)]:
        m = metrics(r)
        print(f"{name:<24}{m['年化']*100:>7.1f}%{m['夏普']:>7.2f}{m['回撤']*100:>7.1f}%{m['Sortino']:>9.2f}")

    # 配对 circular block bootstrap:Δ = SR(rp) − SR(eq), p = P(Δ≤0)
    print(f"\n[block bootstrap 显著性] n={N_BOOT}  block={BLOCK}≈1月  (配对重采样保留自相关)")
    T = len(rc)
    delta_obs = e._ann_sharpe(ret_rp) - e._ann_sharpe(ret_eq)
    rng = np.random.default_rng(7)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = e._circ_block_idx(T, BLOCK, rng)
        boots[b] = e._ann_sharpe(ret_rp[idx]) - e._ann_sharpe(ret_eq[idx])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    p = float((boots <= 0).mean())
    print(f"  Δ夏普(obs)       = {delta_obs:+.3f}   bootstrap 均值 = {boots.mean():+.3f}")
    print(f"  95% CI           = [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"  P(Δ≤0)          = {p:.3f}   (单边,<0.05 = risk-parity 显著优于等权)")

    sig = ci_lo > 0 and p < 0.05
    print(f"\n  判定: {'✓ 显著 —— risk-parity 优于等权,值得落地(动态滚动版)' if sig else '✗ 不显著 —— 保持等权'}")
    if not sig:
        print("  → risk-parity 未显著优于等权,保持等权(省一个'波动估计窗口'过拟合自由度,与 RISK_ADJ/A2 同理)。")
    print("\n  ⚠ 注:权重用全样本 σ(含前视);若显著,落地用滚动 σ(无前视)并再验。")

    # ===== 滚动 risk-parity(无前视):每个 t 用过去 VOL_WINDOW 日 σ 算 t-1 权重 =====
    VOL_WINDOW = 63
    print(f"\n{'='*64}")
    print(f"[滚动 risk-parity(无前视)vol_window={VOL_WINDOW} 日]")
    print(f"{'='*64}")
    s_rc = pd.Series(rc, index=common)
    s_rb = pd.Series(rb, index=common)
    sig_cr = s_rc.rolling(VOL_WINDOW).std() * np.sqrt(252)
    sig_br = s_rb.rolling(VOL_WINDOW).std() * np.sqrt(252)
    tot = sig_cr + sig_br
    w_cr = (sig_br / tot).shift(1).fillna(0.5)      # w_cta ∝ 1/σ_c → σ_b/(σ_b+σ_c);t-1 无前视
    w_br = (sig_cr / tot).shift(1).fillna(0.5)
    valid = common[VOL_WINDOW:]                       # 滚动 σ 有效区
    rcl = s_rc.loc[valid].to_numpy(float)
    rbl = s_rb.loc[valid].to_numpy(float)
    rrl = (w_cr.loc[valid] * s_rc.loc[valid] + w_br.loc[valid] * s_rb.loc[valid]).to_numpy(float)
    req = 0.5 * rcl + 0.5 * rbl

    def m2(r):
        nav = pd.Series(np.cumprod(1 + r), index=valid)
        nav = nav / nav.iloc[0]
        return e.perf(nav, pd.Series(r, index=valid))
    print(f"  滚动权重均值: CTA {w_cr.loc[valid].mean()*100:.1f}%  国债 {w_br.loc[valid].mean()*100:.1f}%")
    me, mr = m2(req), m2(rrl)
    print(f"  {'组合':<20}{'年化':>8}{'夏普':>7}{'回撤':>8}")
    print(f"  {'等权 50/50':<20}{me['年化']*100:>7.1f}%{me['夏普']:>7.2f}{me['回撤']*100:>7.1f}%")
    print(f"  {'滚动风险平价':<20}{mr['年化']*100:>7.1f}%{mr['夏普']:>7.2f}{mr['回撤']*100:>7.1f}%")

    T2 = len(rrl)
    rng2 = np.random.default_rng(11)
    boots2 = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = e._circ_block_idx(T2, BLOCK, rng2)
        boots2[b] = e._ann_sharpe(rrl[idx]) - e._ann_sharpe(req[idx])
    ci2 = np.percentile(boots2, [2.5, 97.5])
    p2 = float((boots2 <= 0).mean())
    print(f"  Δ夏普(obs)={e._ann_sharpe(rrl)-e._ann_sharpe(req):+.3f}  "
          f"95%CI=[{ci2[0]:+.3f},{ci2[1]:+.3f}]  P(Δ≤0)={p2:.3f}")
    sig2 = ci2[0] > 0 and p2 < 0.05
    print(f"  判定: {'✓ 滚动版(无前视)仍显著 → 可放心落地滚动 risk-parity' if sig2 else '△ 滚动版减弱(静态的显著部分来自前视)→ 落地需谨慎,静态权重作近似'}")

    # ===== backtest 口径(月频 RiskParityMulti):现实可部署版,对等权的显著性 =====
    print(f"\n{'='*64}")
    print(f"[backtest 口径(月频 RiskParityMulti):现实可部署版 vs 等权]")
    print(f"{'='*64}")
    from multi_strategy import MultiStrategy, RiskParityMulti
    rp_inst = RiskParityMulti([CTAStrategy(**cfg), BondMomentumStrategy()])
    eq_inst = MultiStrategy([CTAStrategy(**cfg), BondMomentumStrategy()], allocs=[0.5, 0.5])
    _, req_bt, _, _ = e.backtest(px, strategy=eq_inst, cant_buy=cb, cant_sell=cs)
    _, rrp_bt, _, _ = e.backtest(px, strategy=rp_inst, cant_buy=cb, cant_sell=cs)
    cb3 = req_bt.index.intersection(rrp_bt.index)
    a = req_bt.loc[cb3].to_numpy(float)
    b = rrp_bt.loc[cb3].to_numpy(float)
    T3 = len(a)
    print(f"  等权(月频 backtest) SR={e._ann_sharpe(a):.3f}   risk-parity(月频) SR={e._ann_sharpe(b):.3f}")
    rng3 = np.random.default_rng(13)
    boots3 = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = e._circ_block_idx(T3, BLOCK, rng3)
        boots3[i] = e._ann_sharpe(b[idx]) - e._ann_sharpe(a[idx])
    ci3 = np.percentile(boots3, [2.5, 97.5])
    p3 = float((boots3 <= 0).mean())
    print(f"  Δ夏普(obs)={e._ann_sharpe(b)-e._ann_sharpe(a):+.3f}  95%CI=[{ci3[0]:+.3f},{ci3[1]:+.3f}]  P(Δ≤0)={p3:.3f}")
    sig3 = ci3[0] > 0 and p3 < 0.05
    print(f"  判定: {'✓ 月频仍显著 → 落地 risk-parity' if sig3 else '△ 月频不显著(日频诊断显著但月频再平衡滞后吃掉优势)→ 保持等权;概念在更高频调仓下才显现'}")
    print("  注:日频诊断(上)是再平衡上界;月频(项目调仓频率)是现实值。risk-parity 的价值依赖再平衡频率。")


if __name__ == "__main__":
    main()
