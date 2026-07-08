# -*- coding: utf-8 -*-
"""单元测试（改进项 H5）：固化关键逻辑的回归保护。
python test_momentum.py 跑全部；pytest 兼容（test_ 函数）。
覆盖：G1(norm/DSR)、F4(CAPM)、F1(Sortino 钳制)、C4(VaR/CVaR)、H1(reconcile)、D5(整手四舍五入)。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import momentum_core as mc
import etf_momentum as e
import etf_momentum_live as L


def test_norm_roundtrip():
    """G1: _norm_ppf(_norm_cdf(x)) ≈ x（标准正态分位往返）。"""
    for x in [-3.0, -1.5, 0.0, 1.5, 3.0]:
        assert abs(e._norm_ppf(e._norm_cdf(x)) - x) < 1e-6


def test_factor_attribution_capm():
    """F4: 合成 CAPM 数据回收 beta/alpha（n=5000 压采样方差）。"""
    rng = np.random.default_rng(42)
    n = 5000
    b = rng.normal(0.0003, 0.01, n)
    p = 0.0004 + 0.5 * b + rng.normal(0, 0.005, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    a = e.factor_attribution(pd.Series(p, index=idx), pd.Series(b, index=idx))
    assert abs(a["beta"] - 0.5) < 0.03
    assert abs(a["年化alpha"] - 0.0004 * 252) < 0.04


def test_factor_attribution_multi_recovery():
    """F8: 合成 6 因子数据回收 beta/alpha（n=5000）。px 价格面板列名用真实代码，
    因子由 _MFACTORS 默认 spec 构造（MKT/SMB/VMG/BND/GLD/NSDQ）。"""
    rng = np.random.default_rng(42)
    n = 5000
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # 6 个原始代码日收益（独立、各异漂移），构造价格面板
    rets = {c: rng.normal(m, 0.01, n) for c, m in
            [("510300", 0.0003), ("512100", 0.0004), ("510880", 0.00035),
             ("511010", 0.0001), ("518880", 0.0002), ("513100", 0.0005)]}
    px = pd.DataFrame({c: np.cumprod(1 + r) for c, r in rets.items()}, index=idx)
    # 策略：真日 alpha 0.0003 + 已知 beta 暴露（MKT 0.5 / BND 0.2 / GLD 0.1，其余 0）
    y = (0.0003 + 0.5 * rets["510300"] + 0.2 * rets["511010"]
         + 0.1 * rets["518880"] + rng.normal(0, 0.005, n))
    a = e.factor_attribution_multi(pd.Series(y, index=idx), px)
    assert a is not None
    assert abs(a["betas"]["MKT"] - 0.5) < 0.03
    assert abs(a["betas"]["BND"] - 0.2) < 0.03
    assert abs(a["betas"]["GLD"] - 0.1) < 0.03
    assert abs(a["alpha_ann"] - 0.0003 * 252) < 0.04


def test_mfat_strips_spurious_alpha():
    """F8 核心论点：真 alpha=0 但有债券/黄金 beta → CAPM 算出假阳性 alpha，多因子剥回近 0。
    证明「多因子能剥离 CAPM 误判为 alpha 的非权益 beta」——本条改进的存在理由。
    市场均值压低（驱动 CAPM 假阳性的只有债/金漂移），消减截距 SE 的均值杠杆。"""
    rng = np.random.default_rng(7)
    n = 5000
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rm = rng.normal(0.00010, 0.011, n)      # 市场：低均值（截距 SE 不被均值杠杆放大）
    rb = rng.normal(0.00030, 0.002, n)      # 国债：正漂移 ~7.5%/yr（驱动假 CAPM α）
    rg = rng.normal(0.00040, 0.007, n)      # 黄金：正漂移 ~10%/yr（驱动假 CAPM α）
    rnoise = {c: rng.normal(0, 0.01, n) for c in ["512100", "510880", "513100"]}
    px = pd.DataFrame({c: np.cumprod(1 + r) for c, r in
                       {"510300": rm, "511010": rb, "518880": rg, **rnoise}.items()}, index=idx)
    y = 0.3 * rm + 0.4 * rb + 0.2 * rg + rng.normal(0, 0.002, n)   # 真 alpha=0
    daily = pd.Series(y, index=idx)
    # ① CAPM（只对市场回归）→ alpha 显著 > 0（债/金正漂移被误读成 alpha）
    capm = e.factor_attribution(daily, pd.Series(rm, index=idx))
    assert capm["年化alpha"] > 0.025
    # ② 多因子 → alpha 近 0（债/金暴露被 BND/GLD 因子 beta 吸收）；并明确剥离了 ≥1pp 假 α
    mf = e.factor_attribution_multi(daily, px)
    assert abs(mf["alpha_ann"]) < 0.02
    assert mf["alpha_ann"] < capm["年化alpha"] - 0.01
    assert abs(mf["betas"]["BND"] - 0.4) < 0.03
    assert abs(mf["betas"]["GLD"] - 0.2) < 0.03


def test_sortino_clamp():
    """F1: 全正收益 → 下行偏差=0 → Sortino 钳极大。"""
    pos = pd.Series([0.001] * 200, index=pd.date_range("2020-01-01", periods=200, freq="B"))
    nav = (1 + pos).cumprod()
    p = e.perf(nav, pos)
    assert p["Sortino"] > 100  # 下行偏差 0 被钳制，Sortino 极大


def test_var_cvar():
    """C4: 合成正态日收益，VaR/CVaR 满足 CVaR<VaR<0、99% 甚于 95%。"""
    rng = np.random.default_rng(7)
    daily = pd.Series(rng.normal(0.0005, 0.01, 2000),
                      index=pd.date_range("2020-01-01", periods=2000, freq="B"))
    nav = (1 + daily).cumprod()
    p = e.perf(nav, daily)
    assert p["CVaR95"] < p["VaR95"] < 0
    assert p["VaR99"] < p["VaR95"]


def test_reconcile():
    """H1: 一致→小偏离；偏离→告警。"""
    price = {"510300": 4.0, "511010": 100.0}
    target = {"510300": 0.33, "511010": 0.66}
    m_ok = L.reconcile(target, {"510300": 8250, "511010": 660}, 100000.0, price)
    assert m_ok < 0.05
    m_bad = L.reconcile(target, {"510300": 5000, "511010": 800}, 100000.0, price)
    assert m_bad > 0.10


def test_load_hold_all():
    """J2 落地:live_config.json 的 hold_all 覆盖默认。临时写配置文件、跑完删,验证三级回退。"""
    import json, tempfile, crossmarket as cm  # noqa: F401 (cm 仅触发其 import 链)
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_config.json")
    existed = os.path.exists(cfg)
    saved = open(cfg).read() if existed else None
    try:
        for val, want in ((True, True), (False, False)):
            with open(cfg, "w") as f:
                json.dump({"hold_all": val}, f)
            assert L._load_hold_all() is want
        # 无 hold_all 键 → 回退默认 HOLD_ALL(False)
        with open(cfg, "w") as f:
            json.dump({}, f)
        assert L._load_hold_all() is False
    finally:
        if existed:
            open(cfg, "w").write(saved)
        elif os.path.exists(cfg):
            os.remove(cfg)


def test_build_orders_rounding():
    """D5: 整手四舍五入（12.5 lot → 1300，非向下 1200）。"""
    recent = {"510300": [4.0] * 10}
    orders = L.build_orders({"510300": 0.5}, recent, 10000.0, {})
    assert any(c == "510300" and s == "BUY" and sh == 1300 for c, s, sh in orders)


def test_limit_pct():
    """D2: 涨跌停档——显式表(创业板159915=20%)、科创板前缀(588=20%)、跨境/商品/主板默认10%。
    关键回归：159941 纳指跨境虽 159 开头但不是创业板 → 必须 10%（防未来误加 159 前缀把跨境判成20%）。"""
    assert mc._limit("159915") == 0.20   # 创业板（显式表）
    assert mc._limit("588000") == 0.20   # 科创板（588 前缀兜底）
    assert mc._limit("588050") == 0.20   # 科创板
    assert mc._limit("510300") == 0.10   # 主板沪深300
    assert mc._limit("513100") == 0.10   # 跨境 QDII 纳指
    assert mc._limit("159941") == 0.10   # 159 段跨境纳指（防前缀误判为 20%）
    assert mc._limit("518880") == 0.10   # 商品黄金
    assert mc._limit("511010") == 0.10   # 国债


def test_hold_all_equal_weight():
    """J2: hold_all=True → 等权持有全部有历史的候选(不选 top_n、不切防守)；
    False → 仅 top_n。关闭风控叠加(vol_target/trend)以纯测选股 vs 全池的权重结构。
    合成 7 只单调上涨、斜率递增的收盘 → 动量全正、排序确定。"""
    n = 200
    # 斜率随 enumerate 递增：动量全为正、且后入池的标的动量更高（top_n 选股可确定性判定）
    base = {c: np.linspace(1.0, 1.0 + i * 0.3, n).tolist() for i, c in enumerate(mc.POOL)}
    # hold_all：全部 7 只、等权、无防守资产
    t_hold, _ = mc.decide_targets(base, vol_target=None, hold_all=True)
    assert set(t_hold) == set(mc.POOL)                  # 全池，未切防守(511010)
    ws = list(t_hold.values())
    assert max(ws) - min(ws) < 1e-9                     # 等权
    assert abs(sum(ws) - mc.CASH_BUFFER) < 1e-9
    # 选股：动量全正 → 仅持 top_n、无防守，且是动量最高的那 top_n 只
    t_sel, p_sel = mc.decide_targets(base, vol_target=None, hold_all=False)
    top = set(sorted(mc.POOL, key=lambda c: base[c][-1] / base[c][0], reverse=True)[:mc.TOP_N])
    assert set(t_sel) == top
    assert len(p_sel) == mc.TOP_N


def test_crossmarket_globals_swap():
    """J1: _us_globals 临时换美股池跑通回测,退出后恢复 A 股池。合成价格验证整条跨市场链路。
    关键回归:trend_code 必须显式传(默认在 def 时绑定沪深300,改全局无效);退出后池子须原样还原。"""
    import crossmarket as cm
    rng = np.random.default_rng(11)
    cols = list(cm.POOL_US) + [cm.DEFENSE_US[0]]
    idx = pd.date_range("2010-01-01", periods=1500, freq="B")
    # 各标的随机游走 + 不同漂移,给动量排序提供信号;波动 0.011 保夏普有限
    px = pd.DataFrame(
        {c: 100 * np.cumprod(1 + rng.normal(0.0004 + i * 0.0001, 0.011, len(idx)))
         for i, c in enumerate(cols)}, index=idx)
    before = dict(mc.POOL)
    with cm._us_globals():
        assert set(mc.POOL) == set(cm.POOL_US)                  # 进入→美股池
        nav, ret, _, _ = e.backtest(px, hold_all=False, trend_ma=200, trend_code=cm.BENCH_US)
        assert len(nav) > 1000
        assert np.isfinite(e.perf(nav, ret)["夏普"])
    assert mc.POOL == before                                     # 退出→A 股池原样还原
    cm.run_crossmarket(px)                                       # 整条链路(含 bootstrap)不报错


def test_apply_target_with_limits_limit_up():
    """D2 回测可信度核心:目标要新建仓(w>old)但成交日封涨停 → 买不进,维持旧仓。
    合成 cant_buy 掩码命中,验证 _apply_target_with_limits 跳过加仓;对照不封板则正常落地。"""
    cols = ["510300", "511010"]
    cur = pd.Series([0.0, 0.0], index=cols)              # 旧仓空
    tgt = {"510300": 0.33}                               # 目标:新建沪深300
    fill_day = pd.Timestamp("2024-01-31")
    cb = pd.DataFrame(False, index=[fill_day], columns=cols)
    cb.loc[fill_day, "510300"] = True                    # 当日沪深300 封涨停 → 买不进
    cs = pd.DataFrame(False, index=[fill_day], columns=cols)
    new = e._apply_target_with_limits(tgt, cur, fill_day, cb, cs)
    assert new["510300"] == 0.0                          # 买不进 → 维持旧仓 0
    new2 = e._apply_target_with_limits(tgt, cur, fill_day,
                                       pd.DataFrame(False, index=[fill_day], columns=cols), cs)
    assert abs(new2["510300"] - 0.33) < 1e-9             # 不封板 → 正常落地


def test_apply_target_with_limits_limit_down():
    """D2:旧仓要清掉但成交日封跌停 → 卖不出,维持旧仓;对照不封板则清掉。"""
    cols = ["510300", "511010"]
    cur = pd.Series([0.33, 0.0], index=cols)             # 旧仓持有沪深300
    tgt = {}                                            # 目标:清仓(不在 target)
    fill_day = pd.Timestamp("2024-01-31")
    cb = pd.DataFrame(False, index=[fill_day], columns=cols)
    cs = pd.DataFrame(False, index=[fill_day], columns=cols)
    cs.loc[fill_day, "510300"] = True                    # 当日沪深300 封跌停 → 卖不出
    new = e._apply_target_with_limits(tgt, cur, fill_day, cb, cs)
    assert abs(new["510300"] - 0.33) < 1e-9              # 卖不出 → 维持旧仓
    new2 = e._apply_target_with_limits(tgt, cur, fill_day, cb,
                                       pd.DataFrame(False, index=[fill_day], columns=cols))
    assert new2["510300"] == 0.0                         # 不封板 → 清掉


def test_limit_masks():
    """D2:合成未复权价,验证 cant_buy/cant_sell 掩码——触及涨停价→cant_buy,触及跌停价→cant_sell。"""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    raw = pd.DataFrame({"510300": [10.0, 11.0, 9.9]}, index=idx)   # day1 +10% 涨停,day2 -10% 跌停
    pre = pd.DataFrame({"510300": [10.0, 10.0, 11.0]}, index=idx)  # pre_close(正常日=昨收)
    cb, cs = e.limit_masks(raw, pre)
    assert bool(cb.loc[idx[1], "510300"]) is True        # day1: close=11≥round(10*1.1,2)=11 → 涨停
    assert bool(cs.loc[idx[1], "510300"]) is False
    assert bool(cs.loc[idx[2], "510300"]) is True        # day2: close=9.9≤round(11*0.9,2)=9.9 → 跌停
    assert bool(cb.loc[idx[2], "510300"]) is False
    assert bool(cb.loc[idx[0], "510300"]) is False       # day0: 平盘不封板
    assert bool(cs.loc[idx[0], "510300"]) is False


def test_qfq_from_pre_close():
    """E3 复权:除权日 pre_close≠昨收 → 前复权把历史价格按因子拉回,消除跳变;最新价不变。
    合成 5 日,day2 除权(昨收 10→今日 pre_close=8、close=8,0.8 因子)。"""
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    close = pd.Series([10.0, 10.0, 8.0, 8.4, 8.82], index=idx)    # day2 除权后正常涨
    pre = pd.Series([10.0, 10.0, 8.0, 8.0, 8.4], index=idx)       # day2 pre_close=8(≠昨收 10)=除权基准
    adj = e._qfq_from_pre_close(close, pre)
    assert abs(adj.iloc[0] - 8.0) < 1e-6                 # day0~1 被回溯因子 0.8 拉回:10*0.8=8
    assert abs(adj.iloc[1] - 8.0) < 1e-6
    assert abs(adj.iloc[-1] - close.iloc[-1]) < 1e-9     # 最新价 = close 末值不变
    assert abs(adj.iloc[1] - adj.iloc[2]) < 1e-9         # 除权日无跳变(adj[1]=8, adj[2]=8)


def test_multi_k1_equivariance():
    """阶段1 不变量:单策略等权组合(K=1)经 strategy 路径的 NAV ==
    decide_targets 直连路径的 NAV。证明重构只换壳不换行为。
    合成价格(7 POOL + 国债, 600 营业日含月末调仓), 同配置两路径 NAV 逐点相等。
    另验 K=2 叠加自身(两个相同 CTA 等权)== 单 CTA(权重各自减半, 合并=原)。"""
    from multi_strategy import CTAStrategy, MultiStrategy
    rng = np.random.default_rng(3)
    cols = list(mc.POOL) + [mc.DEFENSE[0]]
    idx = pd.date_range("2018-01-01", periods=600, freq="B")
    px = pd.DataFrame({c: 100 * np.cumprod(1 + rng.normal(0.0003 + i * 0.0001, 0.011, len(idx)))
                       for i, c in enumerate(cols)}, index=idx)
    cfg = dict(vol_target=mc.VOL_TARGET, trend_ma=mc.TREND_MA, hold_all=True)
    nav0, ret0, _, _ = e.backtest(px, **cfg)                       # 直连路径(decide_targets)
    multi = MultiStrategy([CTAStrategy(**cfg)])
    nav1, ret1, _, _ = e.backtest(px, strategy=multi)              # 策略路径(K=1)
    assert np.allclose(nav0.values, nav1.values, atol=1e-9)        # K=1 逐点守恒
    multi2 = MultiStrategy([CTAStrategy(**cfg), CTAStrategy(**cfg)])
    nav2, _, _, _ = e.backtest(px, strategy=multi2)                # K=2 两个相同 CTA 等权
    assert np.allclose(nav1.values, nav2.values, atol=1e-9)        # K=2 合并 == 单 CTA


def test_bond_strategy_equivariance():
    """阶段2:BondMomentumStrategy 经 backtest(strategy=) 的 NAV == 内联向量化(同口径)。
    合成国债价格(含上涨/回调段让信号切换),无涨跌停 → 逐点一致。
    口径要点:backtest 是「T 日信号、T+1 落地、新仓 T+2 起吃收益」(weights.shift(1)),
    故内联 weights=sig_ffill.shift(1)、gross=weights.shift(1)*ret,精确复现(非 diag_bonds 的 T+1)。"""
    from multi_strategy import BondMomentumStrategy
    rng = np.random.default_rng(7)
    bond = mc.DEFENSE[0]
    idx = pd.date_range("2018-01-01", periods=600, freq="B")
    drift = np.concatenate([np.full(300, 0.0002), np.full(150, -0.0004), np.full(150, 0.0002)])
    px = pd.DataFrame({bond: 100 * np.cumprod(1 + drift + rng.normal(0, 0.002, len(idx)))}, index=idx)
    nav_s, _, _, _ = e.backtest(px, strategy=BondMomentumStrategy())   # 策略路径
    # 内联向量化(精确对齐 backtest:月末采样、T+1 落地、T+2 吃收益、扣换手成本)
    p = px[bond]
    month_ends = px.resample("ME").last().index
    rebal = [px.index[px.index <= me][-1] for me in month_ends if (px.index <= me).any()]
    rebal = [d for d in rebal if d >= px.index[mc.MAX_LOOKBACK]]
    rd_set = set(rebal)
    mom = (p / p.shift(21) - 1 + p / p.shift(63) - 1 + p / p.shift(126) - 1) / 3
    sig = ((mom > 0) & (p > p.rolling(200).mean())).astype(float)
    weights_v = sig.where(sig.index.isin(rd_set)).ffill().shift(1).fillna(0.0)   # = backtest weights
    w_lag = weights_v.shift(1).fillna(0.0)
    ret = p.pct_change().fillna(0.0)
    gross = w_lag * ret
    turnover = (weights_v - weights_v.shift(1).fillna(0.0)).abs()
    net = (gross - turnover * (e.COMMISSION + e.SLIPPAGE)).fillna(0.0)
    nav_v = (1 + net).cumprod().loc[px.index[mc.MAX_LOOKBACK]:]
    nav_v = nav_v / nav_v.iloc[0]
    assert np.allclose(nav_s.values, nav_v.values, atol=1e-9), \
        f"Bond 策略路径与向量化不一致,max diff={abs(nav_s.values - nav_v.values).max():.2e}"


def test_multi_k2_diversification():
    """阶段2:K=2 组合(CTA + 国债时序动量)跑通,且分散有效——组合年化波动 < 单 CTA
    (不同 beta 叠加降低总风险)。合成价格(POOL + 低波国债,足够长含月末调仓)。"""
    from multi_strategy import CTAStrategy, BondMomentumStrategy, MultiStrategy
    rng = np.random.default_rng(11)
    cols = list(mc.POOL) + [mc.DEFENSE[0]]
    idx = pd.date_range("2016-01-01", periods=700, freq="B")
    data = {}
    for c in cols:
        if c == mc.DEFENSE[0]:
            data[c] = 100 * np.cumprod(1 + rng.normal(0.0001, 0.003, len(idx)))     # 国债低波
        else:
            data[c] = 100 * np.cumprod(1 + rng.normal(0.0002, 0.012, len(idx)))     # 权益高波
    px = pd.DataFrame(data, index=idx)
    cfg = dict(vol_target=mc.VOL_TARGET, trend_ma=mc.TREND_MA, hold_all=True)
    cta, bond = CTAStrategy(**cfg), BondMomentumStrategy()
    _, ret_c, _, _ = e.backtest(px, strategy=cta)
    _, ret_k2, _, _ = e.backtest(px, strategy=MultiStrategy([cta, bond], allocs=[0.5, 0.5]))
    vol_c = ret_c.std() * np.sqrt(252)
    vol_k2 = ret_k2.std() * np.sqrt(252)
    assert vol_k2 < vol_c, f"K=2 波动 {vol_k2:.4f} 未低于单 CTA {vol_c:.4f},分散失效"


def test_riskparity_k1_equivariance():
    """阶段2:RiskParityMulti 单策略(K=1)经 backtest 的 NAV == CTAStrategy 直连。
    单策略 risk-parity 权重恒为 1(1/σ 归一),故退化成纯 CTA——证明 RiskParityMulti 不破坏 K=1 守恒。"""
    from multi_strategy import CTAStrategy, RiskParityMulti
    rng = np.random.default_rng(3)
    cols = list(mc.POOL) + [mc.DEFENSE[0]]
    idx = pd.date_range("2018-01-01", periods=600, freq="B")
    px = pd.DataFrame({c: 100 * np.cumprod(1 + rng.normal(0.0003 + i * 0.0001, 0.011, len(idx)))
                       for i, c in enumerate(cols)}, index=idx)
    cfg = dict(vol_target=mc.VOL_TARGET, trend_ma=mc.TREND_MA, hold_all=True)
    nav0, _, _, _ = e.backtest(px, strategy=CTAStrategy(**cfg))
    nav1, _, _, _ = e.backtest(px, strategy=RiskParityMulti([CTAStrategy(**cfg)]))
    assert np.allclose(nav0.values, nav1.values, atol=1e-9), \
        f"RiskParity K=1 不守恒,max diff={abs(nav0.values - nav1.values).max():.2e}"


def test_riskparity_k2_diversification():
    """阶段2:RiskParityMulti K=2(CTA+国债)跑通,组合年化波动 < 单 CTA
    (risk-parity 给低波国债高权重,降低组合总风险)。"""
    from multi_strategy import CTAStrategy, BondMomentumStrategy, RiskParityMulti
    rng = np.random.default_rng(11)
    cols = list(mc.POOL) + [mc.DEFENSE[0]]
    idx = pd.date_range("2016-01-01", periods=700, freq="B")
    data = {}
    for c in cols:
        if c == mc.DEFENSE[0]:
            data[c] = 100 * np.cumprod(1 + rng.normal(0.0001, 0.003, len(idx)))
        else:
            data[c] = 100 * np.cumprod(1 + rng.normal(0.0002, 0.012, len(idx)))
    px = pd.DataFrame(data, index=idx)
    cfg = dict(vol_target=mc.VOL_TARGET, trend_ma=mc.TREND_MA, hold_all=True)
    cta, bond = CTAStrategy(**cfg), BondMomentumStrategy()
    _, ret_c, _, _ = e.backtest(px, strategy=cta)
    _, ret_rp, _, _ = e.backtest(px, strategy=RiskParityMulti([cta, bond]))
    assert ret_rp.std() * np.sqrt(252) < ret_c.std() * np.sqrt(252), "risk-parity 组合波动未低于单 CTA"


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    npass = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            npass += 1
        except AssertionError as ex:
            print(f"  FAIL  {t.__name__}: {ex}")
    print(f"\n{npass}/{len(_TESTS)} passed")
    sys.exit(0 if npass == len(_TESTS) else 1)
