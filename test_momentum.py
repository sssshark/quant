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
