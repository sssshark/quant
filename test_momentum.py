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


def test_build_orders_rounding():
    """D5: 整手四舍五入（12.5 lot → 1300，非向下 1200）。"""
    recent = {"510300": [4.0] * 10}
    orders = L.build_orders({"510300": 0.5}, recent, 10000.0, {})
    assert any(c == "510300" and s == "BUY" and sh == 1300 for c, s, sh in orders)


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
