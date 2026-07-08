# -*- coding: utf-8 -*-
"""多策略(B 型)组合框架 —— 阶段1:统一接口 + CTA 瘦策略包装 + 等权组合器。

设计原则(前序讨论已定):
  · 每个 Strategy = 一个可命名的独立 alpha 来源(瘦策略:单信号 + 自带风控)。
  · 多 alpha 来源 = 多个 Strategy 在 MultiStrategy 层组合,而非塞进单个策略内部。
  · CTA 内部不做多因子:当前 decide_targets 已是"单信号(blended_momentum)+ 多风控
    (vol/trend/crash/drawdown)"的正确形态;选股信号已证无效(PBO=0.67),CTAStrategy
    默认 hold_all=True 剥离它,只留有效的时序风控部分。

阶段1 范围:仅 CTA 一个策略,接口闭环 + K=1 口径守恒。资金分配/相关性/组合 PBO
留给阶段2(等有第二个策略才有意义)。
"""
import numpy as np
from typing import Protocol

from cta import decide_targets, POOL, DEFENSE


class Strategy(Protocol):
    """统一策略接口。所有子策略(及组合器自身)实现它,backtest 据此可插拔决策来源。

    target 返回目标权重 {code: weight},权重和≈1(含该策略自己的防守/现金)。
    recent = {code: 收盘价序列(list,时间升序)} 由 backtest 在每个调仓日准备好喂入,
    与 decide_targets 完全同口径(故 CTAStrategy 可零摩擦复用 decide_targets)。
    """
    name: str
    universe: list          # 该策略可能输出的全部 code(含防守资产);backtest 据此准备 recent

    def target(self, recent: dict, t) -> dict:
        """给定截至 t 的收盘价面板,返回目标权重 {code: weight}。"""
        ...


class CTAStrategy:
    """CTA 瘦策略:复用 decide_targets(单信号 + 风控),不重写逻辑、口径不变。

    hold_all 默认 True —— 剥离已证无效的横截面选股(PBO=0.67),只保留有效的时序风控
    (vol_target / 趋势过滤 / 崩溃保护 / 回撤控制)。需要回退到部署版选股行为时传 hold_all=False。
    decide_kwargs 透传给 decide_targets:默认值由 decide_targets 自己定(唯一真相源),
    避免在此重复 20 个参数默认值造成两处漂移。这也保证 K=1 守恒——只要传入的 decide_kwargs
    与 backtest 直连路径的参数一致,两条路径产出的 target 必然逐点相同。
    """
    def __init__(self, name="CTA(等权+风控)", hold_all=True, **decide_kwargs):
        self.name = name
        self.hold_all = hold_all
        self.decide_kwargs = decide_kwargs
        # 可能输出的 code:持仓候选(POOL)+ 防守资产(风控挪仓目标)。backtest 据此准备 recent。
        # 注:decide_targets 内部仅遍历模块级 POOL、不读 defense 价格,故 recent 多含一个 defense
        #     key 不影响其输出 —— 这是 K=1 守恒成立的关键(decision 与 execution 完全可分离)。
        self.universe = list(POOL) + [DEFENSE[0]]

    def target(self, recent, t):
        tg, _picks = decide_targets(recent, hold_all=self.hold_all, **self.decide_kwargs)
        return tg


class BondMomentumStrategy:
    """国债时序动量瘦策略:单一 alpha 源 = 国债趋势(做多处于上涨趋势的国债)。

    diag_bonds 实测(2026-07-08):夏普 1.5、回撤 -4.4%,与 CTA 相关性 -0.06~-0.11
    (股债跷跷板,熊市 -0.10~-0.15),50/50 组合夏普 1.29 > 单 CTA 1.01 —— 是三候选
    (均值回归 / 配对 / 国债)里唯一通过"互补 + 自身 + 组合增益"三道关的第二策略。

    target 逻辑(对齐 diag_bonds.bond_momentum;月末调仓由 backtest 的 rebal_days 决定、T+1 成交):
      截至 t 的国债混合动量(21/63/126 日涨幅均值)>0 且(若开趋势)价>MA200 → 满仓国债;
      否则空仓(返回 {bond:0.0},显式 0 触发清仓;⚠ 空 dict {} 会被 backtest 当无操作、维持旧仓)。
      空仓即降低久期敞口、持现金——单标的时序策略的标准避险形态,
      与 CTA"挪到防守资产"不同(国债策略本身就在债里,避险 = 降仓,不切换资产)。

    universe=[bond]:只可能输出国债一个 code。backtest 据此只喂国债价格(轻量)。
    """
    def __init__(self, name="国债时序动量", bond=DEFENSE[0], with_trend=True,
                 lookbacks=(21, 63, 126)):
        self.name = name
        self.bond = bond
        self.with_trend = with_trend
        self.lookbacks = tuple(lookbacks)
        self.universe = [bond]

    def target(self, recent, t):
        p = recent.get(self.bond, [])
        # 历史下限:动量需 max(lookbacks)+1 个点;趋势过滤需 200 日 MA
        need = 201 if self.with_trend else (max(self.lookbacks) + 1)
        # ⚠ 空仓一律返回 {bond:0.0}(非空 dict),绝不能返回 {} —— backtest 的 `if target:`
        # 把空 dict 当"无操作"(不设 pending、不调仓),返回 {} 会维持旧仓、清不掉,策略退化成
        # 买入持有。显式 0.0 经 _apply_target_with_limits 正确清仓到现金。
        if len(p) < need:
            return {self.bond: 0.0}
        last = p[-1]
        mom = sum(last / p[-1 - lb] - 1 for lb in self.lookbacks) / len(self.lookbacks)
        if mom <= 0:
            return {self.bond: 0.0}
        if self.with_trend and last <= sum(p[-200:]) / 200.0:
            return {self.bond: 0.0}
        return {self.bond: 1.0}


class MultiStrategy:
    """等权组合器(阶段1)。allocs=None → 每策略 1/K;传入则归一化到和为 1。

    重叠标的 → 加权叠加(多策略常态:两个策略都要配同一只,权重相加)。
    MultiStrategy 自身满足 Strategy 接口(组合即策略),可被同一 backtest 跑、可嵌套。
    阶段2 将在此扩展资金分配(risk-parity-across-strategies)与策略间相关性诊断。
    """
    def __init__(self, strategies, allocs=None):
        n = len(strategies)
        if n == 0:
            raise ValueError("MultiStrategy 至少需要一个策略")
        self.strategies = strategies
        a = allocs if allocs is not None else [1 / n] * n
        if len(a) != n:
            raise ValueError(f"allocs 长度 {len(a)} != 策略数 {n}")
        s = sum(a)
        if s <= 0:
            raise ValueError("allocs 之和必须 > 0")
        self.allocs = [x / s for x in a]

    @property
    def name(self):
        return "Multi(" + "+".join(s.name for s in self.strategies) + ")"

    @property
    def universe(self):
        # 总资产域 = 所有子策略 universe 的并集;backtest 据此准备 recent 覆盖全部子策略所需。
        return sorted({c for s in self.strategies for c in s.universe})

    def target(self, recent, t):
        out = {}
        for s, a in zip(self.strategies, self.allocs):
            for c, w in s.target(recent, t).items():
                out[c] = out.get(c, 0.0) + a * w
        return out


class RiskParityMulti:
    """滚动 risk-parity 组合器(阶段2 资金分配)。每月调仓时,用各子策略近 VOL_WINDOW 日
    已实现波动算权重 w_i ∝ 1/σ_i(波动反比,各策略对组合风险贡献趋均衡)。

    diag_riskparity 实测(2026-07-08):risk-parity 组合夏普 1.91 vs 等权 1.29,配对 block
    bootstrap Δ=+0.60、P=0.001(高度显著);滚动(无前视)版 1.91 ≈ 静态 1.89(波动比稳定),
    证明提升非前视幻觉。权重稳定在 CTA~15% / 国债~85%。

    实现:有状态(_prev_targets)。每次 target 用 recent 末尾 VOL_WINDOW 天 + 上期各子策略
    target,算各子策略「按上期持仓」的近期日收益序列 → σ → risk-parity 权重。用末尾对齐
    (取所有相关 code 的末尾共同长度)规避 recent 各 code 长度不等(dropna)的错位。历史不足
    VOL_WINDOW/2 或首次(无上期)退等权。w_i 经 min_weight 钳制、再归一。

    满足 Strategy 接口(组合即策略);universe = 子策略并集。"""
    def __init__(self, strategies, vol_window=63, min_weight=0.05):
        n = len(strategies)
        if n == 0:
            raise ValueError("RiskParityMulti 至少需要一个策略")
        self.strategies = strategies
        self.vol_window = vol_window
        self.min_weight = min_weight
        self.name = "RiskParity(" + "+".join(s.name for s in strategies) + ")"
        self.universe = sorted({c for s in strategies for c in s.universe})
        self._prev_targets = None          # 上期各子策略 target(算子策略近期收益用)

    def target(self, recent, t):
        sub = [s.target(recent, t) for s in self.strategies]
        n = len(self.strategies)
        if self._prev_targets is None:
            allocs = [1.0 / n] * n                       # 首次无上期 → 等权
        else:
            codes = set()
            for pt in self._prev_targets:
                codes |= set(pt.keys())
            lens = [len(recent[c]) for c in codes if c in recent]
            if not lens:
                allocs = [1.0 / n] * n
            else:
                L = min(min(lens) - 1, self.vol_window)   # 末尾共同长度(L 个日收益)
                sigmas = []
                for ci in range(n):
                    pt = self._prev_targets[ci]
                    strat_ret = []
                    for c, w in pt.items():
                        arr = recent.get(c)
                        if arr is None or len(arr) <= L:
                            continue
                        tail = arr[-L - 1:]               # L+1 价 → L 个日收益
                        strat_ret.append([w * (tail[k] / tail[k - 1] - 1.0) for k in range(1, L + 1)])
                    if strat_ret:
                        sret = [sum(col) for col in zip(*strat_ret)]   # 各 code 同日求和(末尾对齐)
                        sigmas.append(float(np.std(sret, ddof=1)) * np.sqrt(252)
                                      if len(sret) >= self.vol_window // 2 else None)
                    else:
                        sigmas.append(None)
                if any(s is None or s < 1e-9 for s in sigmas):
                    allocs = [1.0 / n] * n
                else:
                    inv = [1.0 / s for s in sigmas]
                    tot = sum(inv)
                    allocs = [max(i / tot, self.min_weight) for i in inv]
                    s2 = sum(allocs)
                    allocs = [a / s2 for a in allocs]
        self._prev_targets = sub
        out = {}
        for st, a in zip(sub, allocs):
            for c, w in st.items():
                out[c] = out.get(c, 0.0) + a * w
        return out
