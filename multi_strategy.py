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
from typing import Protocol

from momentum_core import decide_targets, POOL, DEFENSE


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
