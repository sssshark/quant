# -*- coding: utf-8 -*-
"""
A股 ETF 动量轮动回测 —— Backtrader 版（对照向量化版 engine.py，做更贴近实盘的撮合）。

选股逻辑与 engine.py、实盘共用同一个 cta.decide_targets；这里负责更真实
的成交：每月第一个交易日按当日收盘价（cheat-on-close）调仓，broker 自动算量、扣手续费/滑点。

为什么也保留这个框架版：
  - 撮合/计费交给 broker：order_target_percent 自动算下单量，setcommission/set_slippage 扣成本；
  - 调仓按当日价成交、先卖后买，更贴近真实账户；向量化版则适合快速调参研究。

运行：
  python bt.py            # 真实数据（需 akshare）
"""
import numpy as np
import pandas as pd
import backtrader as bt

# 策略参数与“选股大脑”统一从核心模块导入，回测/实盘共用一份逻辑
from cta import (POOL, DEFENSE, MAX_LOOKBACK,
                           COMMISSION, SLIPPAGE, BENCH, TREND_MA, _limit, decide_targets)

START_CASH = 1_000_000
# 喂给决策大脑的历史长度要同时够"动量回看"和"大盘趋势均线"两者，取较大值。
# 否则只喂 MAX_LOOKBACK(126) 根，趋势过滤(需 TREND_MA=200 根)会因数据不足静默失效。
HISTORY = max(MAX_LOOKBACK, TREND_MA)

# ---------------- 数据获取（返回 OHLCV，按各自上市日） ----------------
def load_real():
    """用 akshare 拉真实日线（前复权 OHLCV），每只 ETF 用自身上市后的区间。
    改进项 E2：带重试 + 指数退避，应对东财接口限频（详见 engine.load_real 注释）。"""
    import akshare as ak
    import time
    codes = list(POOL.keys()) + [DEFENSE[0]]   # 池中所有股票 ETF + 防守资产
    out = {}                                    # {代码: OHLCV 的 DataFrame}
    end_date = pd.Timestamp.today().strftime("%Y%m%d")   # 动态截止日：跑到哪天拉到哪天，不再锁死 2025 年底
    for c in codes:
        raw = None
        for attempt in range(4):                          # 重试 + 退避，应对东财限频
            try:
                # 拉日线行情；adjust="qfq" 表示前复权（消除分红/拆分造成的价格跳变）
                raw = ak.fund_etf_hist_em(symbol=c, period="daily",
                                          start_date="20140101", end_date=end_date, adjust="qfq")
                if len(raw):
                    break
                raw = None
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"{c} 拉取失败（东财限频？）: {e}") from e
            time.sleep(5 * (attempt + 1))
        time.sleep(0.3)
        raw["日期"] = pd.to_datetime(raw["日期"])      # 字符串日期转成时间戳
        raw = raw.set_index("日期").sort_index()       # 用日期当索引并按时间排序
        # 把 akshare 的中文列名映射成 backtrader 认的英文 OHLCV 列名
        df = pd.DataFrame({
            "open": raw["开盘"], "high": raw["最高"], "low": raw["最低"],
            "close": raw["收盘"], "volume": raw["成交量"],
        })
        out[c] = df.dropna()                    # 去掉任何含缺失值的行
    return out

# ---------------- 策略 ----------------
class MomentumRotation(bt.Strategy):
    # rebal_days：预计算好的"每月最后一个交易日"集合（date 对象）。只在这些日子调仓，
    # 与向量化版 engine.py 的月末调仓口径对齐。传 None 则退回"每月第一个交易日"。
    params = (("rebal_days", None),)

    def __init__(self):
        # self.datas 是 backtrader 注入的所有数据源列表；d._name 是添加时设的代码。
        # 这里挑出“股票 ETF”那几只（排除防守国债），动量排序只在它们之间做。
        self.stock_feeds = [d for d in self.datas if d._name in POOL]
        self.rebal_set = set(self.p.rebal_days) if self.p.rebal_days else None
        self._last_month = None      # 仅在没传 rebal_days 时用（退回月初调仓的兜底逻辑）
        self.holdings_log = []       # 记录每次调仓持有了什么（用于打印/核对）

    def prenext(self):
        # 多数据源默认要等“最晚上市”的标的（512100 到 2016 才有数据）才进 next，
        # 会白白丢掉前几年行情。手动让 prenext 也执行调仓，未上市标的因收盘序列长度
        # 不足会被 decide_targets 自动跳过，等有数据后再纳入排序。
        self.next()

    def _recent_closes(self):
        """为核心大脑准备 {code: 收盘价升序序列}，只放数据够长的标的。"""
        out = {}
        for d in self.stock_feeds:
            if len(d) > MAX_LOOKBACK:        # 已有的历史 bar 数够长才参与动量排序
                # 想喂 HISTORY 根（够趋势均线用），但早期 bar 还没攒够，就有多少给多少，
                # 用 min 防止 d.close[-k] 越界（k 最多到 len(d)-1，即最早一根）。
                k = min(len(d) - 1, HISTORY)
                # d.close[0] 是当前收盘，d.close[-1] 是昨天……负得越多越早。
                # range(k, -1, -1) 生成 k,k-1,...,1,0，于是按时间升序取价。
                out[d._name] = [d.close[-i] for i in range(k, -1, -1)]
        return out

    def _at_limit(self, d):
        """当日 d 是否收盘封涨停/跌停（复权 close 环比近似）。返回 (up, down)。
        BT 拿不到 next bar 数据，成交虽在 T+1 但用 T 日近似判定——与向量化版精确 T+1 判定
        有 1 bar 差异，BT 仅作慢速粗对照。除权日复权因子会扭曲环比，ETF 除权日极少可接受。"""
        if len(d) < 2:
            return False, False
        lim = _limit(d._name)
        prev = d.close[-1]
        if not prev:
            return False, False
        pct = d.close[0] / prev - 1.0
        return pct >= lim - 1e-4, pct <= -lim + 1e-4

    def next(self):
        # next() 每根 bar（每个交易日）被调用一次；datetime.date(0) 是“当前这根”的日期。
        dt = self.datas[0].datetime.date(0)
        if self.rebal_set is not None:
            # 月末调仓口径：只在"每月最后一个交易日"动手，其余日子直接返回。
            if dt not in self.rebal_set:
                return
        else:
            # 兜底：没预算调仓日时，退回"每月第一个交易日"（月份变了才调）。
            if dt.month == self._last_month:
                return
            self._last_month = dt.month

        # === 选股大脑：与实盘共用同一个 decide_targets ===
        # 显式开启大盘趋势过滤（trend_ma），与向量化回测/实盘保持一致；波动率目标默认已开。
        target, names = decide_targets(self._recent_closes(), trend_ma=TREND_MA)
        if not target:                       # 可选标的不足（回测初期）→ 本期不动
            return
        self.holdings_log.append((dt, names))

        # 调仓分两轮，先卖后买（D2：封板方向跳过——涨停买不进、跌停卖不出，维持原仓）：
        # 第一轮：把“目标里不再持有、但当前还有仓”的标的清掉（封跌停卖不出则保留）
        for d in self.datas:
            if target.get(d._name, 0.0) == 0.0 and self.getposition(d).size != 0:
                _, dn = self._at_limit(d)
                if not dn:
                    self.order_target_percent(d, target=0.0)
        # 第二轮：对目标里要持有的，按目标权重下单（封涨停买不进则跳过）
        for d in self.datas:
            tgt = target.get(d._name, 0.0)
            if tgt > 0.0:
                up, _ = self._at_limit(d)
                if not up:
                    self.order_target_percent(d, target=tgt)

# ---------------- 运行 ----------------
def add_feeds(cerebro, data: dict):
    # 把每只 ETF 的 DataFrame 包成 backtrader 的数据源(feed)并喂给引擎(cerebro)。
    for code, df in data.items():
        # PandasData 把 DataFrame 的列对应到 backtrader 的 OHLCV；
        # openinterest=-1 表示“没有持仓量这一列”，让它别去找。
        feed = bt.feeds.PandasData(dataname=df, name=code,
                                   open="open", high="high", low="low",
                                   close="close", volume="volume", openinterest=-1)
        cerebro.adddata(feed, name=code)       # name 即上面策略里用的 d._name

def month_end_trading_days(data):
    """从行情里算出"每月最后一个交易日"集合（用基准的交易日历，纯日历、无前视）。"""
    idx = data[BENCH].index                          # 基准 510300 历史最长，用它当交易日历
    last_per_month = idx.to_series().resample("ME").last()   # 每月最后一个交易日
    return set(last_per_month.dropna().dt.date)       # 转成 date 集合，供策略按日比对


def run_strategy(data):
    cerebro = bt.Cerebro()                      # Cerebro 是 backtrader 的总引擎/大脑
    add_feeds(cerebro, data)                    # 喂入各 ETF 行情
    # 传入月末调仓日，与向量化版口径一致（每月最后一个交易日决策）
    cerebro.addstrategy(MomentumRotation, rebal_days=month_end_trading_days(data))
    cerebro.broker.setcash(START_CASH)          # 设初始资金
    cerebro.broker.setcommission(commission=COMMISSION)   # 设手续费率
    cerebro.broker.set_slippage_perc(perc=SLIPPAGE)       # 设滑点率
    # D1：不用 cheat-on-close——订单在下一根 bar 开盘成交（月末 T 日信号 → T+1 成交），
    #     与向量化版 T+1 口径对齐（向量化按 T+1 收盘，BT 按 T+1 开盘，差半天，可接受）。
    #     先卖后买在同一执行 bar 内顺序撮合，卖出回笼现金立即可用于买入。
    #     原 set_coc(True) 已停用：它让订单按当日收盘成交 = “收盘价信号+收盘价成交”的乐观口径。
    # 挂 analyzer（分析器）：引擎跑完后能取出对应指标。_name 是取结果时用的键。
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")          # 回撤
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",   # 夏普
                        timeframe=bt.TimeFrame.Days, riskfreerate=0.0,
                        annualize=True)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="daily",     # 每日收益序列
                        timeframe=bt.TimeFrame.Days)
    strat = cerebro.run()[0]                    # 跑回测；返回策略实例列表，取第[0]个
    final = cerebro.broker.getvalue()           # 期末总资产
    return strat, final

def benchmark_stats(close: pd.Series):
    """买入持有 510300 的对照指标。"""
    nav = close / close.iloc[0]                # 价格除以首日价 → 归一化净值（从1.0起）
    daily = nav.pct_change().dropna()          # 净值转日收益，丢掉首日的 NaN
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1/years) - 1                                   # 年化收益
    sharpe = (daily.mean() * 252) / (daily.std() * np.sqrt(252) + 1e-12)   # 夏普（+1e-12 防除0）
    mdd = ((nav / nav.cummax()) - 1).min()     # 最大回撤：净值/历史最高 - 1 的最小值
    return dict(总收益=nav.iloc[-1]-1, 年化=cagr, 夏普=sharpe, 最大回撤=mdd, 年数=years)

def main():
    data = load_real()
    strat, final = run_strategy(data)

    # --- 从回测结果里取策略指标 ---
    total_ret = final / START_CASH - 1                   # 总收益率
    dd = strat.analyzers.dd.get_analysis()               # 取回撤分析器结果
    mdd = -dd["max"]["drawdown"] / 100.0                 # 它给的是正数百分比，转成负小数
    # 夏普可能为 None（极端情况），用 or 兜底成 NaN，避免后面格式化报错
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio") or float("nan")
    # 年化收益用总收益整体复合（而非分年平均）
    bench_close = data[BENCH]["close"]                   # 基准（沪深300）的收盘价序列
    # 用策略的每日收益序列拿到实际回测的起止日期，算年数
    daily = pd.Series(strat.analyzers.daily.get_analysis())
    daily.index = pd.to_datetime(daily.index)            # 索引转成时间戳便于取日期
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1/years) - 1

    pb = benchmark_stats(bench_close)                    # 基准指标

    print(f"回测区间: {daily.index[0].date()} ~ {daily.index[-1].date()}  共{years:.1f}年  调仓{len(strat.holdings_log)}次")
    print(f"{'指标':<8}{'轮动策略(BT)':>16}{'买入持有沪深300':>18}")
    # 内嵌小函数：打印一行（指标名 + 策略值 + 基准值）。pct=True 按百分比、否则按2位小数。
    def row(k, sp, bp, pct=True):
        f = (lambda x: f"{x*100:.1f}%") if pct else (lambda x: f"{x:.2f}")
        print(f"{k:<8}{f(sp):>16}{f(bp):>18}")
    row("总收益", total_ret, pb["总收益"])
    row("年化收益", cagr, pb["年化"])
    row("最大回撤", mdd, pb["最大回撤"])
    row("夏普比率", sharpe, pb["夏普"], pct=False)
    print(f"期末资产: {final:,.0f} 元（初始 {START_CASH:,.0f}） （真实数据）")

if __name__ == "__main__":
    main()
