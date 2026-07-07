# -*- coding: utf-8 -*-
"""
A股 ETF 动量轮动 —— 向量化回测 / 研究工具。

策略逻辑统一来自 momentum_core.decide_targets（与 backtrader 回测、实盘共用一份），
这里只负责“喂历史数据 + 算净值 + 出指标/图”，速度快，适合调参研究。

运行:
  python etf_momentum.py            # 真实数据：改进版 vs 无风控版 vs 基准 + 出图
  python etf_momentum.py sweep      # 参数稳健性扫描（不同回看窗口/持仓数）
  python etf_momentum.py robust     # 单参数扰动稳健性（看是不是"平台"，查过拟合）
  python etf_momentum.py wf         # walk-forward 滚动样本外（量化样本外夏普衰减）
  python etf_momentum.py boot       # RISK_ADJ 夏普提升的 bootstrap 显著性检验
  python etf_momentum.py bootmom    # A2 混合动量加权 的 A/B + bootstrap 显著性检验
  python etf_momentum.py dsr        # G1 Deflated Sharpe：多重比较下的夏普可信度
  python etf_momentum.py universe   # 标的池消融：踢掉黄金/纳指，量化 alpha 对池子的依赖
  python etf_momentum.py nomomentum # J2 对照：等权全池+风控不选股 vs 动量轮动，量化选股边际
  python etf_momentum.py bondstress # J4 防守资产债牛敏感性：国债收益替换 0%/−2% 重算回撤与 Calmar
  python etf_momentum.py freezetest # J5 样本外冻结期检验：近期段选股边际+风控稳健性,PBO=0.67 再审视
"""
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from momentum_core import (POOL, DEFENSE, BENCH, MAX_LOOKBACK, LOOKBACKS, TOP_N,
                           VOL_TARGET, VOL_WINDOW, TREND_MA, TREND_CUT, TREND_CODE, SKIP_RECENT,
                           RISK_ADJ, LOOKBACK_WEIGHTS, COMMISSION, SLIPPAGE, WEIGHTING, INV_VOL_WINDOW,
                           CRASH_PROT, CRASH_LOOKBACK, CRASH_THR, CRASH_CUT,
                           DRAWDOWN_PROT, DD_WINDOW, DD_THR, DD_CUT,
                           _limit, decide_targets)

ALL_CODES = list(POOL) + [DEFENSE[0]]

# 历史遗留：曾用 _SOURCE_SHIFT 显式截断 tushare 源的价格断层（510500/512100/513100 等），
# 该方案已被 _load_via_tushare 的 pre_close 对齐前复权法替代（能正确处理份额拆分、且恢复了这些
# ETF 的早期历史）。字典已删除；前复权原理见 _qfq_from_pre_close。详见 IMPROVEMENTS E3/E4。


# ---------------- 数据 ----------------
def load_real(with_limits=False):
    """拉真实前复权日线。优先 tushare（根治 E2），未配置 TUSHARE_TOKEN 则回退东财 akshare（带重试）。

    改进项 E2：tushare 走 fund_daily + fund_div 手动前复权（pro_bar 的 adj 对 ETF 不生效），
    不限频、复权准确，是唯一稳定可靠的 ETF 复权源。东财 fund_etf_hist_em 复权但限频；
    新浪 fund_etf_hist_sina 不复权、baostock/股票接口不支持 ETF。
    改进项 D2：with_limits=True 时额外返回 (cant_buy, cant_sell) 涨跌停掩码（收盘封板→
    对应方向无法成交）。掩码用未复权真实价算（前复权会扭曲涨跌幅）；未配置时返回纯 px（向后兼容）。"""
    res = _load_via_tushare(with_limits)
    if res is None:
        res = _load_via_eastmoney(with_limits)
    px = res[0] if isinstance(res, tuple) else res
    _data_quality_guard(px)          # E4：清洗后仍残留的不可成交伪迹大声告警
    return res


def _qfq_from_pre_close(close, pre_close):
    """前复权（pre_close 对齐法，改进项 E3）：用 pre_close 列对齐除权/拆分，统一处理现金分红与份额拆分。

    替代旧 fund_div + _SOURCE_SHIFT 方案（两缺陷：① fund_div 不记录份额拆分，致拆分日 close 断崖
    漏过复权；② _SOURCE_SHIFT 把"pre_close 准但 close 断点"误判为源永久断层、过度截断丢早期历史）。
    实测 pre_close 对除权/拆分准确：涨跌停时 pre_close=昨收，仅除权/拆分时 pre_close≠close[t-1]。
      f[t] = pre_close[t] / close[t-1]        （偏离 1 = 除权/拆分）
      g[i] = ∏_{t≥i} f[t]
      adj[i] = close[i] × ∏_{t>i} f[t] = close[i] × g.shift(-1)   （最新价 = close 末值不变）
    close / pre_close：同 index 的 pandas Series（未复权）；返回前复权 Series。纯函数，可单测（test_qfq）。
    """
    prev = close.shift(1)
    f = (pre_close / prev).fillna(1.0)            # f[t]=pre_close[t]/close[t-1]；偏离 1 = 除权/拆分
    g = f[::-1].cumprod()[::-1]                  # g[i]=∏_{t≥i}f[t]
    return close * g.shift(-1).fillna(1.0)        # adj[i]=close[i]×∏_{t>i}f[t]，最新价=close 末值不变


def _load_via_tushare(with_limits=False):
    """tushare 前复权（根治 E2）。代理 URL 由 TUSHARE_API 环境变量指定（可选，
    默认代理 fastapic.stockai888.top）。token 优先 TUSHARE_TOKEN 环境变量，其次本地
    .tushare_token 文件（gitignore，不入库，方便不设环境变量直接跑）；两者都无返回 None。

    关键：tushare 的 pro_bar(adj='qfq') 对 ETF(asset='FD') 不生效（返回不复权），故用
    fund_daily(不复权 close) + fund_div(分红 ex_date/div_cash) 手动算前复权：
    每个除权日之前的价格 × (前收盘-分红)/前收盘，累积即得连续前复权序列。
    D2：with_limits=True 时额外用未复权 close/pre_close 算涨跌停掩码（fund_daily 本就不复权，
    pre_close 列直接可用，判定最准）。"""
    import os
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        # 回退本地 .tushare_token 文件（gitignore，不入库；方便不设环境变量直接跑）
        _tf = os.path.join(os.path.dirname(__file__), ".tushare_token")
        if os.path.exists(_tf):
            token = open(_tf, encoding="utf-8").read().strip()
    if not token:
        return None
    import requests, time
    api_url = os.environ.get("TUSHARE_API", "https://fastapic.stockai888.top")
    def _ts_post(api_name, params, fields):
        """tushare Pro HTTP POST（服务商文档“方式二”，绕开 SDK 的 lxml 依赖链——
        本机 termux 装 tushare SDK 卡在 lxml 源码编译、无 cp314 aarch64 wheel，SDK 顶层
        import 即失败）。token 直接进请求体，不再受 SDK 层环境变量干扰代理的坑影响。"""
        r = requests.post(api_url, json={"api_name": api_name, "token": token,
                                         "params": params, "fields": fields},
                          headers={"Accept-Encoding": "gzip"}, timeout=30)
        j = r.json()
        if j.get("code") != 0:
            raise RuntimeError(f"tushare {api_name} 调用失败: {j.get('msg')}")
        d = j["data"]
        return pd.DataFrame(d["items"], columns=d["fields"])
    def tc(c): return c + (".SH" if c[0] in "5" else ".SZ")
    end = pd.Timestamp.today().strftime("%Y%m%d")
    series = {}
    raw_d = {}; pre_d = {}                                 # 未复权（涨跌停判定用，D2）
    for c in ALL_CODES:
        fd = _ts_post("fund_daily", {"ts_code": tc(c), "start_date": "20130101", "end_date": end},
                      "trade_date,close,pre_close")
        fd["date"] = pd.to_datetime(fd["trade_date"])
        fd = fd.set_index("date").sort_index()             # fund_daily 返回本就是未复权价
        close = fd["close"].astype(float)                  # 未复权收盘（前复权由下面手动算）
        raw_d[c] = close
        pre_d[c] = fd["pre_close"].astype(float) if "pre_close" in fd else close.shift(1)
        time.sleep(0.6)                                    # tushare 限速 100次/分，留余量
        # 前复权（pre_close 对齐法，详见 _qfq_from_pre_close）：替代旧 fund_div + _SOURCE_SHIFT，
        # 能正确处理份额拆分、且恢复了 510500/512100/513100 的早期历史。
        adj = _qfq_from_pre_close(close, pre_d[c])
        adj.name = c
        series[c] = adj
    px = pd.concat(series, axis=1, sort=False).sort_index().ffill().dropna(how="all")
    px = _anchor_start(px)
    if with_limits:
        raw = pd.concat(raw_d, axis=1).sort_index().ffill()
        pc = pd.concat(pre_d, axis=1).sort_index().ffill()
        raw = raw.reindex(px.index)[px.columns]            # 对齐到 px 的索引/列（同源，reindex 不引入 NaN）
        pc = pc.reindex(px.index)[px.columns]
        cb, cs = limit_masks(raw, pc)
        return px, cb.fillna(False), cs.fillna(False)
    return px


def _load_via_eastmoney(with_limits=False):
    """东财 fund_etf_hist_em 前复权，带 4 次重试 + 指数退避（应对东财 IP 限频）。

    D2：with_limits=True 时用复权收盘环比近似涨跌停（akshare 回退路径，再拉一次 adjust=""
    会翻倍调用、东财本就限频不划算）。除权日复权因子会扭曲单日环比、可能误判封板，
    但 ETF 除权日极少、影响单日单标的，回测整体可忽略。主力走 tushare 即无此问题。"""
    import akshare as ak, time
    end_date = pd.Timestamp.today().strftime("%Y%m%d")
    series = {}
    for c in ALL_CODES:
        df = None
        for attempt in range(4):
            try:
                df = ak.fund_etf_hist_em(symbol=c, period="daily",
                                         start_date="20140101", end_date=end_date, adjust="qfq")
                if len(df):
                    break
                df = None
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(
                        f"{c} 拉取失败（东财限频；建议配置 TUSHARE_TOKEN 走 tushare）: {e}") from e
            time.sleep(5 * (attempt + 1))
        time.sleep(0.3)
        df["日期"] = pd.to_datetime(df["日期"])
        series[c] = df.set_index("日期")["收盘"].astype(float).rename(c)
    px = pd.concat(series.values(), axis=1).sort_index()
    px = px.dropna(how="all").ffill()
    px = _anchor_start(px)
    if with_limits:
        cb, cs = limit_masks(px.copy(), px.shift(1))      # 复权环比近似
        return px, cb.fillna(False), cs.fillna(False)
    return px


# ---------------- 涨跌停掩码（D2，向量化专用） ----------------
def limit_masks(raw_close, raw_pre_close):
    """用未复权真实价算"收盘封板"掩码（改进项 D2，向量化回测专用）。

    raw_close / raw_pre_close：同形状 DataFrame[date, code]，未复权真实价。
    返回 (cant_buy, cant_sell)：bool DataFrame，True = 当日该标的收盘封涨停/跌停，
    对应方向无法成交。涨停价 = round(pre_close*(1+limit), 2)（A 股按分取整）；
    close ≥ 涨停价 → 收盘封死 → 买不进；close ≤ 跌停价 → 卖不出。
    用 close 而非 high==low：ETF 封板通常尾盘仍封着，close==板价是"收盘封死"的合理代理；
    盘中封板尾盘开板（close 未触板）则视作能成交。"""
    cant_buy = pd.DataFrame(False, index=raw_close.index, columns=raw_close.columns)
    cant_sell = pd.DataFrame(False, index=raw_close.index, columns=raw_close.columns)
    for code in raw_close.columns:
        lim = _limit(code)
        pc = raw_pre_close[code]
        cl = raw_close[code]
        cant_buy[code] = cl >= (pc * (1 + lim)).round(2)
        cant_sell[code] = cl <= (pc * (1 - lim)).round(2)
    return cant_buy, cant_sell


def _data_quality_guard(px, thr_mult=1.5):
    """数据质量告警（改进项 E4）：清洗（去重复权 + 源断层截断）后若仍残留不可成交的伪日收益
    （|环比|>thr_mult×涨跌停，ETF 最高 ±20%），大声告警——可能还有未发现的源 bug。仅告警不修改
    （清洗在 loader 里做）。正常清洗后返回空清单；非空则需人工核查/补截断。"""
    bad = []
    for c in px.columns:
        s = px[c].dropna()
        if len(s) < 2:
            continue
        lim = _limit(c) * thr_mult
        for d, v in s.pct_change().dropna()[lambda x: x.abs() > lim].items():
            bad.append((c, pd.Timestamp(d).date(), v))
    if bad:
        print(f"[DQ告警] 清洗后仍发现 {len(bad)} 个不可成交伪迹（|日收益|>{thr_mult}×涨跌停）:")
        for c, d, v in bad:
            print(f"    {c}  {d}  {v*100:+.1f}%  （源数据 bug？需人工核查/截断）")
    return bad


def _anchor_start(px):
    """起点锚定（E5 审视 2026-07-05）：回测从 BENCH+DEFENSE 都有数据的首个交易日开始。

    纯数据首日驱动（首个非 NaN，ffill 不改首日），不依赖收益 → 无硬前视。
    截到此处是有意设计：保证起点时核心池基本就绪，避免早期"只有少数早上市标的"
    的不代表性段（若提前起点，POOL 多数标的未上市 → 动量凑不齐 top_n → 长期空仓扭曲统计，
    且相当于又一轮重定基线、方向错误）。微弱"前视"仅存于方法论层：BENCH/DEFENSE 是人为选定，
    起点随资产选择间接固定（见 B1 前视教训），代码层无法根除。本函数打印审计行让隐性锚定
    显性化，便于复现/审计/未来 E1 起点敏感性测试。"""
    ready = px.index[px[[BENCH, DEFENSE[0]]].notna().all(axis=1)]
    start = ready[0]
    missing = [c for c in POOL if px[c].loc[:start].isna().all()]   # 起点时尚未上市的 POOL 标的
    print(f"[E5 起点] ready[0]={start.date()}（{BENCH}+{DEFENSE[0]} 首个共同非NaN日，"
          f"数据驱动非收益→无前视）；起点时 POOL 未就绪: {missing or '无'}"
          f"（上市晚于起点，动量排序 dropna 自动跳过）")
    return px.loc[start:]


def _hit(code, day, mask):
    """mask 在 [day, code] 处是否 True；mask 为 None 或索引/列缺失一律视作 False（保守=能成交）。"""
    if mask is None:
        return False
    try:
        return bool(mask.loc[day, code])
    except (KeyError, IndexError):
        return False


def _apply_target_with_limits(tgt_dict, prev_cur, fill_day, cant_buy, cant_sell):
    """把目标权重 tgt_dict 在成交日 fill_day 落地为实际持仓 Series，按当日涨跌停过滤（D2）。

    返回新的持仓 Series。规则：
      - 不在 target 的旧仓：默认清零；当日封跌停（卖不出）则维持。
      - target 中要加仓/新建（w>old）且当日封涨停 → 买不进，维持 old。
      - target 中要减仓（w<old）且当日封跌停 → 卖不出，维持 old。
    被过滤滞留的权重自然落到现金（1 - Σ持仓），向量化框架里现金无收益列 = 0 收益。"""
    new_cur = prev_cur.copy()                       # 先默认维持现状，再按目标/封板调整
    for c in [c for c in new_cur.index if new_cur[c] > 1e-12 and c not in tgt_dict]:
        if not _hit(c, fill_day, cant_sell):        # 没封跌停才清得掉
            new_cur[c] = 0.0
    for c, w in tgt_dict.items():
        old = float(prev_cur.get(c, 0.0))
        if w > old + 1e-9 and _hit(c, fill_day, cant_buy):
            continue                                # 涨停买不进，维持 old（new_cur[c] 已是 old）
        if w < old - 1e-9 and _hit(c, fill_day, cant_sell):
            continue                                # 跌停卖不出，维持 old
        new_cur[c] = w
    return new_cur


# ---------------- 回测引擎（调用核心大脑） ----------------
def backtest(px, lookbacks=LOOKBACKS, top_n=TOP_N, vol_target=VOL_TARGET, trend_ma=None,
             trend_code=None, trend_cut=TREND_CUT, vol_window=VOL_WINDOW, skip_recent=SKIP_RECENT,
             risk_adj=RISK_ADJ, mom_weights=LOOKBACK_WEIGHTS,
             weighting=WEIGHTING, inv_vol_window=INV_VOL_WINDOW,
             crash_prot=CRASH_PROT, crash_lookback=CRASH_LOOKBACK,
             crash_thr=CRASH_THR, crash_cut=CRASH_CUT,
             drawdown_prot=DRAWDOWN_PROT, dd_window=DD_WINDOW,
             dd_thr=DD_THR, dd_cut=DD_CUT,
             cant_buy=None, cant_sell=None, defense_cash=None, max_weight=None, return_turnover=False,
             hold_all=False):
    """月末调仓，权重由 decide_targets 决定。返回 (策略净值, 日收益, 调仓次数, 持仓日志)。
    成交口径（D1）：月末 T 日收盘算信号，次日（T+1）才成交——避免"收盘价信号 + 收盘价成交"
        的前视/乐观偏误。新权重自 T+2 起吃收益（shift(1) 自洽）。原 coc/当日收盘口径已被替换。
    涨跌停（D2）：cant_buy/cant_sell 为 load_real(with_limits=True) 返回的"收盘封板"掩码
        （True=当日该标的封涨停/跌停，对应方向无法成交，维持原仓）。None=不限制（向后兼容，
        sweep/robust/wf 等保持原行为）。
    trend_ma:    大盘趋势过滤均线天数（None=关闭），用于对比加/不加趋势择时的效果。
    trend_cut:   趋势下行时股票仓的"保留比例"（透传给 decide_targets，供 robust 扫描）。
    vol_target:  年化波动目标（None=关闭）；vol_window: 估计近期波动的回看窗口。
    skip_recent: 跳过最近 N 日再算动量（21≈1个月）；risk_adj: 是否用风险调整动量。两者用于 A/B 测试。"""
    rets = px.pct_change().fillna(0.0)                      # 每只标的的每日收益率矩阵

    # 启动期所需历史 = max(全局 MAX_LOOKBACK, 本次最长回看窗口 + skip_recent)。
    #   不写死 MAX_LOOKBACK：否则含 252 日窗口或大 skip 的组合在 day127~need 段历史不足被判
    #   None、凑不齐 top_n → 系统性空仓，年化被不公平拉低（sweep/wf 的长窗口候选即此症）。
    #   默认 (21,63,126)+skip0 时 need=126，与原行为完全一致。
    need = max(MAX_LOOKBACK, max(lookbacks) + skip_recent)

    # 1) 找出每个月最后一个交易日作为调仓日；并跳过历史不足 need 的初期
    month_ends = px.resample("ME").last().index
    rebal_days = [px.index[px.index <= me][-1] for me in month_ends if (px.index <= me).any()]
    rebal_days = [d for d in rebal_days if d >= px.index[need]]
    rd_set = set(rebal_days)

    # 2) 逐日推进，构造一张“每天持有什么权重”的表。D1：信号日(T)算目标→pending，次日(T+1，
    #    成交日)才落地为 cur（防未来函数/收盘瞬时成交偏误）；D2：落地时按成交日涨跌停过滤。
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    cur = pd.Series(0.0, index=px.columns)                  # 当前实际持仓权重（已含涨跌停过滤）
    holdings_log = []
    pending = None                                          # (target_dict, picks) 信号日算出、待次日成交
    for d in px.index:
        if pending is not None:                            # 今日 = 上一调仓日的成交日(T+1)：落地 pending
            tgt_dict, picks = pending
            cur = _apply_target_with_limits(tgt_dict, cur, d, cant_buy, cant_sell)
            pending = None
        if d in rd_set:                                    # 今日 = 信号日(T)：算信号，延迟到次日成交
            # 把截至当日的收盘价喂给核心大脑（与实盘喂法一致）
            recent = {c: px[c].loc[:d].dropna().tolist() for c in POOL}
            target, picks = decide_targets(recent, lookbacks=lookbacks,
                                           top_n=top_n, vol_target=vol_target,
                                           vol_window=vol_window, trend_ma=trend_ma,
                                           trend_code=(trend_code or TREND_CODE),
                                           trend_cut=trend_cut, skip_recent=skip_recent,
                                           risk_adj=risk_adj, mom_weights=mom_weights,
                                           weighting=weighting,
                                           inv_vol_window=inv_vol_window,
                                           crash_prot=crash_prot,
                                           crash_lookback=crash_lookback,
                                           crash_thr=crash_thr, crash_cut=crash_cut,
                                           drawdown_prot=drawdown_prot,
                                           dd_window=dd_window,
                                           dd_thr=dd_thr, dd_cut=dd_cut, defense_cash=defense_cash,
                                           max_weight=max_weight, hold_all=hold_all)
            if target:
                pending = (target, picks)                  # 不立即写 cur，等下一日成交
                holdings_log.append((d.date(), picks))
        weights.loc[d] = cur.values

    # 3) 关键防未来函数：今天的收益用“昨天收盘时持有的权重”（shift(1)）来计，
    #    即调仓日当天还按旧权重，次日才用新权重。
    w_lag = weights.shift(1).fillna(0.0)
    gross = (w_lag * rets).sum(axis=1)                      # 组合每日毛收益

    # 4) 交易成本 = 换手率 ×（手续费+滑点）。换手率 = 权重变化的绝对值之和
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (COMMISSION + SLIPPAGE)
    net = gross - cost                                      # 净收益

    # 5) 从有足够历史的起点开始累乘成净值曲线，并归一化到 1.0
    start = px.index[need]
    nav = (1 + net).cumprod().loc[start:]
    nav = nav / nav.iloc[0]
    if return_turnover:                  # F5：暴露逐日换手（权重总绝对变动），供算年换手/成本敏感性
        return nav, net.loc[start:], len(rebal_days), holdings_log, turnover.loc[start:]
    return nav, net.loc[start:], len(rebal_days), holdings_log


def bench_nav(px):
    start = px.index[MAX_LOOKBACK]
    nav = (1 + px[BENCH].pct_change().fillna(0.0)).cumprod().loc[start:]
    return nav / nav.iloc[0]


def bench_6040(px):
    """F7 股债 60-40 基准(60% 沪深300 + 40% 国债 ETF),经典被动组合对照。
    比纯沪深300 更能体现"被动持有"的可比基准(含债券缓冲)。"""
    start = px.index[MAX_LOOKBACK]
    stock = px[BENCH].pct_change().fillna(0.0)
    bond = px[DEFENSE[0]].pct_change().fillna(0.0)
    nav = (1 + 0.6 * stock + 0.4 * bond).cumprod().loc[start:]
    return nav / nav.iloc[0]


# ---------------- 绩效指标 ----------------
def perf(nav, daily):
    """根据净值曲线 nav 和日收益 daily 算常用绩效指标。
    无风险利率/MAR/Omega 阈值统一按 0（与现有夏普口径一致，便于横向比较）。"""
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1/years) - 1               # 年化收益（几何平均）
    vol = daily.std() * np.sqrt(252)                   # 年化波动
    sharpe = (daily.mean() * 252) / (daily.std() * np.sqrt(252) + 1e-12)  # 夏普（无风险利率按0）
    # Sortino：分母换成下行偏差（仅惩罚负收益，MAR=0），比夏普更能戳穿"靠几次大涨堆出来的夏普"
    dd_dev = np.sqrt((daily.clip(upper=0.0) ** 2).mean()) * np.sqrt(252)  # 年化下行偏差
    sortino = (daily.mean() * 252) / (dd_dev + 1e-12)
    mdd = ((nav / nav.cummax()) - 1).min()             # 最大回撤 = 距历史最高点的最大跌幅
    calmar = cagr / (abs(mdd) + 1e-12)                 # Calmar = 年化/|最大回撤|（风险调整回撤）
    # Omega（阈值0，不年化）：正收益之和/|负收益之和|，>1 即盈利侧占优
    gains = daily[daily > 0].sum()
    losses = -daily[daily < 0].sum()
    omega = gains / (losses + 1e-12)
    # C4 尾部风险：VaR(α 分位的日损失)、CVaR(尾部条件期望=平均最大日损失)
    var_95 = float(daily.quantile(0.05)); var_99 = float(daily.quantile(0.01))
    cvar_95 = float(daily[daily <= var_95].mean()); cvar_99 = float(daily[daily <= var_99].mean())
    return dict(总收益=nav.iloc[-1]-1, 年化=cagr, 波动=vol, 夏普=sharpe,
                Sortino=sortino, Calmar=calmar, Omega=omega, 回撤=mdd, 年数=years,
                VaR95=var_95, CVaR95=cvar_95, VaR99=var_99, CVaR99=cvar_99)


def factor_attribution(daily, bench_daily):
    """F4 因子归因：策略日收益对基准做 CAPM 单因子回归，拆 alpha/beta + 信息比率。
    回答"跑赢基准多少是真 alpha、多少是 beta 暴露"。无风险利率/MAR 按 0（与 perf 口径一致）。
    返回 dict（None=数据不足或基准零波动）：年化alpha、beta、IR、R²、年化跟踪误差、
    策略/基准/超额年化、相关性。"""
    common = daily.index.intersection(bench_daily.index)
    p = daily.loc[common].to_numpy()
    b = bench_daily.loc[common].to_numpy()
    if len(p) < 2 or b.var(ddof=1) <= 1e-12:
        return None
    beta = float(np.cov(p, b, ddof=1)[0, 1] / b.var(ddof=1))     # 市场暴露（斜率）
    alpha_d = float(p.mean() - beta * b.mean())                  # 日 alpha（Jensen 截距）
    alpha_ann = alpha_d * 252                                     # 年化（线性，与夏普口径一致）
    active = p - b                                                # 主动收益 = 策略 − 基准
    te_d = float(active.std(ddof=1))                             # 日跟踪误差
    ir = float((active.mean() / (te_d + 1e-12)) * np.sqrt(252)) if te_d > 0 else 0.0
    corr = float(np.corrcoef(p, b)[0, 1])
    years = (daily.index[-1] - daily.index[0]).days / 365.25    # 与 perf 口径一致（自然日）
    def _ann(x):
        return (np.prod(1 + x) ** (1 / years) - 1) if years > 0 else 0.0
    return dict(年化alpha=alpha_ann, beta=beta, IR=ir, R2=corr * corr,
                年化跟踪误差=te_d * np.sqrt(252), 相关性=corr,
                策略年化=_ann(p), 基准年化=_ann(b), 超额年化=_ann(p) - _ann(b))


# F8 多因子归因的因子定义（全部从已加载的 px 面板构造，无需外部因子数据）。
# pre_close 前复权法已恢复 512100/513100/518880 的早期历史，6 因子全可用（无需外部因子数据）。
# 单代码 = 该资产日收益；(c1, c2) = 零投资 spread = r(c1) − r(c2)。
_MFACTORS = [
    ("MKT",  BENCH),                  # 沪深300 市场（CAPM 已有的市场因子）
    ("SMB",  ("512100", BENCH)),      # 中证1000 − 沪深300（规模：小盘 − 大盘）
    ("VMG",  ("510880", BENCH)),      # 红利 − 市场（价值/红利倾斜）
    ("BND",  DEFENSE[0]),             # 国债（水平）★核心：剥离策略大量持有的债券暴露
    ("GLD",  "518880"),               # 黄金（水平）★核心：剥离黄金暴露
    ("NSDQ", ("513100", BENCH)),      # 纳指 − 市场（国际成长 QDII）
]


def _factor_returns(px, spec):
    """从价格面板 px 构造因子日收益 DataFrame。spec: [(名字, 单代码 或 (长, 短)), ...]。
    单代码 → pct_change；(c1, c2) → r(c1) − r(c2)。全 fillna(0)（停牌/早期未上市段→0 收益）。"""
    rs = px.pct_change().fillna(0.0)
    out = {}
    for name, d in spec:
        if isinstance(d, str):
            out[name] = rs[d] if d in rs else pd.Series(0.0, index=rs.index)
        else:
            c1, c2 = d
            out[name] = (rs[c1] if c1 in rs else 0.0) - (rs[c2] if c2 in rs else 0.0)
    return pd.DataFrame(out, index=rs.index)


def factor_attribution_multi(daily, px, factor_spec=None):
    """F8 多因子归因：策略日收益对 6 因子（MKT/SMB/VMG/BND/GLD/NSDQ）做 OLS 回归，拆 alpha + 各因子
    beta/年化贡献/t-stat/R²。回答"扣掉策略实际持有的资产类别暴露后，真 alpha 还剩多少"——
    F4 的 CAPM 只控制市场因子，会把债券/黄金等非权益收益误判成 alpha（本条核心论点，见 test_mfat_strips_spurious_alpha）。
    无风险利率按 0（与 F4/perf 一致）。返回 dict（观测数不足返回 None）：
    alpha_ann(年化)、betas/tstats/contribs(各因子,贡献已年化)、R2/R2_adj、corr(因子相关矩阵)、n_obs、sigma_ann(残差年化波动)。"""
    spec = factor_spec if factor_spec is not None else _MFACTORS
    fr = _factor_returns(px, spec)
    common = daily.index.intersection(fr.index)
    y = daily.loc[common].to_numpy()
    Xf = fr.loc[common].to_numpy()
    n, k = len(y), len(spec)
    if n < k + 5:                                     # 观测数 ≤ 因子数，OLS 无意义
        return None
    X = np.column_stack([np.ones(n), Xf])             # 含截距列
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)      # OLS：beta = (X'X)^-1 X'y
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if (n - k - 1) > 0 else r2
    sigma2 = ss_res / (n - k - 1)                     # 残差方差（k 个因子 + 1 截距）
    xtx_inv = np.linalg.pinv(X.T @ X)                 # pinv：近共线性时不崩（beta 用 lstsq 已稳）
    se = np.sqrt(np.diag(xtx_inv) * sigma2)           # 各系数标准误
    tstat = beta / np.where(se > 0, se, np.nan)
    alpha_d = float(beta[0])                          # 截距 = 日 alpha
    fac_betas = beta[1:]                              # 各因子 beta（不含截距）
    contribs_d = fac_betas * Xf.mean(axis=0)          # 日贡献 = beta × 因子均值
    corr = np.corrcoef(Xf, rowvar=False)              # 因子相关矩阵（共线性诊断）
    names = [s[0] for s in spec]
    return dict(
        alpha_ann=alpha_d * 252,                      # 年化 alpha（线性 ×252，与 F4 口径一致）
        betas={names[i]: float(fac_betas[i]) for i in range(k)},
        tstats={names[i]: float(tstat[i + 1]) for i in range(k)},
        contribs={names[i]: float(contribs_d[i] * 252) for i in range(k)},   # 年化贡献
        R2=float(r2), R2_adj=float(r2_adj),
        corr={names[i]: {names[j]: float(corr[i, j]) for j in range(k)} for i in range(k)},
        n_obs=int(n), sigma_ann=float(np.sqrt(sigma2) * np.sqrt(252)),
    )


def run_attribution(px, lim=None, title=""):
    """F4 因子归因子命令：全风控口径（vol+trend）跑一次 backtest，对基准做 CAPM 回归拆 alpha/beta/IR。
    与 run_regime 同源（都用全风控 backtest + 基准对照）。打印归因表，返回 dict。"""
    lim = lim or {}
    nav, daily, n, _ = backtest(px, vol_target=VOL_TARGET, trend_ma=TREND_MA, **lim)
    bench = px[BENCH].pct_change().fillna(0.0).reindex(daily.index).fillna(0.0)
    a = factor_attribution(daily, bench)
    if a is None:
        print("归因失败：数据不足或基准零波动")
        return None
    print(f"\n=== 因子归因{'（' + title + '）' if title else ''}（CAPM 单因子 vs {BENCH} 沪深300）===")
    print(f"  策略年化 {a['策略年化']*100:.1f}%  vs  基准年化 {a['基准年化']*100:.1f}%"
          f"  →  超额年化 {a['超额年化']*100:+.1f}%")
    print(f"  beta = {a['beta']:.2f}   （市场暴露：基准每涨1%，策略理论涨 {a['beta']:.2f}%）")
    print(f"  年化 alpha = {a['年化alpha']*100:+.1f}%   （Jensen 超额：扣除 beta 暴露后的纯 alpha）")
    print(f"  信息比率 IR = {a['IR']:.2f}   （主动收益 / 跟踪误差；>0.5 优秀、>1.0 顶尖）")
    print(f"  R² = {a['R2']:.2f}   相关性 {a['相关性']:.2f}   年化跟踪误差 {a['年化跟踪误差']*100:.1f}%")
    beta_drag = (a['beta'] - 1) * a['基准年化']                  # 低 beta 拖累（beta<1 为负：基准涨时策略少赚）
    print(f"  → 超额 {a['超额年化']*100:+.1f}% ≈ alpha {a['年化alpha']*100:+.1f}% + beta 拖累 {beta_drag*100:+.1f}%"
          f"（beta={a['beta']:.2f}：低市场暴露，基准上涨时少赚，须靠 alpha 补回）")
    return a


def run_mfattribution(px, lim=None, title=""):
    """F8 多因子归因子命令：全风控口径（vol+trend）跑一次 backtest，对 6 因子做 OLS 拆 alpha/beta/贡献。
    与 run_attribution（F4 CAPM）同源（同 backtest、同 daily 口径），但补上债券/黄金/小盘/价值/纳指因子——
    剥离 CAPM 误判为 alpha 的非权益 beta。打印因子表 + CAPM α 对照，返回 dict。"""
    lim = lim or {}
    nav, daily, n, _ = backtest(px, vol_target=VOL_TARGET, trend_ma=TREND_MA, **lim)
    bench = px[BENCH].pct_change().fillna(0.0).reindex(daily.index).fillna(0.0)
    capm = factor_attribution(daily, bench)                    # F4 CAPM α 做对照
    mf = factor_attribution_multi(daily, px)
    if mf is None:
        print("多因子归因失败：观测数不足")
        return None
    capm_alpha = capm["年化alpha"] if capm else 0.0
    capm_r2 = capm["R2"] if capm else 0.0
    stripped = capm_alpha - mf["alpha_ann"]                    # 被"剥离"的部分
    print(f"\n=== 多因子归因{'（' + title + '）' if title else ''}"
          f"（6 因子 MKT/SMB/VMG/BND/GLD/NSDQ，日频 OLS，n={mf['n_obs']}）===")
    print(f"  CAPM 年化α {capm_alpha*100:+.1f}%  →  多因子年化α {mf['alpha_ann']*100:+.1f}%"
          f"   被「剥离」{stripped*100:+.1f}%（CAPM 误归为 alpha 的债/金/风格 beta）")
    print(f"  {'因子':<8}{'beta':>9}{'t-stat':>9}{'年化贡献':>11}")
    note = {"MKT": " ← 市场暴露", "BND": " ← 债券暴露", "GLD": " ← 黄金暴露"}
    for name in [s[0] for s in _MFACTORS]:
        print(f"  {name:<8}{mf['betas'][name]:>9.2f}{mf['tstats'][name]:>9.1f}"
              f"{mf['contribs'][name]*100:>10.1f}%{note.get(name, '')}")
    print(f"  R² = {mf['R2']:.2f}（CAPM R²={capm_r2:.2f}）  调整 R² = {mf['R2_adj']:.2f}"
          f"  残差年化波动 {mf['sigma_ann']*100:.1f}%")
    names = list(mf["corr"])
    hi = [(names[i], names[j], mf["corr"][names[i]][names[j]])
          for i in range(len(names)) for j in range(i + 1, len(names))
          if abs(mf["corr"][names[i]][names[j]]) > 0.7]
    if hi:
        print("  ⚠ 高相关（|corr|>0.7，单 beta 标准误被抬高，alpha 仍无偏）："
              + "  ".join(f"{a}~{b}={c:+.2f}" for a, b, c in hi))
    return mf


def rolling_metrics(daily, window=252):
    """滚动 window 日（默认1年）的年化夏普/波动 + 逐日回撤（水下曲线）序列。
    全期单个夏普会掩盖“某段靠运气”，看滚动序列才能判断策略是否随时间稳定（改进项 F2）。"""
    roll_sharpe = daily.rolling(window).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252) + 1e-12), raw=True)
    roll_vol = daily.rolling(window).std() * np.sqrt(252)
    nav = (1 + daily).cumprod()
    drawdown = nav / nav.cummax() - 1                         # 水下曲线：距历史最高点的跌幅
    return roll_sharpe, roll_vol, drawdown


def _print_table(rows):
    """rows: [(名称, perf字典)]。Sortino/Calmar/Omega 见 F1;VaR95/CVaR95 见 C4(尾部风险)。"""
    print(f"{'策略':<22}{'总收益':>9}{'年化':>8}{'波动':>8}{'最大回撤':>9}{'夏普':>7}{'Sortino':>9}{'Calmar':>8}{'Omega':>8}{'VaR95':>8}{'CVaR95':>9}")
    for name, p in rows:
        print(f"{name:<22}{p['总收益']*100:>8.1f}%{p['年化']*100:>7.1f}%"
              f"{p['波动']*100:>7.1f}%{p['回撤']*100:>8.1f}%{p['夏普']:>7.2f}"
              f"{p['Sortino']:>9.2f}{p['Calmar']:>8.2f}{p['Omega']:>8.2f}"
              f"{p['VaR95']*100:>7.2f}%{p['CVaR95']*100:>8.2f}%")


# ---------------- 主流程 ----------------
def run_sweep(px):
    # 与部署口径一致（vol_target + 趋势过滤），否则 sweep 跑的是"无风控"版、其 lookback/top_n 最优
    # 选择对部署版无效甚至误导。WF_GRID / run_robust 同样显式带 trend_ma=TREND_MA。
    cfg = dict(vol_target=VOL_TARGET, trend_ma=TREND_MA)
    print("参数稳健性扫描（全风控口径：vol_target + 趋势过滤；年化% / 最大回撤% / 夏普）:\n")
    lb_opts = {"单60日": (60,), "单126日": (126,), "混合1/3/6月": (21, 63, 126), "混合3/6/12月": (63, 126, 252)}
    print(f"{'回看窗口 \\ 持仓数':<18}" + "".join(f"{'TopN='+str(n):>20}" for n in (1, 2, 3)))
    for lname, lb in lb_opts.items(): # lb是回看多少天
        cells = []
        for n in (1, 2, 3):
            nav, daily, _, _ = backtest(px, lookbacks=lb, top_n=n, **cfg)
            p = perf(nav, daily)
            cells.append(f"{p['年化']*100:5.1f}/{p['回撤']*100:6.1f}/{p['夏普']:.2f}")
        print(f"{lname:<18}" + "".join(f"{c:>20}" for c in cells))
    bn = bench_nav(px)
    pb = perf(bn, bn.pct_change().fillna(0))
    print(f"\n基准 买入持有沪深300: 年化 {pb['年化']*100:.1f}% / 回撤 {pb['回撤']*100:.1f}% / 夏普 {pb['夏普']:.2f}")


# =====================================================================
# 过拟合检验工具集（robust / wf / boot）—— 三个互补视角：
#   robust: 单参数扰动，看指标是不是"平台"（孤峰 = 过拟合）
#   wf    : 滚动样本外，量化样本外相对全样本内的夏普衰减
#   boot  : 配对 block bootstrap，检验 RISK_ADJ 的微提升是否显著
# =====================================================================

# ---- 通用：用一段日收益重建从 1.0 起的子段净值并算指标 ----
def _seg_perf(daily_seg):
    """子段切片不是从 1.0 起，直接 perf() 会算错年化/总收益；这里重建归一化净值再算。"""
    nav = (1 + daily_seg).cumprod()
    nav = nav / nav.iloc[0]
    return perf(nav, daily_seg)


# ---------------- F3：分 regime（牛/熊/震荡）表现拆解 ----------------
def regime_labels(px, bear_dd=-0.20, trend_ma=200):
    """按基准（沪深300）市场状态给每日打 regime 标签（改进项 F3）：
    熊市=从历史高点回撤超 bear_dd（默认20%）；牛市=非熊 且净值在 trend_ma 日均线之上；
    震荡=其余。用基准而非策略净值，刻画的是"市场 regime"本身、与持仓无关。"""
    bench = px[BENCH].ffill().dropna()
    nav = (1 + bench.pct_change().fillna(0)).cumprod()
    nav = nav / nav.iloc[0]
    dd = nav / nav.cummax() - 1                       # 回撤序列（负值）
    bear = dd < bear_dd                               # 回撤超阈值=熊
    ma = nav.rolling(trend_ma).mean()
    bull = (~bear) & (nav > ma)                       # 非熊 + 站上均线=牛
    lab = pd.Series("震荡", index=nav.index)
    lab[bear] = "熊市"; lab[bull] = "牛市"
    return lab


def run_regime(px, daily=None, title="", lim=None):
    """分 regime 表现：把日收益按基准 regime 切片，每段 _seg_perf 重建净值算指标。
    daily 已给则直接拆（复用 real 的 ret_tr/ret_iv），否则用默认参数（等权+趋势+波动目标）
    跑一次 backtest（传 lim 走 D1/D2 新口径，与 real 一致）。同时打印基准同 regime 对照，
    看策略在哪种行情跑赢/跑输基准。"""
    if daily is None:
        _, daily, _, _ = backtest(px, vol_target=VOL_TARGET, trend_ma=TREND_MA, **(lim or {}))
    lab = regime_labels(px).reindex(daily.index).fillna("震荡")
    bench_daily = px[BENCH].pct_change().fillna(0).reindex(daily.index).fillna(0)
    head = f"{'regime':<10}{'天数':>7}{'占比':>7}{'年化':>9}{'夏普':>7}{'最大回撤':>10}{'Sortino':>9}"
    def _rows(series):
        out = []
        for r in ["牛市", "熊市", "震荡"]:
            seg = series[lab == r]
            if len(seg) < 2:
                continue
            out.append((r, len(seg), len(seg) / len(daily), _seg_perf(seg)))
        return out
    print(f"\n=== 分 regime 表现{'（' + title + '）' if title else ''} ===")
    print(head)
    for name, n, share, p in _rows(daily):
        print(f"{name:<10}{n:>7}{share*100:>6.1f}%{p['年化']*100:>8.1f}%"
              f"{p['夏普']:>7.2f}{p['回撤']*100:>9.1f}%{p['Sortino']:>9.2f}")
    print("  —— 基准(沪深300)同 regime 对照 ——")
    print(head)
    for name, n, share, p in _rows(bench_daily):
        print(f"{name+'(基准)':<10}{n:>7}{share*100:>6.1f}%{p['年化']*100:>8.1f}%"
              f"{p['夏普']:>7.2f}{p['回撤']*100:>9.1f}%{p['Sortino']:>9.2f}")


# ---------------- robust：单参数扰动稳健性 ----------------
def run_robust(px):
    """
    one-at-a-time 稳健性扫描：固定当前默认参数为基准，每次只扰动一个维度，
    看年化/回撤/夏普是否随参数"平滑"变化。相邻取值接近 = 稳健（平台）；
    某个取值是明显孤峰 = 过拟合警告。相比 sweep 的二维网格更易读、组合不爆炸。
    注：各维度共用同一段历史；lookbacks 候选均 ≤ MAX_LOOKBACK(126) 日，避免长窗口
    版本因头部历史不足空仓而被系统性低估。
    """
    base = dict(lookbacks=LOOKBACKS, top_n=TOP_N, vol_target=VOL_TARGET,
                trend_ma=TREND_MA, trend_cut=TREND_CUT,
                skip_recent=SKIP_RECENT, risk_adj=RISK_ADJ)
    # (展示名, 参数 key, 候选取值, 基准值)
    sweeps = [
        ("大盘趋势砍仓比例 trend_cut", "trend_cut",
            [0.0, 0.3, 0.5, 0.7, 0.9], TREND_CUT),
        ("年化波动目标 vol_target", "vol_target",
            [None, 0.10, 0.15, 0.20, 0.25], VOL_TARGET),
        ("持仓数 top_n", "top_n",
            [1, 2, 3, 4, 5], TOP_N),
        ("动量回看窗口 lookbacks", "lookbacks",
            [(60,), (126,), (21, 63, 126), (21, 126), (63, 126)], LOOKBACKS),
        ("跳过最近 N 日 skip_recent", "skip_recent",
            [0, 5, 10, 21], SKIP_RECENT),
        ("风险调整动量 risk_adj", "risk_adj",
            [True, False], RISK_ADJ),
        ("趋势均线天数 trend_ma", "trend_ma",
            [None, 120, 200, 250], TREND_MA),
    ]
    for label, key, vals, base_v in sweeps:
        print(f"\n=== 扰动 {label}（基准 = {base_v}）===")
        print(f"{'取值':<24}{'年化':>9}{'最大回撤':>10}{'夏普':>8}  备注")
        for v in vals:
            p = dict(base)
            p[key] = v
            nav, daily, _, _ = backtest(px, **p)
            pf = perf(nav, daily)
            tag = "← 基准" if v == base_v else ""
            shown = ("None" if v is None
                     else ("/".join(map(str, v)) if isinstance(v, tuple) else str(v)))
            print(f"{shown:<24}{pf['年化']*100:>8.1f}%{pf['回撤']*100:>9.1f}%"
                  f"{pf['夏普']:>8.2f}  {tag}")
    print("\n判读：相邻取值指标接近 → 稳健（平台）；某个取值明显跳成孤峰 → 该参数被过拟合。")


# ---------------- wf：walk-forward 滚动样本外 ----------------
# 训练段选参用的候选网格（控制在 8 个，覆盖关键维度，避免组合爆炸）。
# 每条都显式带 trend_ma=TREND_MA —— 与 main 部署版同口径；否则 backtest 默认 trend_ma=None，
# WF 评估的就成了“无趋势过滤”策略族，样本外衰减率无法挂到实际部署的策略上。
# 基线用 risk_adj=False，与 momentum_core.RISK_ADJ 当前默认一致。
WF_GRID = [
    dict(lookbacks=(21, 63, 126), top_n=3, trend_cut=0.5, risk_adj=False, trend_ma=TREND_MA),  # 部署默认
    dict(lookbacks=(21, 63, 126), top_n=3, trend_cut=0.5, risk_adj=True,  trend_ma=TREND_MA),  # 开风险调整对照
    dict(lookbacks=(126,),        top_n=3, trend_cut=0.5, risk_adj=False, trend_ma=TREND_MA),  # 单窗口
    dict(lookbacks=(63, 126, 252),top_n=3, trend_cut=0.5, risk_adj=False, trend_ma=TREND_MA),  # 更长混合(3/6/12月)
    dict(lookbacks=(21, 63, 126), top_n=2, trend_cut=0.5, risk_adj=False, trend_ma=TREND_MA),  # 更集中
    dict(lookbacks=(21, 63, 126), top_n=3, trend_cut=0.3, risk_adj=False, trend_ma=TREND_MA),  # 轻择时
    dict(lookbacks=(21, 63, 126), top_n=3, trend_cut=0.7, risk_adj=False, trend_ma=TREND_MA),  # 重择时
    dict(lookbacks=(21, 63, 126), top_n=3, trend_cut=0.5, risk_adj=False, trend_ma=None),      # 关趋势过滤对照
]


def _near_trading_day(dates, target):
    """把 target 日期对齐到 dates 中 ≤ target 的最后一个交易日；越界返回 None。"""
    sub = dates[dates <= target]
    return sub[-1] if len(sub) else None


def walk_forward(px, train_years=5, test_years=1, grid=None, metric="夏普"):
    """
    滚动样本外：每个测试窗口前用其专属"训练窗口"在 grid 上按 metric 选最优参数，
    冻结参数在测试窗口上跑；拼接所有测试窗口的日收益得到"样本外净值"。再对比
    "全样本内 grid 最优"——样本外相对全样本内的夏普衰减，就是过拟合的直接量化：
    衰减越多，回测越不可信。

    性能：每个 grid 参数只在全期上回测一次（cached），训练/测试段都从这份结果切片，
    所以总成本 ≈ len(grid) 次回测，而非 (段数 × grid) 次。

    返回 (oos_nav, oos_daily, report, full_best, oos_metrics)：
      report    : [(测试起, 测试止, 选中参数, 训练 metric, 测试 metric), ...]
      full_best : (metric 值, 参数, 全样本内 perf) —— 数据窥探上限
    """
    if grid is None:
        grid = WF_GRID
    dates = px.index
    start = dates[MAX_LOOKBACK]
    end = dates[-1]

    # 生成滚动 (train_start, train_end, test_start, test_end)。步长 = test_years：窗口整体每年
    # 前进一段、训练/测试逐年滑动重叠。切勿写成 t0=test_start——那样步长会变成 train_years，
    # 12 年数据只能切出 2 段测试，衰减率失去意义（曾因此误报“衰减 1.35”）。
    windows = []
    t0 = start
    while True:
        train_end = _near_trading_day(dates, t0 + pd.DateOffset(years=train_years))
        test_start = _near_trading_day(dates, t0 + pd.DateOffset(years=train_years)
                                       + pd.Timedelta(days=1))
        test_end = _near_trading_day(dates, t0 + pd.DateOffset(years=train_years + test_years))
        if test_start is None or test_end is None or test_start >= end:
            break
        windows.append((t0, train_end, test_start, test_end))
        t0 = t0 + pd.DateOffset(years=test_years)   # 整体前进 test_years（重叠滚动）

    # 每个 grid 参数全期回测一次并缓存（nav 已从 1.0 归一化；daily 是日净收益）
    cached = [(p, ) + backtest(px, **p)[:2] for p in grid]   # [(params, nav, daily), ...]

    oos_pieces, report = [], []
    for tr_s, tr_e, te_s, te_e in windows:
        # 训练段：按 metric 选最优参数
        best = None
        for params, nav, daily in cached:
            score = _seg_perf(daily.loc[tr_s:tr_e])[metric]
            if best is None or score > best[0]:
                best = (score, params, daily)
        tr_score, best_params, best_daily = best
        # 测试段：冻结参数，切测试段评估
        te_score = _seg_perf(best_daily.loc[te_s:te_e])[metric]
        oos_pieces.append(best_daily.loc[te_s:te_e])
        report.append((te_s, te_e, best_params, tr_score, te_score))

    oos_daily = pd.concat(oos_pieces)
    oos_nav = (1 + oos_daily).cumprod()
    oos_nav = oos_nav / oos_nav.iloc[0]
    oos_metrics = perf(oos_nav, oos_daily)

    # 全样本内 grid 最优（数据窥探上限）
    full_best = None
    for params, nav, daily in cached:
        fp = perf(nav, daily)
        if full_best is None or fp[metric] > full_best[0]:
            full_best = (fp[metric], params, fp)
    return oos_nav, oos_daily, report, full_best, oos_metrics


def _wf_report(oos_nav, oos_daily, report, full_best, oos_metrics):
    fb_score, fb_params, fb_perf = full_best
    print("Walk-forward 滚动样本外（训练 5 年选参 / 测试 1 年 / 滚动）:\n")
    print(f"{'测试期':<24}{'训练夏普':>9}{'测试夏普':>9}  选中参数")
    for te_s, te_e, params, tr_score, te_score in report:
        period = f"{te_s.date()}~{te_e.date()}"
        ps = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"{period:<24}{tr_score:>9.2f}{te_score:>9.2f}  {ps}")
    print(f"\n全样本内 grid 最优（数据窥探上限）: {fb_params}")
    print(f"  → 年化 {fb_perf['年化']*100:.1f}%  回撤 {fb_perf['回撤']*100:.1f}%  夏普 {fb_perf['夏普']:.2f}")
    print(f"\n样本外（拼接所有测试段，{oos_metrics['年数']:.1f} 年）:")
    print(f"  → 年化 {oos_metrics['年化']*100:.1f}%  回撤 {oos_metrics['回撤']*100:.1f}%  夏普 {oos_metrics['夏普']:.2f}")
    decay = oos_metrics['夏普'] / fb_perf['夏普'] if fb_perf['夏普'] > 0 else float("nan")
    print(f"\n夏普衰减 = 样本外 / 全样本内 = {decay:.2f}")
    print("  （≥0.6 算扛得住过拟合；<0.4 说明回测明显拟合到样本内）")


# ---------------- pcv：purged K-fold 交叉验证（G2，López de Prado AFML）----------------
def purged_cv(px, n_splits=6, label_horizon=21, embargo=21, grid=None, metric="夏普"):
    """Combinatorial Purged K-Fold（López de Prado《AFML》）：时间均分 n_splits 折，
    轮流每折作 test、其余作 train，但
      purge:   从 train 剔除 test 边界 ±label_horizon 的样本（动量隐含标签是未来收益，
               train 样本若跨入 test 期即泄漏）
      embargo: test 折后再剔除 embargo 天（防 test 信号经滚动窗口泄漏到后续 train）
    train 在 grid 上选参、冻结到 test 评估；拼接所有 test 段得组合样本外净值。
    与 wf 互补：wf 是重叠滚动单序列，本方法是非重叠 K-fold + purge/embargo（AFDL 标准），
    给"样本外衰减"一个更严、多折交叉的估计。返回 (oos_nav, oos_daily, report, full_best, oos_metrics)。"""
    if grid is None:
        grid = WF_GRID
    dates = px.index
    start = dates[MAX_LOOKBACK]
    end = dates[-1]
    edges = pd.date_range(start, end, periods=n_splits + 1)         # 时间均分 n_splits 折（非重叠）
    folds = [(edges[i], edges[i + 1]) for i in range(n_splits)]
    cached = [(p, ) + backtest(px, **p)[:2] for p in grid]          # 复用 wf 的全期缓存（成本≈len(grid)）
    oos_pieces, report = [], []
    for te_s, te_e in folds:
        pur_lo = te_s - pd.Timedelta(days=label_horizon)            # purge 下界
        pur_hi = te_e + pd.Timedelta(days=embargo)                  # embargo 上界
        def _train_part(daily):
            m = (daily.index < pur_lo) | (daily.index > pur_hi)     # 全期 − test − purge/embargo
            return daily[m]
        best = None
        for params, nav, daily in cached:
            sc = _seg_perf(_train_part(daily))[metric]
            if best is None or sc > best[0]:
                best = (sc, params, daily)
        tr_score, best_params, best_daily = best
        te_daily = best_daily.loc[te_s:te_e]
        te_score = _seg_perf(te_daily)[metric]
        oos_pieces.append(te_daily)
        report.append((te_s, te_e, best_params, tr_score, te_score))
    oos_daily = pd.concat(oos_pieces)
    oos_nav = (1 + oos_daily).cumprod()
    oos_nav = oos_nav / oos_nav.iloc[0]
    oos_metrics = perf(oos_nav, oos_daily)
    full_best = None                                                 # 全样本 grid 最优（数据窥探上限）
    for params, nav, daily in cached:
        fp = perf(nav, daily)
        if full_best is None or fp[metric] > full_best[0]:
            full_best = (fp[metric], params, fp)
    return oos_nav, oos_daily, report, full_best, oos_metrics


def run_pcv(px):
    """G2 子命令：跑 purged_cv 并打印各折 + 衰减。"""
    oos_nav, oos_daily, report, full_best, oos_metrics = purged_cv(px)
    fb_score, fb_params, fb_perf = full_best
    print(f"Purged K-Fold CV（{len(report)} 折，purge 21 日 + embargo 21 日，López de Prado AFML）：\n")
    print(f"{'测试折':<24}{'训练夏普':>9}{'测试夏普':>9}  选中参数")
    for te_s, te_e, params, tr_score, te_score in report:
        period = f"{te_s.date()}~{te_e.date()}"
        ps = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"{period:<24}{tr_score:>9.2f}{te_score:>9.2f}  {ps}")
    print(f"\n全样本内 grid 最优（数据窥探上限）: {fb_params}")
    print(f"  → 年化 {fb_perf['年化']*100:.1f}%  夏普 {fb_perf['夏普']:.2f}")
    print(f"\n样本外（拼接 {len(report)} 个 purged 测试折，{oos_metrics['年数']:.1f} 年）:")
    print(f"  → 年化 {oos_metrics['年化']*100:.1f}%  回撤 {oos_metrics['回撤']*100:.1f}%  夏普 {oos_metrics['夏普']:.2f}")
    decay = oos_metrics['夏普'] / fb_perf['夏普'] if fb_perf['夏普'] > 0 else float("nan")
    print(f"\n夏普衰减 = 样本外 / 全样本内 = {decay:.2f}")
    print("  （≥0.6 扛得住过拟合；<0.4 明显拟合到样本内。purge+embargo 比 wf 更保守严谨）")


# ---------------- pbo：backtest 过拟合概率（G3，Bailey-López de Prado 2017）----------------
def pbo(px, n_splits=6, label_horizon=21, embargo=21, grid=None, metric="夏普"):
    """G3 Probability of Backtest Overfitting（Bailey-López de Prado 2017，简化排名法）：
    purged K-fold 每折：train 选 ISC（样本内）最优参数、看该参数在该折 test 段的 OOS 排名
    （0=最好）。PBO = 最优参数在 test 排名落下半（≥N/2）的折占比。>0.5 = 过拟合
    （样本内最优→样本外系统性差）；<0.5 = 未过拟合。返回 (pbo值, 各折排名列表)。"""
    if grid is None:
        grid = WF_GRID
    dates = px.index
    start = dates[MAX_LOOKBACK]
    end = dates[-1]
    edges = pd.date_range(start, end, periods=n_splits + 1)
    folds = [(edges[i], edges[i + 1]) for i in range(n_splits)]
    cached = [(p, ) + backtest(px, **p)[:2] for p in grid]
    N = len(cached)
    half = N // 2
    ranks = []
    for te_s, te_e in folds:
        pur_lo = te_s - pd.Timedelta(days=label_horizon)
        pur_hi = te_e + pd.Timedelta(days=embargo)
        def _train(daily):
            m = (daily.index < pur_lo) | (daily.index > pur_hi)
            return daily[m]
        tr = [_seg_perf(_train(daily))[metric] for _, _, daily in cached]
        best = int(np.argmax(tr))
        te = [_seg_perf(daily.loc[te_s:te_e])[metric] for _, _, daily in cached]
        order = sorted(range(N), key=lambda i: te[i], reverse=True)
        ranks.append(order.index(best))
    pbo_val = sum(1 for r in ranks if r >= half) / len(ranks)
    return pbo_val, ranks


def run_pbo(px):
    """G3 子命令：跑 pbo 并打印。"""
    pbo_val, ranks = pbo(px)
    N = len(WF_GRID)
    print(f"PBO 过拟合概率（{len(ranks)} 折 purged K-fold + 排名法，Bailey-LdP 2017）：")
    print(f"  各折 ISC 最优参数在 test 的排名（0=最好，共 {N} 个候选）: {ranks}")
    print(f"  落下半（≥{N//2}）的折数: {sum(1 for r in ranks if r >= N//2)}/{len(ranks)}")
    print(f"  PBO = {pbo_val:.2f}")
    print(f"  → {'未过拟合（<0.5）：样本内最优在样本外非系统性垫底' if pbo_val < 0.5 else '过拟合警告（≥0.5）：样本内最优→样本外系统性差'}")
    print("  （与 G1 DSR、wf 衰减、pcv 四重过拟合检验互补）")


# ---------------- boot：RISK_ADJ 提升的配对 block bootstrap ----------------
def _circ_block_idx(T, block, rng):
    """circular block bootstrap 索引：把序列当成环，随机起点取连续 block 个，
    拼到长度 T。保留时序自相关（朴素重抽样会破坏收益序列的自相关结构）。"""
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
    return idx[:T]


def _ann_sharpe(x):
    """日收益序列 → 年化夏普（无风险利率 0）；波动为 0 返回 0。block bootstrap 复用。"""
    sd = x.std(ddof=1)
    return (x.mean() / (sd + 1e-12)) * np.sqrt(252) if sd > 0 else 0.0


def bootstrap_risk_adj(px, n_boot=2000, block=21, seed=7):
    """
    检验"风险调整动量"带来的夏普提升是否统计显著。risk_adj=True / False 共享除该开关
    外的一切，两条日收益高度相关，故对配对 (d1, d0) 做 circular block bootstrap
    （block≈1 个月保留自相关）。Δ = SR(d1) − SR(d0)；p = 重抽样里 Δ≤0 的比例
    （单边，越小越显著）。返回 (delta_obs, boot_mean, ci_lo, ci_hi, p)。
    """
    _, d1, _, _ = backtest(px, risk_adj=True)
    _, d0, _, _ = backtest(px, risk_adj=False)
    common = d1.index.intersection(d0.index)
    a1 = d1.loc[common].to_numpy()
    a0 = d0.loc[common].to_numpy()
    T = len(a1)
    delta_obs = _ann_sharpe(a1) - _ann_sharpe(a0)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _circ_block_idx(T, block, rng)
        boots[b] = _ann_sharpe(a1[idx]) - _ann_sharpe(a0[idx])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return delta_obs, float(boots.mean()), float(ci_lo), float(ci_hi), float((boots <= 0).mean())


def bootstrap_mom_weights(px, weights, n_boot=2000, block=21, seed=7):
    """
    检验"混合动量各窗口加权（A2）"相对等权的夏普提升是否统计显著（改进项 A2）。
    weights 与 LOOKBACKS 对齐；mom_weights=weights 与 =None（等权）共享其余一切、且都用
    部署口径（vol_target + 趋势过滤），两条日收益高度相关，故配对 (dw, deq) 做 circular
    block bootstrap。Δ = SR(dw) − SR(deq)；p = 重抽样里 Δ≤0 的比例（单边，越小越显著）。
    返回 (delta_obs, boot_mean, ci_lo, ci_hi, p)。
    注意：若 weights 是从多个候选里挑出的"最优"，p 值偏乐观（多重比较），判读需打折。
    """
    cfg = dict(trend_ma=TREND_MA)                  # 与部署口径一致（vol_target 取默认）
    _, dw, _, _ = backtest(px, mom_weights=weights, **cfg)
    _, deq, _, _ = backtest(px, mom_weights=None, **cfg)
    common = dw.index.intersection(deq.index)
    aw = dw.loc[common].to_numpy()
    aeq = deq.loc[common].to_numpy()
    T = len(aw)
    delta_obs = _ann_sharpe(aw) - _ann_sharpe(aeq)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _circ_block_idx(T, block, rng)
        boots[b] = _ann_sharpe(aw[idx]) - _ann_sharpe(aeq[idx])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return delta_obs, float(boots.mean()), float(ci_lo), float(ci_hi), float((boots <= 0).mean())


def bootstrap_selection(px, n_boot=2000, block=21, seed=7, extra_cfg=None):
    """J2 检验:动量选股(top_n 排序 + 绝对动量)相对"等权全池不选股"的夏普提升是否显著。
    两条共享 vol_target + 趋势过滤(部署口径),唯一差别是 hold_all——选股版挑 top_n、对照版等权全池。
    高度相关,故对配对 (d_sel, d_hold) 做 circular block bootstrap(block≈1 月保留自相关)。
    Δ = SR(选股) − SR(等权全池);p = 重抽样里 Δ≤0 的比例(单边,越小越显著)。
    extra_cfg: 额外 backtest 配置(跨市场用,如 trend_code="SPY"),并入部署口径。默认 None=A 股口径。
    返回 (delta_obs, boot_mean, ci_lo, ci_hi, p)。"""
    cfg = dict(trend_ma=TREND_MA)                       # 部署口径(vol_target 取默认)
    if extra_cfg:
        cfg.update(extra_cfg)
    _, d_sel, _, _ = backtest(px, hold_all=False, **cfg)
    _, d_hold, _, _ = backtest(px, hold_all=True, **cfg)
    common = d_sel.index.intersection(d_hold.index)
    a1 = d_sel.loc[common].to_numpy()
    a0 = d_hold.loc[common].to_numpy()
    T = len(a1)
    delta_obs = _ann_sharpe(a1) - _ann_sharpe(a0)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _circ_block_idx(T, block, rng)
        boots[b] = _ann_sharpe(a1[idx]) - _ann_sharpe(a0[idx])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return delta_obs, float(boots.mean()), float(ci_lo), float(ci_hi), float((boots <= 0).mean())


def _boot_report(res, title="RISK_ADJ（风险调整动量）夏普提升", on="risk_adj=True", off="False"):
    """配对 block bootstrap 显著性检验的通用报告（默认措辞对应 RISK_ADJ）。
    res = (delta_obs, boot_mean, ci_lo, ci_hi, p)；on/off 是被比较两侧的配置描述。"""
    delta_obs, mean, lo, hi, p = res
    print(f"{title} 的 bootstrap 显著性检验:\n")
    print(f"  观测夏普差 Δ = SR({on}) − SR({off}) = {delta_obs:+.4f}")
    print(f"  配对 block bootstrap（n=2000, block=21 日）:")
    print(f"    均值 {mean:+.4f}    95% CI [{lo:+.4f}, {hi:+.4f}]    P(Δ≤0) = {p:.3f}")
    if lo > 0:
        verdict = "显著为正（CI 整体 > 0）→ 提升可信，可考虑启用"
    elif hi < 0:
        verdict = "显著为负 → 反而拖累，建议关闭"
    else:
        verdict = ("不显著（CI 跨 0）→ 观测到的夏普差更可能是噪声，\n        "
                   "建议保持默认（省一个过拟合自由度）")
    print(f"\n  结论：{verdict}")


def run_bootmom(px):
    """A2 混合动量加权：先 A/B 几种单调加权方案（长窗口更高权）看指标景观，再对主候选
    （权重∝窗口长度，无额外可调参数）做配对 block bootstrap 显著性检验。
    多重比较提醒：主候选从一个小集合中选出，bootstrap p 值偏乐观，判读需打折。"""
    cands = [("等权(基线)", None),
             ("(1,2,3) 线性递增", (1, 2, 3)),
             ("(1,2,4) 半衰期式", (1, 2, 4)),
             ("(21,63,126) ∝长度", (21, 63, 126)),
             ("(1,1,3) 仅抬长窗", (1, 1, 3))]
    rows = []
    for name, w in cands:
        nav, daily, _, _ = backtest(px, mom_weights=w, trend_ma=TREND_MA)
        rows.append((name, perf(nav, daily)))
    print("A2 混合动量加权 A/B（默认全风控：vol_target + 趋势过滤）:\n")
    _print_table(rows)

    primary = (21, 63, 126)     # 权重∝窗口长度：无额外可调参数，最可辩护的"长窗口更高权"方案
    print("\n" + "=" * 64)
    _boot_report(bootstrap_mom_weights(px, primary),
                 title="A2 混合动量加权（权重∝窗口长度 vs 等权）",
                 on="mom_weights=(21,63,126)", off="等权")
    print("  注：主候选从上述 5 元小集合中选出，p 值含多重比较偏误（偏乐观），判读需打折。")


# ---------------- dsr：Deflated Sharpe Ratio（多重比较下的夏普可信度，Bailey-López de Prado 2014）----------------
def _norm_ppf(p):
    """标准正态分位数 Φ⁻¹(p)（Acklam 算法 + 一次 Halley 精修，精度 ~1e-9）。纯 Python，免 scipy。p∈(0,1)。"""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p <= 0.0 or p >= 1.0:
        return float("nan")
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= phigh:
        q = p - 0.5; r = q*q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p        # Halley 一步精修
    u = e * math.sqrt(2 * math.pi) * math.exp(x*x/2)
    return x - u / (1 + x*u/2)


def _norm_cdf(z):
    """标准正态累积分布 Φ(z)（erfc 实现，满精度）。"""
    return 0.5 * math.erfc(-z / math.sqrt(2))


def deflated_sharpe(daily, n_trials, periods=252):
    """Deflated Sharpe Ratio（Bailey & López de Prado, 2014，改进项 G1）：把观测夏普与
    "N 次独立试验的期望最高夏普"比较，得到"观测夏普不只是多重比较运气的概率"。DSR>0.95 才算经得起。

    口径（全用每期/日量，自洽）：
      n      = 日观测数；SR̂ = 每期(日)夏普 = mean/std
      γ₃     = 日收益偏度；γ₄ = 日收益普通峰度(Pearson，正态=3)
      σ(SR̂) = √[(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²)/(n−1)]   （Mertens/Lo 非正态修正，肥尾左偏会放大 σ）
      SR_max = σ · [(1−γ_emc)·Φ⁻¹(1−1/N) + γ_emc·Φ⁻¹(1−1/(N·e))]   （N 次试验期望最高，γ_emc=Euler-Mascheroni≈0.5772）
      DSR    = Φ((SR̂ − SR_max)/σ)；PSR0 = Φ(SR̂/σ)（基准=0，即"真实夏普>0 的概率"）
    返回 dict。注：N 是"等效独立试验数"的保守估计——真实研究里多数试验因不显著被弃、未真正
    用于选参，故真实 N 更小、真实 DSR 更高（此处 N 偏大 → DSR 偏保守，是存疑方向上的下界）。"""
    r = daily.values if hasattr(daily, "values") else list(daily)
    n = len(r)
    mean = sum(r) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (n - 1))
    sr = mean / sd if sd > 0 else 0.0                       # 每期(日)夏普
    g3 = sum((x - mean) ** 3 for x in r) / (n - 1) / sd**3   # 偏度
    g4 = sum((x - mean) ** 4 for x in r) / (n - 1) / sd**4   # Pearson 峰度（正态=3）
    var_sr = (1 - g3*sr + (g4 - 1)/4 * sr**2) / (n - 1)
    sig = math.sqrt(var_sr) if var_sr > 0 else 1e-12        # SR̂ 标准误(每期)
    emc = 0.5772156649015329                                # Euler-Mascheroni
    sr_max = sig * ((1 - emc)*_norm_ppf(1 - 1/n_trials) + emc*_norm_ppf(1 - 1/(n_trials*math.e)))
    return dict(SR_ann=sr*math.sqrt(periods), SR=sr, n=n, skew=g3, kurt=g4,
                sig_ann=sig*math.sqrt(periods), SR_max_ann=sr_max*math.sqrt(periods),
                PSR0=_norm_cdf(sr/sig), DSR=_norm_cdf((sr - sr_max)/sig), N=n_trials)


def _monthly_returns(daily):
    """日收益 → 月收益(月内复利):月末净值环比。月频更接近 i.i.d.,n 从 ~3225 日降到 ~152 月。
    用于 J3:DSR/bootstrap 用日频会夸大自由度(日收益自相关 + edge 集中在少数熊市),月频是更诚实的口径。"""
    nav = (1 + daily).cumprod()
    return nav.resample("ME").last().pct_change().dropna()


def run_dsr(px):
    """对部署策略(等权 + 全风控)与 C1 反向波动版算 Deflated Sharpe Ratio。
    **J3**:同时报日频(n≈3225,自由度被夸大、乐观上界)与月频(n≈152,月收益近 i.i.d.、可信下界)两种口径,
    让"夏普可信度"不被日收益的伪独立观测抬高。N 取 [10,50,100] 多重比较敏感性。"""
    _, ret_tr, _, _ = backtest(px, vol_target=VOL_TARGET, trend_ma=TREND_MA)
    _, ret_iv, _, _ = backtest(px, vol_target=VOL_TARGET, trend_ma=TREND_MA, weighting="inv_vol")
    pairs = [("等权+全风控", ret_tr), ("反向波动+全风控", ret_iv)]
    print("Deflated Sharpe Ratio(Bailey-López de Prado 2014)— 多重比较下的夏普可信度:\n")
    for freq, periods, tag in [("日频", 252, "n≈3225,自由度被夸大 → 乐观上界"),
                               ("月频", 12, "n≈152,月收益近 i.i.d. → J3 可信下界")]:
        print(f"--- {freq}口径({tag})---")
        print(f"{'策略':<20}{'观测夏普':>9}{'N(试验)':>9}{'SR_max':>8}{'DSR':>8}{'PSR(>0)':>9}{'判读':>8}")
        for name, daily in pairs:
            rs = _monthly_returns(daily) if freq == "月频" else daily
            for N in (10, 50, 100):
                r = deflated_sharpe(rs, N, periods=periods)
                verdict = "可信" if r["DSR"] > 0.95 else ("边际" if r["DSR"] > 0.90 else "不足")
                print(f"{name:<20}{r['SR_ann']:>9.3f}{N:>9}{r['SR_max_ann']:>8.3f}"
                      f"{r['DSR']:>8.3f}{r['PSR0']:>9.3f}{verdict:>8}")
        print()
    # J3 直接对比:同一策略日频 vs 月频 DSR
    d = deflated_sharpe(ret_tr, 50)["DSR"]
    m = deflated_sharpe(_monthly_returns(ret_tr), 50, periods=12)["DSR"]
    print(f"  [J3 对比] 等权策略 DSR@N=50:日频 {d:.3f} → 月频 {m:.3f} ({m-d:+.3f})")
    print("  解读:日收益自相关 + edge 集中在少数熊市 regime,日频 n≈3225 高估了独立观测数;")
    print("        月频 n≈152 不依赖'日收益独立'假设,是更诚实的可信度。两口径都 >0.95 才算硬。")
    print("        N(试验)越小扣减越轻;真实独立试验数远小于 100(多数改进验证后被弃),真实 DSR 高于 N=100 列。")


# ---------------- universe：标的池消融（alpha 对池子构成的依赖）----------------
def _with_pool(sub_dict, fn):
    """临时把 momentum_core.POOL 和本模块 POOL 都换成 sub_dict、跑完 fn() 恢复。
    backtest 与 decide_targets 内部都按各自模块的全局 POOL 遍历标的，故两处都要换。"""
    import momentum_core as mc
    g = globals()
    saved = (g["POOL"], mc.POOL)
    g["POOL"] = sub_dict
    mc.POOL = sub_dict
    try:
        return fn()
    finally:
        g["POOL"], mc.POOL = saved


def run_universe(px, drop=("518880", "513100")):
    """
    universe 消融：对比“默认池”与“踢掉指定顺风资产”的指标，量化策略 alpha 对池子构成的依赖
    （selection / universe bias）——这是 robust / wf 都查不到的一类过拟合（它们扫描的全是
    固定池子）。黄金(518880)、纳指(513100) 恰是 2014-2025 最强的两类趋势资产，踢掉它们能
    直接量出“回测好看有多少来自池子里恰好放了事后赢家”。
    """
    full = dict(POOL)
    sub = {c: n for c, n in full.items() if c not in drop}
    px_sub = px[list(sub) + [DEFENSE[0]]]
    print(f"universe 消融：踢掉 {[full[c] for c in drop]}，池中剩 {len(sub)} 只\n")
    rows = []
    nav, daily, _, _ = backtest(px, trend_ma=TREND_MA)
    rows.append((f"默认池(全{len(full)}只)", perf(nav, daily)))
    nav2, daily2, _, _ = _with_pool(sub, lambda: backtest(px_sub, trend_ma=TREND_MA))
    rows.append(("踢掉黄金+纳指", perf(nav2, daily2)))
    bn = bench_nav(px)
    rows.append(("买入持有沪深300", perf(bn, bn.pct_change().fillna(0))))
    bn6040 = bench_6040(px)
    rows.append(("股债60-40(被动)", perf(bn6040, bn6040.pct_change().fillna(0))))
    _print_table(rows)
    d_sharpe = rows[0][1]["夏普"] - rows[1][1]["夏普"]
    d_cagr = (rows[0][1]["年化"] - rows[1][1]["年化"]) * 100
    print(f"\n解读：踢掉这两只，夏普降 {d_sharpe:.2f}、年化降 {d_cagr:.1f} 个百分点。"
          f"\n      这部分就是 alpha 对“池子里恰好有顺风资产”的依赖。"
          f"\n      但踢掉后仍明显跑赢基准（沪深300 夏普 {rows[2][1]['夏普']:.2f}），"
          f"说明动量逻辑本身有 alpha，不是纯靠池子。")


def run_nomomentum(px, lim=None):
    """J2 子命令:动量选股的真实边际。对照 = 等权持有全池 + vol_target + 趋势过滤(不选股、
    无绝对动量),与部署策略同口径、同执行(T+1 + 涨跌停)、同一时点候选集(只差"选不选")。
    若对照夏普≈策略 → "轮动选股"近乎无价值、策略本质是"波动管理+趋势择时的防御性 beta"。
    配对 block bootstrap 检验选股边际的统计显著性(与 C1/C2/A2 同口径)。"""
    lim = lim or {}
    cfg = dict(trend_ma=TREND_MA)                       # 部署口径:vol_target 默认 + 趋势过滤
    nav_sel, ret_sel, n_sel, _ = backtest(px, hold_all=False, **cfg, **lim)
    nav_hold, ret_hold, _, _ = backtest(px, hold_all=True, **cfg, **lim)
    bn = bench_nav(px)
    p_sel = perf(nav_sel, ret_sel)
    p_hold = perf(nav_hold, ret_hold)
    pb = perf(bn, bn.pct_change().fillna(0))

    print("=" * 72)
    print("J2 动量选股的真实边际（等权全池+风控 不选股  vs  部署动量轮动）")
    print("=" * 72)
    print(f"  回测区间: {nav_sel.index[0].date()} ~ {nav_sel.index[-1].date()}  "
          f"({p_sel['年数']:.1f} 年)  调仓 {n_sel} 次\n")
    _print_table([
        ("等权全池+vol+trend(不选股)", p_hold),
        ("动量轮动top3+vol+trend(部署)", p_sel),
        ("买入持有沪深300", pb),
    ])
    d_sharpe = p_sel["夏普"] - p_hold["夏普"]
    d_cagr = (p_sel["年化"] - p_hold["年化"]) * 100
    print(f"\n  [选股边际] 夏普 {p_hold['夏普']:.2f}→{p_sel['夏普']:.2f} ({d_sharpe:+.2f})  "
          f"年化 {p_hold['年化']*100:.1f}%→{p_sel['年化']*100:.1f}% ({d_cagr:+.1f}pp)  "
          f"回撤 {p_hold['回撤']*100:.1f}%→{p_sel['回撤']*100:.1f}%")
    if abs(d_sharpe) < 0.05:
        verdict = ("→ 选股边际≈0:轮动选股近乎无价值,策略本质是\n        "
                   "'波动管理+趋势择时的防御性 beta'(J2 假设成立)")
    elif d_sharpe > 0:
        verdict = f"→ 选股有正贡献(夏普 +{d_sharpe:.2f}),显著性见下方 bootstrap"
    else:
        verdict = f"→ 选股反而拖累(夏普 {d_sharpe:+.2f}),轮动不如等权全池"
    print(f"  {verdict}")

    print("\n" + "=" * 72)
    _boot_report(bootstrap_selection(px),
                 title="J2 动量选股 vs 等权全池(均 vol_target + 趋势过滤)",
                 on="动量选股", off="等权全池")
    print("  注:对照剔除了相对动量(选 top_n)与绝对动量(≤0 切防守)两套信号;若 Δ不显著,")
    print("      说明部署策略相对'等权全池+风控'无可靠选股 alpha,改进方向应转向'持有什么'而非'选谁'。")


def _synth_defense_series(idx, r_ann, anchor):
    """合成"年化 r_ann"的防守资产收盘序列(确定性日复利),用于 J4 债牛敏感性。
    刻意不含波动——J4 要隔离的是'防守资产的漂移'对策略的影响(防守本就低波,波动非主线)。"""
    r_d = (1.0 + r_ann) ** (1.0 / 252) - 1.0
    return pd.Series([anchor * (1.0 + r_d) ** i for i in range(len(idx))], index=idx)


def run_bondstress(px, lim=None):
    """J4:防守资产(国债 511010)吃了 2013–26 十年债牛,策略的 Calmar / 回撤控制有多依赖这个顺风?
    把防守资产收益替换成 0%/年(平价,剔除债牛)与 −2%/年(加息/熊债)重算,对比真实债牛口径。
    回答两件事:① 回撤控制机制是否依赖债牛——若 0% 防守下回撤仍浅,说明机制靠的是'熊市挪进不跌的
    防守'(防守不跌即可),与防守资产涨不涨无关;② 收益/Calmar 水平有多少是债牛顺风——三者差值即债牛贡献。"""
    lim = lim or {}
    dcode = DEFENSE[0]
    d = px[dcode].dropna()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    def_cagr = (d.iloc[-1] / d.iloc[0]) ** (1 / yrs) - 1
    print("=" * 72)
    print("J4 防守资产债牛敏感性(国债 511010 收益替换:真实 vs 0%/年 vs −2%/年)")
    print("=" * 72)
    print(f"  真实国债 {def_cagr*100:+.1f}%/年({d.index[0].date()}~{d.index[-1].date()}, {yrs:.1f}年)——"
          f"{'明显债牛顺风' if def_cagr > 0.015 else '非明显债牛'}\n")
    anchor = float(d.iloc[0])
    scenarios = [("真实国债(债牛)", None),
                 ("国债=0%/年(平价)", 0.0),
                 ("国债=−2%/年(熊债)", -0.02)]
    rows = []
    for name, r_ann in scenarios:
        px_s = px.copy()
        if r_ann is not None:
            px_s[dcode] = _synth_defense_series(px.index, r_ann, anchor)
        nav, ret, _, _ = backtest(px_s, vol_target=VOL_TARGET, trend_ma=TREND_MA, **lim)
        rows.append((name, perf(nav, ret)))
    _print_table(rows)
    base, flat, bear = rows[0][1], rows[1][1], rows[2][1]
    print(f"\n  [剔除债牛(0%)后] 年化 {base['年化']*100:.1f}%→{flat['年化']*100:.1f}% "
          f"({(flat['年化']-base['年化'])*100:+.1f}pp)  夏普 {base['夏普']:.2f}→{flat['夏普']:.2f} "
          f"({flat['夏普']-base['夏普']:+.2f})  回撤 {base['回撤']*100:.1f}%→{flat['回撤']*100:.1f}%")
    print(f"  [熊债(−2%)压力]     年化 {base['年化']*100:.1f}%→{bear['年化']*100:.1f}% "
          f"({(bear['年化']-base['年化'])*100:+.1f}pp)  夏普 {base['夏普']:.2f}→{bear['夏普']:.2f} "
          f"({bear['夏普']-base['夏普']:+.2f})  回撤 {base['回撤']*100:.1f}%→{bear['回撤']*100:.1f}%")
    dd_flat = abs(flat['回撤']) - abs(base['回撤'])
    if dd_flat < 0.03:
        print(f"  → 回撤控制机制稳健:剔除债牛后最大回撤仅变化 {dd_flat*100:+.1f}pp(<3pp)——"
              f"机制靠'熊市挪进不跌的防守',与防守资产涨不涨无关")
    else:
        print(f"  → 回撤对债牛有依赖:剔除债牛后回撤变化 {dd_flat*100:+.1f}pp(>3pp)")
    print(f"  → 收益水平有 {(base['年化']-flat['年化'])*100:.1f}pp/年 是债牛顺风贡献;"
          f"熊债(−2%)再压 {(flat['年化']-bear['年化'])*100:.1f}pp/年")
    print("  注:合成防守序列为确定性日复利、无波动;真实熊债带波动(被 vol_target 部分对冲),")
    print("      此处是'防守漂移'净影响的下界估计,换池/换防守资产时同理适用。")


def run_freezetest(px, lim=None, split="2022-01-01"):
    """J5:样本外冻结期检验 + PBO 再审视。把连续跑的策略净值在某日(默认 2022-01-01)切开,
    看"近期未被全样本 A/B 调参直接优化"的冻结期里:① 部署策略是否还成立(跑赢 B&H);② 选股
    边际是否还在(动量轮动 vs 等权+风控)。回答 G3 PBO=0.67 的担忧——选参过拟合风险是否真被
    "用经验默认值"消除。连续跑后按段切指标,避免分段 warmup 边缘效应。"""
    lim = lim or {}
    split_dt = pd.Timestamp(split)
    cfg = dict(trend_ma=TREND_MA)
    nav_sel, ret_sel, _, _ = backtest(px, hold_all=False, **cfg, **lim)
    _, ret_hold, _, _ = backtest(px, hold_all=True, **cfg, **lim)
    bn = bench_nav(px); ret_bh = bn.pct_change().fillna(0)

    def seg_perf(ret, lo, hi):
        s = ret.loc[lo:hi]
        if len(s) < 60:                                  # 段太短不算
            return None
        nav = (1 + s).cumprod(); nav = nav / nav.iloc[0]
        return perf(nav, s)

    end = ret_sel.index[-1]
    segs = [("全样本", ret_sel.index[0], end),
            (f"调参期(≤{split})", ret_sel.index[0], split_dt),
            (f"冻结期(>{split})", split_dt, end)]
    print("=" * 72)
    print(f"J5 样本外冻结期检验(split={split};冻结期 ≈ {(end - split_dt).days / 365.25:.1f} 年)")
    print("=" * 72)
    rows = []
    for name, lo, hi in segs:
        for label, ret in (("·动量轮动", ret_sel), ("·等权+风控", ret_hold), ("·沪深300", ret_bh)):
            p = seg_perf(ret, lo, hi)
            if p:
                rows.append((name + label, p))
    _print_table(rows)
    oos_sel = seg_perf(ret_sel, split_dt, end)
    oos_hold = seg_perf(ret_hold, split_dt, end)
    oos_bh = seg_perf(ret_bh, split_dt, end)
    if oos_sel and oos_hold and oos_bh:
        d = oos_sel["夏普"] - oos_hold["夏普"]
        print(f"\n  [冻结期选股边际] 动量轮动 {oos_sel['夏普']:.2f} vs 等权+风控 {oos_hold['夏普']:.2f} ({d:+.2f})")
        print(f"  [冻结期策略 vs B&H] {oos_sel['夏普']:.2f} vs 沪深300 {oos_bh['夏普']:.2f} "
              f"({oos_sel['夏普'] - oos_bh['夏普']:+.2f})")
        print("\n  判读(PBO=0.67 再审视):")
        if d <= 0.05:
            print("   → 冻结期选股无优势(≤等权+风控):印证 PBO=0.67——'选股/选参'过拟合,")
            print("     '用经验默认值'没消除选择偏差(整份 backlog 就是 A/B 选参记录)。")
            print("   → 但风控(vol_target/trend)在冻结期仍让策略跑赢 B&H:过拟合风险集中在选股层,")
            print("     风控层稳健(与 WF 衰减 0.89、J4 跨 regime 一致)。落地:切 hold_all 实盘模式即可规避选股过拟合。")
        else:
            print(f"   → 冻结期选股仍有正边际({d:+.2f}):过拟合担忧在该段未显现(单段不足以下定论)。")
    print("\n  注:严格说 backlog 调参用到 2026 全样本,无真正'未见过'的持有期;此'冻结期'是近期段")
    print("      (最少被直接优化的概念持有期),是最可得的事后近似、非真样本外。真样本外须等实盘向前跑出。")


def run_review():
    """D1/D2 改完后，在新口径（T+1 成交 + 涨跌停）下复核 C1/C2 的旧结论是否仍成立。

    旧结论（coc 口径）：C2 单独有效但全风控下与 vol_target+trend 重叠（边际≈0）；
    C1 inv_vol 无可靠夏普提升、不稳健。新口径若改变这两个结论，需重新评估默认开关。"""
    px, cant_buy, cant_sell = load_real(with_limits=True)
    lim = dict(cant_buy=cant_buy, cant_sell=cant_sell)

    print("=" * 64)
    print("C2 回撤控制复核（全风控下 开/关）— 新口径 T+1 + 涨跌停")
    print("=" * 64)
    base = dict(vol_target=VOL_TARGET, trend_ma=TREND_MA, trend_cut=TREND_CUT)
    nav_on, ret_on, _, _ = backtest(px, **base, **lim, drawdown_prot=True,
                                     dd_window=DD_WINDOW, dd_thr=DD_THR, dd_cut=DD_CUT)
    nav_off, ret_off, _, _ = backtest(px, **base, **lim, drawdown_prot=False)
    p_on, p_off = perf(nav_on, ret_on), perf(nav_off, ret_off)
    _print_table([("全风控 + C2关", p_off), ("全风控 + C2开", p_on)])
    d_sh = p_on["夏普"] - p_off["夏普"]
    print(f"  夏普 {p_off['夏普']:.2f}→{p_on['夏普']:.2f} ({d_sh:+.2f})  "
          f"回撤 {p_off['回撤']*100:.1f}%→{p_on['回撤']*100:.1f}%  "
          f"年化 {p_off['年化']*100:.1f}%→{p_on['年化']*100:.1f}%")
    print(f"  → {'边际≈0、与全风控重叠，保持默认关（与旧结论一致）' if abs(d_sh) < 0.03 else '新口径下有显著差异，需重新评估'}")

    print("\n" + "=" * 64)
    print("C1 反向波动加权复核（vol_target × weighting）— 新口径")
    print("=" * 64)
    print(f"{'vol_target':<12}{'equal夏普':>10}{'inv_vol夏普':>12}{'Δ夏普':>8}"
          f"{'equal回撤':>11}{'inv_vol回撤':>12}")
    better = 0; total = 0
    for vt in (None, 0.10, 0.15):
        nav_eq, ret_eq, _, _ = backtest(px, vol_target=vt, trend_ma=TREND_MA, **lim, weighting="equal")
        nav_iv, ret_iv, _, _ = backtest(px, vol_target=vt, trend_ma=TREND_MA, **lim,
                                         weighting="inv_vol", inv_vol_window=INV_VOL_WINDOW)
        pe, pi = perf(nav_eq, ret_eq), perf(nav_iv, ret_iv)
        d = pi["夏普"] - pe["夏普"]
        print(f"{str(vt):<12}{pe['夏普']:>10.2f}{pi['夏普']:>12.2f}{d:>+8.2f}"
              f"{pe['回撤']*100:>10.1f}%{pi['回撤']*100:>11.1f}%")
        better += (d >= 0); total += 1
    print(f"\n  inv_vol 夏普≥等权: {better}/{total}")
    print(f"  → {'不稳健（与旧结论一致），保持默认 equal' if better < total else '新口径下 inv_vol 普遍占优，可考虑启用'}")


def main():
    # 读命令行参数决定运行模式：传 "sweep" 走参数扫描，不传则默认 "real" 跑回测出图。
    # sys.argv[0] 是脚本名，argv[1] 才是用户传的第一个参数，故需先判断长度防越界。
    mode = sys.argv[1] if len(sys.argv) > 1 else "real"
    if mode == "review":
        run_review()
        return
    # sweep/robust/wf/boot/universe 保持原口径（不带涨跌停过滤，与历史对照一致）；
    # real/regime 模式启用 D1/D2 新口径（T+1 成交 + 涨跌停），出图/复核/regime 拆解用。
    loaded = load_real(with_limits=(mode in ("real", "regime", "attrib", "mfat", "nomomentum", "bondstress", "freezetest")))
    if mode in ("real", "regime", "attrib", "mfat", "nomomentum", "bondstress", "freezetest"):
        px, cant_buy, cant_sell = loaded
    else:
        px, cant_buy, cant_sell = loaded, None, None

    if mode == "sweep":
        run_sweep(px)
        return
    if mode == "robust":
        run_robust(px)
        return
    if mode == "wf":
        oos_nav, oos_daily, report, full_best, oos_metrics = walk_forward(px)
        _wf_report(oos_nav, oos_daily, report, full_best, oos_metrics)
        return
    if mode == "pcv":
        run_pcv(px)
        return
    if mode == "pbo":
        run_pbo(px)
        return
    if mode == "boot":
        _boot_report(bootstrap_risk_adj(px))
        return
    if mode == "bootmom":
        run_bootmom(px)
        return
    if mode == "dsr":
        run_dsr(px)
        return
    if mode == "universe":
        run_universe(px)
        return
    if mode == "attrib":
        run_attribution(px, lim=dict(cant_buy=cant_buy, cant_sell=cant_sell))
        return
    if mode == "mfat":
        run_mfattribution(px, lim=dict(cant_buy=cant_buy, cant_sell=cant_sell))
        return
    if mode == "nomomentum":
        run_nomomentum(px, lim=dict(cant_buy=cant_buy, cant_sell=cant_sell))
        return
    if mode == "bondstress":
        run_bondstress(px, lim=dict(cant_buy=cant_buy, cant_sell=cant_sell))
        return
    if mode == "freezetest":
        run_freezetest(px, lim=dict(cant_buy=cant_buy, cant_sell=cant_sell))
        return
    if mode == "regime":
        run_regime(px, lim=dict(cant_buy=cant_buy, cant_sell=cant_sell))
        return

    # 风控逐步叠加（等权）+ C1 反向波动加权对照，全部对比基准（real 模式带 D1/D2 新口径）
    lim = dict(cant_buy=cant_buy, cant_sell=cant_sell)
    nav_base, ret_base, _, _ = backtest(px, vol_target=None, **lim)
    nav_vt, ret_vt, n_vt, _ = backtest(px, vol_target=VOL_TARGET, **lim)
    nav_tr, ret_tr, _, _ = backtest(px, vol_target=VOL_TARGET, trend_ma=TREND_MA, **lim)        # 等权（默认）
    nav_iv, ret_iv, _, log = backtest(px, vol_target=VOL_TARGET, trend_ma=TREND_MA,
                                      weighting="inv_vol", **lim)                               # C1 反向波动
    bn = bench_nav(px)

    p_base, p_vt = perf(nav_base, ret_base), perf(nav_vt, ret_vt)
    p_tr, p_iv = perf(nav_tr, ret_tr), perf(nav_iv, ret_iv)
    pb = perf(bn, bn.pct_change().fillna(0))

    print(f"回测区间: {nav_tr.index[0].date()} ~ {nav_tr.index[-1].date()}  "
          f"共{p_tr['年数']:.1f}年  调仓{n_vt}次  （真实数据）\n")
    _print_table([
        ("改进版(无回撤控制)", p_base),
        (f"改进版+波动目标{int(VOL_TARGET*100)}%", p_vt),
        (f"+大盘趋势过滤{TREND_MA}日(等权)", p_tr),
        (f"+大盘趋势过滤(反向波动C1)", p_iv),
        ("买入持有沪深300", pb),
    ])
    print(f"\n[C1 反向波动 vs 等权] 夏普 {p_tr['夏普']:.2f}→{p_iv['夏普']:.2f} "
          f"({p_iv['夏普']-p_tr['夏普']:+.2f})  年化 {p_tr['年化']*100:.1f}%→{p_iv['年化']*100:.1f}% "
          f"({(p_iv['年化']-p_tr['年化'])*100:+.1f}pp)  回撤 {p_tr['回撤']*100:.1f}%→{p_iv['回撤']*100:.1f}%")
    if log:
        print(f"最近一次调仓 {log[-1][0]}: {log[-1][1]}")

    # 画图：净值 / 水下曲线 / 滚动夏普 三子图（F2 滚动指标）
    rs_tr, _, dd_tr = rolling_metrics(ret_tr)
    rs_iv, _, dd_iv = rolling_metrics(ret_iv)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(nav_tr.index, nav_tr.values, label="等权 + TrendFilt", lw=1.8, color="#1f4e79")
    axes[0].plot(nav_iv.index, nav_iv.values, label="反向波动 + TrendFilt (C1)", lw=1.8, color="#2e7d32")
    axes[0].plot(bn.index, bn.values, label="Buy & Hold CSI300", lw=1.1, color="#c0504d", alpha=0.8)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Net Value (log)")
    axes[0].legend(loc="upper left"); axes[0].grid(alpha=0.3)
    axes[0].set_title("A-share ETF Momentum Rotation [REAL] — C1 反向波动加权 vs 等权")
    axes[1].fill_between(dd_tr.index, dd_tr.values * 100, 0, color="#1f4e79", alpha=0.25, label="等权")
    axes[1].plot(dd_iv.index, dd_iv.values * 100, color="#2e7d32", lw=1.0, label="反向波动")
    axes[1].set_ylabel("Drawdown %"); axes[1].legend(loc="lower left"); axes[1].grid(alpha=0.3)
    axes[2].plot(rs_tr.index, rs_tr.values, color="#1f4e79", lw=1.0, label="等权")
    axes[2].plot(rs_iv.index, rs_iv.values, color="#2e7d32", lw=1.0, label="反向波动")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("Rolling Sharpe (1y)"); axes[2].legend(loc="upper left"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    out = "ETF动量轮动_回测结果.png"
    fig.savefig(out, dpi=130)
    print("\n图已保存:", out)

    print("\n[F2 滚动指标(1年窗口)摘要]")
    for name, rs in (("等权", rs_tr), ("反向波动", rs_iv)):
        rs = rs.dropna()
        if len(rs):
            print(f"  {name}: 滚动夏普 均值{rs.mean():.2f} / 最差{rs.min():.2f} / "
                  f"占比>0 {(rs > 0).mean() * 100:.0f}%")

    # F3 分 regime（牛/熊/震荡）拆解：看策略在哪种行情有效、是否跑赢基准
    run_regime(px, ret_tr, "等权+趋势")
    run_regime(px, ret_iv, "反向波动+趋势")


if __name__ == "__main__":
    main()
