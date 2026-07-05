# -*- coding: utf-8 -*-
"""
动量轮动策略「核心大脑」—— 纯 Python，不依赖 backtrader / xtquant。

回测（etf_momentum.py / etf_momentum_bt.py）和实盘（etf_momentum_live.py）都调用
这里的 decide_targets()，保证“回测怎么选、实盘就怎么选”，逻辑只有一份。

策略要点（相对最初的单一 60 日动量做了三处改进）：
  1. 多周期混合动量：用 1/3/6 个月（约 21/63/126 交易日）涨幅的平均做动量分，
     单窗口噪声大、容易被一两根 K 线带偏，混合后更稳。
  2. 绝对动量过滤：某只入选标的若混合动量≤0，则该仓位切防守资产（国债），
     避免在普跌行情里硬扛 → 控制回撤。
  3. 更分散的标的池：除 A 股宽基外，加入黄金、纳指等与 A 股低相关的趋势资产，
     A 股走弱时动量可轮动到它们，而不是只能缩进国债。
"""
import math

# ---------------- 标的池（唯一来源，回测/实盘共享） ----------------
POOL = {
    "510300": "沪深300",
    "510500": "中证500",
    "159915": "创业板",
    "512100": "中证1000",
    "510880": "红利ETF",
    "518880": "黄金ETF",     # 与 A 股低相关的避险/趋势资产
    "513100": "纳指ETF",     # 海外权益，分散单一市场风险
}
DEFENSE = ("511010", "国债ETF")        # 防守资产

# ---------------- 策略参数 ----------------
LOOKBACKS = (21, 63, 126)              # 混合动量回看窗口（约 1/3/6 个月）
SKIP_RECENT = 0                        # 算动量时跳过最近 N 个交易日（21≈1个月）；避开"短期反转"噪声，0=不跳
RISK_ADJ = False                       # 风险调整动量：动量÷该标的波动，偏好"平滑上涨"而非"剧烈拉升"。
                                       #   配对 block bootstrap（n=2000）已验证其夏普提升不显著
                                       #   （Δ≈+0.04，95%CI 跨 0，P(Δ≤0)≈0.22）→ 判定为噪声而关闭，
                                       #   省一个过拟合自由度。改 True 可回退对照。
MAX_LOOKBACK = max(LOOKBACKS)          # 需要的最少历史长度（注意：跳过期 skip 会额外吃历史，见 blended_momentum）
TOP_N = 3                              # 持有动量最高的前 N 只（等权）
CASH_BUFFER = 0.99                     # 目标仓位上限（留 1% 现金，吸收手续费/滑点）
VOL_TARGET = 0.15                      # 年化目标波动；组合近期波动超此值就降风险仓（None=关闭）
VOL_WINDOW = 20                        # 估计近期波动的回看交易日
COMMISSION = 0.00025                   # 单边手续费
SLIPPAGE = 0.0005                      # 单边滑点
BENCH = "510300"                       # 基准

# ---------------- 大盘趋势过滤（择时开关，纯价格、无前视、不依赖宏观数据） ----------------
# 思路：用沪深300（大盘风向标）自身价格是否跌破长期均线判断"市场大环境"。
# 跌破=下行 → 把股票仓位整体缩小、挪进国债，避开系统性下跌。这是"用价格代替宏观因子"。
TREND_CODE = BENCH                     # 当风向标的标的（沪深300，本身就在 POOL 里，数据现成）
TREND_MA = 200                         # 趋势均线天数；价在均线下方视为下行
TREND_CUT = 0.5                        # 下行时股票仓位"保留比例"（0.5=砍半挪国债，0=清空）

# 仍向后兼容旧名字（个别脚本可能 import LOOKBACK）
LOOKBACK = MAX_LOOKBACK

# ---- 持仓加权方式（改进项 C1：反向波动加权）----
# equal  : 每只入选标的等权（原行为，默认，向后兼容，回测/实盘默认不变）
# inv_vol: 权重 ∝ 1/σ_i 后归一化——波动大的标的（创业板/中证1000）少配、波动小的多配，
#          降低高波动标的对组合风险的过度主导。σ_i 取该标的近 INV_VOL_WINDOW 日日波动。
WEIGHTING = "equal"
INV_VOL_WINDOW = 63                       # 反向波动加权估计日波动的回看窗口（约3个月，比 20 日稳）

# ---- 动量崩溃保护（改进项 A1，Daniel-Moskowitz 2013）----
# 大盘短期急剧反弹（暴跌后的暴力反转）是动量崩溃的典型前兆：前期强势标的滞涨、
# 轮动切换滞后易大亏。检测到风向标近 CRASH_LOOKBACK 日涨幅 ≥ CRASH_THR 即降股票仓。
# 与波动目标/趋势过滤正交：后者看"波动大小/长期方向"，本机制看"短期反弹速度"。
CRASH_PROT = False                        # 默认关（向后兼容）；改 True 开启
CRASH_LOOKBACK = 21                       # 检测"暴力反弹"的短期窗口（约1个月）
CRASH_THR = 0.10                          # 该窗口涨幅 ≥10% 触发降仓（A 股月涨 10% 属急涨）
CRASH_CUT = 0.5                           # 触发时股票仓保留比例（0.5=砍半挪国债）

# ---- 回撤控制（改进项 C2）----
# 波动率目标的盲区是"慢刀阴跌"（低波动但持续下跌，vol_target 不触发）。用大盘（风向标）
# 距近 DD_WINDOW 日高点的累积跌幅捕捉：回撤 ≤ DD_THR 时降股票仓挪国债。与 trend filter
# 同源（都用大盘）但信号不同——trend 看长期均线方向，drawdown 看累积跌幅深度。
DRAWDOWN_PROT = False                     # 默认关（向后兼容）；改 True 开启
DD_WINDOW = 126                           # 回撤计算窗口（约半年）
DD_THR = -0.10                            # 大盘距窗口高点跌幅 ≤ -10% 触发（深跌=慢刀阴跌累积）
DD_CUT = 0.5                              # 触发时股票仓保留比例（0.5=砍半挪国债）

# ---- 涨跌停档位（改进项 D2，回测/实盘共用）----
# A 股 ETF 二级市场涨跌停：普通 ±10%，创业板/科创板 ETF ±20%。回测里"收盘封板"当日该方向
# 无法成交（涨停买不进、跌停卖不出），需维持原仓顺延。当前 POOL 仅 159915 创业板为 20%，
# 其余均 10%；无科创板 588xxx。档位定义集中在此，三处（向量化/BT/实盘）共享，避免散落不一致。
LIMIT_TABLE = {"159915": 0.20}            # 特殊涨跌停档：创业板 ETF ±20%
LIMIT_DEFAULT = 0.10                      # 其余 ETF 默认 ±10%


def _limit(code):
    """某 ETF 的涨跌停幅度（小数，如 0.10/0.20）。LIMIT_TABLE 命中取特殊档，否则 LIMIT_DEFAULT。"""
    return LIMIT_TABLE.get(code, LIMIT_DEFAULT)


def _is_num(x):
    """是否为有效数字（排除 None 和 NaN）——价格序列里可能混入缺失值。"""
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _asset_daily_vol(arr, end_idx, window):
    """
    单只标的最近 window 个日收益的标准差（不年化，仅用于风险调整动量的分母）。
    end_idx 是"截止位置"的下标（含），向前取 window 个日收益。数据不足返回 None。
    """
    seg = arr[end_idx - window: end_idx + 1]      # window+1 个价 → window 个日收益
    if len(seg) < window + 1:
        return None
    rets = []
    for i in range(1, len(seg)):
        a, b = seg[i - 1], seg[i]
        if _is_num(a) and _is_num(b) and a:
            rets.append(b / a - 1.0)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / len(rets)
    return var ** 0.5


def blended_momentum(arr, lookbacks=LOOKBACKS, skip=SKIP_RECENT, risk_adj=RISK_ADJ):
    """
    多周期混合动量：各回看窗口涨幅的平均。历史不足任一窗口则返回 None（该标的本期不参与）。
    arr 是某只标的的收盘价序列（升序），arr[-1] 为当前价。

    skip:     跳过最近 skip 个交易日再算动量。学术界的"12-1 动量"经验——最近一个月常有
              短期反转，跳过它能让趋势信号更干净。end 改为 arr[-1-skip]，各窗口都往前挪 skip。
    risk_adj: True 时把混合动量除以该标的近期日波动，得到"风险调整动量"，
              倾向于选"涨得稳"的而非"涨得猛但很颠"的标的。
    """
    if not arr:                                   # 空序列（如新上市标的早期无数据）
        return None
    end = len(arr) - 1 - skip                     # 计算动量的"终点"下标（跳过最近 skip 日）
    if end < 0 or not _is_num(arr[end]):
        return None
    now = arr[end]
    vals = []
    for lb in lookbacks:                          # 例如 lb=21/63/126
        past_idx = end - lb                       # 终点再往前 lb 天
        if past_idx < 0:                          # 历史长度不够这个窗口（含 skip）→ 整体放弃
            return None
        past = arr[past_idx]
        if not _is_num(past) or past == 0:
            return None
        vals.append(now / past - 1.0)             # 这个窗口的涨幅
    mom = sum(vals) / len(vals)                   # 多窗口取平均 = 混合动量分
    if risk_adj:                                  # 风险调整：动量 ÷ 波动（用最长窗口的日波动做分母）
        vol = _asset_daily_vol(arr, end, max(lookbacks))
        if not vol:
            return None
        mom = mom / vol
    return mom


def _realized_vol(recent_closes, eq_codes, window):
    """估计“持有股票等权组合”近 window 日的年化波动；数据不足返回 None。"""
    # 1) 取每只持仓股票最近 window+1 个收盘价（算 window 个日收益需要多一个点）
    series = []
    for c in eq_codes:
        arr = recent_closes.get(c)
        if arr is None or len(arr) < window + 1:
            return None
        series.append(arr[-(window + 1):])
    # 2) 等权组合每日收益 = 当日各标的收益的平均
    port = []
    for t in range(1, window + 1):
        day = [s[t] / s[t - 1] - 1.0 for s in series if s[t - 1]]
        if day:
            port.append(sum(day) / len(day))
    if len(port) < 2:
        return None
    # 3) 日收益标准差 × √252 → 年化波动
    mean = sum(port) / len(port)
    var = sum((x - mean) ** 2 for x in port) / len(port)
    return (var ** 0.5) * (252 ** 0.5)


def _below_trend(recent_closes, code, ma_window):
    """
    大盘风向标（code）最新收盘价是否跌破其 ma_window 日简单均线。
    数据不足 / 有缺失时一律返回 False（保守：不触发降仓，避免误杀）。
    """
    arr = recent_closes.get(code)
    if not arr or len(arr) < ma_window:        # 历史长度还不够算这条均线
        return False
    window = arr[-ma_window:]                   # 取最近 ma_window 个收盘价
    if any(not _is_num(x) for x in window):     # 窗口内有缺失 → 保守不触发
        return False
    ma = sum(window) / ma_window                # 简单移动平均
    now = arr[-1]
    return _is_num(now) and now < ma            # 现价在均线下方 = 大盘下行


def decide_targets(recent_closes, lookbacks=LOOKBACKS, top_n=TOP_N,
                   cash_buffer=CASH_BUFFER, vol_target=VOL_TARGET,
                   vol_window=VOL_WINDOW, trend_ma=None, trend_code=TREND_CODE,
                   trend_cut=TREND_CUT, skip_recent=SKIP_RECENT, risk_adj=RISK_ADJ,
                   weighting=WEIGHTING, inv_vol_window=INV_VOL_WINDOW,
                   crash_prot=CRASH_PROT, crash_lookback=CRASH_LOOKBACK,
                   crash_thr=CRASH_THR, crash_cut=CRASH_CUT,
                   drawdown_prot=DRAWDOWN_PROT, dd_window=DD_WINDOW,
                   dd_thr=DD_THR, dd_cut=DD_CUT):
    """
    输入:
      recent_closes: {code: 收盘价序列}，按时间升序，最后一个是“当前”。
                     只需放 POOL 里的标的；历史长度不足 max(lookbacks) 的标的
                     （如新上市的）自动跳过。
      vol_target:    年化目标波动（None 关闭波动率目标）。组合近期波动高于目标时，
                     按比例缩小股票仓位、把缩出来的部分挪到防守资产，以压低回撤。
      trend_ma:      大盘趋势过滤的均线天数（None 关闭）。风向标(trend_code，默认沪深300)
                     价跌破该均线视为大盘下行，按 trend_cut 缩小股票仓、其余挪防守资产。
      skip_recent:   算动量时跳过最近 N 个交易日（21≈跳过1个月，避开短期反转），透传给 blended_momentum。
      risk_adj:      是否用风险调整动量（动量÷波动），透传给 blended_momentum。
    输出:
      target: {code: 目标权重}，键可能含防守资产 DEFENSE[0]，权重和 ≈ cash_buffer。
      picks:  [中文名, ...]  本次实际持有的标的（用于日志）。
    规则: 按混合动量排序取前 top_n；某只动量>0→持有它，≤0→该仓位切防守资产；
          最后按波动率目标对股票仓位整体缩放。
    """
    # === 第一步：算每只候选标的的混合动量分 ===
    moms = []
    for code in POOL:
        arr = recent_closes.get(code)
        if arr is None:
            continue
        m = blended_momentum(arr, lookbacks, skip=skip_recent, risk_adj=risk_adj)
        if m is not None:                          # None = 历史不足，跳过
            moms.append((m, code))

    if len(moms) < top_n:
        return {}, []          # 可选标的不足（如回测初期），空仓/保持现状

    # === 第二步：按动量从高到低取前 top_n，分配权重（等权 / 反向波动） ===
    moms.sort(key=lambda x: x[0], reverse=True)
    picks_raw = moms[:top_n]

    target, picks = {}, []
    dcode = DEFENSE[0]
    # 入选标的的目标权重 weights[code]：
    #   equal  : 每只 cash_buffer/top_n（原行为，向后兼容）
    #   inv_vol: 权重 ∝ 1/σ_i 后归一化到 cash_buffer（改进项 C1）。任一只波动数据不足
    #            → 用 1 顶替（与其它同尺度归一化，退化为等权，不报错）。
    if weighting == "inv_vol":
        raw = {}
        for _m, code in picks_raw:
            arr = recent_closes.get(code)
            end = len(arr) - 1 if arr else -1
            v = _asset_daily_vol(arr, end, inv_vol_window) if end >= 0 else None
            raw[code] = (1.0 / v) if (v and v > 0) else 1.0      # 无波动数据 → 等权兜底
        s = sum(raw.values())
        weights = {c: cash_buffer * raw[c] / s for c in raw} if s > 0 else {}
    else:                                                        # equal
        weights = {c: cash_buffer / top_n for _m, c in picks_raw}
    for mom, code in picks_raw:
        w = weights.get(code, cash_buffer / top_n)              # 兜底等权
        if mom > 0:                                             # 绝对动量为正 → 真持有该 ETF
            target[code] = target.get(code, 0.0) + w
            picks.append(POOL[code])
        else:                                                   # 动量≤0 → 这一份切防守资产（国债）
            target[dcode] = target.get(dcode, 0.0) + w
            picks.append(DEFENSE[1])

    # === 第三步：波动率目标。组合近期波动超标 → 整体缩股票仓，缩出来的挪进防守资产 ===
    eq_codes = [c for c in target if c != dcode]   # 真正持有的股票（不含国债）
    if vol_target and eq_codes:
        rv = _realized_vol(recent_closes, eq_codes, vol_window)
        if rv and rv > vol_target:                 # 只在波动超标时降仓（不加杠杆）
            scale = vol_target / rv                # 缩放系数 <1
            for c in eq_codes:
                moved = target[c] * (1 - scale)    # 缩掉的那部分权重
                target[c] *= scale
                target[dcode] = target.get(dcode, 0.0) + moved  # 转入国债

    # === 第四步：大盘趋势过滤。风向标跌破长期均线 → 再整体缩股票仓，挪进防守资产 ===
    #     与波动率目标是两套独立的"减仓"机制，可叠加：波动目标管"波动太大"，
    #     趋势过滤管"大盘方向向下"。两者都只减不加。
    if trend_ma and _below_trend(recent_closes, trend_code, trend_ma):
        eq_codes = [c for c in target if c != dcode]   # 此刻仍持有的股票（可能已被波动目标缩过）
        for c in eq_codes:
            moved = target[c] * (1 - trend_cut)        # 按"保留比例"缩仓
            target[c] *= trend_cut
            target[dcode] = target.get(dcode, 0.0) + moved

    # === 第五步：动量崩溃保护（改进项 A1，Daniel-Moskowitz 2013） ===
    #     大盘短期急剧反弹（暴跌后的暴力反转）是动量崩溃的典型前兆。风向标近
    #     crash_lookback 日涨幅 ≥ crash_thr 时，按 crash_cut 降低股票仓、挪进防守资产。
    #     与波动目标/趋势过滤正交（看的是"反弹速度"，三者只减不加、可叠加）。
    if crash_prot:
        arr = recent_closes.get(trend_code)
        if arr and len(arr) > crash_lookback + 1:
            past, now = arr[-1 - crash_lookback], arr[-1]
            if _is_num(past) and _is_num(now) and past > 0 and now / past - 1 >= crash_thr:
                for c in [c for c in target if c != dcode]:
                    moved = target[c] * (1 - crash_cut)
                    target[c] *= crash_cut
                    target[dcode] = target.get(dcode, 0.0) + moved

    # === 第六步：回撤控制（改进项 C2）。大盘距近期高点深跌 → 降仓挪国债 ===
    #     波动率目标的盲区是"慢刀阴跌"（低波动但持续下跌）。用风向标近 dd_window 日
    #     高点衡量累积跌幅，回撤 ≤ dd_thr 时按 dd_cut 降股票仓。与 trend filter 同源
    #     （都用大盘）但信号不同：trend 看长期均线方向，drawdown 看累积跌幅深度。
    if drawdown_prot:
        arr = recent_closes.get(trend_code)
        if arr and len(arr) > dd_window:
            peak = max(arr[-dd_window:]); now = arr[-1]
            if _is_num(peak) and _is_num(now) and peak > 0 and (now - peak) / peak <= dd_thr:
                for c in [c for c in target if c != dcode]:
                    moved = target[c] * (1 - dd_cut)
                    target[c] *= dd_cut
                    target[dcode] = target.get(dcode, 0.0) + moved
    return target, picks
