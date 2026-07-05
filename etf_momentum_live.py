# -*- coding: utf-8 -*-
"""
动量轮动 —— miniQMT 实盘骨架（与回测共用 momentum_core.decide_targets）。

它做的事：
  1. 取最近 N 日收盘价；
  2. 调用 decide_targets()（和 etf_momentum_bt.py 完全同一套选股逻辑）得到目标权重；
  3. 连接 miniQMT，读取账户总资产 + 当前持仓；
  4. 算出目标股数，生成「先卖后买」的调仓订单；
  5. DRY_RUN=True 时只打印计划、不下单；接通并核对无误后再改 False 真正下单。

★ 重要：
  - 需先在华泰开通 miniQMT/极简模式权限，并让 QMT 客户端在后台登录运行。
  - 运行前务必把下面的 QMT_PATH / ACCOUNT_ID 改成你自己的。
  - 真金白银交易请你本人确认后执行；建议先在模拟账户跑通。
  - 程序化交易需按交易所要求报备。

运行：
  python etf_momentum_live.py          # 默认 DRY_RUN，只打印调仓计划
"""
import os
import sys
import datetime as dt

from momentum_core import (POOL, DEFENSE, MAX_LOOKBACK, COMMISSION, SLIPPAGE,
                           TREND_MA, _limit, decide_targets)

# 取数长度要同时够"动量回看"和"大盘趋势均线"两者，取较大值（趋势用 200 日均线 > 126）
HISTORY_BARS = max(MAX_LOOKBACK, TREND_MA)

# ---------------- 你需要改的配置 ----------------
BROKER = "paper"                                 # "paper"=本地模拟账户跑通；"qmt"=接华泰 miniQMT
DRY_RUN = True                                   # 仅对 qmt 生效：True=只打印不下单；paper 始终模拟成交
LOT = 100                                        # ETF 最小交易单位（股）
REBALANCE_BAND = 0.05                             # 再平衡静默区：偏离<5%总资产的小漂移不调仓
ONCE_PER_MONTH = True                             # True=每月只调一次（适合每日定时触发）
MONTH_END_ONLY = True                             # True=只在"本月最后一个交易日"调仓，与回测月末口径对齐（force 可绕过）
# os.path.dirname(__file__) 是“本脚本所在目录”，拼上文件名 → 标记文件放在脚本旁边，
# 这样不管从哪个目录运行脚本，都能找到同一个文件。
MONTH_MARKER = os.path.join(os.path.dirname(__file__), "last_rebalance.txt")

# --- paper 模拟账户 ---
PAPER_STATE = os.path.join(os.path.dirname(__file__), "paper_account.json")  # 同理，放脚本旁边
PAPER_INIT_CASH = 1_000_000.0                    # 模拟账户初始资金

# --- qmt 真实接入（接入时改这两项） ---
QMT_PATH = r"C:\华泰证券QMT\userdata_mini"       # ← 改成你本机 miniQMT 的 userdata_mini 路径
ACCOUNT_ID = "你的资金账号"                        # ← 改成你的资金账号
SESSION_ID = int(dt.datetime.now().timestamp())  # 任意整数，连接会话号
DEMO_TOTAL = 100_000.0                            # 兜底：完全连不上时的假定总资产


# ---------------- 1) 行情：最近 N 日收盘 ----------------
def get_recent_closes(codes, n):
    """
    用 akshare 取最近 n 个交易日收盘价（独立可用，便于离线核对计划）。
    实盘可改用 xtdata.get_market_data_ex 直接从 QMT 取，少一个数据源依赖。
    """
    import akshare as ak
    # 往前多取些日历日：要 n 个“交易日”，但日历日里有周末/节假日，所以 ×3 再加余量，
    # 保证截取后够 n+1 个交易日收盘价。strftime("%Y%m%d") 把日期转成 akshare 要的 "20260628" 格式。
    start = (dt.date.today() - dt.timedelta(days=n * 3 + 30)).strftime("%Y%m%d")
    end = dt.date.today().strftime("%Y%m%d")
    out = {}
    for c in codes:
        df = ak.fund_etf_hist_em(symbol=c, period="daily",
                                 start_date=start, end_date=end, adjust="qfq")
        if len(df):
            # 取收盘列→转float→转列表，再用 [-(n+1):] 取最后 n+1 个（够算 n 日动量）
            out[c] = df["收盘"].astype(float).tolist()[-(n + 1):]
    return out


# ---------------- 2) 连接券商（paper 本地模拟 / qmt 真实） ----------------
def connect_trader():
    """返回 (kind, handle)；连不上则 None。kind ∈ {'paper','qmt'}。"""
    if BROKER == "paper":
        from paper_broker import PaperTrader
        pt = PaperTrader(PAPER_STATE, PAPER_INIT_CASH, COMMISSION, SLIPPAGE, LOT)
        return ("paper", pt)
    # qmt
    try:
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount
    except ImportError:
        print("[提示] 未安装 xtquant（miniQMT 未配置）→ 演示模式，仅打印计划。")
        return None
    trader = XtQuantTrader(QMT_PATH, SESSION_ID)
    trader.start()
    if trader.connect() != 0:
        print("[提示] 连接 miniQMT 失败（QMT 客户端是否已登录运行？）→ 演示模式。")
        return None
    acc = StockAccount(ACCOUNT_ID)
    trader.subscribe(acc)
    return ("qmt", (trader, acc))


def read_account(ctx, price):
    """返回 (总资产, {code: 持仓股数})。"""
    if ctx is None:
        return DEMO_TOTAL, {}
    kind, h = ctx
    if kind == "paper":
        h.set_prices(price)
        return h.total_asset(), h.positions()
    trader, acc = h
    asset = trader.query_stock_asset(acc)
    total = asset.total_asset if asset else DEMO_TOTAL
    positions = {}
    for p in trader.query_stock_positions(acc) or []:
        positions[p.stock_code.split(".")[0]] = p.volume
    return total, positions


# ---------------- 3) 生成调仓订单 ----------------
def _market(code):
    """补市场后缀：5/1 开头沪市(.SH)，其余(15/16/...)深市(.SZ)。"""
    return code + (".SH" if code[0] in "51" else ".SZ")


def build_orders(target, recent, total, positions, band=REBALANCE_BAND,
                 cant_buy_today=None, cant_sell_today=None):
    """
    对比目标权重与当前持仓，生成 [(code, 'BUY'/'SELL', 股数), ...]，先卖后买。
    再平衡阈值 band：对“调仓前后都持有、仅小幅漂移”的标的设静默区，偏离金额
    不足 band×总资产就不动，避免无谓换手；换信号（清仓/新建仓）照常执行。
    D2：cant_buy_today/cant_sell_today 是当日收盘封涨停/跌停的 code 集合——封板方向
    无法成交（券商本也会拒单），直接跳过该单、维持原仓，省无效挂单。
    """
    cant_buy_today = cant_buy_today or set()
    cant_sell_today = cant_sell_today or set()
    price = {c: arr[-1] for c, arr in recent.items()}   # 各标的现价 = 序列最后一个收盘
    # 目标股数 = 目标市值 / 现价，再向下取整到整百股。
    # 写法解析：total*w/px 是理论股数；// LOT 整除得“多少个100股”；再 *LOT 还原成股数。
    want = {}
    for code, w in target.items():
        px = price.get(code)
        if px:
            want[code] = int(total * w / px // LOT) * LOT

    sells, buys = [], []
    # set(positions) | set(want)：当前持仓代码 与 目标代码 的并集（| 是集合求并），
    # 这样“要清掉的旧仓”和“要新建的仓”都会被遍历到。
    for code in set(positions) | set(want):
        held = positions.get(code, 0)       # 当前持有股数（没有则0）
        tgt = want.get(code, 0)             # 目标股数（不在目标里则0）
        if tgt == held:
            continue                        # 已经一致，跳过
        # 双边都持有、且偏离金额不足 band×总资产 → 落在静默区，不动（省手续费）
        if held > 0 and tgt > 0:
            px = price.get(code, 0.0)
            if px and abs(tgt - held) * px < band * total:
                continue
        if tgt < held:
            if code in cant_sell_today:                 # D2：跌停卖不出，跳过留仓
                continue
            sells.append((code, "SELL", held - tgt))   # 目标比持有少 → 卖差额
        else:
            if code in cant_buy_today:                  # D2：涨停买不进，跳过
                continue
            buys.append((code, "BUY", tgt - held))      # 目标比持有多 → 买差额
    return sells + buys                     # 列表相加 = 先卖单后买单，保证先回笼现金


def place(ctx, code, side, shares):
    """下单：paper 同步模拟成交；qmt 走 xttrader 最新价委托。"""
    kind, h = ctx
    if kind == "paper":
        h.order(_market(code), side, shares)
        return
    from xtquant import xtconstant
    trader, acc = h
    direction = xtconstant.STOCK_BUY if side == "BUY" else xtconstant.STOCK_SELL
    trader.order_stock(acc, _market(code), direction, shares,
                       xtconstant.LATEST_PRICE, 0, "动量轮动", "")


# ---------------- 主流程 ----------------
def _label(code):
    return POOL.get(code, DEFENSE[1] if code == DEFENSE[0] else code)


# ---------------- 每月只调一次的闸 ----------------
def _this_month():
    return dt.date.today().strftime("%Y-%m")    # 当前年月，如 "2026-06"


def already_done_this_month():
    # 关掉了月度闸、或标记文件还不存在 → 视为本月没调过
    if not ONCE_PER_MONTH or not os.path.exists(MONTH_MARKER):
        return False
    # 读标记文件里存的年月，.strip() 去掉首尾空白/换行，与本月比较
    with open(MONTH_MARKER, "r", encoding="utf-8") as f:
        return f.read().strip() == _this_month()


def mark_done_this_month():
    # 把本月年月写进标记文件，下次同月运行就会被 already_done_this_month 拦下
    with open(MONTH_MARKER, "w", encoding="utf-8") as f:
        f.write(_this_month())


def is_month_end_trading_day():
    """
    今天是否为本月最后一个交易日（用 akshare 上交所交易日历判断）。
    与回测的"月末调仓"口径对齐：只有当天是本月最后一个交易日时才真正调仓。
    取不到日历时保守返回 True（不拦截，避免因数据源故障错过调仓）。
    """
    try:
        import akshare as ak
        import pandas as pd
        cal = pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"])  # 全年交易日（含未来）
        today = dt.date.today()
        this_month = cal[(cal.dt.year == today.year) & (cal.dt.month == today.month)]
        if this_month.empty:
            return True
        return today == this_month.max().date()    # 今天 == 本月最后一个交易日？
    except Exception as e:
        print("[warn] 取交易日历失败，跳过月末校验：", e)
        return True


def main():
    force = "force" in sys.argv[1:]               # python etf_momentum_live.py force 可强制重跑
    if MONTH_END_ONLY and not is_month_end_trading_day() and not force:
        print(f"[{BROKER}] 今天不是本月最后一个交易日，按月末调仓口径跳过。加 force 可强制运行。")
        return
    if already_done_this_month() and not force:
        print(f"[{BROKER}] 本月（{_this_month()}）已调仓，跳过。加 force 参数可强制重跑。")
        return

    codes = list(POOL) + [DEFENSE[0]]
    recent = get_recent_closes(codes, HISTORY_BARS)   # 取够 200 根，趋势过滤才不会静默失效
    price = {c: arr[-1] for c, arr in recent.items()}

    target, names = decide_targets(recent, trend_ma=TREND_MA)
    print(f"[{BROKER}] 调仓日:", dt.date.today(), " 目标持有:", names or "（可选标的不足，空仓）")
    if not target:
        return
    print("目标权重:", {c: round(w, 3) for c, w in target.items()})

    # D2：当日收盘封板的标的，对应方向无法成交（券商本也会拒单，这里提前省无效挂单）。
    #     实盘月末收盘后运行，recent 最后一个收盘即今日收盘，用环比近似涨跌停。
    cant_buy_today, cant_sell_today = set(), set()
    for c, arr in recent.items():
        if len(arr) >= 2 and arr[-2] > 0:
            lim = _limit(c)
            pct = arr[-1] / arr[-2] - 1.0
            if pct >= lim - 1e-4:
                cant_buy_today.add(c)
            elif pct <= -lim + 1e-4:
                cant_sell_today.add(c)
    if cant_buy_today or cant_sell_today:
        print(f"[D2] 当日封板：涨停买不进 {cant_buy_today or '无'} / 跌停卖不出 {cant_sell_today or '无'}（跳过对应单）")

    ctx = connect_trader()
    total, positions = read_account(ctx, price)
    print(f"账户总资产: {total:,.0f}  当前持仓: {positions or '无'}")

    orders = build_orders(target, recent, total, positions,
                          cant_buy_today=cant_buy_today, cant_sell_today=cant_sell_today)
    if not orders:
        print("已与目标一致，无需调仓。")
        mark_done_this_month()
        return

    print("\n调仓计划（先卖后买）:")
    for code, side, shares in orders:
        print(f"  {side:4} {code} {_label(code)} {shares} 股")

    kind = ctx[0] if ctx else None
    do_trade = ctx is not None and (kind == "paper" or not DRY_RUN)
    if not do_trade:
        print("\n[DRY_RUN] 未真正下单（不计入本月）。核对无误后改 DRY_RUN=False 并接通 miniQMT 再运行。")
        return

    for code, side, shares in orders:
        place(ctx, code, side, shares)
    mark_done_this_month()

    if kind == "paper":
        total2, pos2 = read_account(ctx, price)
        print(f"\n[paper] 已模拟成交。期末总资产: {total2:,.0f}  持仓: {pos2}")
        print(f"账户状态已保存到 {os.path.basename(PAPER_STATE)}（下次运行会接着用）。")
    else:
        print("\n已提交全部订单。请到 QMT 客户端核对成交。")


if __name__ == "__main__":
    main()
