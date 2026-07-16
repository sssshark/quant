# -*- coding: utf-8 -*-
"""
动量轮动 —— miniQMT 实盘骨架（与回测共用 cta.decide_targets）。

它做的事：
  1. 取最近 N 日收盘价；
  2. 调用 decide_targets()（和 bt.py 完全同一套选股逻辑）得到目标权重；
  3. 连接 miniQMT，读取账户总资产 + 当前持仓；
  4. 算出目标股数，生成「先卖后买」的调仓订单；
  5. DRY_RUN=True 时只打印计划、不下单；接通并核对无误后再改 False 真正下单。

★ 重要：
  - 需先在华泰开通 miniQMT/极简模式权限，并让 QMT 客户端在后台登录运行。
  - 实盘凭证（QMT_PATH / ACCOUNT_ID）不写进代码：从「环境变量 > 本地 live_config.json」
    读取（见 _load_live_secret），配置方法见 live_config.example.json。
  - hold_all 默认 true（等权全池+风控）：研究证明动量选股无可靠 alpha、跨市场甚至显著拖累
    （J1/J2/J5/G3，详见 IMPROVEMENTS），故 2026-07-12 起默认改为 true、与实盘口径一致；
    改 live_config.json 的 "hold_all" 可覆盖（false=动量轮动，诊断/对照用）。
  - 真金白银交易请你本人确认后执行；建议先在模拟账户跑通。
  - 程序化交易需按交易所要求报备。

运行：
  python live.py          # 默认 DRY_RUN，只打印调仓计划
"""
import os
import sys
import json
import time
import datetime as dt

# Windows 系统代理（Clash 等）会把本进程 HTTP 请求路由到代理出口，对 tushare-relay
# 这种 http 明文中转不友好，也叠加东财风控。patch getproxies 让本进程内 requests 直连，
# 不影响 xtquant（它是本地 IPC 不是 HTTP）。注：这只能消除"代理"这一层——东财对直连
# 也会 RST（实测 akshare 直连仍被风控），故 akshare 仅作兜底，主力仍是 tushare relay。
import urllib.request
urllib.request.getproxies = lambda: {}

from cta import (POOL, DEFENSE, MAX_LOOKBACK, COMMISSION, SLIPPAGE,
                           TREND_MA, HOLD_ALL, _limit, decide_targets)

# 取数长度要同时够"动量回看"和"大盘趋势均线"两者，取较大值（趋势用 200 日均线 > 126）
HISTORY_BARS = max(MAX_LOOKBACK, TREND_MA)

# ---------------- 你需要改的配置 ----------------
BROKER = "qmt"                                   # "paper"=本地模拟账户跑通；"qmt"=接华泰 miniQMT
DRY_RUN = False                                  # 仅对 qmt 生效：True=只打印不下单；paper 始终模拟成交
EQUITY_CAP = 30_000.0                             # 策略可用资金上限（None=用账户实际总资产）。设了之后，
                                                  # 即使账户有 100 万，策略按 2 万 × 权重分配，其余资金不动。
                                                  # 用途：实盘小资金测试、子账户隔离。注意：read_account 仍读真实持仓，
                                                  # 所以"账户持仓 > cap×权重"时仍会卖出多余部分。
LOT = 100                                        # ETF 最小交易单位（股）
REBALANCE_BAND = 0.05                             # 再平衡静默区：偏离<5%总资产的小漂移不调仓
RECONCILE_THR = 0.10                              # H1 对账阈值：偏离>10%总资产告警（>band，避免整手/静默区固有偏离误报）
ONCE_PER_MONTH = True                             # True=每月只调一次（适合每日定时触发）
MONTH_END_ONLY = True                             # True=只在"本月最后一个交易日"调仓，与回测月末口径对齐（force 可绕过）
# os.path.dirname(__file__) 是“本脚本所在目录”，拼上文件名 → 标记文件放在脚本旁边，
# 这样不管从哪个目录运行脚本，都能找到同一个文件。
MONTH_MARKER = os.path.join(os.path.dirname(__file__), "last_rebalance.txt")

# --- paper 模拟账户 ---
PAPER_STATE = os.path.join(os.path.dirname(__file__), "paper_account.json")  # 同理，放脚本旁边
PAPER_INIT_CASH = 1_000_000.0                    # 模拟账户初始资金

# --- 实盘凭证（改进项 H2：不硬编码、不入仓库） ---
# QMT_PATH/ACCOUNT_ID 是机器/账户相关的敏感配置，从「环境变量 > 本地 live_config.json」读取，
# 不写进源码、不入 git（live_config.json 已在 .gitignore）。两者都没配时回退占位值，
# connect_trader() 会识别占位并拒绝连真实账户（保持 paper/演示，防误操作）。
_QMT_PATH_DEFAULT = r"C:\华泰证券QMT\userdata_mini"   # 占位：未配置时用，触发"未配置"提示
_ACCOUNT_ID_DEFAULT = "你的资金账号"                   # 同上


def _load_live_secret(key, env_var, default):
    """读实盘凭证，优先级：环境变量 env_var > 本地 live_config.json 的 key > default 占位。"""
    val = os.environ.get(env_var)
    if not val:                                  # 环境变量没给 → 试本地配置文件
        cfg = os.path.join(os.path.dirname(__file__), "live_config.json")
        if os.path.exists(cfg):
            try:
                with open(cfg, "r", encoding="utf-8") as f:
                    val = json.load(f).get(key)
            except Exception as e:
                print(f"[warn] 读取 live_config.json 失败（{e}），用占位值。")
    return val or default                        # 都没给 → 占位（连真实账户前会被拦下）


QMT_PATH = _load_live_secret("qmt_path", "QMT_PATH", _QMT_PATH_DEFAULT)
ACCOUNT_ID = _load_live_secret("account_id", "QMT_ACCOUNT_ID", _ACCOUNT_ID_DEFAULT)
SESSION_ID = int(dt.datetime.now().timestamp())  # 任意整数，连接会话号
DEMO_TOTAL = 100_000.0                            # 兜底：完全连不上时的假定总资产


def _load_hold_all():
    """读部署模式 hold_all:live_config.json 的 "hold_all" 键 > cta.HOLD_ALL 默认。
    True = 等权全池+风控(默认;J1/J2 证两市动量选股均无可靠 alpha,选股还略抬高回撤);
    False = 动量轮动(诊断/对照用)。非敏感的策略开关,故只读本地配置文件(不走环境变量)。"""
    cfg = os.path.join(os.path.dirname(__file__), "live_config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                v = json.load(f).get("hold_all")
                if v is not None:
                    # 只接受真 JSON bool（true/false）。⚠ 旧 `bool(v)` 对字符串
                    # "false"/"0" 返回 True（非空串皆真）→ 用户误把值加引号会静默
                    # 跑成"等权全池"。非 bool 值一律告警并回退默认，杜绝误判。
                    if isinstance(v, bool):
                        return v
                    print(f"[warn] live_config.json 的 hold_all 应为 true/false(bool)，"
                          f"当前为 {v!r}({type(v).__name__})，用默认 {HOLD_ALL}。")
                    return HOLD_ALL
        except Exception as e:
            print(f"[warn] 读 live_config.json 的 hold_all 失败（{e}），用默认 {HOLD_ALL}。")
    return HOLD_ALL


# ---------------- 1) 行情：最近 N 日收盘 ----------------
def get_recent_closes(codes, n):
    """
    取每只最近 n+1 根前复权收盘（够算 n 日动量 + 200 日趋势均线）。
    优先 tushare：复用 engine.load_real（主力源、含 relay 关键标的兜底、且与回测同源
    消除"实盘东财/回测 tushare"的复权口径漂移）；tushare 缺标或失败时回退 akshare 东财。
    实盘接通 QMT 后可改 xtdata.get_market_data_ex 直接从 QMT 取，少一个数据源依赖。
    """
    # 优先 tushare：复用 engine 全量前复权加载，取每只尾部 n+1 根
    try:
        import engine as e
        loaded = e.load_real(with_limits=False)
        px = loaded[0] if isinstance(loaded, tuple) else loaded
        out = {}
        for c in codes:
            if c in px.columns:
                vals = px[c].dropna().tolist()
                if len(vals) >= 2:
                    out[c] = vals[-(n + 1):]
        miss = [c for c in codes if c not in out]
        if not miss:
            return out                                  # tushare 全取到 → 用它（与回测同源）
        print(f"[live] tushare 缺 {miss}，akshare 补")
    except Exception as exc:
        print(f"[live] tushare 加载失败（{type(exc).__name__}: {exc}），回退 akshare 东财")
        out = {}
    # akshare 回退（东财可达时；东财不可达会逐标的抛并跳过）
    import akshare as ak
    # 往前多取日历日：要 n 个"交易日"，周末/节假日占位，×3+余量保证截后够 n+1 根
    start = (dt.date.today() - dt.timedelta(days=n * 3 + 30)).strftime("%Y%m%d")
    end = dt.date.today().strftime("%Y%m%d")
    for c in codes:
        if c in out:
            continue
        try:
            df = ak.fund_etf_hist_em(symbol=c, period="daily",
                                     start_date=start, end_date=end, adjust="qfq")
            if len(df):
                out[c] = df["收盘"].astype(float).tolist()[-(n + 1):]
        except Exception as exc:
            print(f"[live] {c} akshare 也失败：{exc}")
    return out


# ---------------- 2) 连接券商（paper 本地模拟 / qmt 真实） ----------------
def connect_trader():
    """返回 (kind, handle)；连不上则 None。kind ∈ {'paper','qmt'}。"""
    if BROKER == "paper":
        from paper_broker import PaperTrader
        pt = PaperTrader(PAPER_STATE, PAPER_INIT_CASH, COMMISSION, SLIPPAGE, LOT)
        return ("paper", pt)
    # qmt
    if QMT_PATH == _QMT_PATH_DEFAULT or ACCOUNT_ID == _ACCOUNT_ID_DEFAULT:
        # H2：凭证仍是占位值（没配环境变量/live_config.json）→ 不连真实账户，降级演示。
        print("[提示] 未配置实盘凭证 QMT_PATH / QMT_ACCOUNT_ID（环境变量或 live_config.json），"
              "仍是占位值 → 演示模式。配置方法见 live_config.example.json。")
        return None
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
    """返回 (总资产, {code: 持仓股数})。

    若设了 EQUITY_CAP，返回的总资产 = min(账户实际总资产, EQUITY_CAP)。
    持仓仍读真实值——这样 build_orders 算"目标 - 现持"差额时，能正确卖出超额仓位。
    """
    if ctx is None:
        return DEMO_TOTAL, {}
    kind, h = ctx
    if kind == "paper":
        h.set_prices(price)
        total = h.total_asset()
    else:
        trader, acc = h
        asset = trader.query_stock_asset(acc)
        total = asset.total_asset if asset else DEMO_TOTAL
    if EQUITY_CAP and total > EQUITY_CAP:
        total = EQUITY_CAP
    if kind == "paper":
        return total, h.positions()
    positions = {}
    for p in trader.query_stock_positions(acc) or []:
        positions[p.stock_code.split(".")[0]] = p.volume
    return total, positions


# ---------------- 3) 生成调仓订单 ----------------
def _market(code):
    """补市场后缀：5xx/6xx 沪市(.SH)，1xx/0xx/3xx 深市(.SZ)。

    ⚠ 旧实现 `code[0] in "51"` 是子串包含不是集合判断——`"1" in "51"`=True，
    致 159915(创业板)/159941(跨境)等 1xx 深市代码被误判 .SH。paper 路径
    `paper_broker.order` 内部 `code.split('.')[0]` 剥后缀故从不暴露；切到 qmt
    实盘会向上交所下深市代码→拒单。显式按首字符前缀判断（5/6→沪，其余→深）。"""
    return code + (".SH" if code[0] in ("5", "6") else ".SZ")


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
    # 目标股数 = 目标市值 / 现价，再四舍五入到整百股（D5：替代原向下取整，避免长期欠仓累积；
    #   cash_buffer=0.99 留 1% 现金兜底四舍五入的小幅超买）。
    # 写法解析：total*w/px 是理论股数；/LOT 得“多少个100股”；+0.5 再 int 实现四舍五入（五入）。
    want = {}
    for code, w in target.items():
        px = price.get(code)
        if px:
            want[code] = int(total * w / px / LOT + 0.5) * LOT

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

    # 买单现金兜底（deepcode-3 修复）：四舍五入到整百（D5）每只最多多买 50 股，top3 同时
    # 五入可超 1% cash_buffer 兜底，致现金为负（paper）或部分成交（qmt）。先卖后买口径下，
    # 可用现金 = 当前现金 + 卖单回笼（扣滑点）；逐笔买单按可用现金截断股数（向下取整到 LOT），
    # 超出部分自然落现金——宁可小幅欠配，不超支。
    if buys:
        cash = total - sum(positions.get(c, 0) * price.get(c, 0.0)
                           for c in set(positions) | set(want))
        for code, side, sh in sells:                     # 卖单回笼（扣滑点近似）
            cash += sh * price.get(code, 0.0) * (1 - SLIPPAGE)
        capped = []
        for code, side, sh in buys:
            px = price.get(code, 0.0)
            if px <= 0:
                continue
            affordable = int(cash / (px * (1 + SLIPPAGE) * LOT)) * LOT   # 含滑点的可买整百
            sh = min(sh, affordable)
            cash -= sh * px * (1 + SLIPPAGE)            # 扣减已用现金
            if sh > 0:
                capped.append((code, side, sh))
        buys = capped
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


def _weights_from_positions(positions, total, price):
    """实盘持仓 {code: 股数} → 权重 {code: 市值/总资产}，含 '__cash__' 现金币种（H1 对账用）。"""
    w = {}
    for c, vol in positions.items():
        px = price.get(c, 0.0)
        if px and total > 0:
            w[c] = w.get(c, 0.0) + vol * px / total
    if total > 0:
        w["__cash__"] = max(0.0, 1.0 - sum(w.values()))     # 现金 = 1 − 已投（≥0）
    return w


def reconcile(target, positions, total, price, thr=RECONCILE_THR, when=""):
    """H1 回测-实盘对账（reconciliation）：把模型目标权重（target）与实盘实际持仓（positions）
    换算到同一权重口径后逐币种比，偏离 ≥ thr 告警。机构标配——回测再好，实盘偏离就白搭。
      when='调仓前': 实盘累积持仓 vs 当前模型 target → 抓跨月漂移 / 上次调仓未完整执行 / 分红或手工操作。
      when='调仓后': 刚执行完的持仓 vs 本次 target → 抓涨跌停/拒单致下单未全成交（实盘最关键）。
    只读对比、不改交易、不抛异常（对账失败不应阻断调仓）。返回最大绝对偏离，便于落日志。"""
    if total <= 0:
        print(f"[H1 对账·{when}] 总资产≤0，跳过")
        return 0.0
    actual = _weights_from_positions(positions, total, price)
    tgt = dict(target)
    tgt["__cash__"] = max(0.0, 1.0 - sum(tgt.values()))        # 目标也补现金币种，口径对齐
    diffs = [(c, tgt.get(c, 0.0), actual.get(c, 0.0),
              actual.get(c, 0.0) - tgt.get(c, 0.0))
             for c in set(tgt) | set(actual)]
    diffs.sort(key=lambda x: -abs(x[3]))
    max_abs = abs(diffs[0][3]) if diffs else 0.0
    tag = f"[H1 对账·{when}] " if when else "[H1 对账] "
    over = [(c, tw, aw, d) for c, tw, aw, d in diffs if abs(d) >= thr]
    if over:
        print(f"{tag}实盘 vs 模型偏离 ≥ {thr*100:.0f}% 总资产：")
        for c, tw, aw, d in over:
            name = "__现金__" if c == "__cash__" else _label(c)
            print(f"    {str(name):8s} 应有{tw:6.1%}  实际{aw:6.1%}  偏离{d*100:+5.1f}pp "
                  f"({'超配' if d > 0 else '低配'})")
        print("    → 可能：跨月价格漂移 / 上次调仓未完整执行 / 分红或手工操作 / 本次涨跌停跳单")
    else:
        print(f"{tag}实盘与模型一致（最大偏离 {max_abs*100:.1f}% < 阈值 {thr*100:.0f}%）")
    return max_abs


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
    force = "force" in sys.argv[1:]               # python live.py force 可强制重跑
    if MONTH_END_ONLY and not is_month_end_trading_day() and not force:
        print(f"[{BROKER}] 今天不是本月最后一个交易日，按月末调仓口径跳过。加 force 可强制运行。")
        return
    if already_done_this_month() and not force:
        print(f"[{BROKER}] 本月（{_this_month()}）已调仓，跳过。加 force 参数可强制重跑。")
        return

    codes = list(POOL) + [DEFENSE[0]]
    recent = get_recent_closes(codes, HISTORY_BARS)   # 取够 200 根，趋势过滤才不会静默失效
    price = {c: arr[-1] for c, arr in recent.items()}

    hold_all = _load_hold_all()
    mode_name = "等权全池+风控(不选股,J1/J2 稳健版)" if hold_all else "动量轮动(top3+绝对动量)"
    print(f"[模式] {mode_name}")
    if not hold_all:
        print("[⚠ 实盘建议] 当前为动量选股模式。项目研究(J1/J2/J5/G3)证明:动量选股相对'等权全池+风控'")
        print("            无可靠 alpha(A股 bootstrap P=0.148、美股 P=0.999 显著为负)、PBO=0.67 过拟合、")
        print("            冻结期选股夏普 0.74 < 等权+风控 0.79。真金白银实盘建议 hold_all=true,")
        print("            改 live_config.json 的 hold_all 为 true 即可。详见 IMPROVEMENTS J1/J2/J5。")
    target, names = decide_targets(recent, trend_ma=TREND_MA, hold_all=hold_all)
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
    reconcile(target, positions, total, price, when="调仓前")   # H1：调仓前先对账（抓跨月漂移/上次未完整执行）

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

    if kind == "paper":
        mark_done_this_month()                                 # paper 同步模拟成交，必达成
        total2, pos2 = read_account(ctx, price)
        print(f"\n[paper] 已模拟成交。期末总资产: {total2:,.0f}  持仓: {pos2}")
        reconcile(target, pos2, total2, price, when="调仓后")   # H1：调仓后对账（paper 全成交，应≈一致）
        print(f"账户状态已保存到 {os.path.basename(PAPER_STATE)}（下次运行会接着用）。")
    else:
        # qmt：place 是异步委托，要等几秒让服务端撮合，再读真实持仓算偏离。
        # 偏离 < 阈值 = 实际达成目标 → 写月度标记；否则不写，下次运行会按"目标-当前持仓"
        # 差额自动补单。这避免了"place 都发了就误标记为完成"的设计 bug。
        print("\n[等 5 秒] 给 QMT 服务端撮合时间...")
        time.sleep(5)
        total2, pos2 = read_account(ctx, price)
        max_diff = reconcile(target, pos2, total2, price, when="调仓后")
        if max_diff < RECONCILE_THR:
            mark_done_this_month()
            print(f"偏离 {max_diff*100:.1f}% < 阈值 {RECONCILE_THR*100:.0f}%，标记本月已完成。")
        else:
            print(f"\n[警告] 偏离 {max_diff*100:.1f}% ≥ 阈值 {RECONCILE_THR*100:.0f}%，有单未完整成交。")
            print("不写月度标记 → 下次运行会自动算差额补单。")
            print("如确认本次不补，请手工核对 QMT 客户端「交易」面板后用 force 参数强制重跑。")
        print("请到 QMT 客户端「交易」面板核对成交明细。")


if __name__ == "__main__":
    main()
